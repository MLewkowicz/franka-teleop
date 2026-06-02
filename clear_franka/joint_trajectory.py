import numpy as np
from functools import lru_cache
from scipy.interpolate import make_interp_spline
from typing import Tuple


def _rdp_indices(waypts: np.ndarray, tol: float) -> np.ndarray:
    """Iterative Ramer-Douglas-Peucker simplification in joint space.

    Returns the indices of the waypoints to keep so that the maximum
    perpendicular deviation from any original waypoint to the simplified
    piecewise-linear path is below `tol` (L2 across joints).
    """
    n = len(waypts)
    if n <= 2:
        return np.arange(n)

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo <= 1:
            continue

        p_lo = waypts[lo]
        chord = waypts[hi] - p_lo
        chord_sq = float(np.dot(chord, chord))
        diff = waypts[lo + 1 : hi] - p_lo

        if chord_sq < 1e-24:
            dists = np.linalg.norm(diff, axis=1)
        else:
            t = np.clip((diff @ chord) / chord_sq, 0.0, 1.0)
            dists = np.linalg.norm(diff - t[:, None] * chord, axis=1)

        pivot_rel = int(np.argmax(dists))
        if dists[pivot_rel] > tol:
            pivot = lo + 1 + pivot_rel
            keep[pivot] = True
            stack.append((lo, pivot))
            stack.append((pivot, hi))

    return np.where(keep)[0]


@lru_cache(maxsize=64)
def _compute_deform_H(n):
    """
    Compute deformation basis H for width n (cached).

    H is the impulse response of the minimum-acceleration deformation.
    Only depends on n, so we cache it to avoid repeated O(n³) matrix inversions.
    """
    A = np.zeros((n+2, n))
    np.fill_diagonal(A, 1)
    for i in range(n):
        A[i+1, i] = -2
        A[i+2, i] = 1
    R = A.T @ A
    Rinv = np.linalg.inv(R)
    Uh = np.zeros((n, 1))
    Uh[0] = 1
    return Rinv @ Uh * (np.sqrt(n) / np.linalg.norm(Rinv @ Uh))


class Trajectory(object):
    """
    This class represents a trajectory object, supporting operations such as
    interpolating, downsampling waypoints, upsampling waypoints, etc.
    """
    def __init__(self, waypts, waypts_time):
        self.waypts = np.asarray(waypts)
        self.waypts_time = np.asarray(waypts_time, dtype=float)
        self.num_waypts = len(waypts)
        self.num_joints = self.waypts.shape[1]

    def waypoint_index_at(self, t) -> int:
        """
        Get the nearest waypoint index for a given time.

        Params:
            t [float] -- The time to find the nearest waypoint for.

        Returns:
            int -- Index of the nearest waypoint, clamped to valid range.
        """
        if t <= self.waypts_time[0]:
            return 0
        if t >= self.waypts_time[-1]:
            return self.num_waypts - 1
        idx = np.searchsorted(self.waypts_time, t, side='right') - 1
        # Return whichever neighbor is closer
        if idx < self.num_waypts - 1:
            if abs(t - self.waypts_time[idx + 1]) < abs(t - self.waypts_time[idx]):
                idx += 1
        return idx

    @property
    def _spline(self):
        """Lazily build and cache a clamped cubic spline through waypoints."""
        if not hasattr(self, '_cached_spline'):
            if self.num_waypts >= 4:
                self._cached_spline = make_interp_spline(
                    self.waypts_time, self.waypts, k=3, bc_type='clamped'
                )
            else:
                # Fall back to linear for too few points
                self._cached_spline = make_interp_spline(
                    self.waypts_time, self.waypts, k=1
                )
        return self._cached_spline

    def interpolate(self, t):
        """
        Gets the desired position along trajectory at time t using cubic spline.

        Params:
            t [float] -- The time of desired interpolation along path.

        Returns:
            waypt [array] -- Interpolated waypoint at time t.
        """
        assert len(self.waypts) > 1, "Cannot interpolate a one-waypoint trajectory."

        t_clamped = np.clip(t, self.waypts_time[0], self.waypts_time[-1])
        waypt = self._spline(t_clamped)

        return np.array(waypt).reshape((self.num_joints, 1))

    def deform(self, u_h, t, alpha, n):
        """
        Deforms the next n waypoints of the trajectory

        Params:
            u_h -- Deformation torque (num_joints, 1) or (num_joints,).
            t [float] -- The time of deformation.
            alpha -- Alpha deformation parameter (magnitude).
            n -- Width of deformation (number of waypoints affected).

        Returns:
            trajectory -- Deformed trajectory.
        """
        H = _compute_deform_H(n)
        assert len(u_h) == self.num_joints, "Deformation torque u_h has incorrect shape."
        deform_waypt_idx = self.waypoint_index_at(t)
        if (deform_waypt_idx + n) > self.num_waypts:
            print("Deforming too close to end. Returning same trajectory")
            return Trajectory(self.waypts.copy(), self.waypts_time)

        # Vectorized: H is (n, 1), u_h is (num_joints,) or (num_joints, 1)
        # Result gamma is (n, num_joints)
        u_h_flat = np.asarray(u_h).flatten()
        gamma = alpha * (H @ u_h_flat[np.newaxis, :])  # (n, 1) @ (1, num_joints) -> (n, num_joints)

        waypts_deform = self.waypts.copy()
        waypts_deform[deform_waypt_idx : deform_waypt_idx + n, :] += gamma
        return Trajectory(waypts_deform, self.waypts_time)

    def sample_with_velocities(self, hz: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample trajectory at a control rate, returning positions and velocities.

        Params:
            hz [int] -- Control rate in Hz (default 1000).

        Returns:
            positions [ndarray] -- (num_samples, num_joints) resampled positions.
            velocities [ndarray] -- (num_samples, num_joints) resampled velocities.
        """
        assert self.num_waypts > 1, "Cannot resample a one-waypoint trajectory."

        duration = self.waypts_time[-1] - self.waypts_time[0]
        num_samples = max(int(hz * duration) + 1, 2)
        t_resampled = np.linspace(self.waypts_time[0], self.waypts_time[-1], num_samples)

        positions = self._spline(t_resampled)
        velocities = self._spline.derivative()(t_resampled)

        return positions, velocities

    def trim(self, time_window: float = 0.3, threshold: float = 0.01) -> "Trajectory":
        """
        Trim stationary segments from the start and end of the trajectory.

        Removes waypoints where the robot didn't move significantly, which often
        occurs when the human pauses before/after demonstration.

        Params:
            time_window [float] -- Time window in seconds to check for movement (default 0.3).
            threshold [float] -- Minimum displacement to consider as movement (default 0.01).

        Returns:
            Trajectory -- New trajectory with stationary ends removed.
        """
        assert self.num_waypts > 1, "Cannot trim a one-waypoint trajectory."

        times = np.array(self.waypts_time)
        duration = times[-1] - times[0]
        hz = (self.num_waypts - 1) / duration if duration > 0 else 1.0
        n_samples = max(1, int(round(time_window * hz)))

        N = self.num_waypts
        lo = 0
        hi = N - 1

        # Find first index where robot starts moving
        while (lo + n_samples < N) and (np.linalg.norm(self.waypts[lo + n_samples] - self.waypts[lo]) < threshold):
            lo += 1

        # Find last index where robot stops moving
        while (hi - n_samples >= 0) and (np.linalg.norm(self.waypts[hi] - self.waypts[hi - n_samples]) < threshold):
            hi -= 1

        # Ensure we have at least 2 waypoints
        if lo >= hi:
            lo = max(0, lo - 1)
            hi = min(N - 1, lo + 1)

        trimmed_waypts = self.waypts[lo:hi + 1, :]
        trimmed_times = times[lo:hi + 1]

        return Trajectory(trimmed_waypts, trimmed_times)

    def retime(self, max_vel: np.ndarray, max_accel: np.ndarray,
               sample_uniform: bool = False) -> "Trajectory":
        """
        Retime trajectory using TOPPRA time-optimal path parameterization.

        Computes the fastest traversal of this geometric path subject to
        velocity and acceleration limits.

        Params:
            max_vel [ndarray] -- Maximum velocity per joint (num_joints,)
            max_accel [ndarray] -- Maximum acceleration per joint (num_joints,)
            sample_uniform [bool] -- If True, sample TOPPRA's trajectory at uniform
                                    time intervals (avoids spline overshoot). If False,
                                    preserve original waypoints with new timestamps.

        Returns:
            Trajectory -- Retimed trajectory. If sample_uniform=False, returns same
                         waypoints with new timestamps. If sample_uniform=True, returns
                         waypoints sampled from TOPPRA's trajectory at uniform time intervals.
        """
        assert self.num_waypts > 1, "Cannot retime a one-waypoint trajectory."

        from scipy.optimize import minimize_scalar
        import toppra as ta
        import toppra.constraint as ta_constraint
        import toppra.algorithm as ta_algorithm

        max_vel = np.asarray(max_vel)
        max_accel = np.asarray(max_accel)

        # TOPPRA expects a geometric path parameter, not the demonstrated time.
        # Using timestamps here makes the retimer inherit the original timing and
        # can make compute_trajectory fail on slow/noisy demonstrations.
        path_param = np.linspace(0.0, 1.0, self.num_waypts)
        path = ta.SplineInterpolator(path_param, self.waypts, bc_type='clamped')

        # Symmetric joint limits: (dof, 2) with [-limit, +limit]
        vlim = np.column_stack([-max_vel, max_vel])
        alim = np.column_stack([-max_accel, max_accel])

        constraints = [
            ta_constraint.JointVelocityConstraint(vlim),
            ta_constraint.JointAccelerationConstraint(alim),
        ]

        instance = ta_algorithm.TOPPRA(constraints, path)
        jnt_traj = instance.compute_trajectory(0, 0)

        if jnt_traj is None:
            raise RuntimeError("TOPPRA failed to compute a feasible retimed trajectory")

        duration = jnt_traj.duration

        if sample_uniform:
            # Sample TOPPRA's trajectory directly at uniform time intervals
            uniform_times = np.linspace(0, duration, self.num_waypts)
            new_waypts = np.array([jnt_traj(t) for t in uniform_times])
            return Trajectory(new_waypts, uniform_times)

        # Find time at each waypoint by locating when the trajectory
        # passes through each waypoint position. The path is traversed
        # monotonically so each waypoint has a unique time.
        new_times = np.zeros(self.num_waypts)
        new_times[-1] = duration

        for i in range(1, self.num_waypts - 1):
            t_lo = new_times[i - 1]

            def obj(t, target=self.waypts[i]):
                return np.linalg.norm(np.array(jnt_traj(t)) - target)

            result = minimize_scalar(obj, bounds=(t_lo, duration), method='bounded')
            new_times[i] = result.x

        return Trajectory(self.waypts.copy(), new_times)

    def resample_uniform(self, num_waypts: int = None, hz: float = None,
                         linear: bool = False) -> "Trajectory":
        """
        Resample trajectory at uniformly spaced time intervals.

        Redistributes waypoints along the path so they are equally spaced
        in time. Useful after retime() to get waypoints that reflect where
        the robot spends time (denser in slow regions).

        Params:
            num_waypts [int] -- Number of output waypoints (default: same as current).
                               Mutually exclusive with hz.
            hz [float] -- Sample rate in Hz. If provided, num_waypts is computed
                         from the trajectory duration. Mutually exclusive with num_waypts.
            linear [bool] -- If True, use linear interpolation (k=1 spline).
                            If False (default), use cubic spline interpolation.

        Returns:
            Trajectory -- New trajectory with uniform time spacing.
        """
        assert self.num_waypts > 1, "Cannot resample a one-waypoint trajectory."
        assert not (num_waypts is not None and hz is not None), \
            "Cannot specify both num_waypts and hz"

        total_duration = self.waypts_time[-1] - self.waypts_time[0]
        if total_duration <= 0:
            assert False

        # Determine number of waypoints
        if hz is not None:
            num_waypts = max(int(hz * total_duration) + 1, 2)
        elif num_waypts is None:
            num_waypts = self.num_waypts

        uniform_times = np.linspace(self.waypts_time[0], self.waypts_time[-1], num_waypts)

        if linear:
            # Linear interpolation (k=1)
            linear_spline = make_interp_spline(self.waypts_time, self.waypts, k=1)
            new_waypts = linear_spline(uniform_times)
        else:
            # Cubic spline (default)
            new_waypts = self._spline(uniform_times)

        return Trajectory(new_waypts, uniform_times)

    def simplify(self, tol: float) -> "Trajectory":
        """Simplify path using Ramer-Douglas-Peucker in joint space.

        Keeps only the waypoints needed so the maximum perpendicular deviation
        from any original waypoint to the simplified piecewise-linear path is
        below `tol` radians (L2 across joints). Returned waypts_time values are
        exact members of self.waypts_time, serving as source-time indices into
        the original dense trajectory.
        """
        indices = _rdp_indices(self.waypts, tol)
        return Trajectory(self.waypts[indices], self.waypts_time[indices])

    def smooth(self, max_vel: np.ndarray, max_accel: np.ndarray, max_jerk: np.ndarray,
               dt: float = 0.001) -> "Trajectory":
        """
        Generate a dense, smooth trajectory using Ruckig to track a moving target.

        Uses an online tracking approach: a reference target moves along the original
        trajectory, and Ruckig smoothly chases it. This naturally produces smooth
        motion without needing to specify intermediate velocities.

        Params:
            max_vel [ndarray] -- Maximum velocity per joint (num_joints,)
            max_accel [ndarray] -- Maximum acceleration per joint (num_joints,)
            max_jerk [ndarray] -- Maximum jerk per joint (num_joints,)
            dt [float] -- Time step for trajectory output (default 1ms)

        Returns:
            Trajectory -- Dense trajectory with Ruckig's actual motion profile.
        """
        assert self.num_waypts > 1, "Cannot smooth a one-waypoint trajectory."

        from ruckig import InputParameter, OutputParameter, Result, Ruckig

        max_vel = np.asarray(max_vel)
        max_accel = np.asarray(max_accel)
        max_jerk = np.asarray(max_jerk)

        ref_duration = self.waypts_time[-1] - self.waypts_time[0]

        all_positions = [self.waypts[0].copy()]
        all_times = [0.0]

        otg = Ruckig(self.num_joints, dt)
        inp = InputParameter(self.num_joints)
        out = OutputParameter(self.num_joints)

        inp.current_position = self.waypts[0].tolist()
        inp.current_velocity = [0.0] * self.num_joints
        inp.current_acceleration = [0.0] * self.num_joints
        inp.max_velocity = max_vel.tolist()
        inp.max_acceleration = max_accel.tolist()
        inp.max_jerk = max_jerk.tolist()

        ref_time = 0.0
        current_time = 0.0
        max_iterations = int((ref_duration * 3) / dt) + 1000  # Safety limit

        for _ in range(max_iterations):
            # Advance reference target along path
            ref_time = min(ref_time + dt, ref_duration)
            target = self.interpolate(self.waypts_time[0] + ref_time).flatten()

            inp.target_position = target.tolist()
            inp.target_velocity = [0.0] * self.num_joints

            result = otg.update(inp, out)
            current_time += dt
            all_positions.append(np.array(out.new_position))
            all_times.append(current_time)
            out.pass_to_input(inp)

            # Stop when reference reached end and Ruckig finished
            if ref_time >= ref_duration and result == Result.Finished:
                break

        return Trajectory(np.array(all_positions), np.array(all_times))
