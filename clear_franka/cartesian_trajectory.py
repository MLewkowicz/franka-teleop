from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


class CartesianTrajectory:
    """Time-indexed Cartesian pose trajectory.

    Translation is interpolated in R3. Orientation is interpolated on SO3 using
    RotationSpline by default, with Slerp available for piecewise geodesic
    interpolation.
    """

    def __init__(
        self,
        positions: np.ndarray,
        rotations: Rotation | np.ndarray,
        waypts_time: np.ndarray,
        *,
        smooth_orientation: bool = True,
    ):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.waypts_time = np.asarray(waypts_time, dtype=np.float64)
        self.num_waypts = len(self.positions)

        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {self.positions.shape}")
        if self.waypts_time.shape != (self.num_waypts,):
            raise ValueError(
                f"waypts_time must have shape ({self.num_waypts},), got {self.waypts_time.shape}"
            )
        if self.num_waypts < 2:
            raise ValueError("Cannot interpolate a one-waypoint Cartesian trajectory.")
        if np.any(np.diff(self.waypts_time) <= 0):
            raise ValueError("waypts_time must be strictly increasing.")

        if isinstance(rotations, Rotation):
            self.rotations = rotations
        else:
            rot_mats = np.asarray(rotations, dtype=np.float64)
            if rot_mats.shape != (self.num_waypts, 3, 3):
                raise ValueError(
                    f"rotations must have shape (N, 3, 3), got {rot_mats.shape}"
                )
            self.rotations = Rotation.from_matrix(rot_mats)
        if len(self.rotations) != self.num_waypts:
            raise ValueError("positions, rotations, and waypts_time must have the same length.")

        self.smooth_orientation = bool(smooth_orientation)

    @classmethod
    def from_euler_xyz(
        cls,
        waypts: np.ndarray,
        waypts_time: np.ndarray,
        *,
        euler_to_matrix_fn=None,
        smooth_orientation: bool = True,
    ) -> "CartesianTrajectory":
        """Build from waypoints shaped (N, 6): xyz + XYZ Euler angles."""
        waypts = np.asarray(waypts, dtype=np.float64)
        if waypts.ndim != 2 or waypts.shape[1] < 6:
            raise ValueError(f"waypts must have shape (N, >=6), got {waypts.shape}")

        if euler_to_matrix_fn is None:
            rotations = Rotation.from_euler("XYZ", waypts[:, 3:6])
        else:
            rotations = Rotation.from_matrix(
                np.stack([euler_to_matrix_fn(euler) for euler in waypts[:, 3:6]], axis=0)
            )
        return cls(
            waypts[:, :3],
            rotations,
            waypts_time,
            smooth_orientation=smooth_orientation,
        )

    @classmethod
    def from_transforms(
        cls,
        transforms: np.ndarray,
        waypts_time: np.ndarray,
        *,
        smooth_orientation: bool = True,
    ) -> "CartesianTrajectory":
        """Build from homogeneous transforms shaped (N, 4, 4)."""
        transforms = np.asarray(transforms, dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(f"transforms must have shape (N, 4, 4), got {transforms.shape}")
        return cls(
            transforms[:, :3, 3],
            transforms[:, :3, :3],
            waypts_time,
            smooth_orientation=smooth_orientation,
        )

    @property
    def duration(self) -> float:
        return float(self.waypts_time[-1] - self.waypts_time[0])

    @property
    def _position_spline(self):
        if not hasattr(self, "_cached_position_spline"):
            k = 3 if self.num_waypts >= 4 else 1
            kwargs = {"bc_type": "clamped"} if k == 3 else {}
            self._cached_position_spline = make_interp_spline(
                self.waypts_time, self.positions, k=k, **kwargs
            )
        return self._cached_position_spline

    @property
    def _rotation_interpolator(self):
        if not hasattr(self, "_cached_rotation_interpolator"):
            if self.smooth_orientation and self.num_waypts >= 3:
                self._cached_rotation_interpolator = RotationSpline(
                    self.waypts_time, self.rotations
                )
            else:
                self._cached_rotation_interpolator = Slerp(
                    self.waypts_time, self.rotations
                )
        return self._cached_rotation_interpolator

    def waypoint_index_at(self, t: float) -> int:
        if t <= self.waypts_time[0]:
            return 0
        if t >= self.waypts_time[-1]:
            return self.num_waypts - 1
        idx = np.searchsorted(self.waypts_time, t, side="right") - 1
        if idx < self.num_waypts - 1:
            if abs(t - self.waypts_time[idx + 1]) < abs(t - self.waypts_time[idx]):
                idx += 1
        return int(idx)

    def interpolate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, rotation_matrix) at time t."""
        t_clamped = float(np.clip(t, self.waypts_time[0], self.waypts_time[-1]))
        position = np.asarray(self._position_spline(t_clamped), dtype=np.float64).reshape(3)
        rotation = self._rotation_interpolator(t_clamped).as_matrix()
        return position, rotation

    def velocity(self, t: float, dt: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
        """Return linear and base-frame angular velocity at trajectory time t."""
        t_clamped = float(np.clip(t, self.waypts_time[0], self.waypts_time[-1]))
        linear = np.asarray(self._position_spline.derivative()(t_clamped), dtype=np.float64).reshape(3)

        t0 = max(self.waypts_time[0], t_clamped - dt)
        t1 = min(self.waypts_time[-1], t_clamped + dt)
        if t1 <= t0:
            return linear, np.zeros(3, dtype=np.float64)

        r0 = self._rotation_interpolator(t0)
        r1 = self._rotation_interpolator(t1)
        angular = ((r1 * r0.inv()).as_rotvec() / (t1 - t0)).astype(np.float64)
        return linear, angular

    def retime(
        self,
        *,
        max_linear_vel: float,
        max_angular_vel: float,
        min_segment_dt: float = 0.001,
        max_linear_accel: float | None = None,
        max_angular_accel: float | None = None,
    ) -> "CartesianTrajectory":
        """Return the same Cartesian waypoints with velocity (and optionally
        acceleration) limited timing.

        Without acceleration limits, each segment duration is:
            dt_i = max(||dp_i|| / max_linear_vel,
                       angle(R_i^-1 R_{i+1}) / max_angular_vel,
                       min_segment_dt)

        When max_linear_accel / max_angular_accel are provided, a
        forward-backward pass enforces a trapezoidal speed profile so the
        reference velocity never changes faster than the robot can follow.
        The trajectory starts and ends at zero speed and ramps through
        waypoints smoothly, eliminating the instantaneous velocity jumps at
        waypoint boundaries that cause inertial tracking transients.

        Segment timing derivation (constant acceleration assumption):
            For boundary speeds v_a → v_b over distance d:
                t = 2·d / (v_a + v_b)
            Forward bound on v[i]:
                v[i] ≤ sqrt(v[i-1]² + 2·a_max·dist[i-1])
            Backward bound (must decelerate to final):
                v[i] ≤ sqrt(v[i+1]² + 2·a_max·dist[i])
        """
        if max_linear_vel <= 0:
            raise ValueError("max_linear_vel must be positive.")
        if max_angular_vel <= 0:
            raise ValueError("max_angular_vel must be positive.")
        if min_segment_dt <= 0:
            raise ValueError("min_segment_dt must be positive.")

        dp = np.diff(self.positions, axis=0)
        linear_dist = np.linalg.norm(dp, axis=1)

        relative_rot = self.rotations[:-1].inv() * self.rotations[1:]
        angular_dist = relative_rot.magnitude()

        N = self.num_waypts
        M = N - 1

        if max_linear_accel is not None or max_angular_accel is not None:
            # Forward-backward pass to compute boundary speeds subject to both
            # velocity and acceleration limits, then derive segment times.
            def _accel_limited_times(dist, v_max, a_max):
                """Trapezoidal speed profile for a sequence of segments."""
                # Forward pass: max achievable speed at each waypoint boundary.
                v = np.zeros(N)
                for i in range(M):
                    if dist[i] < 1e-9:
                        v[i + 1] = v[i]
                    else:
                        v[i + 1] = min(
                            v_max,
                            float(np.sqrt(max(0.0, v[i] ** 2 + 2.0 * a_max * dist[i]))),
                        )
                # Backward pass: must decelerate to zero at the end.
                v[N - 1] = 0.0
                for i in range(M - 1, -1, -1):
                    if dist[i] < 1e-9:
                        v[i] = min(v[i], v[i + 1])
                    else:
                        v[i] = min(
                            v[i],
                            float(np.sqrt(max(0.0, v[i + 1] ** 2 + 2.0 * a_max * dist[i]))),
                        )
                # Segment time: t = 2·d / (v_start + v_end).
                v_sum = v[:-1] + v[1:]
                dt = np.where(
                    v_sum > 1e-10,
                    2.0 * dist / v_sum,
                    np.full(M, min_segment_dt),
                )
                return np.maximum(dt, min_segment_dt)

            lin_a = float(max_linear_accel) if max_linear_accel is not None else 1e9
            ang_a = float(max_angular_accel) if max_angular_accel is not None else 1e9

            dt_lin = _accel_limited_times(linear_dist, float(max_linear_vel), lin_a)
            dt_ang = _accel_limited_times(angular_dist, float(max_angular_vel), ang_a)
            segment_dt = np.maximum(dt_lin, dt_ang)
        else:
            segment_dt = np.maximum.reduce([
                linear_dist / float(max_linear_vel),
                angular_dist / float(max_angular_vel),
                np.full(M, float(min_segment_dt), dtype=np.float64),
            ])

        new_times = np.concatenate([[0.0], np.cumsum(segment_dt)])
        new_times += self.waypts_time[0]

        return CartesianTrajectory(
            self.positions.copy(),
            self.rotations,
            new_times,
            smooth_orientation=self.smooth_orientation,
        )

    def transform(self, t: float) -> np.ndarray:
        position, rotation = self.interpolate(t)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    def resample_uniform(
        self,
        num_waypts: int | None = None,
        hz: float | None = None,
    ) -> "CartesianTrajectory":
        """Resample at uniformly spaced times."""
        if num_waypts is not None and hz is not None:
            raise ValueError("Cannot specify both num_waypts and hz.")
        if hz is not None:
            num_waypts = max(int(hz * self.duration) + 1, 2)
        elif num_waypts is None:
            num_waypts = self.num_waypts

        times = np.linspace(self.waypts_time[0], self.waypts_time[-1], num_waypts)
        positions = np.asarray(self._position_spline(times), dtype=np.float64)
        rotations = self._rotation_interpolator(times)
        return CartesianTrajectory(
            positions,
            rotations,
            times,
            smooth_orientation=self.smooth_orientation,
        )
