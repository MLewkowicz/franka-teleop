"""Inverse kinematics for the joint tracker (``deploy.tracker=joint``).

Converts policy Cartesian waypoints (EE pose in the ``fr3_link0`` frame, the same
convention as the demonstrated ``O_T_EE`` and :func:`clear_franka.franka.fk_ee_poses`)
into FR3 arm joint configurations, via a pyroki / jaxls Levenberg-Marquardt solver.

Design notes
------------
* **Same URDF as FK.** We build the pyroki robot from the *same* baked Cortado
  URDF that :func:`fk_ee_poses` uses, so IK and FK share one kinematic model.
* **Only the 7 arm joints are free.** The Cortado model exposes 14 actuated
  joints: ``column_joint_0`` (prismatic), ``fr3_joint1..7``, and 6 Robotiq finger
  joints. If pyroki were free to move all of them it would "cheat" by sliding the
  prismatic column. We rewrite every non-arm actuated joint to ``fixed`` (pinned at
  its q=0 origin) before handing the URDF to pyroki, leaving exactly ``fr3_joint1..7``.
  The TCP (``robotiq_arg2f_tcp``) hangs off the wrist through fixed joints only, so
  pinning the finger joints does not change the EE pose — this matches FK exactly,
  which evaluates the arm at ``cfg0 = zeros`` for all non-arm joints.
* **Frame parity.** pyroki's forward kinematics is rooted at the URDF root link
  (``root``), not ``fr3_link0``. Targets arrive in ``fr3_link0``; we left-multiply
  them by the constant ``T_root__fr3link0 = urdf.get_transform("fr3_link0")`` (with
  the column pinned at 0) to express them in the root frame before solving. This is
  the inverse of the ``T_root_to_base`` left-multiply in ``fk_ee_poses``.
* **Continuity via seed chaining.** The 7-DOF arm is redundant; independent
  per-waypoint IK can hop between elbow branches even when the Cartesian path is
  smooth. :meth:`solve_chained` seeds each waypoint's solve from the previous
  waypoint's solution (and waypoint 0 from the measured joint config), keeping the
  whole plan on one branch.

Run the deploy process with ``JAX_PLATFORMS=cpu``: ~25 tiny solves per plan are
well within the 10 Hz plan budget on CPU, and CPU jax avoids GPU init/contention
stalling the realtime franky control loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from clear_franka.franka import (
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_LOWER_JOINT_LIMITS,
    DEFAULT_UPPER_JOINT_LIMITS,
    bake_cortado_urdf,
    get_cortado_description,
)

logger = logging.getLogger(__name__)

_EE_LINK = "robotiq_arg2f_tcp"
_BASE_LINK = "fr3_link0"


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> unit quaternion ``[w, x, y, z]`` (jaxlie convention).

    Shepperd's method; picks the largest-diagonal branch for numerical stability.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` -> rotation matrix (3,3)."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _load_arm_only_cortado_urdf():
    """Load the baked Cortado URDF with every non-arm actuated joint pinned to fixed.

    Returns a ``yourdfpy.URDF`` whose only actuated joints are ``fr3_joint1..7``.

    The modified URDF is written next to the original baked file so any relative
    paths resolve identically, then loaded with the same call ``fk_ee_poses`` uses
    (``load_meshes=False``).
    """
    import xml.etree.ElementTree as ET

    import yourdfpy

    desc = get_cortado_description()
    urdf_path = bake_cortado_urdf(Path(desc.REPOSITORY_PATH))

    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    arm = set(DEFAULT_ARM_JOINT_NAMES)
    pinned = []
    for joint in root.findall("joint"):
        jtype = joint.get("type")
        if jtype in ("revolute", "prismatic", "continuous") and joint.get("name") not in arm:
            joint.set("type", "fixed")
            # A fixed joint ignores axis/limit/mimic and holds at its <origin> (q=0).
            for tag in ("axis", "limit", "mimic"):
                el = joint.find(tag)
                if el is not None:
                    joint.remove(el)
            pinned.append(joint.get("name"))
    logger.info("CortadoIK: pinned %d non-arm joints to fixed: %s", len(pinned), pinned)

    out_path = urdf_path.with_name(urdf_path.stem + "_armonly_ik.urdf")
    tree.write(str(out_path), encoding="unicode", xml_declaration=False)
    return yourdfpy.URDF.load(str(out_path), load_meshes=False)


class CortadoIK:
    """Batched pyroki IK targeting ``robotiq_arg2f_tcp`` in the ``fr3_link0`` frame.

    Inputs/outputs are numpy. Poses are expressed in ``fr3_link0`` (same as
    :func:`fk_ee_poses`). Joint vectors are 7-vectors ordered as
    :data:`DEFAULT_ARM_JOINT_NAMES` (``fr3_joint1..7``).
    """

    def __init__(
        self,
        *,
        pos_weight: float = 10.0,
        ori_weight: float = 5.0,
        limit_weight: float = 50.0,
        max_iterations: int = 30,
        num_seeds: int = 4,
    ):
        import jax.numpy as jnp
        import jaxlie
        import pyroki as pk

        self.pos_weight = float(pos_weight)
        self.ori_weight = float(ori_weight)
        self.limit_weight = float(limit_weight)
        self.max_iterations = int(max_iterations)
        self.num_seeds = int(num_seeds)

        urdf = _load_arm_only_cortado_urdf()
        self.robot = pk.Robot.from_urdf(urdf)

        # Exactly the 7 arm joints must remain actuated (no column / gripper DOF).
        act_names = tuple(self.robot.joints.actuated_names)
        if set(act_names) != set(DEFAULT_ARM_JOINT_NAMES):
            raise RuntimeError(
                "CortadoIK expected the actuated joint set to be "
                f"{set(DEFAULT_ARM_JOINT_NAMES)} but pyroki reports {set(act_names)}. "
                "The arm-only URDF rewrite is wrong."
            )
        self.n_joints = int(self.robot.joints.num_actuated_joints)
        assert self.n_joints == 7, self.n_joints

        # Permutations between the public franka order (DEFAULT_ARM_JOINT_NAMES,
        # i.e. fr3_joint1..7 — the order of robot.latest_state["q"] and set_joint_
        # reference) and pyroki's internal actuated order. Identity in practice, but
        # making it explicit keeps the API robust to any pyroki/URDF reordering.
        #   pyroki-order vec = default-order vec[_perm_in]
        #   default-order vec = pyroki-order vec[_perm_out]
        self._perm_in = np.array(
            [DEFAULT_ARM_JOINT_NAMES.index(a) for a in act_names], dtype=int
        )
        self._perm_out = np.array(
            [act_names.index(d) for d in DEFAULT_ARM_JOINT_NAMES], dtype=int
        )

        # End-effector link index.
        link_names = list(self.robot.links.names)
        if _EE_LINK not in link_names:
            raise RuntimeError(f"EE link {_EE_LINK!r} not found in URDF links: {link_names}")
        self.ee_link_idx = jnp.array(link_names.index(_EE_LINK))

        # Constant fr3_link0 -> root transform (column pinned at 0). yourdfpy
        # get_transform(frame_to) returns the pose of frame_to in the base/root frame.
        urdf_T = urdf.get_transform(_BASE_LINK)  # (4,4) pose of fr3_link0 in root frame
        self._R_root_base = np.asarray(urdf_T[:3, :3], dtype=np.float64)
        self._t_root_base = np.asarray(urdf_T[:3, 3], dtype=np.float64)

        # Deterministic random restart seeds (in the public franka order), drawn
        # within the franka joint limits.
        import jax

        self._lower = np.asarray(DEFAULT_LOWER_JOINT_LIMITS, dtype=np.float64)
        self._upper = np.asarray(DEFAULT_UPPER_JOINT_LIMITS, dtype=np.float64)
        key = jax.random.PRNGKey(0)
        u = np.asarray(jax.random.uniform(key, (max(self.num_seeds, 1), self.n_joints)))
        self._random_seeds = (self._lower + u * (self._upper - self._lower)).astype(np.float64)

        self._jaxlie = jaxlie
        self._pk = pk
        self._solver_cache: dict = {}
        logger.info(
            "CortadoIK ready: 7-DOF FR3 -> %s, %d link(s), max_iter=%d, num_seeds=%d",
            _EE_LINK, len(link_names), self.max_iterations, self.num_seeds,
        )

    # -- internal ---------------------------------------------------------------

    def _get_solver(self):
        """Build (or fetch) the JIT-compiled vmapped solver for the current settings."""
        key = (self.max_iterations,)
        if key in self._solver_cache:
            return self._solver_cache[key]

        import jax
        import jaxls

        jaxlie = self._jaxlie
        pk = self._pk
        robot = self.robot
        ee_idx = self.ee_link_idx
        pos_w, ori_w, lim_w = self.pos_weight, self.ori_weight, self.limit_weight
        max_iter = self.max_iterations

        joint_var = robot.joint_var_cls(0)

        def solve_ik_single(target_pos, target_quat, initial_q):
            target_rot = jaxlie.SO3(target_quat)
            target_se3 = jaxlie.SE3.from_rotation_and_translation(target_rot, target_pos)
            factors = [
                pk.costs.pose_cost_analytic_jac(
                    robot, joint_var, target_se3, ee_idx,
                    pos_weight=pos_w, ori_weight=ori_w,
                ),
                pk.costs.limit_cost(robot, joint_var, weight=lim_w),
            ]
            sol, summary = jaxls.LeastSquaresProblem(factors, [joint_var]).analyze().solve(
                initial_vals=jaxls.VarValues.make([joint_var.with_value(initial_q)]),
                verbose=False,
                linear_solver="dense_cholesky",
                termination=jaxls.TerminationConfig(
                    max_iterations=max_iter, early_termination=False,
                ),
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                return_summary=True,
            )
            return sol[joint_var], summary.cost_history, summary.iterations

        # outer vmap over targets (axis 0), inner vmap over seeds (axis 0).
        vmapped = jax.vmap(
            jax.vmap(solve_ik_single, in_axes=(None, None, 0)),
            in_axes=(0, 0, None),
        )
        compiled = jax.jit(vmapped)
        self._solver_cache[key] = compiled
        return compiled

    def _to_root_frame(self, pos_b: np.ndarray, rot_b: np.ndarray):
        """fr3_link0-frame (pos (B,3), rot (B,3,3)) -> root-frame (pos (B,3), quat_wxyz (B,4))."""
        Rrb, trb = self._R_root_base, self._t_root_base
        pos_root = pos_b @ Rrb.T + trb  # (B,3)
        rot_root = np.einsum("ij,bjk->bik", Rrb, rot_b)  # (B,3,3)
        quat_root = np.stack([_matrix_to_quat_wxyz(r) for r in rot_root], axis=0)
        return pos_root.astype(np.float64), quat_root.astype(np.float64)

    # -- public -----------------------------------------------------------------

    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray, seed_q: np.ndarray):
        """Solve IK for a batch of targets.

        Parameters
        ----------
        target_pos : (B,3) or (3,) EE position in the ``fr3_link0`` frame.
        target_rot : (B,3,3) or (3,3) EE rotation matrix in the ``fr3_link0`` frame.
        seed_q     : (7,) or (S,7) warm-start configuration(s). Each target is solved
                     from every seed; the minimum-cost branch is returned.

        Returns
        -------
        q    : (B,7) joint configs (``fr3_joint1..7`` order). Squeezed to (7,) if a
               single target was given.
        cost : (B,) final solver cost per target (use for reachability checks).
        """
        import jax.numpy as jnp

        pos = np.asarray(target_pos, dtype=np.float64)
        rot = np.asarray(target_rot, dtype=np.float64)
        single = pos.ndim == 1
        if single:
            pos = pos[None, :]
            rot = rot[None, :, :]

        seeds = np.asarray(seed_q, dtype=np.float64)
        if seeds.ndim == 1:
            seeds = seeds[None, :]
        seeds_pk = seeds[:, self._perm_in]  # franka order -> pyroki order

        pos_root, quat_root = self._to_root_frame(pos, rot)

        solver = self._get_solver()
        solutions, cost_history, iterations = solver(
            jnp.asarray(pos_root), jnp.asarray(quat_root), jnp.asarray(seeds_pk)
        )
        B, S = pos.shape[0], seeds.shape[0]
        final_costs = cost_history[
            jnp.arange(B)[:, None], jnp.arange(S)[None, :], iterations
        ]
        best = jnp.argmin(final_costs, axis=1)
        q_pk = np.asarray(solutions[jnp.arange(B), best], dtype=np.float64)
        q = q_pk[:, self._perm_out]  # pyroki order -> franka order
        cost = np.asarray(final_costs[jnp.arange(B), best], dtype=np.float64)
        if single:
            return q[0], cost[0]
        return q, cost

    def solve_chained(self, pos: np.ndarray, rot: np.ndarray, current_q: np.ndarray):
        """Solve a sequence of targets, seeding each from the previous solution.

        Parameters
        ----------
        pos : (N,3) EE positions in ``fr3_link0``.
        rot : (N,3,3) EE rotation matrices in ``fr3_link0``.
        current_q : (7,) measured joint config; seeds waypoint 0.

        Returns
        -------
        q    : (N,7) joint configs, kept on one IK branch.
        cost : (N,) per-waypoint final cost.
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        rot = np.asarray(rot, dtype=np.float64).reshape(-1, 3, 3)
        n = pos.shape[0]
        q_out = np.empty((n, 7), dtype=np.float64)
        cost_out = np.empty((n,), dtype=np.float64)

        # A couple of random restarts in addition to the chained seed, so a genuinely
        # better branch can still be found if the previous solution was a poor seed.
        extra = self._random_seeds[: max(self.num_seeds - 1, 0)]
        seed = np.asarray(current_q, dtype=np.float64).reshape(7)
        for i in range(n):
            seeds = np.vstack([seed[None, :], extra]) if extra.size else seed[None, :]
            qi, ci = self.solve(pos[i], rot[i], seeds)
            q_out[i] = qi
            cost_out[i] = ci
            seed = qi
        return q_out, cost_out

    def fk_ee(self, q: np.ndarray):
        """Forward kinematics: joint config -> EE pose in the ``fr3_link0`` frame.

        Uses pyroki's JIT-compiled FK (sub-ms after warmup), so it is cheap enough
        to call inside the realtime joint streaming loop for tracking-error logging.

        Parameters
        ----------
        q : (7,) or (N,7) joint configs (``fr3_joint1..7`` order).

        Returns
        -------
        pos : (3,) or (N,3) EE position in ``fr3_link0``.
        rot : (3,3) or (N,3,3) EE rotation matrix in ``fr3_link0``.
        """
        import jax.numpy as jnp

        q = np.asarray(q, dtype=np.float64)
        single = q.ndim == 1
        cfg = q[None, :] if single else q
        cfg = cfg[:, self._perm_in]  # franka order -> pyroki order
        poses = np.asarray(self.robot.forward_kinematics(jnp.asarray(cfg)))  # (B,L,7) root frame
        ee = poses[:, int(self.ee_link_idx)]  # (B,7) wxyz_xyz
        Rbr = self._R_root_base.T  # root -> fr3_link0 rotation
        out_pos = np.empty((cfg.shape[0], 3), dtype=np.float64)
        out_rot = np.empty((cfg.shape[0], 3, 3), dtype=np.float64)
        for i in range(cfg.shape[0]):
            R_root = _quat_wxyz_to_matrix(ee[i, :4])
            out_pos[i] = Rbr @ (ee[i, 4:] - self._t_root_base)
            out_rot[i] = Rbr @ R_root
        if single:
            return out_pos[0], out_rot[0]
        return out_pos, out_rot

    def warmup(self, seed_q: np.ndarray | None = None) -> None:
        """Trigger JIT compilation so the first real plan does not pay the compile cost.

        Compiles the (B=1, S=num_seeds) shape used by :meth:`solve_chained`.
        """
        q0 = (
            np.asarray(seed_q, dtype=np.float64).reshape(7)
            if seed_q is not None
            else 0.5 * (self._lower + self._upper)
        )
        from clear_franka.franka import fk_ee_poses

        pos, rot = fk_ee_poses(q0[None, :])
        seeds = np.vstack([q0[None, :], self._random_seeds[: max(self.num_seeds - 1, 0)]]) \
            if self.num_seeds > 1 else q0[None, :]
        q, cost = self.solve(pos[0], rot[0], seeds)
        self.fk_ee(q0)  # compile the FK path too
        logger.info("CortadoIK warmup done (round-trip cost=%.3e).", float(cost))
