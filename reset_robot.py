"""Reset the Franka: clear errors, then move back to the home joint config.

Prerequisite: the zero_franky server must be running on the vector machine
(`kernel`, 172.16.0.1):

    ssh kernel
    source ~/franka/franky/.venv/bin/activate
    zero-franky-server

Then, from this machine:

    python reset_robot.py                  # recover errors + home the arm
    python reset_robot.py +recover_only=true    # clear errors, no motion

The home pose is teleop.reset_joint_config from conf/config.yaml (override with
e.g. `reset_joint_config="[...]"`). Motion runs at 10% dynamics, like deploy/teleop.
"""

import numpy as np
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from zero_franky import Robot, setup_zero_franky
    from franky import JointMotion, JointState

    setup_zero_franky(cfg.zero_franky.ip, cfg.zero_franky.port)

    robot = Robot(cfg.robot.ip)

    print("Clearing errors (recover_from_errors)...")
    robot.recover_from_errors()
    # Drain any leftover motion so the next move starts cleanly.
    try:
        robot.join_motion(2)
    except Exception:
        pass

    if cfg.get("recover_only", False):
        print("recover_only=true — errors cleared, not moving the arm.")
        return

    reset_joint_config = np.asarray(cfg.teleop.reset_joint_config, dtype=float)
    if reset_joint_config.shape != (7,):
        raise ValueError(
            f"reset_joint_config must have 7 joint angles, got "
            f"{reset_joint_config.shape[0]}: {reset_joint_config.tolist()}"
        )

    print(f"Homing to {reset_joint_config.tolist()} ...")
    robot.move(
        JointMotion(JointState(reset_joint_config), relative_dynamics_factor=0.1),
        asynchronous=False,
    )
    print("Done.")


if __name__ == "__main__":
    main()
