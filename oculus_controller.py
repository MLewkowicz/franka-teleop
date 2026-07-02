"""Wrap an Oculus/Meta Quest VR controller (via oculus_reader) so it can stand
in for the SpaceMouse in `teleop.py`.

Exposes the same polling interface as `threed_mouse.ThreeDMouse` —
`run()`, `close()`, `get_controller_state()` returning a `ThreeDMouseData` —
so `run_teleop` can use either device without branching in the control loop.

Coordinate-frame handling and the grip-clutch/origin-reset scheme are adapted
from droid's `VRPolicy` (droid/controllers/oculus_controller.py), which drives
a robot from an Oculus controller the same way.

Button mapping (mirrors the SpaceMouse's two-button interface in teleop.py):
  buttons[0] — grip trigger. Acts as a clutch: while held, hand motion since
    the grip was squeezed is reported as a position/rotation offset. Tap it
    to toggle motion enabled/disabled (same tap/hold logic as the SpaceMouse's
    left button); the robot only actually moves while grip is also held down.
  buttons[1] — index trigger. Acts like the SpaceMouse's right button
    (tap to toggle the gripper).
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
from oculus_reader.reader import OculusReader

from threed_mouse.device import ThreeDMouseData
from threed_mouse.geometry import R_to_rot_vector

logger = logging.getLogger(__name__)

# Poll rate (Hz) for reading from the headset in the background thread.
TELEOP_CONTROL_RATE = 50

TRIGGER_PRESS_THRESHOLD = 0.5


class OculusController:
    def __init__(
        self,
        control_rate: float = TELEOP_CONTROL_RATE,
        right_controller: bool = True,
        ip_address: Optional[str] = None,
        pos_offset_gain: float = 3.0,
        rot_offset_gain: float = 1.0,
    ):
        self._control_rate = control_rate
        self._ip_address = ip_address
        self.controller_id = "r" if right_controller else "l"
        self.trigger_key = "rightTrig" if self.controller_id == "r" else "leftTrig"
        self.grip_key = self.controller_id.upper() + "G"

        # Scales the clutch position/rotation offset (in meters/radians) up into
        # the roughly [-1, 1] analog-stick range that ThreeDMouseFilter expects.
        self.pos_offset_gain = pos_offset_gain
        self.rot_offset_gain = rot_offset_gain

        # Same naming/semantics as ThreeDMouse, so teleop.py can override these
        # if the headset's frame doesn't line up with the robot base frame.
        self._frame_rotation_linear = np.eye(3, dtype=float)
        self._frame_rotation_angular = np.eye(3, dtype=float)

        self.reader: Optional[OculusReader] = None
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._control_lock = threading.Lock()
        self._control: Optional[ThreeDMouseData] = None

        self._grip_origin: Optional[np.ndarray] = None
        self._prev_grip = False

    def __del__(self):
        if self.is_running:
            self.stop()

    @property
    def is_running(self) -> bool:
        return self.thread is not None

    def run(self):
        if self.thread:
            return

        try:
            self.reader = OculusReader(ip_address=self._ip_address)
        except Exception as e:
            logger.error("Unable to open Oculus controller: %s", e)
            raise RuntimeError("Couldn't open device") from e

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def get_controller_state(self) -> Optional[ThreeDMouseData]:
        with self._control_lock:
            return self._control

    def stop(self):
        if not self.is_running:
            return
        self._stop_event.set()
        if self.thread:
            self.thread.join()
        self._stop_event.clear()
        self.thread = None

    def close(self):
        self.stop()
        if self.reader is not None:
            stop = getattr(self.reader, "stop", None)
            if callable(stop):
                stop()
            self.reader = None

    def _run_loop(self):
        sleep_s = max(0.001, 1.0 / float(self._control_rate))

        with self._control_lock:
            self._control = ThreeDMouseData(
                -1.0,
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                np.zeros(2, dtype=int),
            )

        while not self._stop_event.is_set():
            try:
                poses, buttons = self.reader.get_transformations_and_buttons()
            except Exception:
                logger.warning("Lost connection to Oculus controller.")
                time.sleep(sleep_s)
                continue

            if poses and self.controller_id in poses:
                self._update_control(poses, buttons)

            time.sleep(sleep_s)

    def _update_control(self, poses, buttons):
        pose = np.asarray(poses[self.controller_id], dtype=float)
        grip = bool(buttons.get(self.grip_key, False))
        trigger = float(buttons.get(self.trigger_key, (0.0,))[0]) > TRIGGER_PRESS_THRESHOLD

        if grip and not self._prev_grip:
            # Rising edge: re-clutch from the current hand pose so motion
            # resumes smoothly instead of jumping to the accumulated offset.
            self._grip_origin = pose
        self._prev_grip = grip

        xyz = np.zeros(3, dtype=float)
        rpy = np.zeros(3, dtype=float)
        if grip and self._grip_origin is not None:
            rel = np.linalg.inv(self._grip_origin) @ pose
            with self._control_lock:
                r_linear = self._frame_rotation_linear.copy()
                r_angular = self._frame_rotation_angular.copy()
            xyz = np.clip(r_linear @ rel[:3, 3] * self.pos_offset_gain, -1.0, 1.0)
            rpy = np.clip(r_angular @ R_to_rot_vector(rel[:3, :3]) * self.rot_offset_gain, -1.0, 1.0)

        control = ThreeDMouseData(
            time.monotonic(),
            xyz,
            rpy,
            np.array([int(grip), int(trigger)], dtype=int),
        )
        with self._control_lock:
            self._control = control