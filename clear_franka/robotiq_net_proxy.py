"""Small RPyC proxy for pyRobotiqGripper.

Run an RPyC classic server on the robot-side host, where the Robotiq gripper is
physically reachable:

    rpyc_classic -p 18812 --host 0.0.0.0

Then use this file from another machine:

    from clear_franka.robotiq_net_proxy import RobotiqGripperProxy

    gripper = RobotiqGripperProxy(
        server_host="robot-hostname-or-ip",
        com_port="/dev/ttyUSB0",
    )
    gripper.activate()
    gripper.open()
    gripper.close()

The remote host must run `rpyc_classic` from an environment where
`pyrobotiqgripper` is importable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_RPYC_PORT = 18812
DEFAULT_GRIPPER_MAX_WIDTH_M = 0.085


@dataclass(frozen=True)
class RobotiqConnectionConfig:
    """Configuration for the robot-side pyRobotiqGripper object."""

    com_port: str = "auto"
    device_id: int = 9
    connection_type: str = "RTU"
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 54321
    debug: bool = False


def _copy_remote_value(value: Any) -> Any:
    """Convert common RPyC netrefs into plain local Python values."""
    if isinstance(value, dict):
        return {key: _copy_remote_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_remote_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_remote_value(item) for item in value)
    return value


def _clamp_byte(value: float | int, name: str) -> int:
    value_int = int(round(value))
    if not 0 <= value_int <= 255:
        raise ValueError(f"{name} must be in [0, 255], got {value!r}")
    return value_int


def _width_m_to_position(width_m: float, max_width_m: float) -> int:
    if max_width_m <= 0.0:
        raise ValueError(f"max_width_m must be positive, got {max_width_m!r}")
    width_m = min(max(float(width_m), 0.0), max_width_m)
    return int(round((1.0 - (width_m / max_width_m)) * 255.0))


def _position_to_width_m(position: float | int, max_width_m: float) -> float:
    position = _clamp_byte(position, "position")
    return (1.0 - (position / 255.0)) * max_width_m


class RobotiqGripperProxy:
    """Network proxy for pyRobotiqGripper through an RPyC classic server.

    This intentionally mirrors `net_franky`'s model: the gripper driver object
    is created on the machine running `rpyc_classic`, and client-side calls are
    forwarded to that remote object.
    """

    def __init__(
        self,
        server_host: str,
        server_port: int = DEFAULT_RPYC_PORT,
        *,
        config: RobotiqConnectionConfig | None = None,
        com_port: str = "auto",
        device_id: int = 9,
        connection_type: str = "RTU",
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 54321,
        debug: bool = False,
        auto_activate: bool = False,
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        self.config = config or RobotiqConnectionConfig(
            com_port=com_port,
            device_id=device_id,
            connection_type=connection_type,
            tcp_host=tcp_host,
            tcp_port=tcp_port,
            debug=debug,
        )

        import rpyc

        self._conn = rpyc.classic.connect(server_host, server_port)

        remote_module = self._conn.modules["pyrobotiqgripper"]
        self._gripper = remote_module.RobotiqGripper(
            com_port=self.config.com_port,
            device_id=self.config.device_id,
            connection_type=self.config.connection_type,
            tcp_host=self.config.tcp_host,
            tcp_port=self.config.tcp_port,
            debug=self.config.debug,
        )

        if auto_activate:
            self.activate()

    def __enter__(self) -> "RobotiqGripperProxy":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.disconnect()

    def __getattr__(self, name: str) -> Any:
        """Delegate pyRobotiqGripper methods that are not wrapped here."""
        return getattr(self._gripper, name)

    def disconnect(self) -> None:
        """Disconnect the remote gripper object and close the RPyC connection."""
        try:
            self._gripper.disconnect()
        finally:
            self._conn.close()

    def activate(self, reset: bool = True, start: bool = True, refresh_status: bool = True) -> None:
        self._gripper.activate(reset=reset, start=start, refreshStatus=refresh_status)

    def start(self, refresh_status: bool = True) -> None:
        self._gripper.start(refreshStatus=refresh_status)

    def reset(self) -> None:
        self._gripper.reset()

    def stop(self) -> None:
        self._gripper.stop()

    def open(
        self,
        speed: int = 255,
        force: int = 255,
        wait: bool = True,
        read_status: bool = True,
        refresh_status: bool = False,
    ) -> None:
        self._gripper.open(
            speed=_clamp_byte(speed, "speed"),
            force=_clamp_byte(force, "force"),
            wait=wait,
            readStatus=read_status,
            refreshStatus=refresh_status,
        )

    def close(
        self,
        speed: int = 255,
        force: int = 255,
        wait: bool = True,
        read_status: bool = True,
        refresh_status: bool = False,
    ) -> None:
        self._gripper.close(
            speed=_clamp_byte(speed, "speed"),
            force=_clamp_byte(force, "force"),
            wait=wait,
            readStatus=read_status,
            refreshStatus=refresh_status,
        )

    def move_position(
        self,
        position: int,
        speed: int = 255,
        force: int = 255,
        wait: bool = True,
        read_status: bool = True,
        refresh_status: bool = False,
    ) -> None:
        """Move using pyRobotiqGripper's native 0=open, 255=closed units."""
        self._gripper.move(
            position=_clamp_byte(position, "position"),
            speed=_clamp_byte(speed, "speed"),
            force=_clamp_byte(force, "force"),
            wait=wait,
            readStatus=read_status,
            refreshStatus=refresh_status,
        )

    def move_mm(
        self,
        width_mm: float,
        speed: int = 255,
        force: int = 255,
        wait: bool = True,
        read_status: bool = True,
        refresh_status: bool = False,
    ) -> None:
        """Move by opening width in millimeters.

        This calls pyRobotiqGripper's calibrated `move_mm`, so the remote object
        must have been calibrated with `calibrate_mm(...)` first.
        """
        self._gripper.move_mm(
            positionmm=width_mm,
            speed=_clamp_byte(speed, "speed"),
            force=_clamp_byte(force, "force"),
            wait=wait,
            readStatus=read_status,
            refreshStatus=refresh_status,
        )

    def move_width(
        self,
        width_m: float,
        speed: int = 255,
        force: int = 255,
        wait: bool = True,
        max_width_m: float = DEFAULT_GRIPPER_MAX_WIDTH_M,
    ) -> None:
        """Move by opening width in meters without requiring calibration."""
        self.move_position(
            _width_m_to_position(width_m, max_width_m),
            speed=speed,
            force=force,
            wait=wait,
        )

    def calibrate_mm(self, close_mm: float = 0.0, open_mm: float = 85.0) -> None:
        """Calibrate millimeter control on the remote driver.

        If bit calibration has not happened, pyRobotiqGripper will open and close
        the gripper during this call.
        """
        self._gripper.calibrate_mm(closemm=close_mm, openmm=open_mm)

    def position(self, refresh_status: bool = True) -> int | None:
        return _copy_remote_value(self._gripper.position(refreshStatus=refresh_status))

    def position_mm(self, refresh_status: bool = True) -> float:
        return _copy_remote_value(self._gripper.position_mm(refreshStatus=refresh_status))

    def position_width(
        self,
        refresh_status: bool = True,
        max_width_m: float = DEFAULT_GRIPPER_MAX_WIDTH_M,
    ) -> float | None:
        position = self.position(refresh_status=refresh_status)
        if position is None:
            return None
        return _position_to_width_m(position, max_width_m)

    def status(self, refresh_status: bool = True) -> dict[str, Any]:
        return _copy_remote_value(self._gripper.status(refreshStatus=refresh_status))

    def object_detection(self, refresh_status: bool = True) -> int:
        return _copy_remote_value(self._gripper.objectDetection(refreshStatus=refresh_status))


def connect(
    server_host: str,
    server_port: int = DEFAULT_RPYC_PORT,
    **kwargs: Any,
) -> RobotiqGripperProxy:
    """Convenience constructor matching the style of simple client libraries."""
    return RobotiqGripperProxy(server_host=server_host, server_port=server_port, **kwargs)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Control a Robotiq gripper through RPyC.")
    parser.add_argument("command", choices=["activate", "open", "close", "status", "move-width", "move-position"])
    parser.add_argument("--host", required=True, help="RPyC classic server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_RPYC_PORT, help="RPyC classic server port.")
    parser.add_argument("--com-port", default="auto", help="Remote serial port, for example /dev/ttyUSB0.")
    parser.add_argument("--device-id", type=int, default=9)
    parser.add_argument("--connection-type", default="RTU", choices=["RTU", "RTU_VIA_TCP"])
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=54321)
    parser.add_argument("--speed", type=int, default=255)
    parser.add_argument("--force", type=int, default=255)
    parser.add_argument("--width-m", type=float, help="Opening width in meters for move-width.")
    parser.add_argument("--position", type=int, help="Native 0=open, 255=closed position for move-position.")
    args = parser.parse_args()

    config = RobotiqConnectionConfig(
        com_port=args.com_port,
        device_id=args.device_id,
        connection_type=args.connection_type,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
    )

    with RobotiqGripperProxy(
        server_host=args.host,
        server_port=args.port,
        config=config,
    ) as gripper:
        if args.command == "activate":
            gripper.activate()
        elif args.command == "open":
            gripper.open(speed=args.speed, force=args.force)
        elif args.command == "close":
            gripper.close(speed=args.speed, force=args.force)
        elif args.command == "status":
            print(gripper.status())
        elif args.command == "move-width":
            if args.width_m is None:
                parser.error("--width-m is required for move-width")
            gripper.move_width(width_m=args.width_m, speed=args.speed, force=args.force)
        elif args.command == "move-position":
            if args.position is None:
                parser.error("--position is required for move-position")
            gripper.move_position(position=args.position, speed=args.speed, force=args.force)


if __name__ == "__main__":
    _main()
