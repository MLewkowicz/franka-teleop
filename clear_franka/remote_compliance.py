"""Client helpers for robot-side net_franky compliance sessions."""

from __future__ import annotations


class RemoteCompliantHold:
    def __init__(self, session):
        self._session = session

    def get_pose(self) -> dict:
        return dict(self._session.get_pose())

    def stop(self) -> None:
        self._session.stop()


def _net_franky_connection():
    import net_franky.franky as remote_franky

    try:
        return remote_franky.conn
    except AttributeError as exc:
        raise RuntimeError("net_franky connection is not available") from exc


def start_remote_compliant_hold(
    robot_ip: str,
    *,
    translational_stiffness: float,
    rotational_stiffness: float,
    nullspace_stiffness: float,
    lower_joint_limits,
    upper_joint_limits,
    period: float,
    timeout_s: float = 5.0,
) -> RemoteCompliantHold:
    """Start a net_franky robot-side compliant hold session."""
    conn = _net_franky_connection()
    session = conn.modules["net_franky.cb_robot"].start_compliant_hold(
        robot_ip,
        float(translational_stiffness),
        float(rotational_stiffness),
        float(nullspace_stiffness),
        list(lower_joint_limits),
        list(upper_joint_limits),
        float(period),
    )
    session.wait_ready(float(timeout_s))
    return RemoteCompliantHold(session)
