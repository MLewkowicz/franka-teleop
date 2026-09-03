"""Regenerates clear_franka/proto/trajectory_pb2.py from proto/trajectory.proto at build time."""

import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ProtocBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        proto_file = root / "proto" / "trajectory.proto"
        out_dir = root / "clear_franka" / "proto"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{proto_file.parent}",
                f"--python_out={out_dir}",
                str(proto_file),
            ],
            check=True,
            cwd=root,
        )
