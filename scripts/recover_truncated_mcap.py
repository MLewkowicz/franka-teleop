"""Rebuild a truncated (unfinalized) episode MCAP into a valid, readable file.

Background
----------
`Recorder.stop()` enqueues a "finish" item that writes the `episode_summary`
metadata record and then calls `writer.finish()`, which emits the summary
section, footer and closing magic (see clear_franka/recorder.py). If the
process dies before that runs, the file ends mid-record: every message written
so far is on disk and intact, but there is no footer, so `make_reader()` (and
therefore `load_episode`) fails outright with RecordLengthLimitExceeded.

This tool stream-reads whatever survived, rewrites it as a well-formed MCAP,
and reconstructs the lost `episode_summary`.

What is recovered exactly vs. reconstructed
-------------------------------------------
Exact (copied byte-for-byte from the damaged file):
  * header profile/library, schemas, channels
  * every decodable message, with original log_time / publish_time / sequence
  * the opening `episode` metadata record

Reconstructed (the `episode_summary` record, which was never written):
  * num_steps, duration_ns      -- measured from the recovered messages
  * camera.*.frame_count        -- measured by decoding the SVO sidecar
  * camera.*.{first,last}_zed_image_time_ns
                                -- measured from the SVO sidecar. These are
                                   ground truth for what the video actually
                                   contains, which is not always what the
                                   original metadata claimed (the ZED writer
                                   can drop frames at the start of recording).
  * camera.*.{first,last}_host_monotonic_time_ns
                                -- DERIVED, not measured. ZED image stamps are
                                   wall-clock, so these are back-computed as
                                   zed_ns - (start_wall_ns - start_monotonic_ns)
                                   using the surviving `episode` metadata.
                                   Calibration against intact episodes shows a
                                   systematic capture-latency bias of roughly
                                   -110 ms (hand) / -66 ms (third_person) with
                                   ~50 ms spread, i.e. a few frame periods.

Prefer wall-clock alignment over the derived monotonic anchors: a trajectory
sample's wall time is `start_wall_time_ns + episode_time_ns`, and SVO frame
timestamps are on that same clock, so frames and samples can be matched
directly with no anchor involved.

A `recovery` metadata record is written into the output recording all of the
above, so a reconstructed file is never mistaken for a pristine one.

Usage
-----
    # Report what is damaged, change nothing:
    uv run python scripts/recover_truncated_mcap.py --dir data/mug_bowl/training_data --dry-run

    # Recover every truncated episode in a directory (writes *.recovered.mcap):
    uv run python scripts/recover_truncated_mcap.py --dir data/mug_bowl/training_data

    # Recover specific files into a separate directory:
    uv run python scripts/recover_truncated_mcap.py a.mcap b.mcap --output-dir recovered/

Originals are never modified or deleted.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mcap.records import Channel, Footer, Header, Message, Metadata, Schema
from mcap.stream_reader import StreamReader
from mcap.writer import Writer

MCAP_MAGIC = b"\x89MCAP0\r\n"
SUMMARY_METADATA_NAME = "episode_summary"
EPISODE_METADATA_NAME = "episode"
RECOVERY_METADATA_NAME = "recovery"

# Measured on the four intact episodes in data/mug_bowl/training_data:
# (zed_image_ns - host_monotonic_ns) - (start_wall_ns - start_monotonic_ns).
# Recorded for provenance only -- not applied, the spread (~50 ms) is larger
# than a frame period so subtracting it would not make the anchors truer.
_KNOWN_LATENCY_BIAS_MS = {"hand": -112.9, "third_person": -65.9}


@dataclass
class RecoveredEpisode:
    """Everything salvaged from a damaged file, before rewriting."""

    header: Header | None = None
    schemas: dict[int, Schema] = field(default_factory=dict)
    channels: dict[int, Channel] = field(default_factory=dict)
    metadata: list[Metadata] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    had_footer: bool = False
    stream_error: str | None = None


def is_truncated(path: Path) -> bool:
    """True if the file does not end with the MCAP closing magic."""
    size = path.stat().st_size
    if size < len(MCAP_MAGIC) * 2:
        return True
    with open(path, "rb") as handle:
        handle.seek(-len(MCAP_MAGIC), 2)
        return handle.read(len(MCAP_MAGIC)) != MCAP_MAGIC


def scan(path: Path) -> RecoveredEpisode:
    """Stream-read a (possibly damaged) MCAP, stopping at the first bad record.

    The stream reader raises once it walks off the end of the truncated tail;
    everything yielded before that point is valid and is kept.
    """
    out = RecoveredEpisode()
    try:
        with open(path, "rb") as handle:
            for record in StreamReader(handle).records:
                if isinstance(record, Header):
                    out.header = record
                elif isinstance(record, Schema):
                    out.schemas[record.id] = record
                elif isinstance(record, Channel):
                    out.channels[record.id] = record
                elif isinstance(record, Metadata):
                    out.metadata.append(record)
                elif isinstance(record, Message):
                    out.messages.append(record)
                elif isinstance(record, Footer):
                    out.had_footer = True
    except Exception as error:  # truncated tail -- expected for damaged files
        out.stream_error = f"{type(error).__name__}: {error}"
    return out


def episode_attrs(recovered: RecoveredEpisode) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for record in recovered.metadata:
        attrs.update(record.metadata)
    return attrs


def probe_svo(path: Path) -> dict[str, int] | None:
    """Decode an SVO sidecar and report its true frame count and time bounds.

    Returns None if the ZED SDK is unavailable or the file will not open.
    """
    try:
        import pyzed.sl as sl
    except ImportError:
        return None
    if not path.exists():
        return None

    camera = sl.Camera()
    params = sl.InitParameters()
    params.set_from_svo_file(str(path))
    params.svo_real_time_mode = False
    if camera.open(params) != sl.ERROR_CODE.SUCCESS:
        return None
    stamps: list[int] = []
    try:
        while camera.grab() == sl.ERROR_CODE.SUCCESS:
            stamps.append(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
    finally:
        camera.close()
    if not stamps:
        return None
    return {
        "frame_count": len(stamps),
        "first_zed_image_time_ns": stamps[0],
        "last_zed_image_time_ns": stamps[-1],
    }


def camera_names(attrs: dict[str, str]) -> list[str]:
    """Camera names advertised by the surviving `episode` metadata."""
    names = set()
    for key in attrs:
        if key.startswith("camera.") and key.endswith(".video_file"):
            names.add(key[len("camera.") : -len(".video_file")])
    return sorted(names)


def rebuild_summary(
    source: Path, recovered: RecoveredEpisode, attrs: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Reconstruct the `episode_summary` metadata that was never written."""
    notes: list[str] = []
    summary: dict[str, str] = {"num_steps": str(len(recovered.messages))}

    if recovered.messages:
        span_ns = recovered.messages[-1].log_time - recovered.messages[0].log_time
        summary["duration_ns"] = str(span_ns)
        notes.append(
            "duration_ns measured across recovered messages; it is the trajectory "
            "that survived, not necessarily the full demonstration."
        )

    clock_offset: int | None = None
    if "start_wall_time_ns" in attrs and "start_monotonic_time_ns" in attrs:
        clock_offset = int(attrs["start_wall_time_ns"]) - int(attrs["start_monotonic_time_ns"])
    else:
        notes.append(
            "start_wall_time_ns/start_monotonic_time_ns missing -- host monotonic "
            "anchors could not be derived."
        )

    for name in camera_names(attrs):
        video = attrs.get(f"camera.{name}.video_file", "")
        probed = probe_svo(source.parent / Path(video).name) if video else None
        if probed is None:
            notes.append(f"camera.{name}: sidecar not probed (missing file or no ZED SDK).")
            continue
        for key, value in probed.items():
            summary[f"camera.{name}.{key}"] = str(value)
        if clock_offset is not None:
            for label in ("first", "last"):
                zed_ns = probed[f"{label}_zed_image_time_ns"]
                summary[f"camera.{name}.{label}_host_monotonic_time_ns"] = str(zed_ns - clock_offset)
            bias = _KNOWN_LATENCY_BIAS_MS.get(name)
            bias_note = f" Expect ~{bias:+.0f} ms capture-latency bias." if bias else ""
            notes.append(
                f"camera.{name}: host_monotonic anchors DERIVED from wall clock, "
                f"not measured.{bias_note}"
            )

    return summary, notes


def write_recovered(
    source: Path, dest: Path, recovered: RecoveredEpisode, summary: dict[str, str], notes: list[str]
) -> None:
    """Write a well-formed MCAP containing everything salvaged."""
    attrs = episode_attrs(recovered)
    with open(dest, "wb") as handle:
        writer = Writer(handle)
        writer.start(
            profile=recovered.header.profile if recovered.header else "protobuf",
            library=recovered.header.library if recovered.header else "franka-teleop",
        )

        # Old ids are only meaningful within the source file; remap to new ones.
        schema_ids: dict[int, int] = {}
        for old_id, schema in recovered.schemas.items():
            schema_ids[old_id] = writer.register_schema(
                name=schema.name, encoding=schema.encoding, data=schema.data
            )
        channel_ids: dict[int, int] = {}
        for old_id, channel in recovered.channels.items():
            channel_ids[old_id] = writer.register_channel(
                topic=channel.topic,
                message_encoding=channel.message_encoding,
                schema_id=schema_ids.get(channel.schema_id, 0),
                metadata=channel.metadata,
            )

        for record in recovered.metadata:
            writer.add_metadata(record.name, record.metadata)

        for message in recovered.messages:
            writer.add_message(
                channel_id=channel_ids[message.channel_id],
                log_time=message.log_time,
                data=message.data,
                publish_time=message.publish_time,
                sequence=message.sequence,
            )

        writer.add_metadata(SUMMARY_METADATA_NAME, summary)
        writer.add_metadata(
            RECOVERY_METADATA_NAME,
            {
                "recovered_by": "scripts/recover_truncated_mcap.py",
                "source_file": source.name,
                "source_bytes": str(source.stat().st_size),
                "source_stream_error": recovered.stream_error or "",
                "messages_recovered": str(len(recovered.messages)),
                "episode_summary": "reconstructed",
                "episode_id": attrs.get("episode_id", ""),
                "notes": " | ".join(notes),
            },
        )
        writer.finish()


def verify(dest: Path, expected_messages: int) -> str:
    """Re-open the rebuilt file through the project's own reader."""
    try:
        from clear_franka.episode_io import load_episode

        data = load_episode(dest)
    except Exception as error:
        return f"FAILED to reload: {type(error).__name__}: {error}"
    samples = len(data["timestamps"])
    attrs = data["attrs"]
    detail = f"loads cleanly, {samples} samples"
    if samples > expected_messages:
        detail += f" (expected <= {expected_messages} trajectory messages)"
    missing = [k for k in ("num_steps", "duration_ns") if k not in attrs]
    if missing:
        detail += f", MISSING summary keys {missing}"
    return detail


def recover_one(source: Path, output_dir: Path | None, dry_run: bool) -> bool:
    print("=" * 74)
    print(source.name)

    if not is_truncated(source):
        print("  intact (closing magic present) -- nothing to do")
        return True

    recovered = scan(source)
    attrs = episode_attrs(recovered)
    print(f"  TRUNCATED: {recovered.stream_error}")
    print(
        f"  salvaged: {len(recovered.messages)} messages, "
        f"{len(recovered.schemas)} schemas, {len(recovered.channels)} channels, "
        f"{len(recovered.metadata)} metadata record(s)"
    )
    if not recovered.messages:
        print("  nothing to rebuild -- no decodable messages")
        return False
    if not any(r.name == EPISODE_METADATA_NAME for r in recovered.metadata):
        print("  warning: opening `episode` metadata missing; summary will be partial")

    summary, notes = rebuild_summary(source, recovered, attrs)
    print(f"  reconstructed episode_summary with {len(summary)} fields:")
    for key in sorted(summary):
        print(f"      {key} = {summary[key]}")
    for note in notes:
        print(f"    note: {note}")

    dest_dir = output_dir or source.parent
    dest = dest_dir / f"{source.stem}.recovered.mcap"
    if dry_run:
        print(f"  [dry-run] would write {dest}")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    write_recovered(source, dest, recovered, summary, notes)
    print(f"  wrote {dest} ({dest.stat().st_size} bytes)")
    print(f"  verify: {verify(dest, len(recovered.messages))}")
    print(f"  original left untouched: {source}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="episode .mcap files to recover")
    parser.add_argument("--dir", type=Path, help="recover every truncated episode_*.mcap in this directory")
    parser.add_argument("--output-dir", type=Path, help="write recovered files here (default: alongside the source)")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args(argv)

    targets = list(args.files)
    if args.dir:
        targets.extend(sorted(args.dir.glob("episode_*.mcap")))
    targets = [p for p in targets if not p.name.endswith(".recovered.mcap")]
    if not targets:
        parser.error("no input files (pass paths or --dir)")

    ok = True
    for path in targets:
        if not path.exists():
            print(f"missing: {path}")
            ok = False
            continue
        ok &= recover_one(path, args.output_dir, args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
