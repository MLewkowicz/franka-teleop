import select
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np


LOOP_RATE_WINDOW_S = 5.0
LOOP_RATE_PRINT_PERIOD_S = 0.25


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{suffix}] ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def wait_for_enter(timeout_s: float) -> bool:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def announce(text: str) -> None:
    try:
        subprocess.Popen(
            ["espeak", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        print(f"  [voice] {text}")


class LoopRatePrinter:
    def __init__(
        self,
        window_s: float = LOOP_RATE_WINDOW_S,
        print_period_s: float = LOOP_RATE_PRINT_PERIOD_S,
    ):
        self.window_s = window_s
        self.print_period_s = print_period_s
        self._tick_times = []
        self._tick_durations = []
        self._stage_durations = {}
        self._tick_stage_total = 0.0
        self._last_tick_start = None
        self._last_print = 0.0
        self._last_line_len = 0

    def start_tick(self) -> float:
        now = time.monotonic()
        self._last_tick_start = now
        self._tick_stage_total = 0.0
        return now

    def finish_tick(self) -> None:
        now = time.monotonic()
        if self._last_tick_start is None:
            return

        tick_duration = now - self._last_tick_start
        unaccounted = max(0.0, tick_duration - self._tick_stage_total)
        if unaccounted > 0.0:
            self.record_stage("unaccounted", unaccounted)
        self._tick_times.append(now)
        self._tick_durations.append(tick_duration)

        cutoff = now - self.window_s
        while self._tick_times and self._tick_times[0] < cutoff:
            self._tick_times.pop(0)
            self._tick_durations.pop(0)
        for durations in self._stage_durations.values():
            while len(durations) > len(self._tick_durations):
                durations.pop(0)

        if now - self._last_print < self.print_period_s:
            return

        self._last_print = now
        elapsed = self._tick_times[-1] - self._tick_times[0] if len(self._tick_times) > 1 else 0.0
        hz = (len(self._tick_times) - 1) / elapsed if elapsed > 0.0 else 0.0
        avg_ms = np.mean(self._tick_durations) * 1000.0 if self._tick_durations else 0.0
        latest_ms = tick_duration * 1000.0
        stage_parts = []
        for name, durations in self._stage_durations.items():
            if not durations:
                continue
            latest_stage_ms = durations[-1] * 1000.0
            avg_stage_ms = np.mean(durations) * 1000.0
            stage_parts.append((avg_stage_ms, f"{name} {latest_stage_ms:.1f}/{avg_stage_ms:.1f}"))
        stages = " | " + ", ".join(
            part for _avg, part in sorted(stage_parts, reverse=True)[:6]
        ) if stage_parts else ""
        line = (
            f"  [loop] {hz:7.1f} Hz avg over {self.window_s:.0f}s | "
            f"tick {latest_ms:6.2f}/{avg_ms:6.2f} ms latest/avg{stages}"
        )
        padding = max(0, self._last_line_len - len(line))
        print(f"\r{line}{' ' * padding}", end="", flush=True)
        self._last_line_len = len(line)

    def record_stage(self, name: str, duration_s: float) -> None:
        self._stage_durations.setdefault(name, []).append(duration_s)

    def time_call(self, name: str, func, *args, **kwargs):
        start = time.monotonic()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.monotonic() - start
            self._tick_stage_total += duration
            self.record_stage(name, duration)

    @contextmanager
    def time_block(self, name: str):
        start = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start
            self._tick_stage_total += duration
            self.record_stage(name, duration)

    def newline(self) -> None:
        if self._last_line_len:
            print()
            self._last_line_len = 0
