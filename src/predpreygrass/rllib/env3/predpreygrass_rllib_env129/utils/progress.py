"""Tiny progress bar helper (no external dependencies)."""

from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, total: int, *, label: str = "", width: int = 30, stream=None) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.width = max(10, int(width))
        self.stream = stream or sys.stderr
        self.start_time = time.time()
        self.last_render_len = 0

    def update(self, current: float, detail: str | None = None) -> None:
        current_f = float(current)
        current_f = max(0.0, min(current_f, float(self.total)))
        frac = current_f / float(self.total)
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start_time
        eta = 0.0
        if current_f > 1e-9:
            eta = elapsed * (float(self.total) - current_f) / max(current_f, 1e-9)
        cur_display = f"{current_f:.2f}" if abs(current_f - round(current_f)) > 1e-9 else f"{int(round(current_f))}"
        msg = (
            f"{self.label}[{bar}] {cur_display}/{self.total} ({frac*100:5.1f}%)"
            f" elapsed={elapsed:6.1f}s eta={eta:6.1f}s"
        )
        if detail:
            msg = msg + f" | {detail}"
        pad = max(0, self.last_render_len - len(msg))
        self.stream.write("\r" + msg + (" " * pad))
        self.stream.flush()
        self.last_render_len = len(msg)

    def close(self) -> None:
        self.stream.write("\n")
        self.stream.flush()
