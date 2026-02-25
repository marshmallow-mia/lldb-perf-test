"""Hardware monitoring for llama-bench TUI.

Polls GPU (NVIDIA via nvidia-smi, AMD via sysfs), CPU (/proc/stat), and
RAM (/proc/meminfo) in a background daemon thread.  No external dependencies.
"""
from __future__ import annotations

import glob
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GPUStats:
    name: str = "Unknown"
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    util_pct: float = 0.0
    temp_c: float = 0.0


@dataclass
class HWSnapshot:
    gpus: list[GPUStats] = field(default_factory=list)
    cpu_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------


def _find_amd_drm_cards() -> list[str]:
    """Return sorted list of /sys/class/drm card names that have AMD VRAM sysfs files."""
    cards = []
    import os
    try:
        for entry in sorted(os.listdir("/sys/class/drm")):
            if not entry.startswith("card"):
                continue
            # Only plain cardN (no hyphen = not a connector)
            rest = entry[4:]
            if not rest.isdigit():
                continue
            base = f"/sys/class/drm/{entry}/device"
            if os.path.exists(f"{base}/mem_info_vram_used"):
                cards.append(entry)
    except OSError:
        pass
    return cards


def _read_amd_name(base: str) -> str:
    """Try to read a human-readable GPU name from sysfs."""
    for candidate in (f"{base}/product_name", f"{base}/name"):
        try:
            name = open(candidate).read().strip()
            if name:
                return name
        except OSError:
            pass
    # Fall back to parsing /sys/class/drm/cardN/device/uevent for DRIVER
    try:
        for line in open(f"{base}/uevent").read().splitlines():
            if line.startswith("PCI_ID="):
                return f"AMD GPU ({line.split('=',1)[1]})"
    except OSError:
        pass
    return "AMD GPU"


def shorten_gpu_name(name: str) -> str:
    """Strip vendor prefix for compact TUI display."""
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon RX ", "AMD Radeon ", "AMD "):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            # Also strip trailing architecture notes like "(0000:...)"
            paren = suffix.find("(")
            if paren > 0:
                suffix = suffix[:paren].rstrip()
            return suffix
    return name[:18]


# ---------------------------------------------------------------------------
# HWMonitor
# ---------------------------------------------------------------------------


class HWMonitor:
    """Background thread that polls hardware every *poll_interval* seconds.

    Usage::

        monitor = HWMonitor()
        monitor.start()
        snap = monitor.latest()   # HWSnapshot with latest readings
        monitor.stop()
    """

    def __init__(
        self,
        amd_cards: Optional[list[str]] = None,
        poll_interval: float = 2.0,
    ) -> None:
        # Auto-detect AMD cards if not provided
        self._amd_cards: list[str] = amd_cards if amd_cards is not None else _find_amd_drm_cards()
        self._poll_interval = poll_interval

        self._snapshot: HWSnapshot = HWSnapshot()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="hw-monitor")

        # CPU delta state (for utilisation calculation between polls)
        self._prev_cpu_total: int = 0
        self._prev_cpu_idle: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop (non-blocking)."""
        self._stop_event.set()

    def latest(self) -> HWSnapshot:
        """Return the most recent hardware snapshot (thread-safe copy)."""
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self._poll()
                with self._lock:
                    self._snapshot = snap
            except Exception:  # noqa: BLE001 – never crash the daemon thread
                pass
            self._stop_event.wait(self._poll_interval)

    def _poll(self) -> HWSnapshot:
        gpus: list[GPUStats] = []
        gpus.extend(self._poll_nvidia())
        gpus.extend(self._poll_amd())
        cpu_pct = self._poll_cpu()
        ram_used, ram_total = self._poll_ram()
        return HWSnapshot(
            gpus=gpus,
            cpu_pct=cpu_pct,
            ram_used_gb=ram_used,
            ram_total_gb=ram_total,
        )

    # ------------------------------------------------------------------
    # NVIDIA (nvidia-smi)
    # ------------------------------------------------------------------

    def _poll_nvidia(self) -> list[GPUStats]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            result: list[GPUStats] = []
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    try:
                        result.append(
                            GPUStats(
                                name=parts[0],
                                vram_used_mb=float(parts[1]),
                                vram_total_mb=float(parts[2]),
                                util_pct=float(parts[3]),
                                temp_c=float(parts[4]),
                            )
                        )
                    except ValueError:
                        pass
            return result
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # AMD (sysfs)
    # ------------------------------------------------------------------

    def _poll_amd(self) -> list[GPUStats]:
        result: list[GPUStats] = []
        for card in self._amd_cards:
            base = f"/sys/class/drm/{card}/device"
            try:
                vram_used_mb = int(open(f"{base}/mem_info_vram_used").read().strip()) / (1024 * 1024)
                vram_total_mb = int(open(f"{base}/mem_info_vram_total").read().strip()) / (1024 * 1024)
                util_pct = float(open(f"{base}/gpu_busy_percent").read().strip())

                temp_c = 0.0
                temp_files = glob.glob(f"{base}/hwmon/hwmon*/temp1_input")
                if temp_files:
                    temp_c = float(open(temp_files[0]).read().strip()) / 1000.0

                name = _read_amd_name(base)
                result.append(
                    GPUStats(
                        name=name,
                        vram_used_mb=vram_used_mb,
                        vram_total_mb=vram_total_mb,
                        util_pct=util_pct,
                        temp_c=temp_c,
                    )
                )
            except Exception:  # noqa: BLE001
                pass
        return result

    # ------------------------------------------------------------------
    # CPU (/proc/stat)
    # ------------------------------------------------------------------

    def _poll_cpu(self) -> float:
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            if parts[0] != "cpu" or len(parts) < 6:
                return 0.0
            user, nice, system, idle, iowait = (
                int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            )
            total = user + nice + system + idle + iowait
            idle_total = idle + iowait

            if self._prev_cpu_total == 0:
                # First sample — can't compute delta yet
                self._prev_cpu_total = total
                self._prev_cpu_idle = idle_total
                return 0.0

            dtotal = total - self._prev_cpu_total
            didle = idle_total - self._prev_cpu_idle
            self._prev_cpu_total = total
            self._prev_cpu_idle = idle_total

            if dtotal <= 0:
                return 0.0
            return max(0.0, min(100.0, 100.0 * (1.0 - didle / dtotal)))
        except Exception:  # noqa: BLE001
            return 0.0

    # ------------------------------------------------------------------
    # RAM (/proc/meminfo)
    # ------------------------------------------------------------------

    def _poll_ram(self) -> tuple[float, float]:
        try:
            mem: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, rest = line.partition(":")
                    mem[k.strip()] = int(rest.split()[0])
            total_gb = mem.get("MemTotal", 0) / (1024 * 1024)
            avail_gb = mem.get("MemAvailable", 0) / (1024 * 1024)
            return total_gb - avail_gb, total_gb
        except Exception:  # noqa: BLE001
            return 0.0, 0.0
