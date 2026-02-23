"""GPU discovery helpers for llama-bench."""
from __future__ import annotations

import os
import re
import subprocess


def discover_vulkan_gpus() -> list[dict]:
    """Probe the host for Vulkan-capable GPUs via ``vulkaninfo``.

    Returns a list of ``{"index": int, "name": str}`` dicts, one per GPU.
    Returns an empty list if ``vulkaninfo`` is not installed or produces no
    parseable output.
    """
    for args in (
        ["vulkaninfo", "--summary"],
        ["vulkaninfo"],
    ):
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

        gpus: list[dict] = []

        # --summary format: lines like "GPU0: deviceName = NVIDIA GeForce RTX 4090"
        for m in re.finditer(r"GPU(\d+):\s+deviceName\s*=\s*(.+)", output):
            gpus.append({"index": int(m.group(1)), "name": m.group(2).strip()})

        if gpus:
            return gpus

        # Fallback: full vulkaninfo format
        # "deviceName     = NVIDIA GeForce RTX 4090"
        device_names = re.findall(r"deviceName\s*=\s*(.+)", output)
        for i, name in enumerate(device_names):
            gpus.append({"index": i, "name": name.strip()})

        if gpus:
            return gpus

    return []


def default_vk_devices(gpus: list[dict]) -> str:
    """Return a comma-separated device index string for all discovered GPUs.

    Falls back to ``"0"`` if no GPUs were discovered.
    """
    if not gpus:
        return "0"
    return ",".join(str(g["index"]) for g in gpus)


def build_env(vk_visible_devices: str) -> dict:
    """Return a copy of :data:`os.environ` with ``GGML_VK_VISIBLE_DEVICES`` set."""
    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = vk_visible_devices
    return env
