"""GPU discovery helpers for llama-bench."""
from __future__ import annotations

import logging
import os
import re
import subprocess

from typing import Optional

logger = logging.getLogger(__name__)

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


def list_devices_from_server(binary: str) -> dict[int, str]:
    """Run ``binary --list-devices`` and parse the "Available devices:" section.

    Returns a dict mapping numeric index → device name, e.g.::

        {0: "Vulkan0", 1: "Vulkan1"}

    Lines like ``Vulkan0: NVIDIA GeForce RTX 2080 Ti ...`` are matched.
    Returns an empty dict if the command fails or produces no parseable output.
    """
    try:
        result = subprocess.run(
            [binary, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout + result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as exc:
        logger.debug("list_devices_from_server: failed to run %r: %s", binary, exc)
        return {}

    devices: dict[int, str] = {}
    in_section = False
    for line in output.splitlines():
        if re.search(r"Available devices\s*:", line, re.IGNORECASE):
            in_section = True
            continue
        if in_section:
            # Match "Vulkan0: ..." or "  Vulkan1: ..." (with leading whitespace)
            m = re.match(r"\s*(Vulkan(\d+))\s*:", line)
            if m:
                name = m.group(1)   # e.g. "Vulkan0"
                idx = int(m.group(2))  # e.g. 0
                devices[idx] = name
            elif line.strip() and re.match(r"^\S", line):
                # Non-indented non-blank line ends the section
                break
    return devices


def build_env(vk_visible_devices: Optional[str]) -> dict:
    """Return a copy of :data:`os.environ`.

    If *vk_visible_devices* is not ``None``, sets ``GGML_VK_VISIBLE_DEVICES``.
    When omitted (``None``) the environment variable is left unset so the server
    uses all available Vulkan devices.
    """
    env = os.environ.copy()
    if vk_visible_devices is not None:
        env["GGML_VK_VISIBLE_DEVICES"] = vk_visible_devices
    return env
