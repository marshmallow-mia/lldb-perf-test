"""llama-bench: llama.cpp llama-server benchmarking and optimizer CLI."""
import subprocess
import os

__version__ = "0.1"
EXPECTED_LLAMA_SERVER_VERSION = "8133"  # numeric build version; update to match your llama-server build


def _git_commit_timestamp() -> str:
    """Return the last git commit timestamp as a short string, e.g. '2026-02-25 10:18 UTC'.

    Falls back to the mtime of this file if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "-C", os.path.dirname(__file__), "log", "-1", "--format=%ai"],
            capture_output=True, text=True, timeout=3,
        )
        ts = result.stdout.strip()
        if ts:
            # '2026-02-25 10:18:19 +0000' → '2026-02-25 10:18 UTC'
            parts = ts.split()
            return f"{parts[0]} {parts[1][:5]} UTC"
    except Exception:
        pass
    # fallback: mtime of this file
    try:
        mtime = os.path.getmtime(__file__)
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "unknown"


GIT_TIMESTAMP = _git_commit_timestamp()
