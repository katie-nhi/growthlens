"""Run every test file in this directory and aggregate the results.

Usage:  python tests/run_all.py

Exits non-zero if any test file fails. No real LLM API key is required: all
AI code paths are exercised with monkeypatched clients.
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    test_files = sorted(
        name for name in os.listdir(HERE) if name.startswith("test_") and name.endswith(".py")
    )
    failed_files = []
    for name in test_files:
        print("=" * 70)
        print("Running", name)
        print("=" * 70)
        result = subprocess.run([sys.executable, os.path.join(HERE, name)])
        if result.returncode != 0:
            failed_files.append(name)

    print("=" * 70)
    if failed_files:
        print("FAILED files:", ", ".join(failed_files))
        return 1
    print(f"ALL {len(test_files)} test files passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
