#!/usr/bin/env python3
"""Runner script that generates all SVG diagrams for the documentation."""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "generate_selection_flowchart.py",
    "generate_smbo_loop.py",
    "generate_category_diagrams.py",
    "generate_algorithm_flowcharts.py",
]


def main():
    failed = []
    for script_name in SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        print(f"Running {script_name}...")
        result = subprocess.run(  # noqa: S603
            [sys.executable, script_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr.strip()}")
            failed.append(script_name)
        else:
            print(f"  {result.stdout.strip()}")

    print()
    if failed:
        print(f"FAILED scripts: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All diagrams generated successfully.")


if __name__ == "__main__":
    main()
