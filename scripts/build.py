#!/usr/bin/env python3
"""Build script: compile translations and run PyInstaller."""
from __future__ import annotations

import os
import subprocess
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSLATIONS_DIR = os.path.join(ROOT, "translations")
SPEC_FILE = os.path.join(ROOT, "UltrasonicWeldMaster.spec")


def compile_translations():
    """Compile all .ts files to .qm using pyside6-lrelease."""
    ts_files = glob.glob(os.path.join(TRANSLATIONS_DIR, "*.ts"))
    if not ts_files:
        print("No .ts files found in translations/")
        return

    for ts_file in ts_files:
        qm_file = ts_file.replace(".ts", ".qm")
        print("Compiling %s -> %s" % (os.path.basename(ts_file), os.path.basename(qm_file)))
        result = subprocess.run(
            ["pyside6-lrelease", ts_file, "-qm", qm_file],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("ERROR: %s" % result.stderr)
            sys.exit(1)
        print("  %s" % result.stdout.strip())


def run_pyinstaller():
    """Run PyInstaller with the spec file."""
    print("\nRunning PyInstaller...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", SPEC_FILE],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("PyInstaller build failed!")
        sys.exit(1)
    print("\nBuild complete! Output in dist/UltrasonicWeldMaster/")
    if sys.platform == "darwin":
        print("macOS app bundle: dist/UltrasonicWeldMaster.app")


def main():
    print("=" * 60)
    print("UltrasonicWeldMaster Build Script")
    print("=" * 60)

    print("\n[1/2] Compiling translations...")
    compile_translations()

    print("\n[2/2] Building application...")
    run_pyinstaller()

    print("\nDone!")


if __name__ == "__main__":
    main()
