"""CLI interface tests."""
from __future__ import annotations

import subprocess
import sys
import os

import pytest

_CLI_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cli.py")


class TestCLI:
    def test_version(self):
        result = subprocess.run(
            [sys.executable, _CLI_PATH, "--version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_help(self):
        result = subprocess.run(
            [sys.executable, _CLI_PATH, "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "calculate" in result.stdout

    def test_calculate_default(self):
        result = subprocess.run(
            [sys.executable, _CLI_PATH, "calculate"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "amplitude" in result.stdout.lower()

    def test_calculate_with_args(self):
        result = subprocess.run(
            [sys.executable, _CLI_PATH, "calculate",
             "--application", "li_battery_tab",
             "--upper-material", "Al", "--upper-thickness", "0.012", "--upper-layers", "40",
             "--lower-material", "Cu", "--lower-thickness", "0.3",
             "--width", "5", "--length", "25"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "amplitude" in result.stdout.lower()
        assert "energy" in result.stdout.lower()

    def test_calculate_general_metal(self):
        result = subprocess.run(
            [sys.executable, _CLI_PATH, "calculate",
             "--application", "general_metal",
             "--upper-material", "Cu", "--upper-thickness", "0.5",
             "--lower-material", "Steel", "--lower-thickness", "1.0"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "amplitude" in result.stdout.lower()

    def test_calculate_with_json_output(self, tmp_path):
        result = subprocess.run(
            [sys.executable, _CLI_PATH, "calculate",
             "--output", str(tmp_path), "--format", "json"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        json_files = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
        assert len(json_files) == 1
