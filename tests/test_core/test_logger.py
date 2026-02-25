from __future__ import annotations
import json
import os
import pytest
from ultrasonic_weld_master.core.logger import StructuredLogger

class TestStructuredLogger:
    def test_log_operation(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = StructuredLogger(log_dir=log_dir)
        logger.log_operation(session_id="sess1", event_type="calculation.started",
                             user_action="generate_parameters", data={"inputs": {"material": "Cu"}})
        ops_file = os.path.join(log_dir, "operations.jsonl")
        assert os.path.exists(ops_file)
        with open(ops_file) as f:
            record = json.loads(f.readline())
        assert record["event_type"] == "calculation.started"
        assert record["session_id"] == "sess1"
        assert "timestamp" in record

    def test_log_calculation(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = StructuredLogger(log_dir=log_dir)
        logger.log_calculation(session_id="sess2", inputs={"material": "Al", "layers": 40},
                               outputs={"amplitude_um": 30.0}, intermediate={"power_density": 2.5})
        calc_file = os.path.join(log_dir, "calculations.jsonl")
        assert os.path.exists(calc_file)
        with open(calc_file) as f:
            record = json.loads(f.readline())
        assert record["inputs"]["material"] == "Al"
        assert record["outputs"]["amplitude_um"] == 30.0

    def test_multiple_entries(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = StructuredLogger(log_dir=log_dir)
        for i in range(5):
            logger.log_operation(session_id="s%d" % i, event_type="test", data={"i": i})
        ops_file = os.path.join(log_dir, "operations.jsonl")
        with open(ops_file) as f:
            lines = f.readlines()
        assert len(lines) == 5
