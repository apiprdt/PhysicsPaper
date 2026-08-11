import os
import json
import datetime
from typing import Any, Dict


class AuditLogger:

    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    def log_experiment(self, run_data: Dict[str, Any]):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            **run_data
        }
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
