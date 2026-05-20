import random
import string
from datetime import datetime
from pathlib import Path

import yaml


def create_run_dir(log_dir: str = "logs") -> Path:
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    run_dir = Path(log_dir) / f"{timestamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_hparams(config: dict, run_dir: Path) -> None:
    with open(run_dir / "hparams.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
