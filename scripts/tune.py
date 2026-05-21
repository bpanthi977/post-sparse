import argparse
import json
import os
import sys
from pathlib import Path

import wandb
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train import train

POSTSPARSE_SWEEP = {
    "method": "bayes",
    "metric": {"name": "val/i2t_r1", "goal": "maximize"},
    "parameters": {
        "lr":             {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
        # "weight_decay":   {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
        # "proj_dim":       {"values": [4096, 8192, 16384]},
        # "batch_size":     {"values": [256, 512, 1024]},
        # "warmup_fraction":{"distribution": "uniform", "min": 0.01, "max": 0.1},
    },
}

SWEEP_PARAM_MAP = {
    "lr":             ("training", "lr"),
    "weight_decay":   ("training", "weight_decay"),
    "batch_size":     ("training", "batch_size"),
    "warmup_fraction":("training", "warmup_fraction"),
    "proj_dim":       ("model",    "proj_dim"),
}


def sweep(study_name: str, config_path: str, sweep_config: dict, count: int = 20):
    sweep_id_file = os.path.join("logs", study_name, "sweep_id")
    base_config_file = os.path.join("logs", study_name, "base_config.yaml")

    if os.path.exists(sweep_id_file):
        with open(sweep_id_file) as f:
            sweep_id = f.read().strip()
        with open(base_config_file) as f:
            base_config = yaml.safe_load(f)
        print(f"Resuming sweep {sweep_id}")
    else:
        os.makedirs(os.path.join("logs", study_name), exist_ok=True)
        with open(config_path) as f:
            base_config = yaml.safe_load(f)

        base_config["logging"]["log_dir"] = base_config["logging"]["log_dir"] + "/" + study_name
        project = base_config["logging"]["wandb_project"]
        sweep_id = wandb.sweep(sweep_config, project=project)

        with open(sweep_id_file, "w") as f:
            f.write(sweep_id)
        with open(base_config_file, "w") as f:
            yaml.dump(base_config, f)
        with open(os.path.join("logs", study_name, "sweep_config.json"), "w") as f:
            json.dump(sweep_config, f, indent=2)

        print(f"Created sweep {sweep_id}")

    project = base_config["logging"]["wandb_project"]

    def trial():
        wandb.init(project=project)
        config = yaml.safe_load(yaml.dump(base_config))  # deep copy via round-trip
        for key, val in wandb.config.items():
            section, param = SWEEP_PARAM_MAP[key]
            config[section][param] = val
        train(config)
        wandb.finish()

    wandb.agent(sweep_id, function=trial, count=count, project=project)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W&B hyperparameter sweep for PostSparse")
    parser.add_argument("config", default="config/base.yaml", nargs="?")
    parser.add_argument("--study", default="sweep")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    sweep(args.study, args.config, POSTSPARSE_SWEEP, args.count)
