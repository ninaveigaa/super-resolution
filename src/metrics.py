"""
src/metrics.py

Per-EPOCH metrics tracking (not per-iteration), organized as:

    measurements/
    ├── args/
    │   └── {model_name}.csv              -- one row per run, one column per
    │                                          argument (all runs of that model)
    └── metrics/
        └── {model_name}/
            └── {run_id}.csv          -- one row per epoch, for that
                                                 specific run

Each epoch's row records both training and validation loss (per stage --
xy and z/final), PSNR and SSIM (per stage), a timestamp, elapsed time, and
RAM usage.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil


def generate_run_id(model_name: str, timestamp: datetime = None) -> str:
    """Builds a unique, sortable run identifier: '{model_name}_{YYMMDD_HHMM}'.
    """
    timestamp = timestamp or datetime.now()
    return f"{model_name}_{timestamp.strftime('%y%m%d_%H%M')}"


def save_args(args, model_name: str, base_dir: str = "measurements") -> str:
    """Given an `argparse.Namespace` (or a plain dict) of training arguments,
    generates a run_id and appends one row to
    {base_dir}/args/{model_name}.csv -- a per-model table, with one row per
    run and one column per argument.

    Returns the generated run_id, to reuse when constructing MetricsTracker.
    """
    run_id = generate_run_id(model_name)
    args_dir = Path(base_dir) / "args"
    args_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args) if not isinstance(args, dict) else dict(args)
    record = {"run_id": run_id, "model_name": model_name,
              "saved_at": datetime.now().isoformat(), **args_dict}

    registry_path = args_dir / f"{model_name}.csv"
    row_df = pd.DataFrame([record])
    header_needed = not registry_path.exists()
    row_df.to_csv(registry_path, mode="a", index=False, header=header_needed)

    return run_id


def get_run_args(model_name: str, run_id: str, base_dir: str = "measurements") -> dict:
    """Looks up a single run's arguments from {base_dir}/args/{model_name}.csv,
    indexed by run_id. Returns the matching row as a dict.

    Raises KeyError if no run with that run_id is found for this model.
    """
    registry = load_model_registry(model_name, base_dir=base_dir)
    if registry.empty:
        raise KeyError(f"No registry found for model '{model_name}' in {base_dir}/args/.")

    match = registry[registry["run_id"] == run_id]
    if match.empty:
        raise KeyError(f"run_id '{run_id}' not found in {model_name}.csv.")

    return match.iloc[0].to_dict()


def load_model_registry(model_name: str, base_dir: str = "measurements") -> pd.DataFrame:
    """Reads back the full registry of past runs for a given model, as a
    pandas DataFrame (one row per run, one column per argument)."""
    registry_path = Path(base_dir) / "args" / f"{model_name}.csv"
    if not registry_path.exists():
        return pd.DataFrame()
    return pd.read_csv(registry_path)


class MetricsTracker:
    """Logs one record per EPOCH (not per iteration) to
    `{base_dir}/metrics/{model_name}/{run_id}.csv`.
    """

    def __init__(self, model_name: str, run_id: str, base_dir: str = "measurements"):
        self.model_name = model_name
        self.run_id = run_id
        self.csv_path = Path(base_dir) / "metrics" / model_name / f"{run_id}_metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._train_start_time = None
        self._process = psutil.Process()
        self._header_written = self.csv_path.exists()

    def start_training(self):
        """Call once, right before the training loop begins."""
        self._train_start_time = time.time()

    def log_epoch(self, epoch: int,
                  train_loss_xy: float = None, train_loss_z: float = None,
                  val_loss_xy: float = None, val_loss_z: float = None,
                  psnr_xy: float = None, psnr_final: float = None,
                  ssim_xy: float = None, ssim_final: float = None) -> dict:
        """Appends one record for the given epoch to the CSV. Returns the
        record written. Any field not applicable can be left as None (kept
        as NaN in the CSV)."""
        now = time.time()
        elapsed_total_sec = (now - self._train_start_time
                              if self._train_start_time is not None else None)
        ram_mb = self._process.memory_info().rss / 1e6

        record = {
            "run_id": self.run_id,
            "epoch": epoch,
            "timestamp": now,
            "elapsed_total_sec": elapsed_total_sec,
            "train_loss_xy": train_loss_xy,
            "train_loss_z": train_loss_z,
            "val_loss_xy": val_loss_xy,
            "val_loss_z": val_loss_z,
            "psnr_xy": psnr_xy,
            "psnr_final": psnr_final,
            "ssim_xy": ssim_xy,
            "ssim_final": ssim_final,
            "ram_mb": ram_mb,
        }

        row_df = pd.DataFrame([record])
        row_df.to_csv(
            self.csv_path, mode="a", index=False,
            header=not self._header_written,
        )
        self._header_written = True

        return record

    def load_history(self) -> pd.DataFrame:
        """Reads back the full history logged so far, as a pandas DataFrame
        (safe to call mid-training)."""
        if not self.csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.csv_path)
