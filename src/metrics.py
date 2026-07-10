"""
src/metrics.py

Per-iteration metrics tracking, designed around a "modelo_data_hora" run
identifier, so that every training session (regardless of model/architecture)
gets a unique, sortable, human-readable ID -- e.g. "dualedsr_20260710_143200".

Each run produces one row in `{log_dir}/{model_name}_runs_registry.csv`
(the "ficha") -- one table per model, one row per run, one column per
argument -- and one file in `{log_dir}/{run_id}_metrics.csv` with the
per-iteration metrics (loss, PSNR, SSIM, time, RAM/GPU), appended
incrementally during training, readable as a pandas DataFrame.

Metrics are split per pipeline stage (XY and Z/final), matching the
convention used in the original DualEDSR training script:
    - loss_xy, loss_z       (per-iteration, from training batches)
    - psnr_xy, psnr_final   (periodic, from validation -- None otherwise)
    - ssim_xy, ssim_final   (periodic, from validation -- None otherwise)
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import torch
import torch.nn.functional as F

try:
    from torchmetrics.functional import structural_similarity_index_measure as _tm_ssim
    _HAS_TORCHMETRICS = True
except ImportError:
    _HAS_TORCHMETRICS = False


def save_args(args, model_name: str, log_dir: str = "logs") -> str:
    """Given an `argparse.Namespace` (or a plain dict) of training arguments,
    generates a run_id and appends one row to
    {log_dir}/{model_name}_runs_registry.csv -- a per-model table, with one
    row per run and one column per argument.

    Each model gets its OWN registry file (since different models have
    different argument sets) -- so all past runs of the SAME model can be
    compared side-by-side in a single, indexable table, without needing a
    separate file per run.

    This establishes a convention: every training script should build its
    args via argparse, then call this function once (right after
    parse_args()) to get back a run_id to use for the rest of that run
    (e.g. when constructing the MetricsTracker).

    Returns the generated run_id.

    NOTE: if different runs of the SAME model use a different SET of
    arguments (e.g., a new hyperparameter was added after some runs were
    already logged), the registry CSV will have missing values for the
    columns that didn't exist in earlier/later runs -- this is expected
    and safe (pandas reads them as NaN), but worth being aware of.
    """
    run_id = generate_run_id(model_name)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args) if not isinstance(args, dict) else dict(args)
    record = {"run_id": run_id, "model_name": model_name,
              "saved_at": datetime.now().isoformat(), **args_dict}

    registry_path = log_dir / f"{model_name}_runs_registry.csv"
    row_df = pd.DataFrame([record])
    header_needed = not registry_path.exists()
    row_df.to_csv(registry_path, mode="a", index=False, header=header_needed)

    return run_id


def get_run_args(model_name: str, run_id: str, log_dir: str = "logs") -> dict:
    """Looks up a single run's arguments from {model_name}_runs_registry.csv,
    indexed by run_id. Returns the matching row as a dict.

    Raises KeyError if no run with that run_id is found for this model.
    """
    registry = load_model_registry(model_name, log_dir=log_dir)
    if registry.empty:
        raise KeyError(f"No registry found for model '{model_name}' in {log_dir}.")

    match = registry[registry["run_id"] == run_id]
    if match.empty:
        raise KeyError(f"run_id '{run_id}' not found in {model_name}_runs_registry.csv.")

    return match.iloc[0].to_dict()


def load_model_registry(model_name: str, log_dir: str = "logs") -> pd.DataFrame:
    """Reads back the full registry of past runs for a given model, as a
    pandas DataFrame (one row per run, one column per argument)."""
    registry_path = Path(log_dir) / f"{model_name}_runs_registry.csv"
    if not registry_path.exists():
        return pd.DataFrame()
    return pd.read_csv(registry_path)


def generate_run_id(model_name: str, timestamp: datetime = None) -> str:
    """Builds a unique, sortable run identifier: '{model_name}_{YYYYMMDD_HHMMSS}'.

    e.g. generate_run_id("dualedsr") -> "dualedsr_20260710_143200"

    Passing `timestamp` explicitly is mainly useful for testing; in normal
    use, it defaults to the current time.
    """
    timestamp = timestamp or datetime.now()
    return f"{model_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"


class MetricsTracker:
    """Logs one record per training iteration, incrementally, to
    `{log_dir}/{run_id}_metrics.csv`.

    Usage:
        run_id = generate_run_id("dualedsr")
        tracker = MetricsTracker(log_dir="logs", run_id=run_id)
        tracker.start_training()

        for iteration in range(total_iterations):
            tracker.start_iteration()
            ... forward, loss, backward ...
            tracker.log_iteration(
                iteration, loss_xy=loss_xy_val, loss_z=loss_z_val,
                psnr_xy=psnr_xy_val, psnr_final=psnr_final_val,
            )

        history_df = tracker.load_history()  # pandas DataFrame, for the dashboard
    """

    def __init__(self, log_dir: str, run_id: str):
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / f"{run_id}_metrics.csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._train_start_time = None
        self._iter_start_time = None
        self._process = psutil.Process()
        self._header_written = self.csv_path.exists()

    def start_training(self):
        """Call once, right before the training loop begins."""
        self._train_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def start_iteration(self):
        """Call at the start of each iteration, to time that iteration."""
        self._iter_start_time = time.time()

    def log_iteration(self, iteration: int, loss_xy: float = None, loss_z: float = None,
                       psnr_xy: float = None, psnr_final: float = None,
                       ssim_xy: float = None, ssim_final: float = None,
                       epoch: int = None, learning_rate: float = None,
                       extra: dict = None) -> dict:
        """Appends one record for the given iteration to the CSV. Returns the
        record written.

        Fields that don't apply to a given iteration (e.g. psnr_xy on an
        iteration where no validation ran) should simply be left as None --
        they will be recorded as empty/NaN in the CSV, and the dashboard
        skips them when plotting.

        `extra` can carry any additional fields (e.g. model-specific losses)
        without changing this function's signature.
        """
        now = time.time()
        iter_time_sec = (now - self._iter_start_time
                          if self._iter_start_time is not None else None)
        elapsed_total_sec = (now - self._train_start_time
                              if self._train_start_time is not None else None)

        ram_mb = self._process.memory_info().rss / 1e6
        gpu_mem_mb = (torch.cuda.max_memory_allocated() / 1e6
                      if torch.cuda.is_available() else None)

        record = {
            "run_id": self.run_id,
            "iteration": iteration,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "loss_xy": loss_xy,
            "loss_z": loss_z,
            "psnr_xy": psnr_xy,
            "psnr_final": psnr_final,
            "ssim_xy": ssim_xy,
            "ssim_final": ssim_final,
            "iter_time_sec": iter_time_sec,
            "elapsed_total_sec": elapsed_total_sec,
            "ram_mb": ram_mb,
            "gpu_mem_mb": gpu_mem_mb,
            "timestamp": now,
        }
        if extra:
            record.update(extra)

        # Append as a single-row CSV write -- avoids holding the full history
        # in memory during training, while still producing a file pandas
        # reads natively.
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


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 2.0) -> float:
    """PSNR between two tensors. Default max_val=2.0 assumes data in [-1, 1]
    (Tanh-normalized), matching the convention used in DualEDSR/DualHAT."""
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


def compute_ssim(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 2.0) -> float:
    """SSIM between two tensors, shape [B, C, H, W]. Uses torchmetrics if
    available (recommended); falls back to a simple manual implementation
    otherwise."""
    if _HAS_TORCHMETRICS:
        return _tm_ssim(sr, hr, data_range=data_range).item()
    return _manual_ssim(sr, hr, data_range=data_range)


def _manual_ssim(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 2.0,
                  window_size: int = 11) -> float:
    """Minimal single-scale SSIM fallback (Gaussian window), used only if
    torchmetrics is not installed."""

    channel = sr.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, dtype=sr.dtype, device=sr.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.T @ g).unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_sr = F.conv2d(sr, window, padding=pad, groups=channel)
    mu_hr = F.conv2d(hr, window, padding=pad, groups=channel)

    mu_sr_sq, mu_hr_sq, mu_sr_hr = mu_sr ** 2, mu_hr ** 2, mu_sr * mu_hr

    sigma_sr_sq = F.conv2d(sr * sr, window, padding=pad, groups=channel) - mu_sr_sq
    sigma_hr_sq = F.conv2d(hr * hr, window, padding=pad, groups=channel) - mu_hr_sq
    sigma_sr_hr = F.conv2d(sr * hr, window, padding=pad, groups=channel) - mu_sr_hr

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_sr_hr + c1) * (2 * sigma_sr_hr + c2)) / \
               ((mu_sr_sq + mu_hr_sq + c1) * (sigma_sr_sq + sigma_hr_sq + c2))
    return ssim_map.mean().item()