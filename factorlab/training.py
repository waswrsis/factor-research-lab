"""Train-only scaling and chronological holdout calibration of residuals."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .dataset import Dataset
from .graph import GraphSpec, from_dict, parameter_count, to_dict
from .models import build_model
from .protocol import Protocol, project_path, save_json


@dataclass
class FittedModel:
    spec: GraphSpec | None
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    residual: torch.nn.Module | None
    alpha: float
    diagnostics: dict

    def predict(self, data: Dataset, device: str = 'cpu') -> np.ndarray:
        if len(data.features) != len(set(data.features)):
            raise ValueError('Duplicate feature names')
        if data.x.ndim != 3 or data.x.shape[0] != len(data.y) or data.scale.shape != data.y.shape:
            raise ValueError('Inconsistent prediction sample shapes')
        if not np.isfinite(data.x).all() or not np.isfinite(data.scale).all() or (data.scale <= 0).any():
            raise ValueError('Prediction input and positive scale must be finite')
        if not len(data.y):
            return np.empty(0, dtype=np.float32)
        columns = [data.features.index(k) for k in self.feature_names]
        x = np.clip((data.x[:, :, columns] - self.mean) / self.std, -20, 20)
        result = x[:, -1] @ self.coef + self.intercept
        if self.residual is not None and self.alpha > 0:
            self.residual.to(device).eval()
            pieces = []
            with torch.no_grad():
                for start in range(0, len(x), 512):
                    xb = torch.from_numpy(x[start:start+512]).to(device)
                    pieces.append(self.residual(xb).cpu().numpy())
            result += self.alpha * np.concatenate(pieces)
        limit = self.diagnostics['prediction_clip']
        return np.clip(result * data.scale, -limit, limit)

    def save(self, directory: str | Path) -> Path:
        path = project_path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / 'arrays.npz', mean=self.mean, std=self.std,
                 coef=self.coef, intercept=np.asarray(self.intercept))
        save_json(path / 'model.json', {'spec': to_dict(self.spec) if self.spec else None,
                                      'features': list(self.feature_names), 'alpha': self.alpha,
                                      'diagnostics': self.diagnostics})
        if self.residual is not None:
            torch.save({k: v.detach().cpu() for k, v in self.residual.state_dict().items()},
                       path / 'weights.pt')
        return path

    @classmethod
    def load(cls, directory: str | Path) -> FittedModel:
        path = project_path(directory)
        metadata = json.loads((path / 'model.json').read_text('utf-8'))
        arrays = np.load(path / 'arrays.npz', allow_pickle=False)
        spec = from_dict(metadata['spec']) if metadata['spec'] else None
        network = build_model(spec, seed=0) if spec else None
        if network is not None:
            network.load_state_dict(torch.load(path / 'weights.pt', weights_only=True,
                                               map_location='cpu'))
        return cls(spec, tuple(metadata['features']), arrays['mean'], arrays['std'],
                   arrays['coef'], float(arrays['intercept']), network,
                   metadata['alpha'], metadata['diagnostics'])


def fit_model(data: Dataset, train: np.ndarray, protocol: Protocol,
              spec: GraphSpec | None = None, seed: int = 17,
              device: str = 'cpu') -> FittedModel:
    started = time.monotonic()
    if train.dtype != np.bool_ or train.ndim != 1 or len(train) != len(data.y):
        raise ValueError('Training mask must be boolean and match samples')
    if data.y.ndim != 1 or data.scale.shape != data.y.shape or len(data.meta) != len(data.y):
        raise ValueError('Targets, scales and metadata must align')
    if data.x.ndim != 3 or data.x.shape != (len(data.y), protocol.lookback, len(data.features)):
        raise ValueError('Input shape differs from frozen lookback/feature schema')
    if data.x.dtype != np.float32 or not np.isfinite(data.x).all():
        raise ValueError('Model input must be finite float32')
    if not np.isfinite(data.y).all() or not np.isfinite(data.scale).all() or (data.scale <= 0).any():
        raise ValueError('Targets and positive causal scales must be finite')
    if spec is not None and (spec.lookback != protocol.lookback or
                             parameter_count(spec) > protocol.max_parameters):
        raise ValueError('Graph violates frozen lookback/parameter budget')
    indices = np.flatnonzero(train)
    if len(indices) < 300:
        raise ValueError('At least 300 training samples required')
    columns = [data.features.index(k) for k in spec.features] if spec else list(
        range(len(data.features)))
    names = tuple(data.features[i] for i in columns)
    fit_indices = indices
    calibration_indices = np.array([], dtype=int)
    if spec is not None:
        times = sorted(data.meta.loc[train, 'available_at'].unique())
        split_time = times[int(len(times) * 0.8)]
        fit_indices = np.flatnonzero(train & (
            data.meta.label_available_at < split_time).to_numpy())
        calibration_indices = np.flatnonzero(train & (
            data.meta.available_at >= split_time).to_numpy())
        if len(fit_indices) < 200 or len(calibration_indices) < 50:
            raise ValueError('Insufficient purged calibration history')
    x_fit = data.x[fit_indices][:, :, columns]
    mean = x_fit.mean(axis=(0, 1)).astype(np.float32)
    std = np.maximum(x_fit.std(axis=(0, 1)), 1e-6).astype(np.float32)
    x_fit = np.clip((x_fit - mean) / std, -20, 20)
    target = data.y[fit_indices] / data.scale[fit_indices]
    base = Ridge(alpha=100.0).fit(x_fit[:, -1], target)
    network = None
    alpha, losses = 0.0, []
    if spec is not None:
        # The network's hidden layers and gate are jointly trained; the final
        # residual amplitude uses a disjoint chronological calibration block.
        torch.manual_seed(seed)
        network = build_model(spec, seed=seed).to(device)
        optimizer_class = torch.optim.AdamW if spec.optimizer == 'adamw' else torch.optim.Adam
        optimizer = optimizer_class(network.parameters(), lr=spec.learning_rate,
                                    weight_decay=1e-4)
        criterion = torch.nn.SmoothL1Loss() if spec.loss == 'huber' else torch.nn.MSELoss()
        residual_target = target - base.predict(x_fit[:, -1])
        rng = np.random.default_rng(seed)
        for _ in range(protocol.epochs):
            permutation = rng.permutation(len(x_fit))
            total = 0.0
            network.train()
            for start in range(0, len(x_fit), protocol.batch_size):
                batch = permutation[start:start+protocol.batch_size]
                xb = torch.from_numpy(x_fit[batch]).to(device)
                yb = torch.from_numpy(residual_target[batch].astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(xb), yb)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            losses.append(total / len(x_fit))
        network.eval()
        cal = np.clip((data.x[calibration_indices][:, :, columns] - mean) / std, -20, 20)
        with torch.no_grad():
            delta = np.concatenate([network(torch.from_numpy(cal[i:i+512]).to(device))
                                    .cpu().numpy() for i in range(0, len(cal), 512)])
        error = data.y[calibration_indices] / data.scale[calibration_indices] - base.predict(
            cal[:, -1])
        alpha = float(np.clip(np.dot(error, delta) / (np.dot(delta, delta) + 1e-6), 0, 1))
        network.cpu()
    diagnostics = {'fit_samples': len(fit_indices), 'calibration_samples': len(calibration_indices),
                   'fit_label_cutoff': str(data.meta.iloc[fit_indices].label_available_at.max()),
                   'calibration_cutoff': (str(data.meta.iloc[calibration_indices]
                                             .label_available_at.max())
                                          if len(calibration_indices) else None),
                   'alpha': alpha, 'losses': losses, 'seed': seed,
                   'elapsed_seconds': time.monotonic()-started,
                   'stacking': 'purged_chronological_holdout_no_refit',
                   'device': device, 'prediction_clip': protocol.prediction_clip}
    return FittedModel(spec, names, mean, std, base.coef_.astype(np.float32),
                       float(base.intercept_), network, alpha, diagnostics)
