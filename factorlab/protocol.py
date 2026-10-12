"""Immutable research protocol: evaluation rules are not search parameters."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# Windows work stays on the user-designated D drive. A Linux deployment is a
# new source identity rooted at its checkout; it never resumes a Windows run.
ROOT = (Path('D:/FactorResearch') if os.name == 'nt' else
        Path(os.environ.get('FACTOR_RESEARCH_ROOT', str(Path(__file__).resolve().parent.parent))))


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def project_path(path: str | Path) -> Path:
    """Resolve paths, including junctions, before any project artifact write."""
    value = Path(path)
    value = (ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    if not value.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Project path must remain within {ROOT}')
    return value


def save_json(path: str | Path, value: object) -> Path:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(target)
    return target


@dataclass(frozen=True)
class Protocol:
    version: str = 'prototype-1'
    symbols: tuple[str, ...] = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
    start: str = '2025-01-01'
    end: str = '2025-07-01'
    test_starts: tuple[str, ...] = ('2025-04-01', '2025-05-01', '2025-06-01')
    horizon_hours: int = 12
    availability_delay_seconds: int = 5
    lookback: int = 48
    max_lookback: int = 168
    epochs: int = 4
    batch_size: int = 256
    seeds: tuple[int, ...] = (17, 29)
    candidates_per_search: int = 4
    max_parameters: int = 1_000_000
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    gross_limit: float = 0.9
    single_name_limit: float = 0.3
    risk_aversion: float = 10.0
    turnover_penalty: float = 0.001
    volatility_floor: float = 0.001
    prediction_clip: float = 0.1
    initial_equity: float = 100_000.0
    bootstrap_block_hours: int = 48
    bootstrap_samples: int = 1000
    promotion_min_blocks: int = 20
    max_drawdown: float = 0.20
    stress_cost_multipliers: tuple[float, ...] = (1.0, 2.0)
    live_trading: bool = False
    universe_mode: str = 'fixed_survivors_engineering_demo'

    def __post_init__(self):
        for name in ('horizon_hours', 'availability_delay_seconds', 'lookback', 'max_lookback',
                     'epochs', 'batch_size', 'candidates_per_search', 'max_parameters',
                     'bootstrap_block_hours', 'bootstrap_samples', 'promotion_min_blocks'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('fee_bps', 'slippage_bps', 'risk_aversion', 'turnover_penalty'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('volatility_floor', 'prediction_clip', 'initial_equity', 'max_drawdown'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not self.seeds or any(type(x) is not int or not 0 <= x < 2**31 for x in self.seeds):
            raise ValueError('Training seeds must be nonnegative integers')
        if len(set(self.symbols)) != len(self.symbols) or not self.symbols:
            raise ValueError('A nonempty unique symbol set is required')
        if any(not math.isfinite(x) or x < 1 for x in self.stress_cost_multipliers):
            raise ValueError('Cost stresses must be finite and at least one')
        if self.live_trading:
            raise ValueError('Live trading is not implemented')
        if self.horizon_hours != 12 or self.availability_delay_seconds <= 0:
            raise ValueError('Prototype task is delayed 12-hour simple returns')
        if not 1 <= self.lookback <= self.max_lookback or self.max_lookback > 672:
            raise ValueError('Lookback resource limits exceeded')
        if not (0 < self.single_name_limit <= self.gross_limit <= 1):
            raise ValueError('Invalid research risk constraints')
        if any(x <= 0 for x in (self.epochs, self.batch_size, self.candidates_per_search)):
            raise ValueError('Research budgets must be positive')
        boundaries = [utc(self.start), *(utc(x) for x in self.test_starts), utc(self.end)]
        if boundaries != sorted(set(boundaries)):
            raise ValueError('Date boundaries must be strictly increasing')

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))

    def freeze(self, directory: str | Path) -> Path:
        target = project_path(directory) / 'protocol.json'
        value = {'sha256': self.digest, 'protocol': asdict(self)}
        if target.exists() and json.loads(target.read_text('utf-8')) != json.loads(
            json.dumps(value)
        ):
            raise ValueError('Existing protocol differs; create a new experiment')
        return save_json(target, value)

    @classmethod
    def read(cls, path: str | Path) -> Protocol:
        value = json.loads(Path(path).read_text('utf-8-sig'))
        spec = value.get('protocol', value)
        for key in ('symbols', 'test_starts', 'seeds', 'stress_cost_multipliers'):
            if key in spec:
                spec[key] = tuple(spec[key])
        obj = cls(**spec)
        if 'sha256' in value and value['sha256'] != obj.digest:
            raise ValueError('Protocol hash mismatch')
        return obj


def utc(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError('A concrete timestamp is required')
    return result.tz_localize('UTC') if result.tzinfo is None else result.tz_convert('UTC')
