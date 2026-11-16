"""
SmartBiz model operations: a small model registry, a drift check, and an idempotent job runner.
Standard library plus numpy and joblib only.
"""
import json
import logging
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np

log = logging.getLogger('smartbiz.ops')


class ModelRegistry:
    """Versioned models on disk. Only a model that passed the trust gate can be promoted."""

    def __init__(self, root='registry'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, name, model, features, metrics, trusted, data_hash=''):
        version = time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1_000_000:06d}'
        folder = self.root / name / version
        folder.mkdir(parents=True)
        joblib.dump({'model': model, 'features': features}, folder / 'model.joblib')
        meta = {'name': name, 'version': version, 'metrics': metrics, 'trusted': bool(trusted),
                'data_hash': data_hash, 'stage': 'candidate'}
        (folder / 'meta.json').write_text(json.dumps(meta, indent=2))
        return version

    def promote(self, name, version):
        meta = json.loads((self.root / name / version / 'meta.json').read_text())
        if not meta['trusted']:
            raise ValueError('Only a model that passed the trust gate can be promoted')
        current = self.current(name)
        (self.root / name / 'CURRENT').write_text(version)
        if current:
            (self.root / name / 'PREVIOUS').write_text(current)

    def rollback(self, name):
        previous = self.root / name / 'PREVIOUS'
        if not previous.exists():
            raise ValueError('Nothing to roll back to')
        (self.root / name / 'CURRENT').write_text(previous.read_text())

    def current(self, name):
        p = self.root / name / 'CURRENT'
        return p.read_text().strip() if p.exists() else None

    def load(self, name, version=None):
        version = version or self.current(name)
        if not version:
            raise FileNotFoundError('No promoted model for ' + name)
        return joblib.load(self.root / name / version / 'model.joblib')


def psi(expected, actual, bins=10):
    """Population stability index. Below 0.1 steady, 0.1 to 0.25 watch, above 0.25 drift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


class JobRunner:
    """Runs named jobs once per day, retries failures, and is safe to start twice."""

    def __init__(self, state_file='job_state.json'):
        self.state_file = Path(state_file)
        self.jobs = {}
        self.done = set(json.loads(self.state_file.read_text())) if self.state_file.exists() else set()

    def job(self, name, retries=2):
        def decorate(fn):
            self.jobs[name] = (fn, retries)
            return fn
        return decorate

    def run(self, name, run_date=None, **kwargs):
        key = f'{name}:{run_date or date.today().isoformat()}'
        if key in self.done:
            log.info('skip %s: already done', key)
            return 'skipped'
        fn, retries = self.jobs[name]
        for attempt in range(retries + 1):
            try:
                fn(**kwargs)
                self.done.add(key)
                self.state_file.write_text(json.dumps(sorted(self.done)))
                return 'ok'
            except Exception:
                log.exception('job %s attempt %d failed', key, attempt + 1)
                time.sleep(min(2 ** attempt, 30))
        return 'failed'
