import tempfile, os
import numpy as np
from sklearn.linear_model import LinearRegression
from ops import ModelRegistry, psi, JobRunner

d = tempfile.mkdtemp()
reg = ModelRegistry(os.path.join(d, 'reg'))
m = LinearRegression().fit([[0], [1], [2]], [0, 1, 2])
bad = reg.register('demand', m, ['x'], {'mae': 9.0}, trusted=False)
try:
    reg.promote('demand', bad); raise SystemExit('should have refused')
except ValueError:
    pass
v1 = reg.register('demand', m, ['x'], {'mae': 8.0}, trusted=True); reg.promote('demand', v1)
v2 = reg.register('demand', m, ['x'], {'mae': 7.5}, trusted=True); reg.promote('demand', v2)
assert reg.current('demand') == v2
reg.rollback('demand'); assert reg.current('demand') == v1
assert abs(reg.load('demand')['model'].predict([[3]])[0] - 3) < 1e-9
print('registry OK')

rng = np.random.default_rng(1)
a = rng.normal(100, 10, 5000)
assert psi(a, rng.normal(100, 10, 5000)) < 0.1
assert psi(a, rng.normal(130, 10, 5000)) > 0.25
assert psi(np.array([1, 1, 2, 2, 3] * 100), np.array([1, 2, 3] * 100)) >= 0
print('drift OK')

calls = {'n': 0, 'flaky': 0}
r = JobRunner(os.path.join(d, 'state.json'))
@r.job('nightly')
def nightly(): calls['n'] += 1
@r.job('flaky', retries=2)
def flaky():
    calls['flaky'] += 1
    if calls['flaky'] < 2: raise RuntimeError('boom')
assert r.run('nightly') == 'ok' and r.run('nightly') == 'skipped' and calls['n'] == 1
assert JobRunner(os.path.join(d, 'state.json')).run('nightly') == 'skipped'
assert r.run('flaky') == 'ok' and calls['flaky'] == 2
print('jobs OK')
print('ALL OK')
