"""Run:  python test_assistant.py    (needs the daily metrics CSV in the same folder or set DATA)"""
import os
import re

import pandas as pd

from smartbiz_assistant import Analyzer, Assistant, numbers_in

DATA = os.environ.get('DATA', '../data10k/smartbiz_daily_business_metrics.csv')
df = pd.read_csv(DATA)
ids = df.groupby('business_type')['business_id'].first().to_dict()

# 1. every business type produces a briefing without errors
for t, bid in ids.items():
    out = Assistant(df[df.business_id == bid]).ask('how did we do yesterday')
    assert 'Last 7 days' in out and 'Looking ahead' in out, t
print('briefing OK for', len(ids), 'business types')

# 2. every number in every suggestion message comes from its facts (no invented figures)
checked = 0
for bid, g in df[df.business_id.isin(ids.values())].groupby('business_id'):
    for s in Analyzer(g).suggestions():
        facts = [float(v) for v in s.facts.values() if isinstance(v, (int, float))]
        for n in numbers_in(s.message):
            assert any(abs(n - f) < 1e-9 for f in facts) or n in (7.0, 4.0, 2.0, 60.0), (s.key, n, s.facts)
        checked += 1
print('number grounding OK on', checked, 'suggestions')

# 3. staff-safe wording and the no-action rule: nothing says it has already done something
bot = Assistant(df[df.business_id == ids['retail']])
text = bot.ask('advice') + bot.ask('why 1')
assert not re.search(r'\b(I have (sent|ordered|changed)|done for you)\b', text)
assert 'Nothing happens until you approve it' in text
print('no-action rule OK')

# 4. a language model that invents a number is ignored
evil = lambda p: 'Great news, you made 999,999 today!'
polite = lambda p: p.split('\n\n', 1)[1].replace('you took in', 'you brought in')
base = Assistant(df[df.business_id == ids['retail']]).ask('sales')
assert Assistant(df[df.business_id == ids['retail']], call_llm=evil).ask('sales') == base
assert 'brought in' in Assistant(df[df.business_id == ids['retail']], call_llm=polite).ask('sales') or 'you took in' not in base
print('LLM number guard OK')

# 5. intents, including Swahili keywords, and dismiss
for q, kw in [('mauzo', 'Last 7 days'), ('ushauri', 'what I would do'), ('hello', 'Hello'), ('xyz', 'not sure')]:
    assert kw.lower() in Assistant(df[df.business_id == ids['retail']]).ask(q).lower(), q
bot = Assistant(df[df.business_id == ids['retail']])
bot.ask('advice'); first = bot.shown[0].key; bot.ask('dismiss 1')
assert first not in [s.key for s in bot._live()]
print('intents and dismiss OK')
print('ALL TESTS PASSED')
