#!/usr/bin/env python3
"""
SmartBiz Assistant: plain-English insights and suggestions from business data.

Design rules (same as the SmartBiz AI function catalog):
  1. Every number is computed in code. A language model may only change the words.
  2. Every suggestion carries its confidence, assumptions and the facts behind it.
  3. The assistant never acts on its own: orders, price changes and messages are
     only ever proposed.
  4. Anything it cannot back with the owner's data, it does not say.

Run a chat in the terminal:
    python smartbiz_assistant.py --data smartbiz_daily_business_metrics.csv --business B001
"""
import argparse
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# ----------------------------------------------------------------- small helpers
def rmoney(x):
    """Round money to a friendly figure: 12,437 -> 12,400. Owners do not need cents."""
    a = abs(float(x))
    step = 10 if a < 1000 else 100 if a < 100_000 else 1000
    return int(round(float(x) / step) * step)


def rnum(x, nd=0):
    return round(float(x), nd) if nd else int(round(float(x)))


def confidence_word(c):
    return 'fairly sure' if c >= 0.8 else 'reasonably sure' if c >= 0.65 else 'a rough guess'


def numbers_in(text):
    return [float(t.replace(',', '').rstrip('.')) for t in re.findall('[0-9][0-9,]*[.]?[0-9]*', text)]


# ----------------------------------------------------------------- suggestion record
@dataclass
class Suggestion:
    key: str                 # e.g. 'restock'
    title: str
    message: str             # plain English, built only from `facts`
    mode: str                # 'Propose' (one tap), 'Ask' (needs one fact), 'Tell'
    impact: float            # money gained or protected, in currency units
    confidence: float        # 0 to 1
    urgency: float           # 0 to 1
    assumptions: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    next_step: str = ''

    @property
    def score(self):
        return self.impact * self.confidence * self.urgency


# ----------------------------------------------------------------- the analyst (all numbers here)
class Analyzer:
    def __init__(self, df, currency='KSh'):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['gross_profit'] = df['revenue'] - df['cost_of_goods']
        df['profit'] = df['gross_profit'] - df['expenses']
        self.df, self.cur = df, currency
        self.btype = str(df['business_type'].iloc[-1])
        self.is_service = bool(df['bookings'].sum() > 0)

    def m(self, n):
        return f'{self.cur} {int(n):,}'

    def left(self, n):
        """Plain wording for profit that can also be negative."""
        n = int(n)
        return (f'about {self.m(n)} was left' if n >= 0
                else f'you spent about {self.m(-n)} more than you earned')

    def day_name(self, ts):
        return f"{WEEKDAYS[ts.weekday()]} {ts.day} {ts.strftime('%b')}"

    def usual(self, col):
        """What this column normally is on the same weekday as the latest day."""
        d = self.df
        last, prev = d.iloc[-1], d.iloc[:-1]
        same = prev[prev['day_of_week'] == last['day_of_week']].tail(8)[col]
        if len(same) < 4:
            same = prev.tail(28)[col]
        return float(same.mean())

    # ---- latest day
    def latest(self):
        last = self.df.iloc[-1]
        usual_rev = self.usual('revenue')
        pct = (last['revenue'] / usual_rev - 1) * 100 if usual_rev > 0 else 0.0
        return dict(day=self.day_name(last['date']), weekday=WEEKDAYS[int(last['day_of_week'])],
                    revenue=rmoney(last['revenue']), usual=rmoney(usual_rev), pct=rnum(pct),
                    profit=rmoney(last['profit']), units=int(last['units_sold']))

    def week(self):
        d = self.df
        a, b = d.tail(7), d.iloc[-14:-7]
        ra, rb = a['revenue'].sum(), b['revenue'].sum()
        pct = (ra / rb - 1) * 100 if rb > 0 else 0.0
        gm = (ra - a['cost_of_goods'].sum()) / ra * 100 if ra > 0 else 0.0
        by_day = d.tail(56).groupby('day_of_week')['revenue'].mean()
        return dict(revenue=rmoney(ra), prev=rmoney(rb), pct=rnum(pct), profit=rmoney(a['profit'].sum()),
                    margin=rnum(gm, 1), best=WEEKDAYS[int(by_day.idxmax())], worst=WEEKDAYS[int(by_day.idxmin())])

    # ---- forecast, with its own honesty check
    def forecast(self, days=7):
        d = self.df

        def seasonal(train, wd):
            same = train[train['day_of_week'] == wd].tail(8)['units_sold']
            if len(same) < 4:
                same = train.tail(28)['units_sold']
            return float(same.mean())

        err_m, err_n = [], []
        for i in range(max(35, len(d) - 28), len(d)):
            train, row = d.iloc[:i], d.iloc[i]
            wk = train[train['date'] == row['date'] - pd.Timedelta(days=7)]
            if len(wk):
                err_m.append(abs(seasonal(train, int(row['day_of_week'])) - row['units_sold']))
                err_n.append(abs(wk['units_sold'].iloc[0] - row['units_sold']))
        ok = len(err_m) >= 14 and np.mean(err_m) <= 0.95 * np.mean(err_n)
        rev_per_unit = d['revenue'].tail(28).sum() / max(1, d['units_sold'].tail(28).sum())
        last_date, per_day = d['date'].max(), []
        for k in range(1, days + 1):
            day = last_date + pd.Timedelta(days=k)
            per_day.append((WEEKDAYS[day.weekday()], seasonal(d, day.weekday())))
        total = sum(u for _, u in per_day)
        best = max(per_day, key=lambda x: x[1])[0]
        return dict(units=rnum(total), revenue=rmoney(total * rev_per_unit), best=best,
                    trusted=bool(ok), per_day=per_day)

    # ---- suggestion skills: each returns a Suggestion or None
    def s_restock(self):
        d, last = self.df, self.df.iloc[-1]
        drops = (d['stock_end'].shift(1) - d['stock_end']).tail(28).dropna()
        drops = drops[drops >= 0]                      # delivery days show a rise, so are left out
        if len(drops) < 7 or drops.mean() <= 0:
            return None
        use, sd = float(drops.mean()), float(drops.std(ddof=0))
        lead, stock = int(last['supplier_lead_days']), float(last['stock_end'])
        buffer = 1.65 * sd * math.sqrt(lead)
        if stock > use * lead + buffer:
            return None
        cover = stock / use
        qty = max(1, math.ceil(use * (lead + 7) + buffer - stock))
        gp_day = float(d['gross_profit'].tail(28).mean())
        impact = max(lead - cover, 0.5) * gp_day
        facts = dict(stock=rnum(stock), cover_days=rnum(cover, 1), lead_days=lead, order_qty=qty)
        msg = (f"You have about {facts['stock']} units left. At your usual pace that lasts roughly "
               f"{facts['cover_days']} days, and your supplier takes {lead} days to deliver, so you could run "
               f"out before the next delivery. I suggest ordering about {qty} units today.")
        return Suggestion('restock', 'Order stock before you run out', msg, 'Propose', impact,
                          float(np.clip(1 - sd / use, 0.5, 0.95)), min(1.0, lead / max(cover, 0.5)),
                          ['Your recent usage pattern continues',
                           'No order is already on the way (I cannot see open orders yet)'], facts,
                          'Tap to send the supplier order after you check the quantity.')

    def s_margin(self):
        d = self.df[self.df['promo_active'] == 0]      # promotions lower margin on purpose
        if len(d) < 42:
            return None
        recent, prior = d.tail(14), d.iloc[-42:-14]

        def gm(x):
            r = x['revenue'].sum()
            return (r - x['cost_of_goods'].sum()) / r * 100 if r > 0 else None

        a, b = gm(recent), gm(prior)
        if a is None or b is None or b - a < 3:
            return None
        cpu_a = recent['cost_of_goods'].sum() / max(1, recent['units_sold'].sum())
        cpu_b = prior['cost_of_goods'].sum() / max(1, prior['units_sold'].sum())
        rise = (cpu_a / cpu_b - 1) * 100 if cpu_b > 0 else 0
        lost = (b - a) / 100 * recent['revenue'].sum()
        facts = dict(margin_then=rnum(b, 1), margin_now=rnum(a, 1), cost_rise_pct=rnum(rise), lost=rmoney(lost))
        cause = (f"Your cost per item is up about {facts['cost_rise_pct']}%."
                 if rise >= 2 else
                 "Your cost per item has not changed much, so the dip may come from lower prices or a change in what you sell.")
        msg = (f"Your profit margin on sales slipped from {facts['margin_then']}% to {facts['margin_now']}% over "
               f"the last two weeks (days without promotions). {cause} That cost you roughly {self.m(facts['lost'])} "
               f"in that time. Check whether a supplier raised prices, and consider a small price change or a cheaper supplier.")
        return Suggestion('margin', 'Your margin is slipping', msg, 'Propose', lost, 0.75, 0.6,
                          ['Promotion days are left out of the comparison',
                           'Recorded supplier costs are accurate'], facts,
                          'Review your supplier prices; I can show the items that changed most once item-level costs are recorded.')

    def s_expense(self):
        d = self.df
        prior = d['expenses'].iloc[:-7].tail(60)
        if len(prior) < 20:
            return None
        med = float(prior.median())
        mad = float(np.median(np.abs(prior - med))) * 1.4826
        thr = med + 4 * max(mad, 0.1 * med)
        hits = d.tail(7)[d.tail(7)['expenses'] > thr]
        if hits.empty:
            return None
        row = hits.sort_values('expenses').iloc[-1]
        facts = dict(amount=rmoney(row['expenses']), usual=rmoney(med), times=rnum(row['expenses'] / med, 1),
                     day_of_month=int(row['date'].day))
        when = self.day_name(row['date'])
        msg = (f"Spending on {when} was {self.m(facts['amount'])}, about {facts['times']} times your usual "
               f"{self.m(facts['usual'])} a day. Was that expected, for example a repair or a bulk purchase?")
        return Suggestion('expense', 'An unusually high expense', msg, 'Ask', float(row['expenses'] - med), 0.8, 0.5,
                          ['Compared with your own last 60 days, not other businesses'], facts,
                          'Reply yes if it was planned, and I will stop flagging it.')

    def s_cash(self):
        r = self.df.tail(7)
        off = r[r['cash_diff'] != 0]
        if off.empty:
            return None
        net, total = float(off['cash_diff'].sum()), float(off['cash_diff'].abs().sum())
        facts = dict(days_off=int(len(off)), net_abs=rmoney(abs(net)), total=rmoney(total))
        direction = 'short' if net < 0 else 'over'
        detail = ('' if facts['net_abs'] == facts['total']
                  else f" (the differences added up to {self.m(facts['total'])})")
        msg = (f"Your till did not match the books on {facts['days_off']} of the last 7 days. Overall you were "
               f"{direction} by {self.m(facts['net_abs'])}{detail}. "
               f"Counting again at closing, with two people, usually finds the cause.")
        return Suggestion('cash', 'Till differences', msg, 'Ask', total, 0.9, 0.4,
                          ['The cash count you entered is correct'], facts,
                          'At closing, count the till again and enter the new total.')

    def s_debts(self):
        d = self.df
        now = float(d['debt_outstanding'].iloc[-1])
        rev_day = float(d['revenue'].tail(28).mean())
        coll = float(d['debt_collected'].tail(28).mean())
        if rev_day <= 0 or now < 2 * rev_day:
            return None
        ago = float(d['debt_outstanding'].iloc[-29]) if len(d) >= 29 else None
        growth = (now / ago - 1) * 100 if ago and ago > 0 else None
        clear = now / coll if coll > 0 else None
        if not ((growth is not None and growth >= 10) or (clear is not None and clear > 30)):
            return None
        facts = dict(owed=rmoney(now), sales_days=rnum(now / rev_day, 1))
        parts = [f"Customers owe you {self.m(facts['owed'])}, which is {facts['sales_days']} days of your sales."]
        if growth is not None and growth >= 10:
            facts['growth_pct'] = rnum(growth)
            parts.append(f"That is up {facts['growth_pct']}% in four weeks.")
        if clear is not None:
            facts['clear_days'] = rnum(clear)
            parts.append(f"At the pace people have been paying, clearing it would take about {facts['clear_days']} days.")
        parts.append("I can draft friendly reminders for the customers who are overdue.")
        impact = max((now - ago) if ago else 0, 0.1 * now)
        return Suggestion('debts', 'Money customers owe you is building up', ' '.join(parts), 'Propose', impact,
                          0.7, 0.55, ['Customers pay at about the same pace as in the last four weeks',
                                       'Impact counts the recent increase, or one tenth of the debt if it has not grown'],
                          facts, 'Tap to review and send reminders to customers with a phone number.')

    def s_noshows(self):
        d = self.df
        if not self.is_service:
            return None
        r = d.tail(14)
        booked, missed = int(r['bookings'].sum()), int(r['no_shows'].sum())
        if booked < 20 or missed / booked < 0.10:
            return None
        recent = d.tail(56)
        rates = (recent.groupby('day_of_week')['no_shows'].sum() /
                 recent.groupby('day_of_week')['bookings'].sum().replace(0, np.nan))
        price = d['revenue'].tail(28).sum() / max(1, d['units_sold'].tail(28).sum())
        facts = dict(per_100=rnum(missed / booked * 100), missed=missed,
                     worst=WEEKDAYS[int(rates.idxmax())], price=rmoney(price))
        msg = (f"In the last two weeks about {facts['per_100']} out of every 100 booked clients did not show up "
               f"({missed} clients). {facts['worst']} is the worst day. Each missed booking costs you about "
               f"{self.m(facts['price'])}. A reminder the day before, and a small deposit from new clients, usually helps.")
        return Suggestion('noshow', 'Clients are missing appointments', msg, 'Propose', missed * price, 0.7, 0.5,
                          ['Each missed booking would have paid your average price'], facts,
                          'Tap to turn on reminders 24 hours and 2 hours before each booking.')

    def s_promo(self):
        d = self.df
        nonp, pr = d[d['promo_active'] == 0], d[d['promo_active'] == 1]
        if len(pr) < 5 or len(nonp) < 28:
            return None
        dow_mean = nonp.groupby('day_of_week')['units_sold'].mean()
        rel = pr['units_sold'] / pr['day_of_week'].map(dow_mean).replace(0, np.nan)
        lift = float(rel.mean() - 1)
        if not lift > 0.05:
            return None
        by_day = nonp.tail(56).groupby('day_of_week')['revenue'].mean()
        slow = int(by_day.idxmin())
        facts = dict(slow_day=WEEKDAYS[slow], lift_pct=rnum(lift * 100), promo_days=int(len(pr)),
                     extra=rmoney(lift * by_day[slow]))
        msg = (f"{facts['slow_day']} is your quietest day. On the {facts['promo_days']} days when you ran a promotion, "
               f"sales were about {facts['lift_pct']}% higher than normal for that weekday. A small {facts['slow_day']} "
               f"offer could bring in roughly {self.m(facts['extra'])} more on that day, before the cost of the discount.")
        return Suggestion('promo', f"Try a {facts['slow_day']} offer", msg, 'Propose', lift * by_day[slow],
                          min(0.8, 0.4 + 0.02 * len(pr)), 0.3,
                          ['A promotion works the same way on this day', 'The discount cost is not included'],
                          facts, 'Tap to draft the offer; nothing is sent until you approve it.')

    def s_repeat(self):
        d = self.df
        if len(d) < 56:
            return None

        def share(x):
            t = (x['new_customers'] + x['returning_customers']).sum()
            return x['returning_customers'].sum() / t if t > 0 else None

        a, b = share(d.tail(28)), share(d.iloc[-56:-28])
        if a is None or b is None or b - a < 0.05:
            return None
        visits = float((d['new_customers'] + d['returning_customers']).tail(28).sum())
        ticket = d['revenue'].tail(28).sum() / max(1.0, d['transactions'].tail(28).sum())
        facts = dict(now_pct=rnum(a * 100), before_pct=rnum(b * 100))
        msg = (f"Repeat customers made up {facts['now_pct']}% of your visits in the last four weeks, down from "
               f"{facts['before_pct']}% before. New faces are good, but regulars spend more reliably. I can draft a "
               f"'we miss you' message for customers who have not been back lately (only those who agreed to messages).")
        return Suggestion('repeat', 'Fewer regulars are coming back', msg, 'Propose', (b - a) * visits * ticket,
                          0.6, 0.5, ['Lost repeat visits would have spent the average amount'], facts,
                          'Tap to see the list of customers overdue for a visit.')

    def s_sales_drop(self):
        d = self.df
        if len(d) < 42:
            return None
        base = float(d['revenue'].iloc[-35:-7].sum() / 4)
        now = float(d['revenue'].tail(7).sum())
        if base <= 0 or now / base - 1 > -0.15:
            return None
        r = d.tail(7)
        facts = dict(pct=rnum((1 - now / base) * 100), now=rmoney(now), usual=rmoney(base))
        causes = []
        so_days, lost = int((r['stockout_flag'] == 1).sum()), int(r['lost_units'].sum())
        if so_days > 0 and lost > 0:
            facts.update(stockout_days=so_days, lost_units=lost)
            causes.append(f"you ran out of stock on {so_days} of the last 7 days (about {lost} sales missed)")
        rain = int((r['rain_mm'] >= 10).sum())
        if rain >= 3:
            facts['rain_days'] = rain
            causes.append(f"it rained heavily on {rain} of the last 7 days")
        if self.is_service and r['bookings'].sum() > 0 and r['no_shows'].sum() / r['bookings'].sum() >= 0.12:
            facts['noshow_pct'] = rnum(r['no_shows'].sum() / r['bookings'].sum() * 100)
            causes.append(f"{facts['noshow_pct']}% of booked clients did not show up")
        if int(r['is_public_holiday'].sum()) > 0:
            causes.append('there was a public holiday')
        why = ('Things I can see in your records: ' + '; '.join(causes) + '.') if causes else \
              'I cannot see a clear cause in your records.'
        msg = (f"Sales over the last 7 days were {facts['pct']}% lower than your recent weekly average "
               f"({self.m(facts['now'])} compared with {self.m(facts['usual'])}). {why} Does that match what you saw?")
        return Suggestion('sales_drop', 'Sales dropped this week', msg, 'Ask', base - now, 0.6, 0.9,
                          ['Your last four weeks are a fair picture of a normal week',
                           'I only know what is in your records, not what happened outside the business'],
                          facts, 'Tell me what changed (a closure, a competitor, stock, weather) and I will take it into account.')

    def suggestions(self):
        out = []
        for fn in (self.s_sales_drop, self.s_restock, self.s_margin, self.s_expense, self.s_cash, self.s_debts,
                   self.s_noshows, self.s_promo, self.s_repeat):
            try:
                s = fn()
            except Exception:       # one broken skill must never break the whole assistant
                s = None
            if s:
                out.append(s)
        return sorted(out, key=lambda s: s.score, reverse=True)


# ----------------------------------------------------------------- the assistant (words and routing)
INTENTS = [
    ('dismiss', r'\b(dismiss|not useful|hide|stop (showing|telling))\b'),
    ('explain', r'\b(why|explain|kwa nini)\b.*\b(\d|first|second|third|that|this|it)\b'),
    ('forecast', r'\b(forecast|expect|predict|next week|tomorrow|coming days|utabiri)\b'),
    ('stock', r'\b(stock|restock|reorder|re-order|supplier|run(ning)? out|inventory|stoo|bidhaa)\b'),
    ('debts', r'\b(debts?|owe|owed|credit|deni|madeni)\b'),
    ('cash', r'\b(cash|till|drawer|pesa)\b'),
    ('profit', r'\b(profit|margin|faida|earn\w*)\b'),
    ('sales', r'\b(sales|sold|revenue|income|takings|mauzo)\b'),
    ('customers', r'\b(customers?|clients?|regulars|repeat|wateja|no-?shows?)\b'),
    ('advice', r'\b(advice|advise|suggest\w*|recommend\w*|improve|grow|what should|ushauri)\b'),
    ('briefing', r'\b(today|yesterday|summary|briefing|how (is|was|did)|report|leo|jana|ripoti)\b'),
    ('thanks', r'\b(thanks|thank you|asante)\b'),
    ('help', r'\b(help|what can you do|msaada)\b'),
    ('hello', r'^(hi|hello|hey|habari|mambo|jambo)\b'),
]


class Assistant:
    def __init__(self, df, currency='KSh', owner=None, call_llm=None):
        self.a = Analyzer(df, currency)
        self.owner = owner
        self.call_llm = call_llm       # optional: function(prompt) -> text, used only to polish wording
        self.shown = []                # suggestions listed last, so "why 2" knows what 2 means
        self.muted = set()             # suggestion keys the owner asked not to see again

    # -- public entry point
    def ask(self, text):
        intent = self._intent(text)
        reply = getattr(self, f'_{intent}')(text)
        return self._polish(reply)

    def _intent(self, text):
        t = text.lower().strip()
        for name, pattern in INTENTS:
            if re.search(pattern, t):
                return name
        return 'unknown'

    # -- optional wording polish with a number check
    def _polish(self, reply):
        if not self.call_llm:
            return reply
        prompt = ('Rewrite the message below for a busy small business owner in warm, simple English. '
                  'Keep every number exactly as written. Do not add numbers, facts or advice. '
                  'Keep it about the same length.\n\n' + reply)
        try:
            new = self.call_llm(prompt)
        except Exception:
            return reply
        allowed = set(numbers_in(reply))
        if new and all(n in allowed for n in numbers_in(new)):
            return new.strip()
        return reply                   # model added or changed a number: use the safe original

    def _live(self):
        return [s for s in self.a.suggestions() if s.key not in self.muted]

    # -- handlers
    def _dismiss(self, text):
        m = re.search(r'\d', text)
        n = int(m.group()) if m else 1
        if not self.shown or not 1 <= n <= len(self.shown):
            return 'Tell me which item to hide, for example "dismiss 1", after I have listed some.'
        s = self.shown[n - 1]
        self.muted.add(s.key)
        return f'Okay. I will stop raising "{s.title}" for now. Ask me for it any time.'

    def _hello(self, _):
        name = f', {self.owner}' if self.owner else ''
        return f'Hello{name}! Ask me about your sales, stock, profit, debts or cash, or say "advice" and I will tell you what to do next.'

    def _thanks(self, _):
        return 'You are welcome. I am here whenever you need me.'

    def _help(self, _):
        return ('You can ask things like:\n'
                '  - How did we do yesterday?\n  - What should I do to grow?\n  - Do I need to order stock?\n'
                '  - How much do customers owe me?\n  - Why is my profit down?\n  - What do you expect next week?\n'
                '  - Why 1?  (to hear how I worked out suggestion 1)')

    def _briefing(self, _):
        L, W = self.a.latest(), self.a.week()
        up = 'above' if L['pct'] >= 0 else 'below'
        lines = [f"{L['day']}: you took in {self.a.m(L['revenue'])}, which is {abs(L['pct'])}% {up} a normal {L['weekday']} "
                 f"({self.a.m(L['usual'])}). After costs and spending, {self.a.left(L['profit'])}.",
                 f"Last 7 days: {self.a.m(W['revenue'])}, {'up' if W['pct'] >= 0 else 'down'} {abs(W['pct'])}% "
                 f"on the 7 days before."]
        top = [s for s in self._live() if s.mode in ('Propose', 'Ask')][:3]
        self.shown = top
        if top:
            lines.append('\nWhat needs your attention:')
            lines += [f'  {i}. {s.title}: {s.message}' for i, s in enumerate(top, 1)]
            lines.append('\nSay "why 1" to hear how I worked something out.')
        else:
            lines.append('\nNothing needs your attention today. Everything I watch looks normal.')
        f = self.a.forecast()
        lines.append(f"\nLooking ahead: I expect about {f['units']} {self._units()} over the next 7 days, "
                     f"roughly {self.a.m(f['revenue'])}, with {f['best']} the busiest."
                     + ('' if f['trusted'] else ' This is a rough estimate because I am still learning your pattern.'))
        return '\n'.join(lines)

    def _units(self):
        return 'services' if self.a.is_service else 'items sold'

    def _sales(self, _):
        L, W = self.a.latest(), self.a.week()
        up = 'up' if W['pct'] >= 0 else 'down'
        return (f"{L['day']}: {self.a.m(L['revenue'])} ({abs(L['pct'])}% {'above' if L['pct'] >= 0 else 'below'} a normal {L['weekday']}).\n"
                f"Last 7 days: {self.a.m(W['revenue'])}, {up} {abs(W['pct'])}% on the 7 days before ({self.a.m(W['prev'])}).\n"
                f"Over the last 8 weeks, {W['best']} is your best day and {W['worst']} is your quietest.")

    def _profit(self, _):
        d = self.a.df
        a, b = d.tail(7), d.iloc[-14:-7]
        W = self.a.week()
        lines = [f"Last 7 days: {self.a.left(W['profit'])} after costs and spending. "
                 f"You keep about {W['margin']}% of each sale after the cost of what you sold."]
        d_profit = a['profit'].sum() - b['profit'].sum()
        d_sales = a['gross_profit'].sum() - b['gross_profit'].sum()    # profit change from selling
        d_spend = a['expenses'].sum() - b['expenses'].sum()
        if abs(d_profit) >= 0.05 * max(abs(b['profit'].sum()), 1):
            word = 'up' if d_profit > 0 else 'down'
            if abs(d_sales) >= abs(d_spend):
                why = (f"mostly because sales {'rose' if d_sales > 0 else 'fell'}, which moved your profit by "
                       f"{self.a.m(rmoney(abs(d_sales)))}")
            else:
                why = (f"mostly because spending {'rose' if d_spend > 0 else 'fell'} by {self.a.m(rmoney(abs(d_spend)))}")
            lines.append(f"Compared with the 7 days before, profit is {word} by {self.a.m(rmoney(abs(d_profit)))}, {why}.")
        else:
            lines.append('That is about the same as the 7 days before.')
        s = next((x for x in self._live() if x.key == 'margin'), None)
        if s:
            self.shown = [s]
            lines.append('Something to look at: ' + s.message)
        else:
            lines.append('Your margin on sales looks steady compared with earlier weeks.')
        return '\n'.join(lines)

    def _stock(self, _):
        last = self.a.df.iloc[-1]
        s = next((x for x in self._live() if x.key == 'restock'), None)
        if s:
            self.shown = [s]
            return s.message
        return (f"Your stock looks fine: about {rnum(last['stock_end'])} units on hand and your supplier delivers "
                f"in {int(last['supplier_lead_days'])} days. I will tell you when it is time to order.")

    def _debts(self, _):
        now = rmoney(self.a.df['debt_outstanding'].iloc[-1])
        s = next((x for x in self._live() if x.key == 'debts'), None)
        if s:
            self.shown = [s]
            return s.message
        return f'Customers owe you {self.a.m(now)}. That is a normal level for your sales, so no action is needed yet.'

    def _cash(self, _):
        s = next((x for x in self._live() if x.key == 'cash'), None)
        if s:
            self.shown = [s]
            return s.message
        return 'Your till matched the books on each of the last 7 days. Well done.'

    def _forecast(self, _):
        f = self.a.forecast()
        days = ', '.join(f'{n[:3]} {rnum(u)}' for n, u in f['per_day'])
        note = ('' if f['trusted'] else '\nPlease treat this as a rough estimate. I am still learning your pattern '
                                         'and it has not yet beaten a simple "same as last week" guess.')
        return (f"Next 7 days I expect about {f['units']} {self._units()}, roughly {self.a.m(f['revenue'])}. "
                f"{f['best']} should be the busiest.\nDay by day: {days}.{note}")

    def _customers(self, _):
        picks = [s for s in self._live() if s.key in ('repeat', 'noshow')]
        self.shown = picks
        if not picks:
            d = self.a.df.tail(28)
            n, r = int(d['new_customers'].sum()), int(d['returning_customers'].sum())
            return f'In the last four weeks you had {n} new and {r} returning customers. Nothing worrying stands out.'
        return '\n'.join(f'{i}. {s.message}' for i, s in enumerate(picks, 1))

    def _advice(self, _):
        top = [s for s in self._live() if s.mode in ('Propose', 'Ask')][:5]
        self.shown = top
        if not top:
            return 'I have nothing urgent for you. Keep recording sales and expenses and I will keep watching.'
        lines = ['Here is what I would do, most valuable first:']
        for i, s in enumerate(top, 1):
            lines.append(f'  {i}. {s.title}\n     {s.message}\n     Next step: {s.next_step}')
        lines.append('Say "why 1" to see how I worked out any of these.')
        return '\n'.join(lines)

    def _explain(self, text):
        if not self.shown:
            return 'Ask me for "advice" first, and then I can explain any numbered item.'
        m = re.search(r'\d', text)
        words = {'first': 1, 'second': 2, 'third': 3}
        n = int(m.group()) if m else next((v for k, v in words.items() if k in text.lower()), 1)
        if not 1 <= n <= len(self.shown):
            return f'I only have {len(self.shown)} item(s) on the list. Pick a number between 1 and {len(self.shown)}.'
        s = self.shown[n - 1]
        facts = ', '.join(f'{k.replace("_", " ")}: {v}' for k, v in s.facts.items())
        return (f"{s.title}\nHow sure I am: {confidence_word(s.confidence)} ({int(s.confidence * 100)}%).\n"
                f"The numbers I used: {facts}.\nWhat I assumed:\n" + '\n'.join(f'  - {a}' for a in s.assumptions) +
                f"\nAbout {self.a.m(rmoney(s.impact))} is at stake, if my assumptions hold.\nNothing happens until you approve it.")

    def _unknown(self, _):
        return ("I am not sure what you mean yet. I can help with sales, profit, stock, debts, cash, customers and "
                'what to expect next week. Say "help" to see example questions.')


# ----------------------------------------------------------------- terminal chat
def load_business(path, business_id=None):
    df = pd.read_csv(path)
    business_id = business_id or df['business_id'].iloc[0]
    sub = df[df['business_id'] == business_id]
    if sub.empty:
        raise SystemExit(f'Business {business_id} not found in {path}')
    return sub


def main():
    p = argparse.ArgumentParser(description='SmartBiz plain-English assistant')
    p.add_argument('--data', required=True, help='daily business metrics CSV')
    p.add_argument('--business', help='business_id, e.g. B001 (default: first in the file)')
    p.add_argument('--currency', default='KSh')
    p.add_argument('--owner', help='owner first name, for a friendlier greeting')
    p.add_argument('--briefing', action='store_true', help='print the morning briefing and exit')
    args = p.parse_args()
    bot = Assistant(load_business(args.data, args.business), args.currency, args.owner)
    if args.briefing:
        print(bot.ask('how did we do yesterday'))
        return
    print('SmartBiz Assistant. Type "help" for ideas, "quit" to leave.\n')
    print(bot.ask('hello'), '\n')
    while True:
        try:
            q = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ('quit', 'exit', 'q'):
            break
        if q:
            print('\nSmartBiz:', bot.ask(q), '\n')


if __name__ == '__main__':
    main()
