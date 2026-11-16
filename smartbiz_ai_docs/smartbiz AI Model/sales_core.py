"""
SmartBiz Sales Book: the core of the "record and track daily sales" module.

Goals: record a sale in a few seconds, work offline, cost almost nothing to run,
catch mistakes before they are saved, and give the owner a trustworthy daily total.
Uses only the Python standard library (SQLite). Swap SQLite for Postgres in production;
the schema and rules stay the same.
"""
import csv
import io
import re
import sqlite3
import statistics
import uuid
from datetime import datetime, timedelta

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS businesses(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT, currency TEXT DEFAULT 'KSh', closing_time TEXT DEFAULT '19:00');
CREATE TABLE IF NOT EXISTS items(
  id INTEGER PRIMARY KEY, business_id TEXT NOT NULL REFERENCES businesses(id), name TEXT NOT NULL,
  price REAL NOT NULL, cost REAL DEFAULT 0, track_stock INTEGER DEFAULT 0, stock_qty REAL DEFAULT 0,
  active INTEGER DEFAULT 1, UNIQUE(business_id, name));
CREATE TABLE IF NOT EXISTS aliases(
  item_id INTEGER NOT NULL REFERENCES items(id), alias TEXT NOT NULL, PRIMARY KEY(item_id, alias));
CREATE TABLE IF NOT EXISTS customers(
  id INTEGER PRIMARY KEY, business_id TEXT NOT NULL, name TEXT NOT NULL, phone TEXT, UNIQUE(business_id, name));
CREATE TABLE IF NOT EXISTS sales(
  id INTEGER PRIMARY KEY, business_id TEXT NOT NULL, client_uuid TEXT NOT NULL, ts TEXT NOT NULL, day TEXT NOT NULL,
  total REAL NOT NULL, payment TEXT NOT NULL CHECK(payment IN ('cash','mpesa','card','credit')),
  customer_id INTEGER, note TEXT, source TEXT NOT NULL, staff TEXT,
  status TEXT NOT NULL DEFAULT 'active', UNIQUE(business_id, client_uuid));
CREATE TABLE IF NOT EXISTS sale_lines(
  id INTEGER PRIMARY KEY, sale_id INTEGER NOT NULL REFERENCES sales(id), item_id INTEGER, name TEXT NOT NULL,
  qty REAL NOT NULL, unit_price REAL NOT NULL, line_total REAL NOT NULL);
CREATE TABLE IF NOT EXISTS debt_payments(
  id INTEGER PRIMARY KEY, business_id TEXT NOT NULL, customer_id INTEGER NOT NULL, day TEXT NOT NULL,
  amount REAL NOT NULL, method TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS day_close(
  business_id TEXT NOT NULL, day TEXT NOT NULL, cash_expected REAL, cash_counted REAL, diff REAL, closed_at TEXT,
  PRIMARY KEY(business_id, day));
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY, ts TEXT, business_id TEXT, action TEXT, sale_id INTEGER, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_sales_day ON sales(business_id, day, status);
"""

NUM_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
             'ten': 10, 'half': 0.5, 'moja': 1, 'mbili': 2, 'tatu': 3, 'nne': 4, 'tano': 5, 'sita': 6, 'saba': 7,
             'nane': 8, 'tisa': 9, 'kumi': 10, 'nusu': 0.5}
PAYMENT_WORDS = [('credit', r'\b(?:on credit|credit|deni|kopa\w*)\b'),
                 ('mpesa', r'\bm-?pesa\b'),
                 ('card', r'\b(?:card|kadi)\b'),
                 ('cash', r'\b(?:cash|taslimu|pesa taslimu)\b')]
FILLER = r'\b(?:sold|sell|sale|nimeuza|niliuza|uza|ksh|kes|shs?|of|i|we|to)\b'


def now_iso():
    return datetime.now().isoformat(timespec='seconds')


def money(x):
    return f'{round(x):,}'


class SalesBook:
    def __init__(self, path=':memory:'):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------ setup
    def add_business(self, bid, name, btype='retail', currency='KSh'):
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO businesses(id,name,type,currency) VALUES(?,?,?,?)',
                            (bid, name, btype, currency))

    def add_item(self, bid, name, price, cost=0, track_stock=False, stock_qty=0, aliases=()):
        with self.db:
            cur = self.db.execute(
                'INSERT INTO items(business_id,name,price,cost,track_stock,stock_qty) VALUES(?,?,?,?,?,?)',
                (bid, name, price, cost, int(track_stock), stock_qty))
            for a in aliases:
                self.db.execute('INSERT OR IGNORE INTO aliases VALUES(?,?)', (cur.lastrowid, a.lower()))
        return cur.lastrowid

    def items(self, bid):
        return self.db.execute('SELECT * FROM items WHERE business_id=? AND active=1', (bid,)).fetchall()

    def _customer_id(self, bid, name, phone=None):
        if not name:
            return None
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO customers(business_id,name,phone) VALUES(?,?,?)', (bid, name, phone))
        return self.db.execute('SELECT id FROM customers WHERE business_id=? AND name=?', (bid, name)).fetchone()['id']

    # ------------------------------------------------------------ mistake catcher (R5)
    def _warnings(self, bid, lines, payment, ts):
        out = []
        for l in lines:
            if l['item_id']:
                it = self.db.execute('SELECT * FROM items WHERE id=?', (l['item_id'],)).fetchone()
                if it['price'] > 0 and abs(l['unit_price'] - it['price']) / it['price'] > 0.25:
                    out.append({'code': 'price', 'message': f"{it['name']} is usually {money(it['price'])} but you entered "
                                                           f"{money(l['unit_price'])}. Is that right?"})
                past = [r['qty'] for r in self.db.execute(
                    'SELECT qty FROM sale_lines sl JOIN sales s ON s.id=sl.sale_id WHERE sl.item_id=? AND s.status="active" '
                    'ORDER BY sl.id DESC LIMIT 30', (l['item_id'],))]
                limit = max(10, 5 * statistics.median(past)) if len(past) >= 5 else 50
                if l['qty'] > limit:
                    out.append({'code': 'quantity', 'message': f"{l['qty']:g} {it['name']} is a lot more than you usually sell at once. Is that right?"})
                if it['track_stock'] and l['qty'] > it['stock_qty']:
                    out.append({'code': 'stock', 'message': f"Your records show only {it['stock_qty']:g} {it['name']} in stock. Is the stock count out of date?"})
        total = sum(l['qty'] * l['unit_price'] for l in lines)
        since = (datetime.fromisoformat(ts) - timedelta(minutes=3)).isoformat(timespec='seconds')
        for r in self.db.execute('SELECT id,total FROM sales WHERE business_id=? AND status="active" AND ts>=? AND payment=?',
                                 (bid, since, payment)):
            if abs(r['total'] - total) < 0.01:
                out.append({'code': 'duplicate', 'message': f'You recorded a sale of {money(total)} a moment ago. Is this a second one?'})
                break
        return out

    # ------------------------------------------------------------ record a sale
    def record_sale(self, bid, lines, payment='cash', client_uuid=None, ts=None, customer=None, note='',
                    source='tile', staff=None, confirmed=False):
        """lines: [{'item': 'Bread', 'qty': 2, 'unit_price': 50 (optional)}] or [{'item_id':..}] or
        [{'name': 'General sale', 'qty': 1, 'unit_price': 500}]. Safe to call twice with the same client_uuid."""
        client_uuid = client_uuid or str(uuid.uuid4())
        dup = self.db.execute('SELECT id FROM sales WHERE business_id=? AND client_uuid=?', (bid, client_uuid)).fetchone()
        if dup:
            return {'status': 'duplicate', 'sale_id': dup['id']}
        ts = ts or now_iso()
        resolved = []
        for l in lines:
            it = None
            if l.get('item_id'):
                it = self.db.execute('SELECT * FROM items WHERE id=?', (l['item_id'],)).fetchone()
            elif l.get('item'):
                it = self.db.execute('SELECT * FROM items WHERE business_id=? AND lower(name)=lower(?)',
                                     (bid, l['item'])).fetchone()
            price = l.get('unit_price', it['price'] if it else None)
            if price is None or float(l['qty']) <= 0 or float(price) < 0:
                return {'status': 'error', 'message': 'Each line needs a quantity above zero and a price.'}
            resolved.append({'item_id': it['id'] if it else None, 'name': it['name'] if it else l.get('name', 'General sale'),
                             'qty': float(l['qty']), 'unit_price': float(price)})
        if not resolved:
            return {'status': 'error', 'message': 'Add at least one item.'}
        if payment == 'credit' and not customer:
            return {'status': 'ask', 'question': 'Who is this credit sale for?'}
        warnings = self._warnings(bid, resolved, payment, ts)
        if warnings and not confirmed:
            return {'status': 'confirm', 'warnings': warnings}
        total = sum(l['qty'] * l['unit_price'] for l in resolved)
        cid = self._customer_id(bid, customer)
        with self.db:
            cur = self.db.execute(
                'INSERT INTO sales(business_id,client_uuid,ts,day,total,payment,customer_id,note,source,staff) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (bid, client_uuid, ts, ts[:10], total, payment, cid, note, source, staff))
            sid = cur.lastrowid
            for l in resolved:
                self.db.execute('INSERT INTO sale_lines(sale_id,item_id,name,qty,unit_price,line_total) VALUES(?,?,?,?,?,?)',
                                (sid, l['item_id'], l['name'], l['qty'], l['unit_price'], l['qty'] * l['unit_price']))
                if l['item_id']:
                    self.db.execute('UPDATE items SET stock_qty=MAX(0, stock_qty-?) WHERE id=? AND track_stock=1',
                                    (l['qty'], l['item_id']))
            self.db.execute('INSERT INTO audit(ts,business_id,action,sale_id,detail) VALUES(?,?,?,?,?)',
                            (ts, bid, 'record', sid, source))
            for w in warnings:   # saved anyway (offline or confirmed): keep the flag for review
                self.db.execute('INSERT INTO audit(ts,business_id,action,sale_id,detail) VALUES(?,?,?,?,?)',
                                (ts, bid, 'flag', sid, w['code'] + ': ' + w['message']))
        return {'status': 'saved', 'sale_id': sid, 'total': total, 'flags': [w['code'] for w in warnings]}

    def void_sale(self, sale_id, reason='undo'):
        s = self.db.execute('SELECT * FROM sales WHERE id=?', (sale_id,)).fetchone()
        if not s or s['status'] != 'active':
            return {'status': 'error', 'message': 'Sale not found or already removed.'}
        with self.db:
            self.db.execute('UPDATE sales SET status="voided" WHERE id=?', (sale_id,))
            for l in self.db.execute('SELECT * FROM sale_lines WHERE sale_id=?', (sale_id,)):
                if l['item_id']:
                    self.db.execute('UPDATE items SET stock_qty=stock_qty+? WHERE id=? AND track_stock=1', (l['qty'], l['item_id']))
            self.db.execute('INSERT INTO audit(ts,business_id,action,sale_id,detail) VALUES(?,?,?,?,?)',
                            (now_iso(), s['business_id'], 'void', sale_id, reason))
        return {'status': 'voided', 'sale_id': sale_id}

    # ------------------------------------------------------------ type or speak a note (R1)
    def parse_note(self, bid, text):
        """Turns 'sold 2 bread and 1 milk mpesa' into lines for the owner to confirm. Never saves anything."""
        t = text.lower().strip()
        payment, customer = 'cash', None
        m = re.search(r'(?:credit|deni|kopa\w*)\s+(?:to\s+|kwa\s+|ya\s+)?([a-z]+)', t)
        for name, pat in PAYMENT_WORDS:
            if re.search(pat, t):
                payment = name
                break
        if m and payment == 'credit' and m.group(1) not in ('to', 'kwa', 'ya'):
            customer = m.group(1).capitalize()
            t = t.replace(m.group(0), ' credit ')
        for _, pat in PAYMENT_WORDS:
            t = re.sub(pat, ' ', t)
        t = re.sub(FILLER, ' ', t)
        for w, n in NUM_WORDS.items():
            t = re.sub(rf'\b{w}\b', str(n), t)

        names = {}
        for it in self.items(bid):
            names[it['name'].lower()] = it
        for r in self.db.execute('SELECT a.alias, i.id FROM aliases a JOIN items i ON i.id=a.item_id WHERE i.business_id=?', (bid,)):
            names[r['alias']] = self.db.execute('SELECT * FROM items WHERE id=?', (r['id'],)).fetchone()
        ordered = sorted(names, key=len, reverse=True)

        lines, unmatched = [], []
        for seg in re.split(r'[,;+]|\band\b|\bna\b', t):
            seg = seg.strip()
            if not seg:
                continue
            item = None
            for key in ordered:
                mm = re.search(rf'\b{re.escape(key)}(?:s|es)?\b', seg)
                if mm:
                    item, seg = names[key], (seg[:mm.start()] + ' ' + seg[mm.end():])
                    break
            unit_price = total_price = None
            for pat, kind in ((r'@\s*(\d+(?:\.\d+)?)', 'unit'), (r'(\d+(?:\.\d+)?)\s*(?:each|kila)', 'unit'),
                              (r'(?:at)\s*(\d+(?:\.\d+)?)', 'unit'), (r'(?:for|kwa)\s*(\d+(?:\.\d+)?)', 'total')):
                pm = re.search(pat, seg)
                if pm:
                    if kind == 'unit':
                        unit_price = float(pm.group(1))
                    else:
                        total_price = float(pm.group(1))
                    seg = seg[:pm.start()] + ' ' + seg[pm.end():]
                    break
            qm = re.search(r'x\s*(\d+(?:\.\d+)?)', seg) or re.search(r'(\d+(?:\.\d+)?)', seg)
            qty = float(qm.group(1)) if qm else 1.0
            if qm:
                seg = seg[:qm.start()] + ' ' + seg[qm.end():]
            leftover = re.sub(r'[^a-z]+', ' ', seg).strip()
            if item is None:
                if qm and not leftover and unit_price is None and total_price is None:
                    lines.append({'name': 'General sale', 'item_id': None, 'qty': 1.0, 'unit_price': qty})
                elif leftover:
                    unmatched.append(leftover)
                continue
            if leftover:
                unmatched.append(leftover)       # words next to a known item that were not understood
            if total_price is not None:
                unit_price = total_price / qty
            lines.append({'name': item['name'], 'item_id': item['id'], 'qty': qty,
                          'unit_price': unit_price if unit_price is not None else item['price']})
        total = sum(l['qty'] * l['unit_price'] for l in lines)
        return {'lines': lines, 'payment': payment, 'customer': customer, 'unmatched': unmatched, 'total': total,
                'confirm_text': ('Record ' + ', '.join(f"{l['qty']:g} {l['name']}" for l in lines) + f" = {money(total)}, paid by {payment}"
                                 + (f' (credit: {customer})' if customer else '') + '?') if lines else 'I could not find anything to record.'}

    # ------------------------------------------------------------ offline sync (H5)
    def sync(self, bid, batch):
        """batch: list of dicts with client_uuid, ts, lines, payment, customer. Oldest first. Safe to resend."""
        res = {'saved': 0, 'duplicates': 0, 'errors': []}
        for e in sorted(batch, key=lambda x: x['ts']):
            r = self.record_sale(bid, e['lines'], e.get('payment', 'cash'), e['client_uuid'], e['ts'],
                                 e.get('customer'), e.get('note', ''), 'offline', e.get('staff'), confirmed=True)
            if r['status'] == 'saved':
                res['saved'] += 1
            elif r['status'] == 'duplicate':
                res['duplicates'] += 1
            else:
                res['errors'].append((e['client_uuid'], r))
        return res

    # ------------------------------------------------------------ track: day, range, debts
    def day_summary(self, bid, day):
        q = lambda sql, *a: self.db.execute(sql, a).fetchall()
        by_pay = {r['payment']: (r['n'], r['t']) for r in q(
            'SELECT payment, COUNT(*) n, SUM(total) t FROM sales WHERE business_id=? AND day=? AND status="active" GROUP BY payment', bid, day)}
        total = sum(t for _, t in by_pay.values())
        top = [(r['name'], r['q'], r['t']) for r in q(
            'SELECT sl.name, SUM(sl.qty) q, SUM(sl.line_total) t FROM sale_lines sl JOIN sales s ON s.id=sl.sale_id '
            'WHERE s.business_id=? AND s.day=? AND s.status="active" GROUP BY sl.name ORDER BY t DESC LIMIT 5', bid, day)]
        d0 = datetime.fromisoformat(day)
        prev = []
        for k in range(1, 9):
            d = (d0 - timedelta(days=7 * k)).date().isoformat()
            r = q('SELECT SUM(total) t FROM sales WHERE business_id=? AND day=? AND status="active"', bid, d)[0]['t']
            if r:
                prev.append(r)
        usual = statistics.mean(prev) if len(prev) >= 3 else None
        return {'day': day, 'total': total, 'count': sum(n for n, _ in by_pay.values()), 'by_payment': by_pay, 'top_items': top,
                'usual': usual, 'vs_usual_pct': round((total / usual - 1) * 100) if usual else None}

    def series(self, bid, start, end):
        rows = {r['day']: r['t'] for r in self.db.execute(
            'SELECT day, SUM(total) t FROM sales WHERE business_id=? AND day BETWEEN ? AND ? AND status="active" GROUP BY day', (bid, start, end))}
        d, out = datetime.fromisoformat(start), []
        while d <= datetime.fromisoformat(end):
            k = d.date().isoformat()
            out.append((k, rows.get(k, 0.0)))
            d += timedelta(days=1)
        return out

    def record_debt_payment(self, bid, customer, amount, method='cash', day=None):
        cid = self._customer_id(bid, customer)
        with self.db:
            self.db.execute('INSERT INTO debt_payments(business_id,customer_id,day,amount,method) VALUES(?,?,?,?,?)',
                            (bid, cid, day or now_iso()[:10], amount, method))

    def debts(self, bid):
        rows = self.db.execute(
            'SELECT c.name, COALESCE((SELECT SUM(total) FROM sales WHERE customer_id=c.id AND payment="credit" AND status="active"),0) '
            '- COALESCE((SELECT SUM(amount) FROM debt_payments WHERE customer_id=c.id),0) AS owed '
            'FROM customers c WHERE c.business_id=?', (bid,)).fetchall()
        return sorted([(r['name'], r['owed']) for r in rows if r['owed'] > 0.001], key=lambda x: -x[1])

    # ------------------------------------------------------------ close the day (M7)
    def close_day(self, bid, day, cash_counted):
        cash_sales = self.db.execute('SELECT COALESCE(SUM(total),0) t FROM sales WHERE business_id=? AND day=? AND payment="cash" AND status="active"', (bid, day)).fetchone()['t']
        cash_debt = self.db.execute('SELECT COALESCE(SUM(amount),0) t FROM debt_payments WHERE business_id=? AND day=? AND method="cash"', (bid, day)).fetchone()['t']
        expected = cash_sales + cash_debt
        diff = cash_counted - expected
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO day_close VALUES(?,?,?,?,?,?)', (bid, day, expected, cash_counted, diff, now_iso()))
        s = self.day_summary(bid, day)
        msg = f"You took in {money(s['total'])} today from {s['count']} sales. "
        msg += ('Your till matches the books.' if abs(diff) < 1 else
                f"Your till is {'short' if diff < 0 else 'over'} by {money(abs(diff))}. Count again with a second person, then check for a sale that was not recorded." if diff < 0 else
                f"Your till is over by {money(diff)}. A sale may have been missed or a change given wrongly.")
        return {'expected': expected, 'counted': cash_counted, 'diff': diff, 'message': msg}

    def export_csv(self, bid, start, end):
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(['date', 'time', 'item', 'qty', 'unit_price', 'line_total', 'payment', 'customer', 'source', 'status'])
        for r in self.db.execute(
                'SELECT s.ts, sl.name, sl.qty, sl.unit_price, sl.line_total, s.payment, c.name cust, s.source, s.status '
                'FROM sales s JOIN sale_lines sl ON sl.sale_id=s.id LEFT JOIN customers c ON c.id=s.customer_id '
                'WHERE s.business_id=? AND s.day BETWEEN ? AND ? ORDER BY s.ts', (bid, start, end)):
            w.writerow([r['ts'][:10], r['ts'][11:], r['name'], r['qty'], r['unit_price'], r['line_total'], r['payment'],
                        r['cust'] or '', r['source'], r['status']])
        return out.getvalue()
