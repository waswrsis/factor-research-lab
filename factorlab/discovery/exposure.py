"""Persistent V7 S/V/C partitions and one-look confirmation reservations.

Calendar reuse by a later batch is legitimate; feedback into the same frozen
claim is not. The controller must register actual information sources and keep
the candidate/rule immutable. This ledger cannot prove that unregistered peeking
did not happen or resist an administrator rewriting every local artifact.
"""
from __future__ import annotations

import json
import math
import sqlite3

import pandas as pd

from factorlab.protocol import fingerprint, project_path

CLAIM_SLOTS = ('library_skill', 'factor1', 'factor2', 'factor3', 'algorithm')
COHORTS = (1,)


def _stamp(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError('Evidence timestamps must be timezone-aware and nonmissing')
    return stamp.tz_convert('UTC')


def _identity(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError('Evidence requires bounded nonempty identities')
    return value


def _symbols(values):
    if not isinstance(values, (list, tuple, set)) or not 1 <= len(values) <= 100:
        raise ValueError('Evidence requires an explicit nonempty symbol collection')
    return sorted({_identity(value) for value in values})


def _interval(start, end):
    begin, finish = _stamp(start), _stamp(end)
    if not begin < finish:
        raise ValueError('Evidence interval must be positive and timezone-aware')
    return begin.tz_convert('UTC').isoformat(), finish.tz_convert('UTC').isoformat()


class ExposureLedger:
    def __init__(self, path, *, clock=None):
        # Only trusted controller/test code injects a clock, never a GPT proposal.
        self.clock = clock or (lambda: pd.Timestamp.now(tz='UTC'))
        self.path = project_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY, source TEXT, symbol TEXT, start TEXT, end TEXT,
                role TEXT, batch TEXT, detail TEXT, observed_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS confirmations (
                batch TEXT PRIMARY KEY, identity TEXT NOT NULL,
                consumed INTEGER NOT NULL DEFAULT 0, result TEXT,
                cohort INTEGER NOT NULL, claim_slot TEXT NOT NULL, reserved_at TEXT NOT NULL,
                UNIQUE(cohort,claim_slot));
            CREATE TABLE IF NOT EXISTS validations (batch TEXT PRIMARY KEY, identity TEXT NOT NULL);
        ''')
        if ('observed_at' not in {row[1] for row in self.db.execute('PRAGMA table_info(evidence)')}
                or 'cohort' not in {row[1] for row in self.db.execute('PRAGMA table_info(confirmations)')}):
            self.close()
            raise ValueError('Pre-release exposure ledger has no frozen clock/slot contract; use a new ledger')

    def close(self):
        self.db.close()

    def record(self, *, source, symbols, start, end, role, batch, detail=None, observed_at=None):
        _identity(source)
        _identity(batch)
        symbols = _symbols(symbols)
        if role not in {'S', 'V', 'C', 'legacy'}:
            raise ValueError('Invalid evidence identity')
        begin, finish = _interval(start, end)
        if role == 'S':
            protected = self.db.execute('SELECT identity FROM validations WHERE batch=?', (batch,)).fetchone()
            if protected:
                value = json.loads(protected[0])
                if (value['source'] == source and set(symbols) & set(value['symbols'])
                        and _stamp(begin) < _stamp(value['end']) and _stamp(finish) > _stamp(value['start'])):
                    raise ValueError('Frozen V cannot become same-batch search feedback')
        now = _stamp(self.clock())
        observed = now if observed_at is None else _stamp(observed_at)
        if observed > now or _stamp(finish) > observed:
            raise ValueError('Exposure must have occurred before recording; future reservations are separate')
        with self.db:
            for symbol in symbols:
                record = {'source': source, 'symbol': symbol, 'start': begin, 'end': finish,
                          'role': role, 'batch': batch, 'detail': detail or {},
                          'observed_at': observed.isoformat(), 'recorded_at': now.isoformat()}
                self.db.execute('INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)',
                                (fingerprint(record), source, symbol, begin, finish, role, batch,
                                 json.dumps(record['detail'], sort_keys=True, allow_nan=False),
                                 record['observed_at'], record['recorded_at']))

    def _overlaps(self, source, symbols, start, end):
        begin, finish = _interval(start, end)
        for symbol in symbols:
            yield from self.db.execute('SELECT role,batch,start,end,symbol,observed_at,recorded_at FROM evidence WHERE '
                                       'source=? AND symbol=? AND start < ? AND end > ?',
                                       (source, symbol, finish, begin))

    def reserve_validation(self, *, batch, source, symbols, start, end,
                           candidate_ids, reference_id, frozen_at):
        begin, finish = _interval(start, end)
        frozen, now = _stamp(frozen_at), _stamp(self.clock())
        symbols = _symbols(symbols)
        for value in (batch, source, reference_id, *candidate_ids):
            _identity(value)
        if (not 1 <= len(candidate_ids) <= 3 or len(set(candidate_ids)) != len(candidate_ids)
                or frozen > now or frozen > _stamp(begin)):
            raise ValueError('Validation must freeze a bounded unique shortlist prospectively')
        record = {'batch': batch, 'source': source, 'symbols': symbols, 'start': begin, 'end': finish,
                  'candidate_ids': list(candidate_ids), 'reference_id': reference_id,
                  'frozen_at': frozen.isoformat()}
        text = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT identity FROM validations WHERE batch=?', (batch,)).fetchone()
            if old:
                if old[0] != text:
                    raise ValueError('Frozen validation identity cannot change')
                return fingerprint(record)
            if now > _stamp(begin) or list(self._overlaps(source, symbols, begin, finish)):
                raise ValueError('New validation must reserve an unexposed future interval')
            self.db.execute('INSERT INTO validations VALUES (?,?)', (batch, text))
        return fingerprint(record)

    def reserve_confirmation(self, *, batch, source, symbols, start, end,
                             candidate_id, reference_id, rule, frozen_at, cohort, claim_slot,
                             alpha=0.01, actual_start=None):
        begin, finish = _interval(start, end)
        symbols = _symbols(symbols)
        for value in (batch, source, candidate_id, reference_id):
            _identity(value)
        if alpha != 0.01 or type(cohort) is not int or cohort not in COHORTS or claim_slot not in CLAIM_SLOTS:
            raise ValueError('V7 freezes per-claim alpha=0.01 and complete identities')
        if not isinstance(rule, dict):
            raise ValueError('Confirmation requires a frozen rule dictionary')  # noqa: TRY004 - protocol rejection API
        horizon = rule.get('horizon_hours', 12)
        if type(horizon) is not int or not 1 <= horizon <= 168:
            raise ValueError('Confirmation horizon must be a positive bounded integer')
        frozen, now = _stamp(frozen_at), _stamp(self.clock())
        actual = _stamp(begin) if actual_start is None else _stamp(actual_start)
        if (actual < _stamp(begin) or actual >= _stamp(finish)
                or actual > _stamp(begin) + pd.Timedelta(hours=24)
                or frozen > actual):
            raise ValueError('Candidate must freeze before confirmation actual activation within 24h')
        scoring_cutoff = _stamp(finish) + pd.Timedelta(hours=horizon + 24)
        record = {'batch': batch, 'source': source, 'symbols': symbols,
                  'start': begin, 'end': finish, 'candidate_id': candidate_id,
                  'reference_id': reference_id, 'rule': {**rule, 'horizon_hours': horizon}, 'alpha': alpha,
                  'frozen_at': frozen.isoformat(), 'cohort': cohort, 'claim_slot': claim_slot,
                  'scoring_cutoff': scoring_cutoff.isoformat()}
        if actual_start is not None:
            record['actual_start'] = actual.isoformat()
            record['nominal_start'] = begin
            record['activation_delay_seconds'] = (actual - _stamp(begin)).total_seconds()
        text = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT identity FROM confirmations WHERE batch=?', (batch,)).fetchone()
            if old:
                if old[0] != text:
                    raise ValueError('Frozen confirmation identity cannot be changed')
                return fingerprint(record)
            if self.db.execute('SELECT 1 FROM confirmations WHERE cohort=? AND claim_slot=?',
                               (cohort, claim_slot)).fetchone():
                raise ValueError('Confirmation claim slot already reserved; renaming cannot reset its budget')
            if frozen > now or now > actual:
                raise ValueError('New confirmation must be reserved prospectively after its actual freeze')
            if list(self._overlaps(source, symbols, start, end)):
                raise ValueError('Confirmation interval was already exposed')
            self.db.execute('''INSERT INTO confirmations(batch,identity,cohort,claim_slot,reserved_at)
                VALUES (?,?,?,?,?)''', (batch, text, cohort, claim_slot, now.isoformat()))
        return fingerprint(record)

    def consume_confirmation(self, batch, *, as_of, receipt_grace_hours=24):
        stamp, now = _stamp(as_of), _stamp(self.clock())
        if receipt_grace_hours != 24 or stamp > now:
            raise ValueError('Confirmation endpoint uses aware time and fixed 24-hour receipt grace')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT identity,consumed FROM confirmations WHERE batch=?', (batch,)).fetchone()
            if row is None or row[1]:
                raise ValueError('Confirmation is missing or already consumed')
            identity = json.loads(row[0])
            if stamp < _stamp(identity['scoring_cutoff']):
                raise ValueError('Confirmation labels/receipt grace have not fully matured')
            for role, feedback_batch, _, _, _, observed_at, _ in self._overlaps(
                    identity['source'], identity['symbols'], identity['start'], identity['end']):
                if feedback_batch == batch or _stamp(observed_at) <= _stamp(identity['frozen_at']):
                    raise ValueError('Confirmation data exposed to this frozen batch')
            # Consume before reading outcomes; failures cannot obtain another look.
            self.db.execute('UPDATE confirmations SET consumed=1 WHERE batch=?', (batch,))
            for symbol in identity['symbols']:
                record = {**identity, 'symbol': symbol, 'role': 'C',
                          'observed_at': now.isoformat(), 'recorded_at': now.isoformat()}
                self.db.execute('INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)',
                                (fingerprint(record), identity['source'], symbol, identity['start'],
                                 identity['end'], 'C', batch, json.dumps(identity, sort_keys=True),
                                 now.isoformat(), now.isoformat()))
        return identity

    def finish_confirmation(self, batch, result):
        value = json.dumps(result, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT consumed,result FROM confirmations WHERE batch=?', (batch,)).fetchone()
            if row is None or not row[0] or row[1] is not None:
                raise ValueError('Only a consumed, unfinished confirmation can record one result')
            self.db.execute('UPDATE confirmations SET result=? WHERE batch=?', (value, batch))

    def assert_prompt_search(self, *, source, symbols, start, end, batch):
        _identity(source)
        _identity(batch)
        symbols = _symbols(symbols)
        begin, finish = _interval(start, end)
        now = _stamp(self.clock())
        rows = list(self._overlaps(source, symbols, start, end))
        current = [row for row in rows if row[1] == batch]
        validation = self.db.execute('SELECT identity FROM validations WHERE batch=?', (batch,)).fetchone()
        if validation:
            value = json.loads(validation[0])
            if (value['source'] == source and set(symbols) & set(value['symbols'])
                    and _stamp(begin) < _stamp(value['end']) and _stamp(finish) > _stamp(value['start'])):
                raise ValueError('Frozen V cannot enter its own batch prompt')
        if any(row[0] in {'V', 'C'} for row in current):
            raise ValueError('Protected V/C cannot become search feedback within its batch')
        reservation = self.db.execute('SELECT identity FROM confirmations WHERE batch=?', (batch,)).fetchone()
        if reservation:
            identity = json.loads(reservation[0])
            if (identity['source'] == source and set(symbols) & set(identity['symbols'])
                    and _stamp(begin) < _stamp(identity['end']) and _stamp(finish) > _stamp(identity['start'])):
                raise ValueError('Frozen confirmation cannot become search feedback within its batch')
        for symbol in symbols:
            cursor = _stamp(begin)
            intervals = sorted((_stamp(row[2]), _stamp(row[3])) for row in current
                               if row[0] == 'S' and row[4] == symbol
                               and _stamp(row[5]) <= now and _stamp(row[6]) <= now)
            for left, right in intervals:
                if left > cursor:
                    break
                cursor = max(cursor, right)
                if cursor >= _stamp(finish):
                    break
            if cursor < _stamp(finish):
                raise ValueError('Each symbol needs complete explicitly recorded search coverage for this batch')
        # Reusing another batch's old V/C needs this batch's explicit S record;
        # all older roles remain in the ledger and cannot confirm this new claim.
        return True

    def budget_status(self):
        allocated = {cohort: 0 for cohort in COHORTS}
        for cohort, count in self.db.execute('SELECT cohort,COUNT(*) FROM confirmations GROUP BY cohort'):
            allocated[cohort] = count
        return {'cohort_limit': 1, 'slots_per_cohort': 5, 'alpha_per_slot': .01,
                'planned_alpha_per_cohort': .05, 'planned_alpha_all_cohorts': .05,
                'reserved_slots': allocated, 'recycling': False,
                'calibrated_individual_tests_required': True, 'lifetime_fwer_05_claimed': False}


def select_on_validation(records, *, minimum_gain=0.005, coverage=0.95, limit=3):
    """A bounded point-estimate ranking, not a significance or alpha claim."""
    if limit != 3 or minimum_gain != 0.005 or coverage != 0.95:
        raise ValueError('Validation gate is frozen in V7')
    if len(records) > limit:
        raise ValueError('Validation candidate budget exceeded')
    eligible = []
    seen = set()
    for row in records:
        if row['role'] != 'V' or row['common_support'] is not True:
            raise ValueError('Validation requires declared common support')
        candidate = _identity(row['candidate_id'])
        if candidate in seen:
            raise ValueError('Validation candidates must have unique identities')
        seen.add(candidate)
        if (type(row['coverage']) not in (int, float) or not math.isfinite(row['coverage'])
                or not 0 <= row['coverage'] <= 1
                or (row['gain'] is not None and (type(row['gain']) not in (int, float)
                                                or not math.isfinite(row['gain'])))
                or type(row['complexity']) is not int or row['complexity'] < 0):
            raise ValueError('Validation scores must be finite with valid coverage and complexity')
        if row['gain'] is None:
            continue  # Zero reference loss has no defined relative gain.
        if row['coverage'] >= coverage and row['gain'] > minimum_gain:
            eligible.append(row)
    return min(eligible, key=lambda r: (-r['gain'], r['complexity'], r['candidate_id'])) if eligible else None
