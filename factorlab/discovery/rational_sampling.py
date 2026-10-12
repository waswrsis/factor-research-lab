"""Finite exact-rational sampling component; no calibrator or launch authority.

Exactness assumes independent uniform b-bit blocks. A sequence supplied for
replay is not evidence of that assumption; secrets is an implementation boundary,
not a mathematical certification of hardware randomness or provider independence.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from pathlib import Path
import json
import os
import secrets

VERSION = 'bounded-exact-rational-sampling-candidate-v1'
MAX_BITS = 4096
MAX_ATTEMPTS = 1024
MAX_ATOMS = 128


@dataclass(frozen=True)
class SamplingPlan:
    weights: tuple[Fraction, ...]
    block_bits: int
    max_attempts: int

    def __post_init__(self):
        if (type(self.weights) is not tuple or not 1 <= len(self.weights) <= MAX_ATOMS
                or type(self.block_bits) is not int or not 1 <= self.block_bits <= MAX_BITS
                or type(self.max_attempts) is not int or not 1 <= self.max_attempts <= MAX_ATTEMPTS):
            raise ValueError('Strict bounded tuple/Fraction and integer parameters required')
        if any(type(w) is not Fraction or not 0 <= w <= 1 or w.denominator.bit_length() > MAX_BITS + 1
               for w in self.weights) or sum(self.weights, Fraction()) != 1:
            raise ValueError('Exact nonnegative Fraction weights summing to one required')
        denominator = 1
        for weight in self.weights:
            denominator = lcm(denominator, weight.denominator)
            if denominator > 1 << self.block_bits:
                raise ValueError('Weight denominator exceeds one registered block; do not enlarge budget')

    @property
    def denominator(self):
        return lcm(*(w.denominator for w in self.weights))

    @property
    def cutoff(self):
        size = 1 << self.block_bits
        return (size // self.denominator) * self.denominator

    @property
    def counts(self):
        return tuple(int(w * self.denominator) for w in self.weights)

    def to_dict(self):
        return {'version': VERSION, 'weights': [[w.numerator, w.denominator] for w in self.weights],
                'block_bits': self.block_bits, 'max_attempts': self.max_attempts,
                'denominator': self.denominator, 'accept_below': self.cutoff,
                'entropy_model': 'independent_uniform_b_bit_blocks_assumed_not_certified',
                'formal_acceptance': False}


def theoretical_law(plan):
    """Exact finite-cap law under the stated block model, not empirical evidence."""
    if type(plan) is not SamplingPlan:
        raise TypeError('Original immutable sampling plan required')
    failure = Fraction((1 << plan.block_bits) - plan.cutoff, 1 << plan.block_bits) ** plan.max_attempts
    return {'failure': failure, 'success': 1-failure,
            'unconditional_atoms': tuple((1-failure)*p for p in plan.weights),
            'conditional_on_success': plan.weights, 'formal_acceptance': False}


def _observation(value, bits):
    # Never call a caller object's repr/int conversion. Invalid objects cannot
    # become randomness, and their invalid type is retained in the failed bill.
    if type(value) is int and value.bit_length() <= MAX_BITS + 1:
        return {'returned_type': 'int', 'block': value}
    if type(value) is bool:
        return {'returned_type': 'bool', 'block': value}
    return {'returned_type': type(value).__name__[:100], 'block': None,
            'invalid_return_omitted': True}


class _Run:
    def __init__(self, plan, origin):
        if type(plan) is not SamplingPlan:
            raise TypeError('Original immutable sampling plan required')
        self.plan = plan
        self.origin = origin
        self.events = []
        self.requests = 0
        self.selected = None
        self.status = None

    def request(self):
        if self.status is not None or self.requests >= self.plan.max_attempts:
            raise ValueError('Terminal or allocated attempt budget cannot be extended')
        self.requests += 1
        event = {'event': 'block_request_charged', 'attempt': self.requests,
                 'requested_bits': self.plan.block_bits}
        self.events.append(event)
        return event

    def observe(self, value):
        event = {'event': 'block_returned', 'attempt': self.requests,
                 **_observation(value, self.plan.block_bits)}
        if type(value) is not int or not 0 <= value < 1 << self.plan.block_bits:
            event['disposition'] = 'invalid_entropy_return'
            self.status = 'invalid_entropy_consumed'
        elif value >= self.plan.cutoff:
            event['disposition'] = 'rejected_remainder'
            if self.requests == self.plan.max_attempts:
                self.status = 'attempt_budget_exhausted'
        else:
            residue = value % self.plan.denominator
            upper = 0
            for index, count in enumerate(self.plan.counts):
                upper += count
                if residue < upper:
                    self.selected = index
                    break
            event.update(disposition='sampled', residue=residue, atom=self.selected)
            self.status = 'sampled'
        self.events.append(event)
        return event

    def failure(self, error):
        self.status = 'entropy_error_consumed'
        event = {'event': 'entropy_error', 'attempt': self.requests, 'error_type': error}
        self.events.append(event)
        return event

    def receipt(self):
        if self.status is None:
            raise ValueError('Missing original terminal; treat reserved allocation as unknown/consumed')
        return {'version': VERSION, 'plan': self.plan.to_dict(), 'origin': self.origin,
                'status': self.status, 'atom': self.selected, 'events': list(self.events),
                'block_requests': self.requests, 'requested_bits': self.requests*self.plan.block_bits,
                'reserved_attempts_consumed': self.plan.max_attempts,
                'unused_attempts': self.plan.max_attempts-self.requests,
                'allocation_consumed': True, 'retry_allowed': False,
                'trial_denominator_contribution': 1, 'formal_acceptance': False}


def sample_frozen(plan, blocks):
    """Pure bounded replay. Repeating this computes the same record, not a redraw.

    The tuple freezes at most m input blocks. Unrequested suffixes are not entropy
    requests. A missing/invalid requested value consumes its already billed slot.
    This function owns no durable trial registry and must never issue one.
    """
    run = _Run(plan, 'frozen_sequence_replay_not_an_entropy_certificate')
    if type(blocks) is not tuple or len(blocks) > plan.max_attempts:
        run.status = 'invalid_sequence_consumed'
        return run.receipt()
    for attempt in range(plan.max_attempts):
        run.request()
        if attempt >= len(blocks):
            run.failure('MissingFrozenBlock')
        else:
            run.observe(blocks[attempt])
        if run.status is not None:
            return run.receipt()
    raise AssertionError('Finite attempt budget must produce a terminal')


def _write_exclusive(path, value):
    body = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _append(stream, value):
    stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())
    stream.flush()
    os.fsync(stream.fileno())


def sample_system_once(directory, plan):
    """One exclusive durable allocation with fixed fresh secrets.randbits calls.

    No caller entropy callback/seed, continuation, retry, resume or replacement is
    accepted. The caller must preregister this directory as the original trial;
    creating another directory is not replacement permission. Missing terminal,
    write error, interruption or killed process means consumed/unknown, not free.
    A charged journal entry is durable before each entropy request. Returned bits
    are recorded before any later request. This is a necessary generator boundary,
    not integration with the production monthly authority or risk certificate.
    """
    run = _Run(plan, 'system_secrets_fresh_blocks_model_assumption')
    directory = Path(directory)
    if (not directory.is_absolute() or directory.resolve() != directory or directory.is_symlink()
            or not directory.parent.is_dir()):
        raise ValueError('Canonical original trial directory with existing parent required')
    directory.mkdir(exist_ok=False)
    _write_exclusive(directory/'allocation.json', {'plan': plan.to_dict(), 'reserved_attempts': plan.max_attempts,
                     'directory': str(directory), 'retry_allowed': False, 'formal_acceptance': False})
    with (directory/'events.jsonl').open('xb') as stream:
        for _ in range(plan.max_attempts):
            _append(stream, run.request())
            try:
                value = secrets.randbits(plan.block_bits)
            except BaseException as error:
                _append(stream, run.failure(type(error).__name__))
            else:
                _append(stream, run.observe(value))
            if run.status is not None:
                break
    receipt = run.receipt()
    _write_exclusive(directory/'terminal.json', receipt)
    return receipt

