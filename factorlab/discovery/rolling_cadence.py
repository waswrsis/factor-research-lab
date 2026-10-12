"""Elapsed-time training cadence, independent of calendar accounting months."""
from dataclasses import asdict,dataclass
import pandas as pd
from factorlab.protocol import fingerprint

VERSION='v7-elapsed90-refit15-refresh1-v1'

def aware_utc(value):
    stamp=pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Elapsed cadence requires a finite aware UTC timestamp')
    try:return stamp.tz_convert('UTC').as_unit('ns')
    except (OverflowError,ValueError) as error:
        raise ValueError('Elapsed cadence requires a bounded nanosecond timestamp') from error

def elapsed_window(cutoff,days=90):
    if type(days) is not int or days!=90:
        raise ValueError('This version freezes an elapsed 90-day training window')
    end=aware_utc(cutoff)
    return end-pd.Timedelta(days=days),end

@dataclass(frozen=True)
class RollingCadence:
    training_days:int=90
    refit_days:int=15
    refresh_days:int=1

    def __post_init__(self):
        if any(type(x) is not int for x in (self.training_days,self.refit_days,self.refresh_days)):
            raise ValueError('Cadence periods must be exact integers')
        if (self.training_days,self.refit_days,self.refresh_days)!=(90,15,1):
            raise ValueError('Frozen cadence is 90-day training, 15-day refit, daily view refresh')

    @property
    def digest(self):
        return fingerprint({'version':VERSION,**asdict(self),
            'anchor':'durably_recorded_actual_first_preparation_time',
            'missed_slots':'never_backfill_predictions_or_reset_anchor'})

    def window(self,cutoff):
        return elapsed_window(cutoff,self.training_days)

    @staticmethod
    def _latest(anchor,as_of,days):
        anchor,as_of=aware_utc(anchor),aware_utc(as_of)
        if as_of<anchor:return None
        step=pd.Timedelta(days=days)
        index=(as_of.value-anchor.value)//step.value
        return pd.Timestamp(anchor.value+index*step.value,tz='UTC')

    def latest_fit_cutoff(self,anchor,as_of):
        return self._latest(anchor,as_of,self.refit_days)

    def latest_refresh_cutoff(self,anchor,as_of):
        return self._latest(anchor,as_of,self.refresh_days)

    def next_fit_cutoff(self,anchor,as_of):
        latest=self.latest_fit_cutoff(anchor,as_of)
        return aware_utc(anchor) if latest is None else latest+pd.Timedelta(days=self.refit_days)

    def fit_cutoffs(self,anchor,start,end,*,include_current_cycle=True):
        """Pure half-open endpoint arithmetic; never grants fit permission."""
        if type(include_current_cycle) is not bool:
            raise ValueError('Exact original-cycle inclusion flag required')
        anchor,start,end=map(aware_utc,(anchor,start,end))
        if start>=end:
            raise ValueError('Nonempty ordered endpoint required')
        first=self.latest_fit_cutoff(anchor,start)
        if first is None:first=anchor
        elif not include_current_cycle and first<start:
            first+=pd.Timedelta(days=self.refit_days)
        if first>=end:return ()
        step=pd.Timedelta(days=self.refit_days)
        count=(end.value-first.value-1)//step.value+1
        if count>8192:
            raise ValueError('Original endpoint exceeds bounded cadence inventory')
        return tuple(pd.Timestamp(first.value+i*step.value,tz='UTC') for i in range(count))

    def slot_identity(self,anchor,cutoff):
        anchor,cutoff=aware_utc(anchor),aware_utc(cutoff)
        latest=self.latest_fit_cutoff(anchor,cutoff)
        if latest is None or latest!=cutoff:
            raise ValueError('Fit cutoff is outside the original anchored 15-day grid')
        return {'version':VERSION,'cadence_hash':self.digest,
            'anchor':anchor.isoformat(),'cutoff':cutoff.isoformat(),
            'index':(cutoff.value-anchor.value)//pd.Timedelta(days=self.refit_days).value,
            'training_start':self.window(cutoff)[0].isoformat()}
