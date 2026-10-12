"""Candidate time-window tests only; no operational or predictive acceptance."""
from pathlib import Path
import importlib.util
import sys,unittest
import numpy as np
import pandas as pd

ROOT=Path('D:/FactorResearch');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from factorlab.discovery.rolling import RollingSpec as OriginalSpec
from factorlab.discovery.known_library import FactorFrame

from factorlab.discovery import rolling_cadence as cadence
from factorlab.discovery import rolling

class Windows(unittest.TestCase):
    def setUp(self):
        self.rule=cadence.RollingCadence()
        self.anchor=pd.Timestamp('2026-10-10T06:34:39.609948Z')

    def test_window_is_2160_hours_across_leap_month(self):
        end=pd.Timestamp('2024-03-10T12:00:00Z')
        start,actual=self.rule.window(end)
        self.assertEqual(actual,end)
        self.assertEqual((actual-start)/pd.Timedelta(hours=1),2160)
        self.assertEqual(start,pd.Timestamp('2023-12-11T12:00:00Z'))

    def test_timezone_conversion_preserves_instant(self):
        end=pd.Timestamp('2026-10-10T15:34:39.609948+09:00')
        self.assertEqual(self.rule.window(end)[1],self.anchor)

    def test_naive_and_missing_timestamps_rejected(self):
        for value in ('2026-10-10',pd.NaT):
            with self.assertRaises(ValueError):self.rule.window(value)

    def test_bool_and_unregistered_periods_rejected(self):
        for field,value in [('training_days',True),('training_days',91),('refit_days',1),('refresh_days',2)]:
            with self.assertRaises(ValueError):cadence.RollingCadence(**{field:value})

    def test_before_anchor_cannot_create_backdated_fit(self):
        before=self.anchor-pd.Timedelta(nanoseconds=1)
        self.assertIsNone(self.rule.latest_fit_cutoff(self.anchor,before))
        self.assertEqual(self.rule.next_fit_cutoff(self.anchor,before),self.anchor)

    def test_first_cutoff_is_actual_anchor(self):
        self.assertEqual(self.rule.latest_fit_cutoff(self.anchor,self.anchor),self.anchor)
        self.assertEqual(self.rule.next_fit_cutoff(self.anchor,self.anchor),self.anchor+pd.Timedelta(days=15))

    def test_fifteen_days_not_first_or_fifteenth_of_calendar_month(self):
        at=self.anchor+pd.Timedelta(days=31)
        self.assertEqual(self.rule.latest_fit_cutoff(self.anchor,at),self.anchor+pd.Timedelta(days=30))
        self.assertEqual(self.rule.latest_fit_cutoff(self.anchor,at).day,9)

    def test_subsecond_boundary_not_rounded_forward(self):
        at=self.anchor+pd.Timedelta(days=15)-pd.Timedelta(nanoseconds=1)
        self.assertEqual(self.rule.latest_fit_cutoff(self.anchor,at),self.anchor)

    def test_restart_uses_original_anchor_and_same_slot(self):
        cutoff=self.anchor+pd.Timedelta(days=45)
        new=cadence.RollingCadence()
        self.assertEqual(new.slot_identity(self.anchor,cutoff),self.rule.slot_identity(self.anchor,cutoff))
        self.assertEqual(new.latest_fit_cutoff(self.anchor,self.anchor+pd.Timedelta(days=49)),cutoff)

    def test_non_grid_cutoff_rejected(self):
        with self.assertRaises(ValueError):self.rule.slot_identity(self.anchor,self.anchor+pd.Timedelta(days=14))

    def test_daily_refresh_does_not_create_daily_refits(self):
        at=self.anchor+pd.Timedelta(days=7,hours=10)
        self.assertEqual(self.rule.latest_refresh_cutoff(self.anchor,at),self.anchor+pd.Timedelta(days=7))
        self.assertEqual(self.rule.latest_fit_cutoff(self.anchor,at),self.anchor)

    def test_legacy_digest_and_window_preserved(self):
        original=OriginalSpec();candidate=rolling.RollingSpec()
        self.assertEqual(candidate.digest,original.digest)
        cutoff=pd.Timestamp('2026-11-01T00:00:00Z')
        self.assertEqual(candidate.window(cutoff),(pd.Timestamp('2026-08-01T00:00:00Z'),cutoff))
        with self.assertRaises(ValueError):candidate.window(self.anchor)

    def test_elapsed_rule_has_separate_identity(self):
        elapsed=rolling.RollingSpec(window_days=90)
        self.assertNotEqual(elapsed.digest,rolling.RollingSpec().digest)
        self.assertEqual(elapsed.window(self.anchor),self.rule.window(self.anchor))

    def data(self,invalid=False):
        origins=pd.DatetimeIndex([self.anchor-pd.Timedelta(days=91),self.anchor-pd.Timedelta(days=90),
            self.anchor-pd.Timedelta(days=89),self.anchor-pd.Timedelta(hours=12),
            self.anchor-pd.Timedelta(hours=1),self.anchor])
        values=np.ones((6,1));observed=np.ones(6,dtype=bool)
        if invalid:values[1]=np.nan;observed[2]=False
        frame=FactorFrame(values,pd.DataFrame({'symbol':['BTCUSDT']*6,'available_at':origins}),
            feature_names=('test_feature',),bypass_mask=(False,))
        return rolling.RollingData(frame,np.ones(6),origins+pd.Timedelta(hours=12),observed,
            rolling.RollingSpec(window_days=90,minimum_train_rows=2),'actual_receipt')

    def test_training_mask_uses_window_and_mature_labels(self):
        np.testing.assert_array_equal(self.data().train_mask(self.anchor),[False,True,True,True,False,False])

    def test_missing_support_and_unobserved_labels_excluded(self):
        np.testing.assert_array_equal(self.data(True).train_mask(self.anchor),[False,False,False,True,False,False])

    def test_training_identity_records_actual_elapsed_dates(self):
        data=self.data();value=data.training_identity(self.anchor)
        self.assertEqual(value['train_start'],(self.anchor-pd.Timedelta(days=90)).isoformat())
        self.assertEqual(value['cutoff'],self.anchor.isoformat())
        self.assertEqual(value['rows'],3)
        self.assertEqual(value['rolling_hash'],data.spec.digest)

if __name__=='__main__':unittest.main(verbosity=2)
