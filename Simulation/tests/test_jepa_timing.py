import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from jepa_timing import configured_plan_age


class TimingChecks(unittest.TestCase):
    def test_source_cap_is_retained_without_explicit_override(self):
        self.assertEqual(configured_plan_age(.2,{},SimpleNamespace(maximum_plan_age_seconds=.4)),.2)

    def test_explicit_override_reaches_execution_age(self):
        settings=dict(mppi_overrides=dict(maximum_plan_age_seconds=.4))
        self.assertEqual(configured_plan_age(.2,settings,SimpleNamespace(maximum_plan_age_seconds=.4)),.4)

    def test_invalid_override_rejected(self):
        settings=dict(mppi_overrides=dict(maximum_plan_age_seconds=.4))
        for age in [0,-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):configured_plan_age(.2,settings,SimpleNamespace(maximum_plan_age_seconds=age))


if __name__=='__main__':unittest.main()
