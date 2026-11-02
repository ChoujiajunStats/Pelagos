"""Apply explicitly configured model-plan age without changing command leases."""
import math


def configured_plan_age(source_age,settings,config):
    # The mounted Stonefish adapter clamps model plans to .2 s. An explicit
    # user-approved MPPI override must also reach this execution-side check.
    if 'maximum_plan_age_seconds' not in settings.get('mppi_overrides',{}):return source_age
    age=float(config.maximum_plan_age_seconds)
    if not math.isfinite(age) or age<=0:raise ValueError('Invalid configured model plan age')
    return age
