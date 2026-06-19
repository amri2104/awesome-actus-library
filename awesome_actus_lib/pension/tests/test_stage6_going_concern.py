"""Offline smoke test for the Stage 6 going-concern regression guards (synthetic fixtures, no ACTUS service)."""

from awesome_actus_lib.pension.analysis.going_concern import (
    _regression_check_against_5c,
)


def test_stage6_offline_regression_guards():
    _regression_check_against_5c()
