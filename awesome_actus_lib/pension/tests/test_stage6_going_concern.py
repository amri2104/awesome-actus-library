"""Offline smoke tests for the Stage 6 going-concern extension.

The helper builds synthetic in-memory fixtures and does not call the ACTUS
service. It covers the key regression guards used by the Stage 6 prototype:
5c delegation, fixed-mix rebalancing, allocation shift, equity_sigma,
bond_yield and SanierungsPolicy logging.
"""

from awesome_actus_lib.pension.analysis.going_concern import (
    _regression_check_against_5c,
)


def test_stage6_offline_regression_guards():
    _regression_check_against_5c()
