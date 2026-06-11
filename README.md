# Awesome ACTUS Library

A modular Python library for financial contract modelling with the ACTUS
standard.

## Bachelor Thesis Extension

This branch contains the implementation work for the bachelor thesis
"Enhanced Portfolio Analytics for Pension Funds using ACTUS and Python".

The extension adds a pension-fund ALM layer on top of the Awesome ACTUS
Library:

- ACTUS/AAL remains the native engine for asset-side contracts and cashflows,
  especially PAM bond portfolios and stochastic risk-factor scenarios.
- Pension-fund liabilities are modelled with a separate Pension/BVG domain
  engine: cohorts, pension policies, salary and conversion-rate dynamics,
  mortality tables, liability paths, funding-ratio calculations and retirement
  loss analysis.
- Asset and liability cashflows meet in shared ALM, liquidity, value and
  funding-ratio analyses.
- Stage 5d adds a going-concern funding-ratio extension with buy-and-hold,
  fixed-mix rebalancing, allocation-shift and asset-only recovery-measure
  scenarios.

Important scope statement: the BVG liability side is not claimed to be fully
ACTUS-native. It is attached as an external domain engine around the ACTUS
asset and cashflow layer, documenting a practical boundary of the ACTUS
standard for Swiss pension-fund ALM.

## Original AAL Features

- Define and simulate ACTUS contracts and portfolios.
- Generate complete event streams from ACTUS contracts.
- Perform income, liquidity and value analysis.
- Visualize cashflows for single contracts or full portfolios.

## Key Pension Examples

- `awesome_actus_lib/pension/examples/Stage4_pension_quickstart.py`
- `awesome_actus_lib/pension/examples/Stage5c_forward_funding_ratio_quickstart.py`
- `awesome_actus_lib/pension/examples/Stage5d_going_concern_quickstart.py`

## Install

```bash
pip install awesome-actus-lib
```

For local thesis runs from the repository root:

```bash
PYTHONPATH=. python3 awesome_actus_lib/pension/examples/Stage5d_going_concern_quickstart.py
```

## Documentation

The original project documentation is available at
https://awesomeactuslibrarydoc.vercel.app. Pension-specific notes live under
`awesome_actus_lib/pension/docs/`.

## ACTUS Foundation

This library is based on the ACTUS Financial Contract Standard. Learn more at
https://www.actusfrf.org.

## Author & Contributions

Original library maintained by [@maierdon](https://github.com/maierdon).
This branch contains bachelor-thesis extensions for pension-fund analytics.
