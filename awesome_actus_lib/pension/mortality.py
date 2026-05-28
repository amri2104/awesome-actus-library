"""Stage 3 — mortality tables and survival probability helper.

Period mortality table EK 2001-2005 (Swiss pension-actuarial reference,
"Erfahrungstafeln Kollektivversicherung", evaluation period 2001-2005).
Constants below are extracted once from

    awesome_actus_lib/pension/mortality_tables_EK.xlsx

sheets "EKM 0105" (men) and "EKF 0105" (women), columns ``x`` and ``qx``
(one-year death probability). The other sheets (EKM 95, EKF 95,
qx-Grafiken) and all commutation columns (lx, dx, Dx, Nx, ...) are
ignored. The Excel file is kept in the repo for traceability; this
module has no runtime Excel dependency.

This is a **period** table: it does not encode calendar-year mortality
improvement. A future stage may layer an improvement factor on top.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class MortalityTable:
    """One-year death probabilities by age and gender.

    ``q_male`` and ``q_female`` map integer age -> q_x in [0, 1].

    Optional generational (Kohorten-) projection: when ``improvement_rate``
    > 0 and a ``base_year`` is set, the stored period ``q_x`` is projected to
    a calendar year via the standard exponential trend
    ``q_x(cal_year) = q_x * (1 - improvement_rate) ** (cal_year - base_year)``.
    The projection is applied only when ``survival_probability`` is called
    with an explicit ``cal_year``; without it the raw period table is used,
    so the simulator's headcount decrement and all Stage 1-5b behaviour are
    unchanged. Defaults (``improvement_rate=0.0``) reproduce the pure period
    table exactly.
    """
    q_male: Dict[int, float]
    q_female: Dict[int, float]
    improvement_rate: float = 0.0
    base_year: Optional[int] = None

    def survival_probability(
        self, age: int, gender: str, cal_year: Optional[int] = None
    ) -> float:
        """Return 1 - q_x for the given age and gender.

        gender:
            "m"      -> uses q_male
            "f"      -> uses q_female
            "unisex" -> uses 0.5 * (q_male + q_female)

        ``cal_year`` (optional): calendar year at which to evaluate mortality.
        When given together with a non-zero ``improvement_rate`` and a
        ``base_year``, the period ``q_x`` is projected generationally
        (lower mortality in later years). Omitting it uses the raw period
        value.

        Out-of-range ages return 0.0 (treat as already dead), mirroring
        the existing ``terminal_age`` boundary in the simulator.
        """
        if gender == "m":
            q = self.q_male.get(age)
        elif gender == "f":
            q = self.q_female.get(age)
        elif gender == "unisex":
            qm = self.q_male.get(age)
            qf = self.q_female.get(age)
            if qm is None or qf is None:
                return 0.0
            q = 0.5 * (qm + qf)
        else:
            raise ValueError(
                f"gender must be 'm', 'f', or 'unisex'; got {gender!r}"
            )
        if q is None:
            return 0.0
        if self.improvement_rate and cal_year is not None and self.base_year is not None:
            q = q * (1.0 - self.improvement_rate) ** (cal_year - self.base_year)
            q = min(max(q, 0.0), 1.0)
        return 1.0 - q

    @classmethod
    def from_excel(
        cls,
        path,
        *,
        male_sheet: Optional[str] = None,
        female_sheet: Optional[str] = None,
        unisex_sheet: Optional[str] = None,
        age_col: str = "x",
        q_col: str = "qx",
    ) -> "MortalityTable":
        """Build a MortalityTable from an Excel workbook.

        Modes (exactly one must be used):
            - gendered: pass both ``male_sheet`` and ``female_sheet``
            - unisex:   pass ``unisex_sheet`` (q_male = q_female).

        If only one of (male_sheet, female_sheet) is given, the supplied
        gender is mirrored to the other and a warning is emitted.

        ``age_col`` and ``q_col`` are the column names within each sheet.
        Defaults match the EK 2001-2005 workbook ("x" and "qx").

        Pandas is imported lazily so the rest of this module stays
        dependency-free.
        """
        import pandas as pd  # lazy

        if unisex_sheet is not None:
            if male_sheet is not None or female_sheet is not None:
                raise ValueError(
                    "from_excel: pass either unisex_sheet OR "
                    "(male_sheet, female_sheet), not both."
                )
            q = _read_excel_qx(pd, path, unisex_sheet, age_col, q_col)
            return cls(q_male=q, q_female=q)

        if male_sheet is None and female_sheet is None:
            raise ValueError(
                "from_excel: must pass unisex_sheet, or at least one of "
                "(male_sheet, female_sheet)."
            )

        if male_sheet is not None and female_sheet is None:
            warnings.warn(
                f"from_excel: only male_sheet={male_sheet!r} given; "
                "mirroring to female.",
                stacklevel=2,
            )
            q = _read_excel_qx(pd, path, male_sheet, age_col, q_col)
            return cls(q_male=q, q_female=q)

        if female_sheet is not None and male_sheet is None:
            warnings.warn(
                f"from_excel: only female_sheet={female_sheet!r} given; "
                "mirroring to male.",
                stacklevel=2,
            )
            q = _read_excel_qx(pd, path, female_sheet, age_col, q_col)
            return cls(q_male=q, q_female=q)

        q_m = _read_excel_qx(pd, path, male_sheet, age_col, q_col)
        q_f = _read_excel_qx(pd, path, female_sheet, age_col, q_col)
        return cls(q_male=q_m, q_female=q_f)

    @classmethod
    def from_csv(
        cls,
        path,
        *,
        age_col: str = "age",
        q_col: str = "qx",
        gender_col: Optional[str] = None,
        male_value: str = "m",
        female_value: str = "f",
    ) -> "MortalityTable":
        """Build a MortalityTable from a CSV file.

        If ``gender_col`` is set, rows are split by that column into
        male/female (matched against ``male_value`` / ``female_value``).
        Otherwise the file is treated as unisex (q_male = q_female).

        Pandas is imported lazily.
        """
        import pandas as pd  # lazy

        df = pd.read_csv(path)
        _require_columns(df, [age_col, q_col] + ([gender_col] if gender_col else []))

        if gender_col is None:
            q = _df_to_qx_dict(df, age_col, q_col)
            return cls(q_male=q, q_female=q)

        male_rows = df[df[gender_col] == male_value]
        female_rows = df[df[gender_col] == female_value]
        if male_rows.empty:
            raise ValueError(
                f"from_csv: no rows with {gender_col}={male_value!r} found."
            )
        if female_rows.empty:
            raise ValueError(
                f"from_csv: no rows with {gender_col}={female_value!r} found."
            )
        return cls(
            q_male=_df_to_qx_dict(male_rows, age_col, q_col),
            q_female=_df_to_qx_dict(female_rows, age_col, q_col),
        )


def _require_columns(df, cols) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"missing required column(s) {missing}; available: {list(df.columns)}"
        )


def _df_to_qx_dict(df, age_col: str, q_col: str) -> Dict[int, float]:
    sub = df[[age_col, q_col]].dropna()
    try:
        ages = sub[age_col].astype(int)
    except (ValueError, TypeError) as e:
        raise ValueError(f"column {age_col!r} could not be coerced to int: {e}")
    qs = sub[q_col].astype(float)
    out: Dict[int, float] = {}
    for a, q in zip(ages, qs):
        if not (0.0 <= q <= 1.0):
            raise ValueError(
                f"q_x out of [0, 1] at age={a}: {q}"
            )
        out[int(a)] = float(q)
    return out


def _read_excel_qx(pd, path, sheet: str, age_col: str, q_col: str) -> Dict[int, float]:
    # EK workbook layout: row 0 = title + "Zinssatz i:" cell, row 1 = column
    # numbers (1..11), row 2 = column headers (x, qx, lx, ...). Use header=2.
    df = pd.read_excel(path, sheet_name=sheet, header=2)
    cols = list(df.columns)
    # Positional fallback: the EK workbook female sheet uses gender-specific
    # column names ("y", "qy") instead of the male sheet's ("x", "qx"). When
    # the requested names are not present but the sheet has >= 2 columns,
    # fall back to the first two columns (age, qx) and warn so the caller
    # knows what happened.
    if age_col not in cols or q_col not in cols:
        if len(cols) >= 2:
            warnings.warn(
                f"from_excel: sheet {sheet!r} does not contain columns "
                f"{age_col!r}/{q_col!r} (available: {cols}); falling back to "
                f"the first two columns ({cols[0]!r}, {cols[1]!r}).",
                stacklevel=4,
            )
            return _df_to_qx_dict(
                df.rename(columns={cols[0]: age_col, cols[1]: q_col}),
                age_col,
                q_col,
            )
        _require_columns(df, [age_col, q_col])  # will raise with clear error
    return _df_to_qx_dict(df, age_col, q_col)


def ek2001_2005(
    improvement_rate: float = 0.0, base_year: int = 2003
) -> MortalityTable:
    """Factory returning the EK 2001-2005 mortality table.

    ``improvement_rate=0.0`` (default) returns the raw period table,
    byte-identical to all earlier stages. A positive ``improvement_rate``
    turns it into a generational table projected from ``base_year`` (the
    2001-2005 observation midpoint, 2003): mortality is reduced by
    ``(1 - improvement_rate)`` per calendar year, applied whenever an
    annuity is valued with an explicit ``cal_year``. Typical Swiss
    occupational-pension improvement assumptions are ~1.0-1.5 % p.a.
    """
    return MortalityTable(
        q_male=_QX_MALE_EK_0105,
        q_female=_QX_FEMALE_EK_0105,
        improvement_rate=improvement_rate,
        base_year=base_year,
    )


# ---------------------------------------------------------------------------
# Hardcoded q_x constants (extracted from the EK 2001-2005 workbook).
# Ages 0..127. Values 0..1. See module docstring for source.
# ---------------------------------------------------------------------------

_QX_MALE_EK_0105: Dict[int, float] = {
    0: 0.005338,
    1: 0.0004031,
    2: 0.0003703,
    3: 0.0003436,
    4: 0.0003234,
    5: 0.0003107,
    6: 0.0003068,
    7: 0.0003132,
    8: 0.0003315,
    9: 0.0003629,
    10: 0.0004079,
    11: 0.0004655,
    12: 0.0005335,
    13: 0.0006086,
    14: 0.0006859,
    15: 0.0007597,
    16: 0.000824,
    17: 0.0008728,
    18: 0.0009012,
    19: 0.0009059,
    20: 0.0008864,
    21: 0.0008455,
    22: 0.0007889,
    23: 0.0007241,
    24: 0.0006586,
    25: 0.0005985,
    26: 0.0005478,
    27: 0.0005084,
    28: 0.0004809,
    29: 0.0004646,
    30: 0.0004583,
    31: 0.0004614,
    32: 0.0004741,
    33: 0.0004973,
    34: 0.0005314,
    35: 0.0005751,
    36: 0.0006255,
    37: 0.0006797,
    38: 0.0007361,
    39: 0.0007958,
    40: 0.0008618,
    41: 0.0009383,
    42: 0.0010291,
    43: 0.001136,
    44: 0.0012585,
    45: 0.0013949,
    46: 0.0015444,
    47: 0.0017099,
    48: 0.0018961,
    49: 0.0021079,
    50: 0.0023467,
    51: 0.0026094,
    52: 0.0028907,
    53: 0.0031872,
    54: 0.0035024,
    55: 0.0038455,
    56: 0.0042255,
    57: 0.0046439,
    58: 0.0050922,
    59: 0.0055558,
    60: 0.0060186,
    61: 0.0064713,
    62: 0.0069176,
    63: 0.0073751,
    64: 0.0078719,
    65: 0.008446,
    66: 0.009142,
    67: 0.0100047,
    68: 0.0110735,
    69: 0.0123802,
    70: 0.0139483,
    71: 0.0157944,
    72: 0.0179288,
    73: 0.0203568,
    74: 0.0230797,
    75: 0.026097,
    76: 0.0294081,
    77: 0.0330124,
    78: 0.0369106,
    79: 0.0411066,
    80: 0.0456104,
    81: 0.05044,
    82: 0.0556238,
    83: 0.061202,
    84: 0.0672275,
    85: 0.0737664,
    86: 0.0808975,
    87: 0.0887121,
    88: 0.0973126,
    89: 0.1068111,
    90: 0.1173277,
    91: 0.1289887,
    92: 0.1419241,
    93: 0.1562658,
    94: 0.1721449,
    95: 0.1896901,
    96: 0.2090256,
    97: 0.2302699,
    98: 0.2535346,
    99: 0.2789231,
    100: 0.3065304,
    101: 0.3364428,
    102: 0.3687373,
    103: 0.4034822,
    104: 0.4407366,
    105: 0.4805506,
    106: 0.5229664,
    107: 0.5680181,
    108: 0.6157327,
    109: 0.666131,
    110: 0.7192282,
    111: 0.7750348,
    112: 0.8335576,
    113: 0.8948004,
    114: 0.9587649,
    115: 0.9957139,
    116: 1.0,
    117: 1.0,
    118: 1.0,
    119: 1.0,
    120: 1.0,
    121: 1.0,
    122: 1.0,
    123: 1.0,
    124: 1.0,
    125: 1.0,
    126: 1.0,
    127: 1.0,
}

_QX_FEMALE_EK_0105: Dict[int, float] = {
    0: 0.004172,
    1: 0.0003501,
    2: 0.0003136,
    3: 0.000286,
    4: 0.0002668,
    5: 0.0002552,
    6: 0.0002502,
    7: 0.0002513,
    8: 0.0002577,
    9: 0.000269,
    10: 0.0002844,
    11: 0.0003035,
    12: 0.0003258,
    13: 0.0003503,
    14: 0.0003749,
    15: 0.0003973,
    16: 0.0004152,
    17: 0.0004266,
    18: 0.0004296,
    19: 0.0004233,
    20: 0.0004075,
    21: 0.0003831,
    22: 0.000352,
    23: 0.000317,
    24: 0.0002814,
    25: 0.0002487,
    26: 0.0002212,
    27: 0.0002004,
    28: 0.0001876,
    29: 0.0001834,
    30: 0.0001876,
    31: 0.0001991,
    32: 0.0002162,
    33: 0.0002367,
    34: 0.0002595,
    35: 0.0002856,
    36: 0.0003178,
    37: 0.0003602,
    38: 0.0004155,
    39: 0.0004829,
    40: 0.000559,
    41: 0.0006393,
    42: 0.0007203,
    43: 0.0008006,
    44: 0.0008805,
    45: 0.0009617,
    46: 0.0010457,
    47: 0.0011356,
    48: 0.0012356,
    49: 0.0013505,
    50: 0.0014852,
    51: 0.0016441,
    52: 0.0018298,
    53: 0.0020413,
    54: 0.0022741,
    55: 0.0025233,
    56: 0.0027834,
    57: 0.0030477,
    58: 0.0033078,
    59: 0.0035561,
    60: 0.0037887,
    61: 0.0040089,
    62: 0.0042285,
    63: 0.0044637,
    64: 0.00473,
    65: 0.0050398,
    66: 0.0054032,
    67: 0.0058288,
    68: 0.0063264,
    69: 0.0069089,
    70: 0.0075945,
    71: 0.0084105,
    72: 0.0093939,
    73: 0.0105904,
    74: 0.0120521,
    75: 0.0138361,
    76: 0.0160023,
    77: 0.0186114,
    78: 0.0217221,
    79: 0.0253889,
    80: 0.0296602,
    81: 0.0345794,
    82: 0.0401856,
    83: 0.0465153,
    84: 0.0536032,
    85: 0.0614846,
    86: 0.0701972,
    87: 0.0797827,
    88: 0.0902858,
    89: 0.1017538,
    90: 0.1142358,
    91: 0.1277825,
    92: 0.1424459,
    93: 0.1582789,
    94: 0.1753344,
    95: 0.1936646,
    96: 0.2133204,
    97: 0.2343511,
    98: 0.2568036,
    99: 0.2807222,
    100: 0.3061481,
    101: 0.3331191,
    102: 0.3616692,
    103: 0.391829,
    104: 0.4236254,
    105: 0.4570816,
    106: 0.4922176,
    107: 0.5290501,
    108: 0.5675928,
    109: 0.607857,
    110: 0.6498515,
    111: 0.6935831,
    112: 0.7390567,
    113: 0.7862758,
    114: 0.8352427,
    115: 0.8859584,
    116: 0.9384237,
    117: 0.9825469,
    118: 1.0,
    119: 1.0,
    120: 1.0,
    121: 1.0,
    122: 1.0,
    123: 1.0,
    124: 1.0,
    125: 1.0,
    126: 1.0,
    127: 1.0,
}
