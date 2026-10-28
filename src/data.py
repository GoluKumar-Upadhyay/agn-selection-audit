"""Data loading, cleaning, feature construction and leakage-safe splitting.

Everything that touches the label or the split lives here so that every
experiment (tabular, image, fusion) uses exactly the same objects.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MAGS = ["u", "g", "r", "i", "z"]
COLOURS = ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i"]
MORPH = ["concentration"]
SHORTCUT = ["redshift", "abs_mag_r"]            # variables that encode the query design
ALL_FEATURES = MAGS + COLOURS + MORPH + SHORTCUT  # the 14 predictors in the manuscript

FEATURE_SETS = {
    "full_14":        ALL_FEATURES,
    "no_redshift":    [c for c in ALL_FEATURES if c != "redshift"],
    "no_z_no_Mr":     [c for c in ALL_FEATURES if c not in SHORTCUT],
    "colours_only":   COLOURS,
    "morphology_only": MORPH + ["petroR50_r"],
    "mags_only":      MAGS,
    "shortcut_only":  SHORTCUT,
    "redshift_only":  ["redshift"],
}


def load_clean(path: str) -> pd.DataFrame:
    """Load the SDSS table; drop SDSS sentinel values (-9999), duplicates, NaNs."""
    df = pd.read_csv(path)
    n0 = len(df)
    df = df.drop_duplicates("objid")
    ok = np.ones(len(df), bool)
    for c in MAGS:
        ok &= ((df[c] > 10) & (df[c] < 25)).values
    df = df[ok].dropna(subset=MAGS + ["petroR50_r", "petroR90_r", "redshift", "label"]).copy()
    df = add_features(df).reset_index(drop=True)
    df.attrs["n_dropped"] = n0 - len(df)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["u_g"] = df.u - df.g
    df["g_r"] = df.g - df.r
    df["r_i"] = df.r - df.i
    df["i_z"] = df.i - df.z
    df["u_r"] = df.u - df.r
    df["g_i"] = df.g - df.i
    df["concentration"] = df.petroR50_r / (df.petroR90_r + 1e-6)
    # crude distance-modulus proxy (no K-correction / extinction): identical to the
    # original manuscript so numbers are comparable.
    d_pc = df.redshift * 3e5 / 70.0 * 1e6
    df["abs_mag_r"] = df.r - 5 * np.log10(d_pc / 10.0 + 1e-6)
    return df


def split_70_15_15(df: pd.DataFrame, seed: int = 42):
    """Stratified 70/15/15 split -> positional (train, val, test) indices."""
    idx = np.arange(len(df))
    y = df["label"].values
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return tr, va, te
