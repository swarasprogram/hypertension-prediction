import pandas as pd
import numpy as np
from .make_target import bp_stage

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")  # Kaggle cardio dataset uses ';'
    return df

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Convert age from days to years
    df["age_years"] = (df["age"] / 365.25).round(1)

    # BMI
    df["height_m"] = df["height"] / 100.0
    df["bmi"] = df["weight"] / (df["height_m"] ** 2)

    # Create target
    df["hypertension_stage"] = df.apply(lambda r: bp_stage(r["ap_hi"], r["ap_lo"]), axis=1)

    # Basic sanity filters (removes crazy outliers)
    df = df[df["ap_hi"].between(70, 260)]
    df = df[df["ap_lo"].between(40, 160)]
    df = df[df["bmi"].between(10, 60)]

    # Drop columns we don’t want as predictors
    df = df.drop(columns=["id", "age", "height_m"], errors="ignore")

    return df