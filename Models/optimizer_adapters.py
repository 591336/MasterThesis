from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from Models.optimizer import SailingTimePredictor, TurnaroundPredictor


def _derive_season(month: float | int | None) -> str:
    if pd.isna(month):
        return "unknown"
    m = int(month)
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "unknown"


class SailingTimeModelAdapter(SailingTimePredictor):
    """Wraps a trained sailing-time sklearn pipeline.

    Contract for the optimiser: `predict()` returns sailing time in **days** for each row.
    """

    def __init__(self, artifact_path: Path, metadata: dict):
        payload = joblib.load(artifact_path)
        self.pipeline = payload["pipeline"]
        self.feature_columns: Sequence[str] = payload["feature_columns"]
        self.metadata = metadata
        self.use_log_target = bool(payload.get("use_log_target", False))
        self.expected_cats = metadata.get("categorical_features", []) + metadata.get(
            "derived_categorical_features", []
        )
        self.expected_nums = metadata.get("numeric_features", [])
        self.target_col = metadata.get("target_column", "HOURS_PER_NM")
        self.log_target_col = metadata.get("log_target_column", "LOG_HOURS_PER_NM")

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()

        # Map optimiser column variants to training feature names.
        if "DAYS_AT_SEA" not in working.columns and "DAYS_TOTAL_AT_SEA" in working.columns:
            working["DAYS_AT_SEA"] = working["DAYS_TOTAL_AT_SEA"]
        if "CANAL_COST" not in working.columns and "VOYAGE_CANAL_COST" in working.columns:
            working["CANAL_COST"] = working["VOYAGE_CANAL_COST"]

        # Derive MONTH_NO and SEASON from laycan/start if missing.
        if "MONTH_NO" not in working.columns:
            if "LAYCAN_START_UTC" in working.columns:
                working["MONTH_NO"] = pd.to_datetime(
                    working["LAYCAN_START_UTC"], errors="coerce"
                ).dt.month
            elif "ESTIMATED_VOYAGE_START_DATE" in working.columns:
                working["MONTH_NO"] = pd.to_datetime(
                    working["ESTIMATED_VOYAGE_START_DATE"], errors="coerce"
                ).dt.month
            else:
                working["MONTH_NO"] = pd.NA

        if "SEASON" not in working.columns:
            working["SEASON"] = working["MONTH_NO"].map(_derive_season)

        # Fill MILES_TOTAL if missing.
        if "MILES_TOTAL" not in working.columns:
            miles_ballast = pd.to_numeric(working.get("MILES_BALLAST"), errors="coerce")
            miles_loaded = pd.to_numeric(working.get("MILES_LOADED"), errors="coerce")
            working["MILES_TOTAL"] = miles_ballast.add(miles_loaded, fill_value=np.nan)
            if "MILES_DIRECT" in working.columns:
                direct = pd.to_numeric(working["MILES_DIRECT"], errors="coerce")
                working["MILES_TOTAL"] = working["MILES_TOTAL"].fillna(direct)

        # BALLAST_FRAC if missing.
        if "BALLAST_FRAC" not in working.columns:
            mt = pd.to_numeric(working.get("MILES_TOTAL"), errors="coerce")
            mb = pd.to_numeric(working.get("MILES_BALLAST"), errors="coerce")
            working["BALLAST_FRAC"] = mb / mt

        # Derived keys
        if "VESSEL_TYPE_CANAL_KEY" not in working.columns:
            vt = pd.to_numeric(working.get("VESSEL_TYPE_ID"), errors="coerce")
            canal = pd.to_numeric(working.get("HAS_CANAL_PASSAGE"), errors="coerce").fillna(0)
            working["VESSEL_TYPE_CANAL_KEY"] = vt.astype("Int64").astype(str) + "__" + canal.astype(int).astype(str)
        if "VESSEL_TYPE_MONTH_KEY" not in working.columns:
            vt = pd.to_numeric(working.get("VESSEL_TYPE_ID"), errors="coerce")
            month = pd.to_numeric(working.get("MONTH_NO"), errors="coerce").fillna(-1)
            working["VESSEL_TYPE_MONTH_KEY"] = vt.astype("Int64").astype(str) + "__" + month.astype(int).astype(str)

        # Ensure all expected columns exist.
        for col in set(self.expected_cats + self.expected_nums):
            if col not in working.columns:
                working[col] = pd.NA

        # Keep only model columns
        missing = [c for c in self.feature_columns if c not in working.columns]
        for col in missing:
            working[col] = pd.NA

        return working[self.feature_columns]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features = self._prepare(df)

        raw = np.asarray(self.pipeline.predict(features), dtype=float)
        hours_per_nm = np.expm1(raw) if self.use_log_target else raw

        # Convert model output (hours per nautical mile) to sailing duration (days).
        miles_total = pd.to_numeric(df.get("MILES_TOTAL"), errors="coerce")
        if miles_total is None or miles_total.isna().all():
            miles_ballast = pd.to_numeric(df.get("MILES_BALLAST"), errors="coerce")
            miles_loaded = pd.to_numeric(df.get("MILES_LOADED"), errors="coerce")
            miles_total = miles_ballast.add(miles_loaded, fill_value=np.nan)
        if miles_total is None or miles_total.isna().all():
            miles_total = pd.to_numeric(df.get("MILES_DIRECT"), errors="coerce")

        if miles_total is None:
            miles_total = pd.Series([np.nan] * len(df), index=df.index)

        days = (hours_per_nm * miles_total.to_numpy(dtype=float)) / 24.0
        days = np.where(np.isfinite(days) & (days >= 0.0), days, np.nan)

        fallback = self.fallback(df)
        return np.where(np.isnan(days), fallback, days).astype(float)

    def fallback(self, df: pd.DataFrame) -> np.ndarray:
        miles_total = pd.to_numeric(df.get("MILES_TOTAL"), errors="coerce")
        if miles_total is None or miles_total.isna().all():
            miles_ballast = pd.to_numeric(df.get("MILES_BALLAST"), errors="coerce")
            miles_loaded = pd.to_numeric(df.get("MILES_LOADED"), errors="coerce")
            miles_total = miles_ballast.add(miles_loaded, fill_value=np.nan)
        if miles_total is None or miles_total.isna().all():
            miles_total = pd.to_numeric(df.get("MILES_DIRECT"), errors="coerce")
        if miles_total is None:
            miles_total = pd.Series([np.nan] * len(df), index=df.index)

        hours = miles_total / 12.0
        days = hours / 24.0
        return days.fillna(1.0).to_numpy(dtype=float)


class TurnaroundModelAdapter(TurnaroundPredictor):
    """Adapter for port turnaround model; falls back to constant."""

    def __init__(self, artifact_path: Path | None = None, metadata: dict | None = None, default_days: float = 1.0):
        self.default_days = default_days
        if artifact_path and metadata:
            payload = joblib.load(artifact_path)
            self.pipeline = payload["pipeline"]
            self.feature_columns: Sequence[str] = payload["feature_columns"]
            self.meta = metadata
            self.use_log_target = bool(payload.get("use_log_target", False))
            self.port_features = set(metadata.get("categorical_features", []) + metadata.get("derived_categorical_features", []))
            self.num_features = set(metadata.get("numeric_features", []) + metadata.get("derived_numeric_features", []))
            self.target_col = metadata.get("target_column", "DAYS_IN_PORT")
        else:
            self.pipeline = None
            self.feature_columns = ()
            self.meta = {}
            self.use_log_target = False
            self.port_features = set()
            self.num_features = set()
            self.target_col = "DAYS_IN_PORT"

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        # Map incoming columns to model features
        # Expect PORT_ID, TERMINAL_ID, IS_BALLAST, MONTH_NO, and derived keys if present.
        if "PORT_ID" not in working.columns and "ORIGIN_PORT_ID" in working.columns:
            working["PORT_ID"] = working["ORIGIN_PORT_ID"]
        for col in ("PORT_ID", "TERMINAL_ID", "IS_BALLAST", "VESSEL_TYPE_ID"):
            if col not in working.columns:
                working[col] = pd.NA

        if "MONTH_NO" not in working.columns:
            if "LAYCAN_START_UTC" in working.columns:
                working["MONTH_NO"] = pd.to_datetime(working["LAYCAN_START_UTC"], errors="coerce").dt.month
            elif "ESTIMATED_VOYAGE_START_DATE" in working.columns:
                working["MONTH_NO"] = pd.to_datetime(working["ESTIMATED_VOYAGE_START_DATE"], errors="coerce").dt.month
            else:
                working["MONTH_NO"] = pd.NA

        # Derived keys if needed
        if "PORT_TERMINAL_KEY" in self.port_features and "PORT_TERMINAL_KEY" not in working.columns:
            pt = working.get("PORT_ID")
            term = working.get("TERMINAL_ID")
            working["PORT_TERMINAL_KEY"] = pt.astype("Int64").astype(str) + "__" + term.astype("Int64").astype(str)
        if "PORT_IS_BALLAST_KEY" in self.port_features and "PORT_IS_BALLAST_KEY" not in working.columns:
            pt = working.get("PORT_ID")
            ballast = pd.to_numeric(working.get("IS_BALLAST"), errors="coerce").fillna(0).astype(int)
            working["PORT_IS_BALLAST_KEY"] = pt.astype("Int64").astype(str) + "__" + ballast.astype(str)

        # Ensure all model columns exist
        for col in self.feature_columns:
            if col not in working.columns:
                working[col] = pd.NA
        return working[self.feature_columns]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            return np.full(len(df), self.default_days, dtype=float)
        features = self._prepare(df)
        raw = np.asarray(self.pipeline.predict(features), dtype=float)
        preds = np.expm1(raw) if self.use_log_target else raw
        return np.asarray(preds, dtype=float)

    def fallback(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.default_days, dtype=float)
