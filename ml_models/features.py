"""
Feature engineering utilities shared across ML models.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    def build_eta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Basic temporal features
        if 'scheduled_arrival' in out:
            out['sched_hour'] = pd.to_datetime(out['scheduled_arrival']).dt.hour
            out['sched_dow'] = pd.to_datetime(out['scheduled_arrival']).dt.dayofweek
        if 'created_at' in out:
            out['obs_hour'] = pd.to_datetime(out['created_at']).dt.hour
            out['obs_dow'] = pd.to_datetime(out['created_at']).dt.dayofweek
        # Weather
        for col in ['temperature', 'humidity', 'wind_speed', 'visibility']:
            wcol = f'weather_{col}'
            if wcol in out:
                out[wcol] = pd.to_numeric(out[wcol], errors='coerce').fillna(out[wcol]).astype(float)
        # Delay prior
        if 'delay_minutes' in out:
            out['delay_clip_60'] = np.clip(out['delay_minutes'], 0, 60)
        return out


