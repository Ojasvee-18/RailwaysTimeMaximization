"""
Station dwell time predictor using random forest.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


@dataclass
class DwellTimeConfig:
    random_state: int = 42
    test_size: float = 0.2


class DwellTimeModel:
    def __init__(self, config: Optional[DwellTimeConfig] = None):
        self.config = config or DwellTimeConfig()
        self.model = RandomForestRegressor(random_state=self.config.random_state, n_estimators=200)
        self.feature_columns: List[str] = []

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        candidates = [
            'platform', 'passenger_volume', 'is_peak_hour', 'is_holiday', 'is_weekend',
            'weather_temperature', 'weather_visibility', 'train_length_cars', 'station_importance'
        ]
        # Convert categorical station_importance if present
        out = df.copy()
        if 'station_importance' in out:
            out['station_importance'] = out['station_importance'].astype('category').cat.codes
        cols = [c for c in candidates if c in out.columns]
        self.feature_columns = cols
        return out[cols]

    def fit(self, df: pd.DataFrame) -> float:
        if 'dwell_minutes' not in df:
            raise ValueError("Expected 'dwell_minutes' as target column")
        X = self._select_features(df)
        y = df['dwell_minutes']
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_columns] if self.feature_columns else self._select_features(df)
        return pd.Series(self.model.predict(X), index=df.index)


