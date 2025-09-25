"""
ETA predictor using gradient boosting with simple preprocessing.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from .features import FeatureEngineer


@dataclass
class ETAPredictorConfig:
    random_state: int = 42
    test_size: float = 0.2


class ETAPredictor:
    def __init__(self, config: Optional[ETAPredictorConfig] = None):
        self.config = config or ETAPredictorConfig()
        self.model = GradientBoostingRegressor(random_state=self.config.random_state)
        self.features = FeatureEngineer()
        self.feature_columns: List[str] = []

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        candidates = [
            'sched_hour', 'sched_dow', 'obs_hour', 'obs_dow',
            'weather_temperature', 'weather_humidity', 'weather_wind_speed', 'weather_visibility',
            'delay_clip_60', 'route_distance', 'avg_speed_kmh'
        ]
        cols = [c for c in candidates if c in df.columns]
        self.feature_columns = cols
        return df[cols]

    def fit(self, df: pd.DataFrame) -> float:
        data = self.features.build_eta_features(df)
        if 'target_delay_minutes' not in data:
            raise ValueError("Expected 'target_delay_minutes' column as target")
        X = self._select_features(data)
        y = data['target_delay_minutes']
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        data = self.features.build_eta_features(df)
        X = data[self.feature_columns] if self.feature_columns else self._select_features(data)
        return pd.Series(self.model.predict(X), index=df.index)


