"""
External Data Merger
====================

Merges railway data with external data sources like weather, holidays, events,
and other contextual information that can affect train operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import requests
import json

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class ExternalDataMerger:
    """
    Merges railway data with external data sources.
    
    This class handles:
    - Weather data integration
    - Holiday and event data
    - Economic indicators
    - Social media sentiment
    - Infrastructure status
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the external data merger.
        
        Args:
            config: Configuration object with external data settings
        """
        self.config = config or Config()
        self.external_data_dir = Path(self.config.get('data.external_dir', 'data/external'))
        self.external_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Weather API configuration
        self.weather_api_key = self.config.get('weather.api_key')
        self.weather_base_url = self.config.get('weather.base_url', 'https://api.openweathermap.org/data/2.5')
        
        # Holiday API configuration
        self.holiday_api_key = self.config.get('holidays.api_key')
        self.holiday_base_url = self.config.get('holidays.base_url', 'https://date.nager.at/api/v3')
        
        # Cache for external data
        self.cache = {}
        self.cache_ttl = self.config.get('external.cache_ttl', 3600)  # 1 hour
    
    def merge_weather_data(self, df: pd.DataFrame, date_column: str = 'created_at') -> pd.DataFrame:
        """
        Merge weather data with railway data.
        
        Args:
            df: Railway DataFrame
            date_column: Column containing date information
            
        Returns:
            DataFrame with weather data merged
        """
        logger.info(f"Merging weather data for {len(df)} records")
        
        # Get unique dates and locations
        df_with_weather = df.copy()
        
        # Extract dates and locations
        df_with_weather[date_column] = pd.to_datetime(df_with_weather[date_column])
        unique_dates = df_with_weather[date_column].dt.date.unique()
        
        # Get weather data for each date
        weather_data = self._fetch_weather_data(unique_dates)
        
        # Merge weather data
        df_with_weather = self._merge_weather_features(df_with_weather, weather_data, date_column)
        
        logger.info("Successfully merged weather data")
        return df_with_weather
    
    def merge_holiday_data(self, df: pd.DataFrame, date_column: str = 'created_at') -> pd.DataFrame:
        """
        Merge holiday and event data with railway data.
        
        Args:
            df: Railway DataFrame
            date_column: Column containing date information
            
        Returns:
            DataFrame with holiday data merged
        """
        logger.info(f"Merging holiday data for {len(df)} records")
        
        df_with_holidays = df.copy()
        df_with_holidays[date_column] = pd.to_datetime(df_with_holidays[date_column])
        
        # Get holiday data
        holiday_data = self._fetch_holiday_data()
        
        # Merge holiday data
        df_with_holidays = self._merge_holiday_features(df_with_holidays, holiday_data, date_column)
        
        logger.info("Successfully merged holiday data")
        return df_with_holidays
    
    def merge_infrastructure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge infrastructure status data with railway data.
        
        Args:
            df: Railway DataFrame
            
        Returns:
            DataFrame with infrastructure data merged
        """
        logger.info(f"Merging infrastructure data for {len(df)} records")
        
        df_with_infra = df.copy()
        
        # Get infrastructure data
        infra_data = self._fetch_infrastructure_data()
        
        # Merge infrastructure data
        df_with_infra = self._merge_infrastructure_features(df_with_infra, infra_data)
        
        logger.info("Successfully merged infrastructure data")
        return df_with_infra
    
    def merge_all_external_data(self, df: pd.DataFrame, date_column: str = 'created_at') -> pd.DataFrame:
        """
        Merge all external data sources with railway data.
        
        Args:
            df: Railway DataFrame
            date_column: Column containing date information
            
        Returns:
            DataFrame with all external data merged
        """
        logger.info(f"Merging all external data for {len(df)} records")
        
        # Start with weather data
        df_merged = self.merge_weather_data(df, date_column)
        
        # Add holiday data
        df_merged = self.merge_holiday_data(df_merged, date_column)
        
        # Add infrastructure data
        df_merged = self.merge_infrastructure_data(df_merged)
        
        # Add derived features
        df_merged = self._add_derived_features(df_merged)
        
        logger.info("Successfully merged all external data")
        return df_merged
    
    def _fetch_weather_data(self, dates: List[datetime.date]) -> Dict:
        """Fetch weather data for given dates."""
        weather_data = {}
        
        for date in dates:
            cache_key = f"weather_{date}"
            if self._is_cached(cache_key):
                weather_data[date] = self.cache[cache_key]['data']
                continue
            
            try:
                # Fetch weather data for major railway cities
                cities = ['New Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad']
                city_weather = {}
                
                for city in cities:
                    weather = self._fetch_city_weather(city, date)
                    if weather:
                        city_weather[city] = weather
                
                weather_data[date] = city_weather
                self._cache_data(cache_key, city_weather)
                
            except Exception as e:
                logger.warning(f"Failed to fetch weather data for {date}: {e}")
                weather_data[date] = {}
        
        return weather_data
    
    def _fetch_city_weather(self, city: str, date: datetime.date) -> Optional[Dict]:
        """Fetch weather data for a specific city and date."""
        if not self.weather_api_key:
            logger.warning("No weather API key provided, using mock data")
            return self._get_mock_weather_data()
        
        try:
            # Convert date to timestamp
            timestamp = int(datetime.combine(date, datetime.min.time()).timestamp())
            
            url = f"{self.weather_base_url}/weather"
            params = {
                'q': city,
                'appid': self.weather_api_key,
                'dt': timestamp
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'weather_condition': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'visibility': data.get('visibility', 10000),
                'cloudiness': data['clouds']['all']
            }
            
        except Exception as e:
            logger.warning(f"Failed to fetch weather for {city} on {date}: {e}")
            return self._get_mock_weather_data()
    
    def _get_mock_weather_data(self) -> Dict:
        """Generate mock weather data for testing."""
        return {
            'temperature': np.random.normal(25, 10),
            'humidity': np.random.uniform(30, 90),
            'pressure': np.random.uniform(1000, 1020),
            'wind_speed': np.random.uniform(0, 20),
            'wind_direction': np.random.uniform(0, 360),
            'weather_condition': np.random.choice(['Clear', 'Clouds', 'Rain', 'Snow', 'Fog']),
            'weather_description': 'Mock weather data',
            'visibility': np.random.uniform(5000, 15000),
            'cloudiness': np.random.uniform(0, 100)
        }
    
    def _fetch_holiday_data(self) -> Dict:
        """Fetch holiday data."""
        cache_key = "holidays"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Fetch Indian holidays
            current_year = datetime.now().year
            url = f"{self.holiday_base_url}/PublicHolidays/{current_year}/IN"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            holidays = response.json()
            
            # Process holidays data
            holiday_data = {}
            for holiday in holidays:
                date = datetime.strptime(holiday['date'], '%Y-%m-%d').date()
                holiday_data[date] = {
                    'name': holiday['name'],
                    'type': holiday.get('type', 'Public'),
                    'is_holiday': True
                }
            
            self._cache_data(cache_key, holiday_data)
            return holiday_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch holiday data: {e}")
            return {}
    
    def _fetch_infrastructure_data(self) -> Dict:
        """Fetch infrastructure status data."""
        cache_key = "infrastructure"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # This would typically fetch from railway infrastructure APIs
            # For now, using mock data
            infra_data = {
                'track_maintenance': {
                    'status': 'normal',
                    'affected_sections': [],
                    'maintenance_schedule': {}
                },
                'signal_system': {
                    'status': 'operational',
                    'outages': [],
                    'upgrades': []
                },
                'power_system': {
                    'status': 'normal',
                    'outages': [],
                    'load': 'normal'
                },
                'communication': {
                    'status': 'operational',
                    'outages': [],
                    'coverage': 'full'
                }
            }
            
            self._cache_data(cache_key, infra_data)
            return infra_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch infrastructure data: {e}")
            return {}
    
    def _merge_weather_features(self, df: pd.DataFrame, weather_data: Dict, date_column: str) -> pd.DataFrame:
        """Merge weather features into DataFrame."""
        df_merged = df.copy()
        
        # Add weather columns
        weather_columns = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
            'weather_condition', 'weather_description', 'visibility', 'cloudiness'
        ]
        
        for col in weather_columns:
            df_merged[f'weather_{col}'] = None
        
        # Fill weather data
        for idx, row in df_merged.iterrows():
            date = row[date_column].date()
            if date in weather_data:
                # For now, use average weather across cities
                # In practice, you'd match by station location
                city_weather = weather_data[date]
                if city_weather:
                    avg_weather = self._calculate_average_weather(city_weather)
                    for col in weather_columns:
                        df_merged.at[idx, f'weather_{col}'] = avg_weather.get(col)
        
        return df_merged
    
    def _merge_holiday_features(self, df: pd.DataFrame, holiday_data: Dict, date_column: str) -> pd.DataFrame:
        """Merge holiday features into DataFrame."""
        df_merged = df.copy()
        
        # Add holiday columns
        df_merged['is_holiday'] = False
        df_merged['holiday_name'] = None
        df_merged['holiday_type'] = None
        df_merged['is_weekend'] = df_merged[date_column].dt.dayofweek.isin([5, 6])
        df_merged['is_peak_season'] = False
        
        # Fill holiday data
        for idx, row in df_merged.iterrows():
            date = row[date_column].date()
            if date in holiday_data:
                holiday_info = holiday_data[date]
                df_merged.at[idx, 'is_holiday'] = holiday_info.get('is_holiday', False)
                df_merged.at[idx, 'holiday_name'] = holiday_info.get('name')
                df_merged.at[idx, 'holiday_type'] = holiday_info.get('type')
        
        # Determine peak season (festive periods)
        df_merged['is_peak_season'] = self._determine_peak_season(df_merged[date_column])
        
        return df_merged
    
    def _merge_infrastructure_features(self, df: pd.DataFrame, infra_data: Dict) -> pd.DataFrame:
        """Merge infrastructure features into DataFrame."""
        df_merged = df.copy()
        
        # Add infrastructure columns
        df_merged['track_maintenance_status'] = infra_data.get('track_maintenance', {}).get('status', 'normal')
        df_merged['signal_system_status'] = infra_data.get('signal_system', {}).get('status', 'operational')
        df_merged['power_system_status'] = infra_data.get('power_system', {}).get('status', 'normal')
        df_merged['communication_status'] = infra_data.get('communication', {}).get('status', 'operational')
        
        # Calculate overall infrastructure health score
        df_merged['infrastructure_health_score'] = self._calculate_infrastructure_health(infra_data)
        
        return df_merged
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from external data."""
        df_derived = df.copy()
        
        # Weather impact features
        if 'weather_temperature' in df_derived.columns:
            df_derived['is_extreme_temperature'] = (
                (df_derived['weather_temperature'] < 5) | 
                (df_derived['weather_temperature'] > 40)
            )
        
        if 'weather_visibility' in df_derived.columns:
            df_derived['is_low_visibility'] = df_derived['weather_visibility'] < 1000
        
        if 'weather_wind_speed' in df_derived.columns:
            df_derived['is_high_wind'] = df_derived['weather_wind_speed'] > 15
        
        # Combined weather severity score
        weather_conditions = [
            'is_extreme_temperature', 'is_low_visibility', 'is_high_wind'
        ]
        df_derived['weather_severity_score'] = df_derived[weather_conditions].sum(axis=1)
        
        # Demand prediction features
        df_derived['expected_demand_multiplier'] = self._calculate_demand_multiplier(df_derived)
        
        # Operational complexity score
        df_derived['operational_complexity_score'] = self._calculate_operational_complexity(df_derived)
        
        return df_derived
    
    def _calculate_average_weather(self, city_weather: Dict) -> Dict:
        """Calculate average weather across cities."""
        if not city_weather:
            return {}
        
        numeric_fields = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'visibility', 'cloudiness']
        avg_weather = {}
        
        for field in numeric_fields:
            values = [weather.get(field) for weather in city_weather.values() if weather.get(field) is not None]
            if values:
                avg_weather[field] = np.mean(values)
        
        # For categorical fields, use the most common value
        categorical_fields = ['weather_condition', 'weather_description']
        for field in categorical_fields:
            values = [weather.get(field) for weather in city_weather.values() if weather.get(field)]
            if values:
                avg_weather[field] = max(set(values), key=values.count)
        
        return avg_weather
    
    def _determine_peak_season(self, dates: pd.Series) -> pd.Series:
        """Determine if dates fall in peak travel season."""
        # Define peak seasons (festive periods, summer vacation, etc.)
        peak_months = [3, 4, 5, 10, 11, 12]  # March-May, Oct-Dec
        return dates.dt.month.isin(peak_months)
    
    def _calculate_infrastructure_health(self, infra_data: Dict) -> float:
        """Calculate overall infrastructure health score (0-1)."""
        status_mapping = {
            'normal': 1.0,
            'operational': 1.0,
            'degraded': 0.7,
            'maintenance': 0.5,
            'outage': 0.0
        }
        
        scores = []
        for system, data in infra_data.items():
            if isinstance(data, dict) and 'status' in data:
                scores.append(status_mapping.get(data['status'], 0.5))
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_demand_multiplier(self, df: pd.DataFrame) -> pd.Series:
        """Calculate expected demand multiplier based on external factors."""
        multiplier = pd.Series(1.0, index=df.index)
        
        # Holiday effect
        if 'is_holiday' in df.columns:
            multiplier += df['is_holiday'].astype(int) * 0.3
        
        # Weekend effect
        if 'is_weekend' in df.columns:
            multiplier += df['is_weekend'].astype(int) * 0.2
        
        # Peak season effect
        if 'is_peak_season' in df.columns:
            multiplier += df['is_peak_season'].astype(int) * 0.4
        
        # Weather effect (bad weather reduces demand)
        if 'weather_severity_score' in df.columns:
            multiplier -= df['weather_severity_score'] * 0.1
        
        return multiplier.clip(0.1, 3.0)  # Cap between 0.1 and 3.0
    
    def _calculate_operational_complexity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate operational complexity score based on external factors."""
        complexity = pd.Series(0.0, index=df.index)
        
        # Weather complexity
        if 'weather_severity_score' in df.columns:
            complexity += df['weather_severity_score'] * 0.3
        
        # Infrastructure complexity
        if 'infrastructure_health_score' in df.columns:
            complexity += (1 - df['infrastructure_health_score']) * 0.4
        
        # Holiday complexity
        if 'is_holiday' in df.columns:
            complexity += df['is_holiday'].astype(int) * 0.2
        
        # Peak season complexity
        if 'is_peak_season' in df.columns:
            complexity += df['is_peak_season'].astype(int) * 0.1
        
        return complexity.clip(0.0, 1.0)
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and not expired."""
        if key not in self.cache:
            return False
        
        cache_entry = self.cache[key]
        return datetime.now().timestamp() - cache_entry['timestamp'] < cache_entry['ttl']
    
    def _cache_data(self, key: str, data: Any, ttl: int = None):
        """Cache data with TTL."""
        if ttl is None:
            ttl = self.cache_ttl
            
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now().timestamp(),
            'ttl': ttl
        }
    
    def save_external_data(self, data: Dict, filename: str, data_type: str = "external"):
        """
        Save external data to disk.
        
        Args:
            data: External data dictionary
            filename: Filename (without extension)
            data_type: Type of data for organizing in folders
        """
        data_type_dir = self.external_data_dir / data_type
        data_type_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = data_type_dir / f"{filename}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved external data to {filepath}")
    
    def clear_cache(self):
        """Clear all cached external data."""
        self.cache.clear()
        logger.info("External data cache cleared")
