"""
Data Preprocessor
=================

Handles data cleaning, validation, and feature engineering for railway data.
Converts raw API responses into structured datasets suitable for ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import re
import json

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocesses railway data for analysis and ML models.
    
    This class handles:
    - Data cleaning and validation
    - Feature engineering
    - Data transformation and normalization
    - Handling missing values and outliers
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object with preprocessing settings
        """
        self.config = config or Config()
        self.processed_data_dir = Path(self.config.get('data.processed_dir', 'data/processed'))
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Station codes mapping
        self.station_codes = self._load_station_codes()
        
        # Train type patterns
        self.train_type_patterns = {
            'express': r'^[0-9]{4,5}$',
            'passenger': r'^[0-9]{5,6}$',
            'superfast': r'^[0-9]{4,5}$',
            'mail': r'^[0-9]{4,5}$',
            'freight': r'^[0-9]{4,5}$'
        }
    
    def _load_station_codes(self) -> Dict[str, str]:
        """Load station codes mapping from configuration or file."""
        # This would typically load from a file or database
        # For now, using a sample mapping
        return {
            'NDLS': 'New Delhi',
            'MUMBAI': 'Mumbai Central',
            'CHENNAI': 'Chennai Central',
            'KOLKATA': 'Howrah',
            'BANGALORE': 'Bangalore City',
            'HYDERABAD': 'Hyderabad Deccan',
            'AHMEDABAD': 'Ahmedabad',
            'PUNE': 'Pune',
            'JAIPUR': 'Jaipur',
            'LUCKNOW': 'Lucknow'
        }
    
    def clean_train_schedule_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Clean and structure train schedule data.
        
        Args:
            raw_data: List of raw train schedule dictionaries
            
        Returns:
            Cleaned DataFrame with train schedule data
        """
        logger.info(f"Cleaning {len(raw_data)} train schedule records")
        
        cleaned_records = []
        
        for record in raw_data:
            try:
                # Extract basic train information
                train_info = {
                    'train_number': record.get('train_number', ''),
                    'train_name': record.get('train_name', ''),
                    'train_type': self._classify_train_type(record.get('train_number', '')),
                    'source_station': record.get('source_station', ''),
                    'destination_station': record.get('destination_station', ''),
                    'departure_time': self._parse_time(record.get('departure_time')),
                    'arrival_time': self._parse_time(record.get('arrival_time')),
                    'journey_duration': self._calculate_duration(
                        record.get('departure_time'), 
                        record.get('arrival_time')
                    ),
                    'days_of_operation': self._parse_days(record.get('days_of_operation', '')),
                    'distance': self._parse_distance(record.get('distance', 0)),
                    'created_at': datetime.now()
                }
                
                # Add route information if available
                if 'route' in record:
                    train_info.update(self._extract_route_info(record['route']))
                
                cleaned_records.append(train_info)
                
            except Exception as e:
                logger.warning(f"Failed to clean train schedule record: {e}")
                continue
        
        df = pd.DataFrame(cleaned_records)
        
        # Additional cleaning
        df = self._clean_dataframe(df)
        
        logger.info(f"Successfully cleaned {len(df)} train schedule records")
        return df
    
    def clean_live_status_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Clean and structure live train status data.
        
        Args:
            raw_data: List of raw live status dictionaries
            
        Returns:
            Cleaned DataFrame with live status data
        """
        logger.info(f"Cleaning {len(raw_data)} live status records")
        
        cleaned_records = []
        
        for record in raw_data:
            try:
                # Extract basic status information
                status_info = {
                    'train_number': record.get('train_number', ''),
                    'current_station': record.get('current_station', ''),
                    'last_station': record.get('last_station', ''),
                    'next_station': record.get('next_station', ''),
                    'current_status': record.get('status', ''),
                    'delay_minutes': self._parse_delay(record.get('delay', 0)),
                    'expected_arrival': self._parse_time(record.get('expected_arrival')),
                    'actual_arrival': self._parse_time(record.get('actual_arrival')),
                    'expected_departure': self._parse_time(record.get('expected_departure')),
                    'actual_departure': self._parse_time(record.get('actual_departure')),
                    'last_updated': self._parse_time(record.get('last_updated')),
                    'created_at': datetime.now()
                }
                
                # Add position information if available
                if 'position' in record:
                    status_info.update(self._extract_position_info(record['position']))
                
                cleaned_records.append(status_info)
                
            except Exception as e:
                logger.warning(f"Failed to clean live status record: {e}")
                continue
        
        df = pd.DataFrame(cleaned_records)
        
        # Additional cleaning
        df = self._clean_dataframe(df)
        
        logger.info(f"Successfully cleaned {len(df)} live status records")
        return df
    
    def clean_station_schedule_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Clean and structure station schedule data.
        
        Args:
            raw_data: List of raw station schedule dictionaries
            
        Returns:
            Cleaned DataFrame with station schedule data
        """
        logger.info(f"Cleaning {len(raw_data)} station schedule records")
        
        cleaned_records = []
        
        for record in raw_data:
            try:
                # Extract basic station information
                station_info = {
                    'station_code': record.get('station_code', ''),
                    'station_name': record.get('station_name', ''),
                    'train_number': record.get('train_number', ''),
                    'train_name': record.get('train_name', ''),
                    'arrival_time': self._parse_time(record.get('arrival_time')),
                    'departure_time': self._parse_time(record.get('departure_time')),
                    'platform': record.get('platform', ''),
                    'delay_minutes': self._parse_delay(record.get('delay', 0)),
                    'status': record.get('status', ''),
                    'created_at': datetime.now()
                }
                
                # Add route information
                if 'route' in record:
                    station_info.update(self._extract_route_info(record['route']))
                
                cleaned_records.append(station_info)
                
            except Exception as e:
                logger.warning(f"Failed to clean station schedule record: {e}")
                continue
        
        df = pd.DataFrame(cleaned_records)
        
        # Additional cleaning
        df = self._clean_dataframe(df)
        
        logger.info(f"Successfully cleaned {len(df)} station schedule records")
        return df
    
    def engineer_features(self, df: pd.DataFrame, data_type: str = "schedule") -> pd.DataFrame:
        """
        Engineer features for ML models.
        
        Args:
            df: Input DataFrame
            data_type: Type of data ('schedule', 'live_status', 'station_schedule')
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Engineering features for {data_type} data with {len(df)} records")
        
        df_features = df.copy()
        
        # Time-based features
        df_features = self._add_time_features(df_features)
        
        # Route-based features
        df_features = self._add_route_features(df_features)
        
        # Train-based features
        df_features = self._add_train_features(df_features)
        
        # Delay-based features
        if 'delay_minutes' in df_features.columns:
            df_features = self._add_delay_features(df_features)
        
        # Station-based features
        df_features = self._add_station_features(df_features)
        
        # Historical features (if historical data is available)
        df_features = self._add_historical_features(df_features, data_type)
        
        logger.info(f"Successfully engineered features. Final shape: {df_features.shape}")
        return df_features
    
    def _classify_train_type(self, train_number: str) -> str:
        """Classify train type based on train number pattern."""
        if not train_number:
            return 'unknown'
        
        for train_type, pattern in self.train_type_patterns.items():
            if re.match(pattern, train_number):
                return train_type
        
        return 'other'
    
    def _parse_time(self, time_str: Any) -> Optional[datetime]:
        """Parse time string to datetime object."""
        if not time_str or pd.isna(time_str):
            return None
        
        try:
            if isinstance(time_str, datetime):
                return time_str
            
            # Handle various time formats
            time_formats = [
                '%H:%M:%S',
                '%H:%M',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%d-%m-%Y %H:%M:%S',
                '%d-%m-%Y %H:%M'
            ]
            
            for fmt in time_formats:
                try:
                    return datetime.strptime(str(time_str), fmt)
                except ValueError:
                    continue
            
            # If no format matches, try pandas parsing
            return pd.to_datetime(time_str)
            
        except Exception as e:
            logger.warning(f"Failed to parse time '{time_str}': {e}")
            return None
    
    def _calculate_duration(self, departure_time: Any, arrival_time: Any) -> Optional[int]:
        """Calculate journey duration in minutes."""
        dep_time = self._parse_time(departure_time)
        arr_time = self._parse_time(arrival_time)
        
        if dep_time and arr_time:
            # Handle overnight journeys
            if arr_time < dep_time:
                arr_time += timedelta(days=1)
            return int((arr_time - dep_time).total_seconds() / 60)
        
        return None
    
    def _parse_days(self, days_str: str) -> List[str]:
        """Parse days of operation string."""
        if not days_str:
            return []
        
        day_mapping = {
            'MON': 'Monday', 'TUE': 'Tuesday', 'WED': 'Wednesday',
            'THU': 'Thursday', 'FRI': 'Friday', 'SAT': 'Saturday', 'SUN': 'Sunday'
        }
        
        days = []
        for day_code in days_str.split(','):
            day_code = day_code.strip().upper()
            if day_code in day_mapping:
                days.append(day_mapping[day_code])
        
        return days
    
    def _parse_distance(self, distance: Any) -> Optional[float]:
        """Parse distance value."""
        if pd.isna(distance) or distance == '':
            return None
        
        try:
            return float(distance)
        except (ValueError, TypeError):
            return None
    
    def _parse_delay(self, delay: Any) -> Optional[int]:
        """Parse delay value in minutes."""
        if pd.isna(delay) or delay == '':
            return 0
        
        try:
            return int(delay)
        except (ValueError, TypeError):
            return 0
    
    def _extract_route_info(self, route_data: Dict) -> Dict:
        """Extract route information from route data."""
        route_info = {}
        
        if isinstance(route_data, dict):
            route_info['total_stations'] = len(route_data.get('stations', []))
            route_info['route_distance'] = route_data.get('total_distance', 0)
            route_info['route_type'] = route_info.get('type', 'unknown')
        
        return route_info
    
    def _extract_position_info(self, position_data: Dict) -> Dict:
        """Extract position information from position data."""
        position_info = {}
        
        if isinstance(position_data, dict):
            position_info['latitude'] = position_data.get('lat')
            position_info['longitude'] = position_data.get('lon')
            position_info['speed'] = position_data.get('speed')
            position_info['direction'] = position_data.get('direction')
        
        return position_info
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply general cleaning operations to DataFrame."""
        # Remove duplicate rows (handle unhashable types like list/dict)
        df_for_dupes = df.copy()
        for col in df_for_dupes.columns:
            if df_for_dupes[col].dtype == 'object':
                has_unhashable = df_for_dupes[col].apply(lambda v: isinstance(v, (list, dict))).any()
                if has_unhashable:
                    df_for_dupes[col] = df_for_dupes[col].apply(
                        lambda v: tuple(v) if isinstance(v, list) else (
                            json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                        )
                    )
        df = df.loc[df_for_dupes.drop_duplicates().index]
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Standardize text fields
        df = self._standardize_text_fields(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # For numeric columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col].fillna(mode_value[0], inplace=True)
                else:
                    df[col].fillna('unknown', inplace=True)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _standardize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text fields."""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            if col not in ['train_number', 'station_code']:  # Keep these as-is
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        time_columns = ['departure_time', 'arrival_time', 'expected_arrival', 
                       'actual_arrival', 'expected_departure', 'actual_departure']
        
        for col in time_columns:
            if col in df.columns:
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_minute'] = df[col].dt.minute
                df[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6])
                df[f'{col}_is_peak_hour'] = df[col].dt.hour.isin([7, 8, 9, 17, 18, 19])
        
        return df
    
    def _add_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add route-based features."""
        if 'total_stations' in df.columns:
            df['stations_per_100km'] = df['total_stations'] / (df['route_distance'] / 100)
        
        if 'journey_duration' in df.columns and 'route_distance' in df.columns:
            df['avg_speed_kmh'] = df['route_distance'] / (df['journey_duration'] / 60)
        
        return df
    
    def _add_train_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add train-based features."""
        if 'train_type' in df.columns:
            df['is_express'] = df['train_type'] == 'express'
            df['is_passenger'] = df['train_type'] == 'passenger'
            df['is_superfast'] = df['train_type'] == 'superfast'
        
        return df
    
    def _add_delay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add delay-based features."""
        if 'delay_minutes' in df.columns:
            df['is_delayed'] = df['delay_minutes'] > 0
            df['delay_category'] = pd.cut(
                df['delay_minutes'], 
                bins=[-np.inf, 0, 15, 30, 60, np.inf],
                labels=['On Time', 'Minor Delay', 'Moderate Delay', 'Major Delay', 'Severe Delay']
            )
        
        return df
    
    def _add_station_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add station-based features."""
        if 'current_station' in df.columns:
            df['station_importance'] = df['current_station'].map(
                lambda x: 'major' if x in ['NDLS', 'MUMBAI', 'CHENNAI', 'KOLKATA'] else 'minor'
            )
        
        return df
    
    def _add_historical_features(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Add historical features (placeholder for future implementation)."""
        # This would typically load historical data and add features like:
        # - Average delay for this train/route
        # - Historical on-time performance
        # - Seasonal patterns
        # For now, just add placeholder columns
        df['historical_avg_delay'] = 0
        df['historical_on_time_rate'] = 0.8
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, data_type: str = "processed"):
        """
        Save processed data to disk.
        
        Args:
            df: Processed DataFrame
            filename: Filename (without extension)
            data_type: Type of data for organizing in folders
        """
        data_type_dir = self.processed_data_dir / data_type
        data_type_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = data_type_dir / f"{filename}_{timestamp}.parquet"
        
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filename: str, data_type: str = "processed") -> pd.DataFrame:
        """
        Load processed data from disk.
        
        Args:
            filename: Filename (without extension)
            data_type: Type of data folder
            
        Returns:
            Loaded DataFrame
        """
        data_type_dir = self.processed_data_dir / data_type
        filepath = data_type_dir / f"{filename}.parquet"
        
        if filepath.exists():
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded processed data from {filepath}")
            return df
        else:
            logger.error(f"Processed data file not found: {filepath}")
            return pd.DataFrame()
