"""
NTES Data Fetcher
=================

Fetches real-time railway data from NTES (National Train Enquiry System)
and other railway APIs. Handles authentication, rate limiting, and data validation.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class NTESDataFetcher:
    """
    Fetches railway data from NTES and other railway APIs.
    
    This class handles:
    - Authentication with railway APIs
    - Rate limiting and retry logic
    - Data validation and cleaning
    - Caching of frequently accessed data
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the NTES data fetcher.
        
        Args:
            config: Configuration object with API credentials and settings
        """
        self.config = config or Config()
        self.session = requests.Session()
        self.rate_limit_delay = self.config.get('ntes.rate_limit_delay', 1.0)
        self.max_retries = self.config.get('ntes.max_retries', 3)
        
        # Setup authentication
        self._setup_authentication()
        
        # Data cache
        self.cache = {}
        self.cache_ttl = self.config.get('ntes.cache_ttl', 300)  # 5 minutes
        
    def _setup_authentication(self):
        """Setup API authentication headers and tokens."""
        api_key = self.config.get('ntes.api_key')
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'RailTrafficAI/1.0'
            })
        else:
            logger.warning("No API key found for NTES. Some endpoints may not work.")
    
    def _make_request(self, url: str, params: Dict = None, retries: int = None) -> Dict:
        """
        Make HTTP request with retry logic and rate limiting.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            retries: Number of retries (uses default if None)
            
        Returns:
            JSON response data
            
        Raises:
            requests.RequestException: If all retries fail
        """
        if retries is None:
            retries = self.max_retries
            
        for attempt in range(retries + 1):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                if attempt == retries:
                    logger.error(f"Failed to fetch data from {url} after {retries} retries: {e}")
                    raise
                else:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
    
    def get_train_schedule(self, train_number: str, date: str = None) -> Dict:
        """
        Fetch train schedule for a specific train number.
        
        Args:
            train_number: Train number (e.g., '12345')
            date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Train schedule data
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        cache_key = f"schedule_{train_number}_{date}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        url = f"{self.config.get('ntes.base_url')}/api/train/schedule"
        params = {
            'train_number': train_number,
            'date': date
        }
        
        data = self._make_request(url, params)
        self._cache_data(cache_key, data)
        
        logger.info(f"Fetched schedule for train {train_number} on {date}")
        return data
    
    def get_live_train_status(self, train_number: str) -> Dict:
        """
        Fetch live status of a train.
        
        Args:
            train_number: Train number
            
        Returns:
            Live train status data
        """
        cache_key = f"live_{train_number}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        url = f"{self.config.get('ntes.base_url')}/api/train/live"
        params = {'train_number': train_number}
        
        data = self._make_request(url, params)
        self._cache_data(cache_key, data, ttl=60)  # Shorter TTL for live data
        
        logger.info(f"Fetched live status for train {train_number}")
        return data
    
    def get_station_schedule(self, station_code: str, date: str = None) -> Dict:
        """
        Fetch all trains arriving/departing from a station.
        
        Args:
            station_code: Station code (e.g., 'NDLS' for New Delhi)
            date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Station schedule data
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        cache_key = f"station_{station_code}_{date}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        url = f"{self.config.get('ntes.base_url')}/api/station/schedule"
        params = {
            'station_code': station_code,
            'date': date
        }
        
        data = self._make_request(url, params)
        self._cache_data(cache_key, data)
        
        logger.info(f"Fetched schedule for station {station_code} on {date}")
        return data
    
    def get_route_trains(self, from_station: str, to_station: str, date: str = None) -> Dict:
        """
        Fetch trains between two stations.
        
        Args:
            from_station: Source station code
            to_station: Destination station code
            date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Route trains data
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        cache_key = f"route_{from_station}_{to_station}_{date}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        url = f"{self.config.get('ntes.base_url')}/api/route/trains"
        params = {
            'from': from_station,
            'to': to_station,
            'date': date
        }
        
        data = self._make_request(url, params)
        self._cache_data(cache_key, data)
        
        logger.info(f"Fetched trains from {from_station} to {to_station} on {date}")
        return data
    
    def get_delay_info(self, train_number: str) -> Dict:
        """
        Fetch delay information for a train.
        
        Args:
            train_number: Train number
            
        Returns:
            Delay information data
        """
        cache_key = f"delay_{train_number}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        url = f"{self.config.get('ntes.base_url')}/api/train/delay"
        params = {'train_number': train_number}
        
        data = self._make_request(url, params)
        self._cache_data(cache_key, data, ttl=120)  # 2 minutes TTL for delay data
        
        logger.info(f"Fetched delay info for train {train_number}")
        return data
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and not expired."""
        if key not in self.cache:
            return False
        
        cache_entry = self.cache[key]
        return time.time() - cache_entry['timestamp'] < cache_entry['ttl']
    
    def _cache_data(self, key: str, data: Any, ttl: int = None):
        """Cache data with TTL."""
        if ttl is None:
            ttl = self.cache_ttl
            
        self.cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    def save_raw_data(self, data: Dict, filename: str, data_type: str = "train_data"):
        """
        Save raw data to disk for backup and analysis.
        
        Args:
            data: Data to save
            filename: Filename (without extension)
            data_type: Type of data for organizing in folders
        """
        raw_data_dir = Path(self.config.get('data.raw_dir', 'data/raw'))
        data_type_dir = raw_data_dir / data_type
        data_type_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = data_type_dir / f"{filename}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved raw data to {filepath}")
    
    def fetch_bulk_data(self, train_numbers: List[str], date_range: tuple = None) -> pd.DataFrame:
        """
        Fetch data for multiple trains in bulk.
        
        Args:
            train_numbers: List of train numbers
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            
        Returns:
            Combined DataFrame with all train data
        """
        if date_range is None:
            start_date = datetime.now().strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        else:
            start_date, end_date = date_range
        
        all_data = []
        
        for train_number in train_numbers:
            try:
                # Fetch schedule data
                schedule_data = self.get_train_schedule(train_number, start_date)
                
                # Fetch live status
                live_data = self.get_live_train_status(train_number)
                
                # Combine data
                combined_data = {
                    'train_number': train_number,
                    'date': start_date,
                    'schedule': schedule_data,
                    'live_status': live_data,
                    'fetched_at': datetime.now().isoformat()
                }
                
                all_data.append(combined_data)
                
                # Save individual raw data
                self.save_raw_data(combined_data, f"train_{train_number}", "schedules")
                
            except Exception as e:
                logger.error(f"Failed to fetch data for train {train_number}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Save bulk data
        bulk_filename = f"bulk_trains_{start_date}_{end_date}"
        self.save_raw_data(df.to_dict('records'), bulk_filename, "bulk_data")
        
        logger.info(f"Fetched bulk data for {len(all_data)} trains")
        return df
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() 
                            if time.time() - entry['timestamp'] >= entry['ttl'])
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'active_entries': total_entries - expired_entries,
            'cache_size_mb': sum(len(str(entry['data'])) for entry in self.cache.values()) / (1024 * 1024)
        }
