"""
Synthetic Data Generator
========================

Generates realistic synthetic railway data for testing and demonstration
when real API data is not available.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """
    Generates synthetic railway data for testing and demonstration.
    
    This class creates realistic train schedules, delays, weather data,
    and other contextual information that mimics real railway operations.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.synthetic_data_dir = Path(self.config.get('data.synthetic_dir', 'data/synthetic'))
        self.synthetic_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Railway network configuration
        self.stations = {
            'NDLS': {'name': 'New Delhi', 'lat': 28.6448, 'lon': 77.2167, 'importance': 'major'},
            'AGC': {'name': 'Agra Cantt', 'lat': 27.1767, 'lon': 78.0081, 'importance': 'major'},
            'CNB': {'name': 'Kanpur Central', 'lat': 26.4499, 'lon': 80.3319, 'importance': 'major'},
            'LKO': {'name': 'Lucknow', 'lat': 26.8467, 'lon': 80.9462, 'importance': 'major'},
            'GKP': {'name': 'Gorakhpur', 'lat': 26.7606, 'lon': 83.3732, 'importance': 'minor'},
            'PNBE': {'name': 'Patna', 'lat': 25.5941, 'lon': 85.1376, 'importance': 'major'},
            'MUMBAI': {'name': 'Mumbai Central', 'lat': 19.0176, 'lon': 72.8562, 'importance': 'major'},
            'CHENNAI': {'name': 'Chennai Central', 'lat': 13.0827, 'lon': 80.2707, 'importance': 'major'},
            'BANGALORE': {'name': 'Bangalore City', 'lat': 12.9716, 'lon': 77.5946, 'importance': 'major'},
            'HYDERABAD': {'name': 'Hyderabad Deccan', 'lat': 17.3850, 'lon': 78.4867, 'importance': 'major'}
        }
        
        # Train types and their characteristics
        self.train_types = {
            'Rajdhani': {'speed': 120, 'delay_probability': 0.15, 'base_delay': 5},
            'Shatabdi': {'speed': 110, 'delay_probability': 0.10, 'base_delay': 2},
            'Duronto': {'speed': 100, 'delay_probability': 0.20, 'base_delay': 8},
            'Express': {'speed': 80, 'delay_probability': 0.25, 'base_delay': 10},
            'Passenger': {'speed': 60, 'delay_probability': 0.35, 'base_delay': 15},
            'Mail': {'speed': 70, 'delay_probability': 0.30, 'base_delay': 12}
        }
        
        # Weather conditions and their impact
        self.weather_conditions = {
            'Clear': {'delay_multiplier': 1.0, 'probability': 0.6},
            'Cloudy': {'delay_multiplier': 1.1, 'probability': 0.2},
            'Rain': {'delay_multiplier': 1.5, 'probability': 0.1},
            'Fog': {'delay_multiplier': 2.0, 'probability': 0.05},
            'Storm': {'delay_multiplier': 3.0, 'probability': 0.02},
            'Snow': {'delay_multiplier': 2.5, 'probability': 0.03}
        }
    
    def generate_train_schedules(self, num_trains: int = 50, days: int = 7) -> pd.DataFrame:
        """
        Generate synthetic train schedules.
        
        Args:
            num_trains: Number of trains to generate
            days: Number of days to generate schedules for
            
        Returns:
            DataFrame with train schedule data
        """
        logger.info(f"Generating {num_trains} train schedules for {days} days")
        
        schedules = []
        train_numbers = [f"{random.randint(12000, 19999)}" for _ in range(num_trains)]
        
        for train_num in train_numbers:
            # Select random train type
            train_type = random.choice(list(self.train_types.keys()))
            train_config = self.train_types[train_type]
            
            # Generate route (2-6 stations)
            route_length = random.randint(2, 6)
            route = random.sample(list(self.stations.keys()), route_length)
            
            # Generate schedule for each day
            for day in range(days):
                schedule_date = datetime.now() + timedelta(days=day)
                
                # Generate departure time (6 AM to 10 PM)
                departure_hour = random.randint(6, 22)
                departure_minute = random.choice([0, 15, 30, 45])
                departure_time = schedule_date.replace(hour=departure_hour, minute=departure_minute)
                
                # Calculate journey duration based on route and speed
                total_distance = self._calculate_route_distance(route)
                journey_duration = (total_distance / train_config['speed']) * 60  # minutes
                arrival_time = departure_time + timedelta(minutes=journey_duration)
                
                # Generate delay
                delay_minutes = self._generate_delay(train_config, schedule_date)
                
                schedule = {
                    'train_number': train_num,
                    'train_name': f"{train_type} Express",
                    'train_type': train_type,
                    'source_station': route[0],
                    'destination_station': route[-1],
                    'route': route,
                    'departure_time': departure_time,
                    'arrival_time': arrival_time,
                    'journey_duration': journey_duration,
                    'distance': total_distance,
                    'delay_minutes': delay_minutes,
                    'days_of_operation': self._generate_days_of_operation(),
                    'created_at': datetime.now()
                }
                
                schedules.append(schedule)
        
        df = pd.DataFrame(schedules)
        logger.info(f"Generated {len(df)} schedule records")
        return df
    
    def generate_live_status(self, num_trains: int = 20) -> pd.DataFrame:
        """
        Generate synthetic live train status data.
        
        Args:
            num_trains: Number of trains to generate status for
            
        Returns:
            DataFrame with live status data
        """
        logger.info(f"Generating live status for {num_trains} trains")
        
        statuses = []
        train_numbers = [f"{random.randint(12000, 19999)}" for _ in range(num_trains)]
        
        for train_num in train_numbers:
            # Select random train type
            train_type = random.choice(list(self.train_types.keys()))
            train_config = self.train_types[train_type]
            
            # Generate route
            route_length = random.randint(2, 6)
            route = random.sample(list(self.stations.keys()), route_length)
            
            # Current position (random station in route)
            current_station_idx = random.randint(0, len(route) - 1)
            current_station = route[current_station_idx]
            next_station = route[current_station_idx + 1] if current_station_idx < len(route) - 1 else None
            
            # Generate realistic status
            delay_minutes = self._generate_delay(train_config, datetime.now())
            
            if delay_minutes == 0:
                status = "ON_TIME"
            elif delay_minutes <= 5:
                status = "ON_TIME"
            elif delay_minutes <= 15:
                status = "DELAYED"
            else:
                status = "DELAYED"
            
            # Generate position data
            station_info = self.stations[current_station]
            position = {
                'latitude': station_info['lat'] + random.uniform(-0.01, 0.01),
                'longitude': station_info['lon'] + random.uniform(-0.01, 0.01),
                'speed': random.uniform(0, train_config['speed']),
                'direction': random.uniform(0, 360)
            }
            
            status_data = {
                'train_number': train_num,
                'train_name': f"{train_type} Express",
                'current_station': current_station,
                'next_station': next_station,
                'status': status,
                'delay_minutes': delay_minutes,
                'position': position,
                'last_updated': datetime.now(),
                'created_at': datetime.now()
            }
            
            statuses.append(status_data)
        
        df = pd.DataFrame(statuses)
        logger.info(f"Generated {len(df)} live status records")
        return df
    
    def generate_weather_data(self, days: int = 7) -> pd.DataFrame:
        """
        Generate synthetic weather data.
        
        Args:
            days: Number of days to generate weather for
            
        Returns:
            DataFrame with weather data
        """
        logger.info(f"Generating weather data for {days} days")
        
        weather_data = []
        
        for day in range(days):
            date = datetime.now() + timedelta(days=day)
            
            # Generate weather for each major station
            for station_code, station_info in self.stations.items():
                if station_info['importance'] == 'major':
                    # Select weather condition based on probabilities
                    rand = random.random()
                    cumulative_prob = 0
                    weather_condition = 'Clear'
                    
                    for condition, config in self.weather_conditions.items():
                        cumulative_prob += config['probability']
                        if rand <= cumulative_prob:
                            weather_condition = condition
                            break
                    
                    # Generate weather parameters
                    base_temp = 25 + random.uniform(-10, 15)  # Base temperature
                    if weather_condition == 'Rain':
                        base_temp -= 5
                    elif weather_condition == 'Snow':
                        base_temp -= 10
                    
                    weather = {
                        'date': date.date(),
                        'station_code': station_code,
                        'station_name': station_info['name'],
                        'temperature': round(base_temp, 1),
                        'humidity': random.randint(30, 90),
                        'pressure': random.randint(1000, 1020),
                        'wind_speed': random.uniform(0, 20),
                        'wind_direction': random.uniform(0, 360),
                        'weather_condition': weather_condition,
                        'visibility': random.randint(1000, 15000),
                        'cloudiness': random.randint(0, 100),
                        'created_at': datetime.now()
                    }
                    
                    weather_data.append(weather)
        
        df = pd.DataFrame(weather_data)
        logger.info(f"Generated {len(df)} weather records")
        return df
    
    def generate_holiday_data(self, year: int = None) -> Dict:
        """
        Generate synthetic holiday data.
        
        Args:
            year: Year to generate holidays for (defaults to current year)
            
        Returns:
            Dictionary with holiday data
        """
        if year is None:
            year = datetime.now().year
        
        logger.info(f"Generating holiday data for {year}")
        
        # Indian holidays
        holidays = {
            f"{year}-01-26": {"name": "Republic Day", "type": "National"},
            f"{year}-03-08": {"name": "Holi", "type": "Religious"},
            f"{year}-04-14": {"name": "Ambedkar Jayanti", "type": "National"},
            f"{year}-05-01": {"name": "Labour Day", "type": "National"},
            f"{year}-08-15": {"name": "Independence Day", "type": "National"},
            f"{year}-10-02": {"name": "Gandhi Jayanti", "type": "National"},
            f"{year}-10-22": {"name": "Dussehra", "type": "Religious"},
            f"{year}-11-14": {"name": "Diwali", "type": "Religious"},
            f"{year}-12-25": {"name": "Christmas", "type": "Religious"}
        }
        
        # Add some random regional holidays
        regional_holidays = [
            {"name": "Regional Festival", "type": "Regional"},
            {"name": "State Day", "type": "State"},
            {"name": "Local Holiday", "type": "Local"}
        ]
        
        for _ in range(5):
            random_date = datetime(year, random.randint(1, 12), random.randint(1, 28))
            date_str = random_date.strftime("%Y-%m-%d")
            if date_str not in holidays:
                holiday = random.choice(regional_holidays)
                holidays[date_str] = holiday
        
        logger.info(f"Generated {len(holidays)} holidays")
        return holidays
    
    def _calculate_route_distance(self, route: List[str]) -> float:
        """Calculate approximate distance for a route."""
        total_distance = 0
        for i in range(len(route) - 1):
            # Approximate distance between stations (simplified)
            total_distance += random.uniform(50, 200)
        return total_distance
    
    def _generate_delay(self, train_config: Dict, date: datetime) -> int:
        """Generate realistic delay based on train type and conditions."""
        base_delay = train_config['base_delay']
        delay_probability = train_config['delay_probability']
        
        # Check if delay should occur
        if random.random() > delay_probability:
            return 0
        
        # Generate delay amount
        delay = base_delay + random.randint(0, 20)
        
        # Weather impact
        weather_condition = self._get_weather_condition(date)
        weather_multiplier = self.weather_conditions[weather_condition]['delay_multiplier']
        delay = int(delay * weather_multiplier)
        
        # Peak hour impact
        if 7 <= date.hour <= 9 or 17 <= date.hour <= 19:
            delay = int(delay * 1.2)
        
        return max(0, delay)
    
    def _get_weather_condition(self, date: datetime) -> str:
        """Get weather condition for a date (simplified)."""
        rand = random.random()
        cumulative_prob = 0
        
        for condition, config in self.weather_conditions.items():
            cumulative_prob += config['probability']
            if rand <= cumulative_prob:
                return condition
        
        return 'Clear'
    
    def _generate_days_of_operation(self) -> List[str]:
        """Generate days of operation for a train."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Most trains run daily, some skip certain days
        if random.random() < 0.8:
            return days
        else:
            # Skip 1-2 random days
            skip_days = random.randint(1, 2)
            return random.sample(days, len(days) - skip_days)
    
    def save_synthetic_data(self, df: pd.DataFrame, filename: str, data_type: str = "synthetic"):
        """
        Save synthetic data to disk.
        
        Args:
            df: DataFrame to save
            filename: Filename (without extension)
            data_type: Type of data for organizing in folders
        """
        data_type_dir = self.synthetic_data_dir / data_type
        data_type_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = data_type_dir / f"{filename}_{timestamp}.parquet"
        
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved synthetic data to {filepath}")
    
    def generate_all_data(self, num_trains: int = 50, days: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Generate all types of synthetic data.
        
        Args:
            num_trains: Number of trains to generate
            days: Number of days to generate data for
            
        Returns:
            Dictionary with all generated datasets
        """
        logger.info(f"Generating complete synthetic dataset: {num_trains} trains, {days} days")
        
        datasets = {}
        
        # Generate train schedules
        datasets['schedules'] = self.generate_train_schedules(num_trains, days)
        self.save_synthetic_data(datasets['schedules'], 'train_schedules')
        
        # Generate live status
        datasets['live_status'] = self.generate_live_status(min(num_trains, 20))
        self.save_synthetic_data(datasets['live_status'], 'live_status')
        
        # Generate weather data
        datasets['weather'] = self.generate_weather_data(days)
        self.save_synthetic_data(datasets['weather'], 'weather_data')
        
        # Generate holiday data
        datasets['holidays'] = self.generate_holiday_data()
        
        # Save holiday data as JSON
        holiday_file = self.synthetic_data_dir / f"holidays_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(holiday_file, 'w') as f:
            json.dump(datasets['holidays'], f, indent=2, default=str)
        
        logger.info(f"Generated complete synthetic dataset with {len(datasets)} data types")
        return datasets





