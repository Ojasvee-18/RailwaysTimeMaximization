"""
Visualization Utilities
=======================

Visualization tools and utilities for railway data analysis and dashboard.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


class RailwayVisualizer:
    """
    Visualization utilities for railway data.
    
    This class provides methods for creating various types of visualizations
    including train schedules, delay analysis, route maps, and performance metrics.
    """
    
    def __init__(self, style: str = "ggplot", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#bcbd22',
            'dark': '#17becf'
        }
        
        # Set up plotting style (safe fallback if unavailable)
        try:
            plt.style.use(style)
        except Exception:
            fallback = "ggplot" if style != "ggplot" else "classic"
            try:
                plt.style.use(fallback)
            except Exception:
                pass
        try:
            sns.set_theme()
            sns.set_palette("husl")
        except Exception:
            pass
    
    def plot_train_schedule(self, df: pd.DataFrame, train_number: str = None, 
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot train schedule timeline.
        
        Args:
            df: DataFrame with train schedule data
            train_number: Specific train number to plot (if None, plots all)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating train schedule plot for {train_number or 'all trains'}")
        
        # Filter data if specific train requested
        if train_number:
            df = df[df['train_number'] == train_number]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create timeline plot
        for idx, row in df.iterrows():
            if pd.notna(row['departure_time']) and pd.notna(row['arrival_time']):
                # Calculate journey duration
                duration = (row['arrival_time'] - row['departure_time']).total_seconds() / 3600
                
                # Plot horizontal bar for journey
                ax.barh(
                    row['train_number'],
                    duration,
                    left=row['departure_time'],
                    height=0.6,
                    alpha=0.7,
                    color=self.colors['primary']
                )
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Train Number')
        ax.set_title(f'Train Schedule Timeline{" - " + train_number if train_number else ""}')
        
        # Format x-axis for time
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved train schedule plot to {save_path}")
        
        return fig
    
    def plot_delay_analysis(self, df: pd.DataFrame, group_by: str = 'train_number',
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot delay analysis charts.
        
        Args:
            df: DataFrame with delay data
            group_by: Column to group delays by
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating delay analysis plot grouped by {group_by}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Delay distribution
        axes[0, 0].hist(df['delay_minutes'], bins=30, alpha=0.7, color=self.colors['primary'])
        axes[0, 0].set_xlabel('Delay (minutes)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Delay Distribution')
        axes[0, 0].axvline(df['delay_minutes'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["delay_minutes"].mean():.1f} min')
        axes[0, 0].legend()
        
        # 2. Average delay by group
        delay_by_group = df.groupby(group_by)['delay_minutes'].mean().sort_values(ascending=False)
        delay_by_group.head(10).plot(kind='bar', ax=axes[0, 1], color=self.colors['secondary'])
        axes[0, 1].set_xlabel(group_by.title())
        axes[0, 1].set_ylabel('Average Delay (minutes)')
        axes[0, 1].set_title(f'Average Delay by {group_by.title()}')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Delay over time
        if 'created_at' in df.columns:
            df_time = df.copy()
            df_time['date'] = pd.to_datetime(df_time['created_at']).dt.date
            daily_delay = df_time.groupby('date')['delay_minutes'].mean()
            daily_delay.plot(ax=axes[1, 0], color=self.colors['success'])
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Average Delay (minutes)')
            axes[1, 0].set_title('Daily Average Delay Trend')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Delay categories
        delay_categories = pd.cut(df['delay_minutes'], 
                                bins=[-np.inf, 0, 15, 30, 60, np.inf],
                                labels=['On Time', 'Minor', 'Moderate', 'Major', 'Severe'])
        delay_categories.value_counts().plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
        axes[1, 1].set_title('Delay Categories')
        axes[1, 1].set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved delay analysis plot to {save_path}")
        
        return fig
    
    def plot_route_map(self, df: pd.DataFrame, route_id: str = None,
                      save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive route map using Plotly.
        
        Args:
            df: DataFrame with route data
            route_id: Specific route to plot
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        logger.info(f"Creating route map for {route_id or 'all routes'}")
        
        # Filter data if specific route requested
        if route_id:
            df = df[df['route_id'] == route_id]
        
        fig = go.Figure()
        
        # Add route lines
        for route in df['route_id'].unique():
            route_data = df[df['route_id'] == route]
            
            fig.add_trace(go.Scattermapbox(
                mode="lines+markers",
                lon=route_data['longitude'],
                lat=route_data['latitude'],
                text=route_data['station_name'],
                name=f"Route {route}",
                line=dict(width=3, color=self.colors['primary']),
                marker=dict(size=8, color=self.colors['secondary'])
            ))
        
        # Update layout
        fig.update_layout(
            title="Railway Route Map",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=20.5937, lon=78.9629),  # Center of India
                zoom=5
            ),
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved route map to {save_path}")
        
        return fig
    
    def plot_performance_metrics(self, metrics: Dict[str, float],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating performance metrics plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Accuracy metrics
        accuracy_metrics = {k: v for k, v in metrics.items() if 'accuracy' in k.lower()}
        if accuracy_metrics:
            axes[0, 0].bar(accuracy_metrics.keys(), accuracy_metrics.values(), 
                          color=self.colors['success'])
            axes[0, 0].set_title('Accuracy Metrics')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Loss metrics
        loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower()}
        if loss_metrics:
            axes[0, 1].bar(loss_metrics.keys(), loss_metrics.values(), 
                          color=self.colors['warning'])
            axes[0, 1].set_title('Loss Metrics')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Precision/Recall
        pr_metrics = {k: v for k, v in metrics.items() if any(x in k.lower() for x in ['precision', 'recall', 'f1'])}
        if pr_metrics:
            axes[1, 0].bar(pr_metrics.keys(), pr_metrics.values(), 
                          color=self.colors['info'])
            axes[1, 0].set_title('Precision/Recall Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Overall performance gauge
        overall_score = metrics.get('overall_score', 0)
        axes[1, 1].pie([overall_score, 1-overall_score], 
                      labels=['Performance', 'Remaining'],
                      colors=[self.colors['success'], self.colors['light']],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[1, 1].set_title(f'Overall Performance: {overall_score:.2f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance metrics plot to {save_path}")
        
        return fig
    
    def plot_heatmap(self, df: pd.DataFrame, x_col: str, y_col: str, value_col: str,
                    title: str = "Heatmap", save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap visualization.
        
        Args:
            df: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            value_col: Column for heatmap values
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating heatmap: {x_col} vs {y_col}")
        
        # Create pivot table
        pivot_data = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel(x_col.title())
        ax.set_ylabel(y_col.title())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix for numerical columns.
        
        Args:
            df: DataFrame with numerical data
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating correlation matrix plot")
        
        # Select only numerical columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f')
        
        ax.set_title('Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to {save_path}")
        
        return fig
    
    def plot_time_series(self, df: pd.DataFrame, time_col: str, value_col: str,
                        group_by: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series data.
        
        Args:
            df: DataFrame with time series data
            time_col: Column containing time data
            value_col: Column containing values to plot
            group_by: Column to group by (optional)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating time series plot: {value_col} over {time_col}")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if group_by:
            # Plot multiple series
            for group in df[group_by].unique():
                group_data = df[df[group_by] == group]
                ax.plot(group_data[time_col], group_data[value_col], 
                       label=str(group), alpha=0.7)
            ax.legend()
        else:
            # Plot single series
            ax.plot(df[time_col], df[value_col], color=self.colors['primary'])
        
        ax.set_xlabel(time_col.title())
        ax.set_ylabel(value_col.title())
        ax.set_title(f'{value_col.title()} Over Time')
        
        # Format x-axis for time
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved time series plot to {save_path}")
        
        return fig
    
    def create_dashboard_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create data for dashboard visualizations.
        
        Args:
            df: DataFrame with railway data
            
        Returns:
            Dictionary with dashboard data
        """
        logger.info("Creating dashboard data")
        
        dashboard_data = {
            'summary_stats': {
                'total_trains': df['train_number'].nunique() if 'train_number' in df.columns else 0,
                'total_delays': len(df[df.get('delay_minutes', 0) > 0]) if 'delay_minutes' in df.columns else 0,
                'avg_delay': df.get('delay_minutes', pd.Series([0])).mean(),
                'on_time_percentage': (len(df[df.get('delay_minutes', 0) == 0]) / len(df) * 100) if len(df) > 0 else 0
            },
            'top_delayed_trains': df.nlargest(10, 'delay_minutes')[['train_number', 'delay_minutes']].to_dict('records') if 'delay_minutes' in df.columns else [],
            'delay_trend': df.groupby(pd.to_datetime(df.get('created_at', pd.Series([datetime.now()]))).dt.date)['delay_minutes'].mean().to_dict() if 'delay_minutes' in df.columns else {},
            'station_performance': df.groupby('current_station')['delay_minutes'].mean().sort_values(ascending=False).head(10).to_dict() if 'current_station' in df.columns else {}
        }
        
        return dashboard_data
    
    def save_plot(self, fig: Union[plt.Figure, go.Figure], 
                  save_path: str, format: str = 'png'):
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib or Plotly figure
            save_path: Path to save the plot
            format: File format ('png', 'jpg', 'pdf', 'html' for Plotly)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, plt.Figure):
            fig.savefig(save_path, format=format, dpi=300, bbox_inches='tight')
        elif isinstance(fig, go.Figure):
            if format == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, format=format)
        
        logger.info(f"Saved plot to {save_path}")
    
    def close_all_figures(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')
        logger.info("Closed all matplotlib figures")
