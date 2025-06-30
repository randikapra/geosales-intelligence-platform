import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, BatchNormalization,
                                   Bidirectional, Attention, Conv1D, MaxPooling1D,
                                   Flatten, Input, concatenate, MultiHeadAttention,
                                   LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

# Scientific Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import scipy.stats as stats

# Utilities
import joblib
import json
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class TimeSeriesConfig:
    """Configuration for time series models"""
    sequence_length: int = 30
    forecast_horizon: int = 7
    validation_split: float = 0.2
    confidence_level: float = 0.95
    seasonality_period: int = 7
    model_save_path: str = "models/timeseries/"

class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series models
    """
    
    def __init__(self, config: TimeSeriesConfig = None):
        self.config = config or TimeSeriesConfig()
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create model directory
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_column: str = 'value') -> None:
        """Fit the model to training data