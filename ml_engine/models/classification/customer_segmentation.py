import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Clustering Libraries
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# RFM Analysis
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Utilities
import joblib
import json
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class SegmentationConfig:
    """Configuration for customer segmentation"""
    n_clusters: int = 5
    random_state: int = 42
    model_save_path: str = "models/segmentation/"
    min_cluster_size: int = 10
    eps: float = 0.5  # For DBSCAN
    rfm_weights: Dict[str, float] = None

class CustomerSegmentation:
    """
    Advanced customer segmentation using multiple clustering techniques and RFM analysis
    """
    
    def __init__(self, config: SegmentationConfig = None):
        self.config = config or SegmentationConfig()
        if self.config.rfm_weights is None:
            self.config.rfm_weights = {'recency': 0.3, 'frequency': 0.35, 'monetary': 0.35}
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.segment_profiles = {}
        self.rfm_data = None
        self.is_fitted = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create model directory
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
    
    def prepare_customer_features(self, sales_data: pd.DataFrame, customer_data: pd.DataFrame = None,
                                gps_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare comprehensive customer features for segmentation
        """
        self.logger.info("Preparing customer features...")
        
        # Ensure proper datetime format
        if 'Date' in sales_data.columns:
            sales_data['Date'] = pd.to_datetime(sales_data['Date'])
        elif 'CreationDate' in sales_data.columns:
            sales_data['Date'] = pd.to_datetime(sales_data['CreationDate'])
        
        # Calculate RFM metrics
        rfm_features = self._calculate_rfm_metrics(sales_data)
        
        # Calculate transactional features
        transactional_features = self._calculate_transactional_features(sales_data)
        
        # Calculate temporal features
        temporal_features = self._calculate_temporal_features(sales_data)
        
        # Calculate product diversity features
        product_features = self._calculate_product_features(sales_data)
        
        # Combine all features
        customer_features = rfm_features.merge(transactional_features, left_index=True, right_index=True, how='outer')
        customer_features = customer_features.merge(temporal_features, left_index=True, right_index=True, how='outer')
        customer_features = customer_features.merge(product_features, left_index=True, right_index=True, how='outer')
        
        # Add geographical features if available
        if customer_data is not None and 'Latitude' in customer_data.columns:
            geo_features = self._calculate_geographical_features(customer_data, gps_data)
            customer_features = customer_features.merge(geo_features, left_index=True, right_index=True, how='left')
        
        # Fill missing values
        customer_features = customer_features.fillna(customer_features.median())
        
        self.feature_names = list(customer_features.columns)
        self.logger.info(f"Created {len(self.feature_names)} features for {len(customer_features)} customers")
        
        return customer_features
    
    def _calculate_rfm_metrics(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) metrics"""
        
        # Determine customer and value columns
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data.columns else 'CustomerID'
        value_col = 'FinalValue' if 'FinalValue' in sales_data.columns else 'Total'
        
        # Current date for recency calculation
        current_date = sales_data['Date'].max()
        
        # Calculate RFM metrics
        rfm = sales_data.groupby(customer_col).agg({
            'Date': lambda x: (current_date - x.max()).days,  # Recency
            value_col: ['count', 'sum', 'mean', 'std']  # Frequency and Monetary
        }).round(2)
        
        # Flatten column names
        rfm.columns = ['recency', 'frequency', 'monetary_total', 'monetary_avg', 'monetary_std']
        rfm['monetary_std'] = rfm['monetary_std'].fillna(0)
        
        # Add RFM scores (1-5 scale)
        rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['monetary_score'] = pd.qcut(rfm['monetary_total'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert scores to numeric
        rfm['recency_score'] = pd.to_numeric(rfm['recency_score'])
        rfm['frequency_score'] = pd.to_numeric(rfm['frequency_score'])
        rfm['monetary_score'] = pd.to_numeric(rfm['monetary_score'])
        
        # Calculate RFM combined score
        rfm['rfm_score'] = (
            rfm['recency_score'] * self.config.rfm_weights['recency'] +
            rfm['frequency_score'] * self.config.rfm_weights['frequency'] +
            rfm['monetary_score'] * self.config.rfm_weights['monetary']
        )
        
        # Customer lifecycle stage
        rfm['customer_lifecycle'] = rfm.apply(self._determine_customer_lifecycle, axis=1)
        
        # Add RFM segments
        rfm['rfm_segment'] = rfm.apply(self._assign_rfm_segment, axis=1)
        
        self.rfm_data = rfm
        return rfm
    
    def _determine_customer_lifecycle(self, row) -> str:
        """Determine customer lifecycle stage based on RFM scores"""
        R, F, M = row['recency_score'], row['frequency_score'], row['monetary_score']
        
        if R >= 4 and F >= 4 and M >= 4:
            return 'Champions'
        elif R >= 3 and F >= 3 and M >= 3:
            return 'Loyal Customers'
        elif R >= 4 and F <= 2:
            return 'New Customers'
        elif R >= 3 and F >= 3 and M <= 2:
            return 'Potential Loyalists'
        elif R <= 2 and F >= 3 and M >= 3:
            return 'At Risk'
        elif R <= 2 and F <= 2 and M >= 3:
            return 'Cannot Lose Them'
        elif R <= 2 and F <= 2 and M <= 2:
            return 'Lost'
        else:
            return 'Others'
    
    def _assign_rfm_segment(self, row) -> str:
        """Assign RFM segment based on scores"""
        score = row['rfm_score']
        
        if score >= 4.5:
            return 'Premium'
        elif score >= 3.5:
            return 'Gold'
        elif score >= 2.5:
            return 'Silver'
        elif score >= 1.5:
            return 'Bronze'
        else:
            return 'Basic'
    
    def _calculate_transactional_features(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate transactional behavior features"""
        
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data.columns else 'CustomerID'
        value_col = 'FinalValue' if 'FinalValue' in sales_data.columns else 'Total'
        
        features = sales_data.groupby(customer_col).agg({
            value_col: [
                'count', 'sum', 'mean', 'std', 'min', 'max',
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 75),
                lambda x: x.std() / x.mean() if x.mean() != 0 else 0  # Coefficient of variation
            ],
            'Date': [
                lambda x: (x.max() - x.min()).days,  # Customer tenure
                lambda x: len(x.unique()),  # Number of unique transaction days
            ]
        }).round(2)
        
        # Flatten column names
        features.columns = [
            'transaction_count', 'total_spent', 'avg_transaction_value', 'transaction_std',
            'min_transaction', 'max_transaction', 'transaction_q25', 'transaction_q75',
            'transaction_cv', 'customer_tenure_days', 'active_days'
        ]
        
        # Calculate additional metrics
        features['avg_days_between_purchases'] = features['customer_tenure_days'] / (features['transaction_count'] - 1)
        features['avg_days_between_purchases'] = features['avg_days_between_purchases'].fillna(0)
        
        features['purchase_frequency'] = features['active_days'] / features['customer_tenure_days']
        features['purchase_frequency'] = features['purchase_frequency'].fillna(0)
        
        # Transaction concentration (Gini coefficient approximation)
        features['transaction_concentration'] = features['transaction_std'] / features['avg_transaction_value']
        features['transaction_concentration'] = features['transaction_concentration'].fillna(0)
        
        return features
    
    def _calculate_temporal_features(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate temporal behavior features"""
        
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data.columns else 'CustomerID'
        
        # Add temporal components
        sales_data['hour'] = sales_data['Date'].dt.hour
        sales_data['day_of_week'] = sales_data['Date'].dt.dayofweek
        sales_data['month'] = sales_data['Date'].dt.month
        sales_data['quarter'] = sales_data['Date'].dt.quarter
        sales_data['is_weekend'] = sales_data['day_of_week'].isin([5, 6]).astype(int)
        
        temporal_features = sales_data.groupby(customer_col).agg({
            'hour': ['mean', 'std'],
            'day_of_week': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
            'month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 1,
            'quarter': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 1,
            'is_weekend': 'mean'
        }).round(2)
        
        # Flatten column names
        temporal_features.columns = [
            'avg_purchase_hour', 'purchase_hour_std', 'preferred_day_of_week',
            'preferred_month', 'preferred_quarter', 'weekend_purchase_ratio'
        ]
        
        # Add seasonality indicators
        sales_monthly = sales_data.groupby([customer_col, 'month']).size().unstack(fill_value=0)
        for month in range(1, 13):
            if month in sales_monthly.columns:
                temporal_features[f'purchases_month_{month}'] = sales_monthly[month]
            else:
                temporal_features[f'purchases_month_{month}'] = 0
        
        return temporal_features
    
    def _calculate_product_features(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate product diversity and preference features"""
        
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data.columns else 'CustomerID'
        
        # If product information is available
        if 'ProductID' in sales_data.columns or 'Code' in sales_data.columns:
            product_col = 'ProductID' if 'ProductID' in sales_data.columns else 'Code'
            
            product_features = sales_data.groupby(customer_col).agg({
                product_col: ['nunique', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown']
            })
            
            product_features.columns = ['product_diversity', 'most_purchased_product']
            
            # Calculate product concentration (how concentrated purchases are on few products)
            product_dist = sales_data.groupby([customer_col, product_col]).size().reset_index(name='count')
            product_concentration = product_dist.groupby(customer_col)['count'].apply(
                lambda x: (x ** 2).sum() / (x.sum() ** 2)  # Herfindahl index
            ).to_frame('product_concentration')
            
            product_features = product_features.merge(product_concentration, left_index=True, right_index=True)
        else:
            # Create dummy features if product info not available
            unique_customers = sales_data[customer_col].unique()
            product_features = pd.DataFrame(index=unique_customers)
            product_features['product_diversity'] = 1
            product_features['most_purchased_product'] = 'Unknown'
            product_features['product_concentration'] = 1.0
        
        return product_features
    
    def _calculate_geographical_features(self, customer_data: pd.DataFrame, 
                                       gps_data: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate geographical behavior features"""
        
        geo_features = pd.DataFrame(index=customer_data.index)
        
        if 'Latitude' in customer_data.columns and 'Longitude' in customer_data.columns:
            geo_features['latitude'] = customer_data['Latitude']
            geo_features['longitude'] = customer_data['Longitude']
            
            # Calculate distance from city center (assuming Colombo as reference)
            colombo_lat, colombo_lon = 6.9271, 79.8612
            geo_features['distance_from_center'] = np.sqrt(
                (geo_features['latitude'] - colombo_lat) ** 2 + 
                (geo_features['longitude'] - colombo_lon) ** 2
            ) * 111  # Approximate km conversion
            
            # Regional clustering (simple grid-based)
            geo_features['region_lat'] = (geo_features['latitude'] * 10).astype(int)
            geo_features['region_lon'] = (geo_features['longitude'] * 10).astype(int)
        
        return geo_features
    
    def fit_clustering_models(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit multiple clustering models and select the best one
        """
        self.logger.info("Fitting clustering models...")
        
        # Prepare data
        X = features.select_dtypes(include=[np.number]).copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Scale features
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        models = {
            'kmeans': KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_state),
            'gaussian_mixture': GaussianMixture(n_components=self.config.n_clusters, random_state=self.config.random_state),
            'hierarchical': AgglomerativeClustering(n_clusters=self.config.n_clusters),
            'dbscan': DBSCAN(eps=self.config.eps, min_samples=self.config.min_cluster_size)
        }
        
        results = {}
        best_score = -1
        best_model = None
        best_scaler = None
        best_labels = None
        
        for scaler_name, scaler in scalers.items():
            X_scaled = scaler.fit_transform(X)
            
            for model_name, model in models.items():
                try:
                    # Fit model
                    if model_name == 'dbscan':
                        labels = model.fit_predict(X_scaled)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        if n_clusters < 2:  # Skip if too few clusters
                            continue
                    else:
                        model.fit(X_scaled)
                        labels = model.predict(X_scaled) if hasattr(model, 'predict') else model.labels_
                        n_clusters = len(set(labels))
                    
                    # Calculate metrics
                    if len(set(labels)) > 1:
                        silhouette = silhouette_score(X_scaled, labels)
                        calinski = calinski_harabasz_score(X_scaled, labels)
                        davies_bouldin = davies_bouldin_score(X_scaled, labels)
                        
                        # Combined score (higher is better)
                        combined_score = silhouette + (calinski / 1000) - davies_bouldin
                        
                        results[f"{model_name}_{scaler_name}"] = {
                            'model': model,
                            'scaler': scaler,
                            'labels': labels,
                            'n_clusters': n_clusters,
                            'silhouette_score': silhouette,
                            'calinski_harabasz_score': calinski,
                            'davies_bouldin_score': davies_bouldin,
                            'combined_score': combined_score
                        }
                        
                        # Update best model
                        if combined_score > best_score:
                            best_score = combined_score
                            best_model = model_name
                            best_scaler = scaler_name
                            best_labels = labels
                            
                        self.logger.info(f"{model_name}_{scaler_name}: Silhouette={silhouette:.3f}, "
                                       f"Calinski-Harabasz={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}")
                        
                except Exception as e:
                    self.logger.warning(f"Error fitting {model_name}_{scaler_name}: {str(e)}")
                    continue
        
        # Store best model
        if best_model:
            best_key = f"{best_model}_{best_scaler}"
            self.models['best'] = results[best_key]['model']
            self.scalers['best'] = results[best_key]['scaler']
            self.labels_ = best_labels
            self.is_fitted = True
            
            self.logger.info(f"Best model: {best_key} with combined score: {best_score:.3f}")
            
            # Create segment profiles
            self._create_segment_profiles(features, best_labels)
        
        return results
    
    def _create_segment_profiles(self, features: pd.DataFrame, labels: np.ndarray):
        """Create detailed profiles for each segment"""
        
        # Add cluster labels to features
        features_with_clusters = features.copy()
        features_with_clusters['cluster'] = labels
        
        # Calculate profiles for each cluster
        profiles = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_data = features_with_clusters[features_with_clusters['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features_with_clusters) * 100,
                'characteristics': {}
            }
            
            # Statistical summary for numerical features
            numerical_cols = features.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                profile['characteristics'][col] = {
                    'mean': cluster_data[col].mean(),
                    'median': cluster_data[col].median(),
                    'std': cluster_data[col].std(),
                    'min': cluster_data[col].min(),
                    'max': cluster_data[col].max()
                }
            
            # Add business insights based on RFM data
            if self.rfm_data is not None:
                cluster_customers = cluster_data.index
                rfm_subset = self.rfm_data.loc[self.rfm_data.index.isin(cluster_customers)]
                
                profile['business_insights'] = {
                    'avg_recency': rfm_subset['recency'].mean(),
                    'avg_frequency': rfm_subset['frequency'].mean(),
                    'avg_monetary': rfm_subset['monetary_total'].mean(),
                    'dominant_lifecycle_stage': rfm_subset['customer_lifecycle'].mode().iloc[0],
                    'dominant_rfm_segment': rfm_subset['rfm_segment'].mode().iloc[0]
                }
            
            profiles[f'cluster_{cluster_id}'] = profile
        
        self.segment_profiles = profiles
        
        # Generate segment names based on characteristics
        self._generate_segment_names()
    
    def _generate_segment_names(self):
        """Generate meaningful names for segments based on their characteristics"""
        
        segment_names = {}
        
        for cluster_key, profile in self.segment_profiles.items():
            if 'business_insights' in profile:
                insights = profile['business_insights']
                lifecycle = insights['dominant_lifecycle_stage']
                rfm_segment = insights['dominant_rfm_segment']
                
                # Generate name based on dominant characteristics
                if lifecycle == 'Champions':
                    name = "VIP Champions"
                elif lifecycle == 'Loyal Customers':
                    name = "Loyal Advocates"
                elif lifecycle == 'At Risk':
                    name = "At-Risk Customers"
                elif lifecycle == 'New Customers':
                    name = "New Prospects"
                elif lifecycle == 'Lost':
                    name = "Win-Back Targets"
                else:
                    name = f"{rfm_segment} Segment"
            else:
                # Fallback naming based on transaction patterns
                chars = profile['characteristics']
                if 'avg_transaction_value' in chars:
                    avg_value = chars['avg_transaction_value']['mean']
                    frequency = chars['frequency']['mean']
                    
                    if avg_value > chars['avg_transaction_value']['median'] * 1.5:
                        if frequency > chars['frequency']['median'] * 1.5:
                            name = "High-Value Frequent"
                        else:
                            name = "High-Value Occasional"
                    else:
                        if frequency > chars['frequency']['median'] * 1.5:
                            name = "Low-Value Frequent"
                        else:
                            name = "Low-Value Occasional"
                else:
                    name = cluster_key.replace('_', ' ').title()
            
            segment_names[cluster_key] = name
        
        self.segment_names = segment_names
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for new data"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        X = features.select_dtypes(include=[np.number]).copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Scale features
        X_scaled = self.scalers['best'].transform(X)
        
        # Predict
        if hasattr(self.models['best'], 'predict'):
            return self.models['best'].predict(X_scaled)
        else:
            # For models without predict method (like DBSCAN), use fit_predict
            return self.models['best'].fit_predict(X_scaled)
    
    def generate_visualizations(self, features: pd.DataFrame) -> Dict[str, go.Figure]:
        """Generate comprehensive visualizations for segmentation analysis"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating visualizations")
        
        figures = {}
        
        # 1. PCA visualization
        figures['pca_clusters'] = self._create_pca_visualization(features)
        
        # 2. RFM analysis visualization
        if self.rfm_data is not None:
            figures['rfm_analysis'] = self._create_rfm_visualization()
        
        # 3. Segment size distribution
        figures['segment_distribution'] = self._create_segment_distribution()
        
        # 4. Feature importance heatmap
        figures['feature_heatmap'] = self._create_feature_heatmap(features)
        
        # 5. Customer lifecycle distribution
        if self.rfm_data is not None:
            figures['lifecycle_distribution'] = self._create_lifecycle_distribution()
        
        return figures
    
    def _create_pca_visualization(self, features: pd.DataFrame) -> go.Figure:
        """Create PCA visualization of clusters"""
        
        # Prepare data
        X = features.select_dtypes(include=[np.number]).copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        X_scaled = self.scalers['best'].transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=self.config.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': [self.segment_names.get(f'cluster_{label}', f'Cluster {label}') 
                       for label in self.labels_]
        })
        
        # Create scatter plot
        fig = px.scatter(
            plot_df, x='PC1', y='PC2', color='Cluster',
            title=f'Customer Segments - PCA Visualization<br><sub>PC1 explains {pca.explained_variance_ratio_[0]:.1%} variance, PC2 explains {pca.explained_variance_ratio_[1]:.1%} variance</sub>',
            labels={'PC1': f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%})',
                   'PC2': f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%})'}
        )
        
        fig.update_layout(
            width=800, height=600,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
        )
        
        return fig
    
    def _create_rfm_visualization(self) -> go.Figure:
        """Create RFM analysis visualization"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RFM Score Distribution', 'Customer Lifecycle Stages',
                          'Recency vs Frequency', 'Frequency vs Monetary'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # RFM Score Distribution
        fig.add_trace(
            go.Histogram(x=self.rfm_data['rfm_score'], name='RFM Score'),
            row=1, col=1
        )
        
        # Customer Lifecycle Stages
        lifecycle_counts = self.rfm_data['customer_lifecycle'].value_counts()
        fig.add_trace(
            go.Bar(x=lifecycle_counts.index, y=lifecycle_counts.values, name='Lifecycle'),
            row=1, col=2
        )
        
        # Recency vs Frequency scatter
        fig.add_trace(
            go.Scatter(
                x=self.rfm_data['recency'], y=self.rfm_data['frequency'],
                mode='markers', name='Customers',
                text=self.rfm_data['customer_lifecycle'],
                hovertemplate='Recency: %{x}<br>Frequency: %{y}<br>Lifecycle: %{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Frequency vs Monetary scatter
        fig.add_trace(
            go.Scatter(
                x=self.rfm_data['frequency'], y=self.rfm_data['monetary_total'],
                mode='markers', name='Customers',
                text=self.rfm_data['customer_lifecycle'],
                hovertemplate='Frequency: %{x}<br>Monetary: %{y}<br>Lifecycle: %{text}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="RFM Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def _create_segment_distribution(self) -> go.Figure:
        """Create segment size distribution chart"""
        
        # Calculate segment sizes
        segment_sizes = []
        segment_labels = []
        
        for cluster_key, profile in self.segment_profiles.items():
            segment_labels.append(self.segment_names.get(cluster_key, cluster_key))
            segment_sizes.append(profile['size'])
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=segment_labels,
            values=segment_sizes,
            hole=0.3
        )])
        
        fig.update_layout(
            title_text="Customer Segment Distribution",
            annotations=[dict(text='Segments', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig
    
    def _create_feature_heatmap(self, features: pd.DataFrame) -> go.Figure:
        """Create feature importance heatmap across segments"""
        
        # Calculate mean values for each feature by cluster
        features_with_clusters = features.copy()
        features_with_clusters['cluster'] = self.labels_
        
        # Get numerical features only
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        
        # Calculate cluster means
        cluster_means = features_with_clusters.groupby('cluster')[numerical_cols].mean()
        
        # Normalize for better visualization
        cluster_means_normalized = (cluster_means - cluster_means.mean()) / cluster_means.std()
        
        # Create segment names for y-axis
        y_labels = [self.segment_names.get(f'cluster_{idx}', f'Cluster {idx}') 
                   for idx in cluster_means_normalized.index]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cluster_means_normalized.values,
            x=cluster_means_normalized.columns,
            y=y_labels,
            colorscale='RdBu',
            zmid=0,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Characteristics by Segment<br><sub>Normalized values (blue=below average, red=above average)</sub>',
            xaxis_title='Features',
            yaxis_title='Segments',
            height=max(400, len(y_labels) * 50)
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def _create_lifecycle_distribution(self) -> go.Figure:
        """Create customer lifecycle distribution by segment"""
        
        # Merge RFM data with cluster labels
        rfm_with_clusters = self.rfm_data.copy()
        rfm_with_clusters['cluster'] = self.labels_
        
        # Create crosstab
        lifecycle_by_cluster = pd.crosstab(
            rfm_with_clusters['cluster'],
            rfm_with_clusters['customer_lifecycle'],
            normalize='index'
        ) * 100
        
        # Create segment names
        segment_labels = [self.segment_names.get(f'cluster_{idx}', f'Cluster {idx}') 
                         for idx in lifecycle_by_cluster.index]
        
        # Create stacked bar chart
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i, lifecycle in enumerate(lifecycle_by_cluster.columns):
            fig.add_trace(go.Bar(
                name=lifecycle,
                x=segment_labels,
                y=lifecycle_by_cluster[lifecycle],
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='Customer Lifecycle Distribution by Segment',
            xaxis_title='Segments',
            yaxis_title='Percentage',
            barmode='stack',
            height=500
        )
        
        return fig
    
    def save_model(self, filepath: str = None):
        """Save the trained model and associated artifacts"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        if filepath is None:
            filepath = Path(self.config.model_save_path) / "customer_segmentation_model.pkl"
        else:
            filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        model_artifacts = {
            'model': self.models['best'],
            'scaler': self.scalers['best'],
            'feature_names': self.feature_names,
            'segment_profiles': self.segment_profiles,
            'segment_names': self.segment_names,
            'config': self.config,
            'labels': self.labels_,
            'rfm_data': self.rfm_data
        }
        
        joblib.dump(model_artifacts, filepath)
        self.logger.info(f"Model saved to {filepath}")
        
        # Save segment profiles as JSON for easy access
        profiles_path = filepath.parent / "segment_profiles.json"
        with open(profiles_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_profiles = {}
            for k, v in self.segment_profiles.items():
                json_profiles[k] = self._convert_to_json_serializable(v)
            json.dump(json_profiles, f, indent=2)
        
        return filepath
    
    def load_model(self, filepath: str):
        """Load a previously saved model"""
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model artifacts
        model_artifacts = joblib.load(filepath)
        
        self.models['best'] = model_artifacts['model']
        self.scalers['best'] = model_artifacts['scaler']
        self.feature_names = model_artifacts['feature_names']
        self.segment_profiles = model_artifacts['segment_profiles']
        self.segment_names = model_artifacts['segment_names']
        self.config = model_artifacts['config']
        self.labels_ = model_artifacts['labels']
        self.rfm_data = model_artifacts['rfm_data']
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_segment_recommendations(self, segment_name: str = None) -> Dict[str, Any]:
        """Get marketing and business recommendations for segments"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting recommendations")
        
        recommendations = {}
        
        # If specific segment requested
        if segment_name:
            segments_to_process = [k for k, v in self.segment_names.items() if v == segment_name]
        else:
            segments_to_process = list(self.segment_profiles.keys())
        
        for cluster_key in segments_to_process:
            if cluster_key not in self.segment_profiles:
                continue
                
            profile = self.segment_profiles[cluster_key]
            segment_name = self.segment_names.get(cluster_key, cluster_key)
            
            # Generate recommendations based on segment characteristics
            recs = {
                'segment_name': segment_name,
                'size': profile['size'],
                'percentage': profile['percentage'],
                'marketing_strategy': self._generate_marketing_strategy(profile),
                'retention_strategy': self._generate_retention_strategy(profile),
                'growth_opportunities': self._generate_growth_opportunities(profile),
                'key_metrics_to_track': self._generate_key_metrics(profile)
            }
            
            recommendations[cluster_key] = recs
        
        return recommendations
    
    def _generate_marketing_strategy(self, profile: Dict) -> List[str]:
        """Generate marketing strategy recommendations"""
        
        strategies = []
        
        if 'business_insights' in profile:
            insights = profile['business_insights']
            lifecycle = insights['dominant_lifecycle_stage']
            
            if lifecycle == 'Champions':
                strategies.extend([
                    "Leverage as brand ambassadors and referral sources",
                    "Offer exclusive premium products and early access to new launches",
                    "Create VIP customer programs with special benefits",
                    "Use testimonials and case studies in marketing materials"
                ])
            elif lifecycle == 'Loyal Customers':
                strategies.extend([
                    "Implement loyalty reward programs",
                    "Cross-sell and upsell complementary products",
                    "Send personalized product recommendations",
                    "Maintain regular communication through newsletters"
                ])
            elif lifecycle == 'At Risk':
                strategies.extend([
                    "Launch win-back campaigns with special offers",
                    "Conduct surveys to understand dissatisfaction",
                    "Provide personalized customer service outreach",
                    "Offer flexible payment terms or discounts"
                ])
            elif lifecycle == 'New Customers':
                strategies.extend([
                    "Implement onboarding campaigns to increase engagement",
                    "Provide educational content about products",
                    "Offer welcome bonuses or first-purchase discounts",
                    "Focus on building trust through excellent service"
                ])
            elif lifecycle == 'Lost':
                strategies.extend([
                    "Deploy aggressive win-back campaigns",
                    "Offer significant discounts or incentives",
                    "Re-engage through different communication channels",
                    "Analyze exit reasons and address pain points"
                ])
        
        # Add general strategies based on transaction patterns
        if 'characteristics' in profile:
            chars = profile['characteristics']
            if 'avg_transaction_value' in chars:
                avg_value = chars['avg_transaction_value']['mean']
                frequency = chars['frequency']['mean']
                
                if avg_value > 50000:  # High value threshold
                    strategies.append("Focus on relationship marketing and personal account management")
                if frequency > 10:  # High frequency threshold
                    strategies.append("Implement bulk purchase incentives and volume discounts")
        
        return strategies
    
    def _generate_retention_strategy(self, profile: Dict) -> List[str]:
        """Generate customer retention strategy recommendations"""
        
        strategies = []
        
        if 'business_insights' in profile:
            insights = profile['business_insights']
            recency = insights['avg_recency']
            frequency = insights['avg_frequency']
            
            if recency > 90:  # Haven't purchased in 90+ days
                strategies.extend([
                    "Implement proactive outreach before churn occurs",
                    "Send personalized 'we miss you' campaigns",
                    "Offer limited-time return incentives"
                ])
            
            if frequency < 3:  # Low frequency customers
                strategies.extend([
                    "Develop engagement campaigns to increase purchase frequency",
                    "Create subscription or auto-replenishment programs",
                    "Send regular product updates and promotions"
                ])
            else:
                strategies.extend([
                    "Maintain consistent service quality",
                    "Reward loyalty with exclusive benefits",
                    "Provide predictive stock recommendations"
                ])
        
        return strategies
    
    def _generate_growth_opportunities(self, profile: Dict) -> List[str]:
        """Generate growth opportunity recommendations"""
        
        opportunities = []
        
        if 'characteristics' in profile:
            chars = profile['characteristics']
            
            # Analyze product diversity
            if 'product_diversity' in chars:
                diversity = chars['product_diversity']['mean']
                if diversity < 3:
                    opportunities.append("Cross-selling opportunity: Low product diversity suggests potential for expanding product range")
            
            # Analyze transaction patterns
            if 'transaction_cv' in chars:
                cv = chars['transaction_cv']['mean']
                if cv > 1:
                    opportunities.append("Standardization opportunity: High transaction variation suggests need for consistent pricing/packages")
            
            # Analyze temporal patterns
            if 'weekend_purchase_ratio' in chars:
                weekend_ratio = chars['weekend_purchase_ratio']['mean']
                if weekend_ratio < 0.2:
                    opportunities.append("Weekend engagement opportunity: Low weekend activity suggests potential for weekend promotions")
        
        # Size-based opportunities
        percentage = profile['percentage']
        if percentage > 30:
            opportunities.append("Scale opportunity: Large segment size allows for dedicated resources and specialized programs")
        elif percentage < 5:
            opportunities.append("Niche opportunity: Small but potentially high-value segment worth specialized attention")
        
        return opportunities
    
    def _generate_key_metrics(self, profile: Dict) -> List[str]:
        """Generate key metrics to track for each segment"""
        
        metrics = [
            "Customer Lifetime Value (CLV)",
            "Retention Rate",
            "Average Order Value",
            "Purchase Frequency"
        ]
        
        if 'business_insights' in profile:
            insights = profile['business_insights']
            lifecycle = insights['dominant_lifecycle_stage']
            
            if lifecycle in ['Champions', 'Loyal Customers']:
                metrics.extend([
                    "Net Promoter Score (NPS)",
                    "Referral Rate",
                    "Share of Wallet"
                ])
            elif lifecycle in ['At Risk', 'Lost']:
                metrics.extend([
                    "Churn Rate",
                    "Win-back Campaign Success Rate",
                    "Time to Re-activation"
                ])
            elif lifecycle == 'New Customers':
                metrics.extend([
                    "Onboarding Completion Rate",
                    "Time to Second Purchase",
                    "Early Engagement Metrics"
                ])
        
        return metrics
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating report")
        
        report = []
        report.append("=" * 60)
        report.append("CUSTOMER SEGMENTATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model summary
        report.append("MODEL SUMMARY:")
        report.append("-" * 20)
        report.append(f"Number of segments identified: {len(self.segment_profiles)}")
        report.append(f"Total customers analyzed: {sum(p['size'] for p in self.segment_profiles.values())}")
        report.append("")
        
        # Segment details
        for cluster_key, profile in self.segment_profiles.items():
            segment_name = self.segment_names.get(cluster_key, cluster_key)
            report.append(f"SEGMENT: {segment_name.upper()}")
            report.append("-" * 30)
            report.append(f"Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
            
            if 'business_insights' in profile:
                insights = profile['business_insights']
                report.append(f"Dominant Lifecycle Stage: {insights['dominant_lifecycle_stage']}")
                report.append(f"Average Recency: {insights['avg_recency']:.1f} days")
                report.append(f"Average Frequency: {insights['avg_frequency']:.1f} transactions")
                report.append(f"Average Monetary Value: ${insights['avg_monetary']:,.2f}")
            
            report.append("")
        
        # Recommendations summary
        recommendations = self.get_segment_recommendations()
        report.append("KEY RECOMMENDATIONS:")
        report.append("-" * 25)
        
        for cluster_key, recs in recommendations.items():
            segment_name = recs['segment_name']
            report.append(f"\n{segment_name}:")
            report.append("  Marketing Strategy:")
            for strategy in recs['marketing_strategy'][:2]:  # Top 2 strategies
                report.append(f"    • {strategy}")
            
            report.append("  Growth Opportunities:")
            for opportunity in recs['growth_opportunities'][:2]:  # Top 2 opportunities
                report.append(f"    • {opportunity}")
        
        report.append("")
        report.append("=" * 60)
        report.append("End of Report")
        report.append("=" * 60)
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    config = SegmentationConfig(n_clusters=5, random_state=42)
    segmenter = CustomerSegmentation(config)
    
    # Load sample data (replace with actual data loading)
    # sales_data = pd.read_excel("SFA_Orders.xlsx", sheet_name="Jan")
    # customer_data = pd.read_csv("customer.csv")
    
    print("Customer Segmentation Model Ready!")
    print("Use the following methods:")
    print("1. segmenter.prepare_customer_features(sales_data, customer_data)")
    print("2. segmenter.fit_clustering_models(features)")
    print("3. segmenter.generate_visualizations(features)")
    print("4. segmenter.get_segment_recommendations()")
    print("5. segmenter.save_model()")
    print("6. segmenter.generate_summary_report()")