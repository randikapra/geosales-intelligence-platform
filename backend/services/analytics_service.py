"""
Advanced Analytics Service for Sales Force Automation
Provides comprehensive KPI calculations, trend analysis, and business intelligence
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, text
import pandas as pd
import numpy as np
from collections import defaultdict
import statistics

from models.sales import Sales
from models.dealer import Dealer
from models.customer import Customer
from models.gps_data import GPSData
from utils.date_utils import DateUtils
from utils.geo_utils import GeoUtils


class AnalyticsService:
    """Comprehensive analytics service for sales performance and insights"""
    
    def __init__(self, db: Session):
        self.db = db
        self.date_utils = DateUtils()
        self.geo_utils = GeoUtils()
    
    def get_sales_kpis(self, start_date: datetime, end_date: datetime, 
                      dealer_code: Optional[str] = None, 
                      territory_code: Optional[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive sales KPIs"""
        
        # Base query
        query = self.db.query(Sales).filter(
            Sales.date.between(start_date, end_date)
        )
        
        if dealer_code:
            query = query.filter(Sales.user_code == dealer_code)
        if territory_code:
            query = query.join(Dealer).filter(Dealer.territory_code == territory_code)
        
        sales_data = query.all()
        
        if not sales_data:
            return self._empty_kpis()
        
        # Calculate KPIs
        total_revenue = sum(sale.final_value for sale in sales_data)
        total_orders = len(sales_data)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        # Unique customers and dealers
        unique_customers = len(set(sale.distributor_code for sale in sales_data))
        unique_dealers = len(set(sale.user_code for sale in sales_data))
        
        # Performance metrics
        revenue_per_dealer = total_revenue / unique_dealers if unique_dealers > 0 else 0
        orders_per_dealer = total_orders / unique_dealers if unique_dealers > 0 else 0
        
        # Time-based analysis
        daily_sales = self._calculate_daily_metrics(sales_data, start_date, end_date)
        growth_rate = self._calculate_growth_rate(start_date, end_date, dealer_code, territory_code)
        
        return {
            "total_revenue": round(total_revenue, 2),
            "total_orders": total_orders,
            "avg_order_value": round(avg_order_value, 2),
            "unique_customers": unique_customers,
            "unique_dealers": unique_dealers,
            "revenue_per_dealer": round(revenue_per_dealer, 2),
            "orders_per_dealer": round(orders_per_dealer, 2),
            "daily_avg_revenue": round(daily_sales["avg_daily_revenue"], 2),
            "daily_avg_orders": round(daily_sales["avg_daily_orders"], 2),
            "growth_rate": round(growth_rate, 2),
            "peak_sales_day": daily_sales["peak_day"],
            "conversion_metrics": self._calculate_conversion_metrics(sales_data)
        }
    
    def get_dealer_performance_analysis(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Comprehensive dealer performance analysis"""
        
        # Get sales data with dealer info
        query = self.db.query(
            Sales.user_code,
            Sales.user_name,
            func.sum(Sales.final_value).label('total_revenue'),
            func.count(Sales.code).label('total_orders'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.count(func.distinct(Sales.distributor_code)).label('unique_customers')
        ).filter(
            Sales.date.between(start_date, end_date)
        ).group_by(Sales.user_code, Sales.user_name)
        
        dealer_stats = query.all()
        
        performance_data = []
        for stat in dealer_stats:
            # Get GPS activity data
            gps_activity = self._get_dealer_gps_activity(stat.user_code, start_date, end_date)
            
            # Calculate efficiency metrics
            efficiency_score = self._calculate_dealer_efficiency(
                stat.total_revenue, 
                stat.total_orders, 
                gps_activity["total_distance"],
                gps_activity["active_days"]
            )
            
            performance_data.append({
                "dealer_code": stat.user_code,
                "dealer_name": stat.user_name,
                "total_revenue": float(stat.total_revenue or 0),
                "total_orders": stat.total_orders,
                "avg_order_value": float(stat.avg_order_value or 0),
                "unique_customers": stat.unique_customers,
                "revenue_per_customer": float(stat.total_revenue / stat.unique_customers) if stat.unique_customers > 0 else 0,
                "gps_metrics": gps_activity,
                "efficiency_score": efficiency_score,
                "performance_rating": self._get_performance_rating(efficiency_score)
            })
        
        # Sort by efficiency score
        performance_data.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        return performance_data
    
    def get_customer_analysis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze customer behavior and segmentation"""
        
        # Customer purchase patterns
        customer_query = self.db.query(
            Sales.distributor_code,
            func.sum(Sales.final_value).label('total_spent'),
            func.count(Sales.code).label('order_frequency'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.min(Sales.date).label('first_purchase'),
            func.max(Sales.date).label('last_purchase')
        ).filter(
            Sales.date.between(start_date, end_date)
        ).group_by(Sales.distributor_code).all()
        
        customers = []
        for customer in customer_query:
            # Calculate customer lifetime value and recency
            days_active = (customer.last_purchase - customer.first_purchase).days + 1
            recency_days = (end_date.date() - customer.last_purchase.date()).days
            
            # Customer segmentation
            segment = self._segment_customer(
                float(customer.total_spent), 
                customer.order_frequency, 
                recency_days
            )
            
            customers.append({
                "customer_code": customer.distributor_code,
                "total_spent": float(customer.total_spent),
                "order_frequency": customer.order_frequency,
                "avg_order_value": float(customer.avg_order_value),
                "days_active": days_active,
                "recency_days": recency_days,
                "customer_segment": segment,
                "clv_score": self._calculate_clv_score(customer)
            })
        
        # Segment analysis
        segment_summary = self._analyze_customer_segments(customers)
        
        return {
            "total_customers": len(customers),
            "customer_details": customers,
            "segment_analysis": segment_summary,
            "churn_risk_customers": [c for c in customers if c["recency_days"] > 30],
            "high_value_customers": [c for c in customers if c["customer_segment"] == "Champion"],
            "geographic_distribution": self._get_customer_geographic_distribution()
        }
    
    def get_territory_performance(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Analyze performance by territory/division"""
        
        # Territory-wise sales analysis
        territory_query = self.db.query(
            Dealer.territory_code,
            Dealer.division_code,
            func.sum(Sales.final_value).label('total_revenue'),
            func.count(Sales.code).label('total_orders'),
            func.count(func.distinct(Sales.user_code)).label('active_dealers'),
            func.count(func.distinct(Sales.distributor_code)).label('customers_served')
        ).join(
            Sales, Dealer.user_code == Sales.user_code
        ).filter(
            Sales.date.between(start_date, end_date)
        ).group_by(Dealer.territory_code, Dealer.division_code).all()
        
        territory_performance = []
        for territory in territory_query:
            # Calculate territory efficiency
            avg_revenue_per_dealer = float(territory.total_revenue) / territory.active_dealers if territory.active_dealers > 0 else 0
            market_penetration = self._calculate_market_penetration(territory.territory_code)
            
            territory_performance.append({
                "territory_code": territory.territory_code,
                "division_code": territory.division_code,
                "total_revenue": float(territory.total_revenue),
                "total_orders": territory.total_orders,
                "active_dealers": territory.active_dealers,
                "customers_served": territory.customers_served,
                "avg_revenue_per_dealer": round(avg_revenue_per_dealer, 2),
                "market_penetration": round(market_penetration, 2),
                "performance_rank": 0  # Will be calculated after sorting
            })
        
        # Rank territories by performance
        territory_performance.sort(key=lambda x: x["total_revenue"], reverse=True)
        for i, territory in enumerate(territory_performance):
            territory["performance_rank"] = i + 1
        
        return territory_performance
    
    def get_trend_analysis(self, start_date: datetime, end_date: datetime, 
                          granularity: str = "daily") -> Dict[str, Any]:
        """Comprehensive trend analysis with multiple granularities"""
        
        if granularity == "daily":
            date_format = "%Y-%m-%d"
            date_trunc = func.date(Sales.date)
        elif granularity == "weekly":
            date_format = "%Y-W%U"
            date_trunc = func.date_trunc('week', Sales.date)
        elif granularity == "monthly":
            date_format = "%Y-%m"
            date_trunc = func.date_trunc('month', Sales.date)
        else:
            date_format = "%Y-%m-%d"
            date_trunc = func.date(Sales.date)
        
        # Sales trends
        trend_query = self.db.query(
            date_trunc.label('period'),
            func.sum(Sales.final_value).label('revenue'),
            func.count(Sales.code).label('orders'),
            func.count(func.distinct(Sales.user_code)).label('active_dealers'),
            func.count(func.distinct(Sales.distributor_code)).label('active_customers')
        ).filter(
            Sales.date.between(start_date, end_date)
        ).group_by(date_trunc).order_by(date_trunc).all()
        
        trends = []
        revenues = []
        orders = []
        
        for trend in trend_query:
            period_data = {
                "period": trend.period.strftime(date_format) if hasattr(trend.period, 'strftime') else str(trend.period),
                "revenue": float(trend.revenue),
                "orders": trend.orders,
                "active_dealers": trend.active_dealers,
                "active_customers": trend.active_customers,
                "avg_order_value": float(trend.revenue) / trend.orders if trend.orders > 0 else 0
            }
            trends.append(period_data)
            revenues.append(float(trend.revenue))
            orders.append(trend.orders)
        
        # Calculate trend statistics
        revenue_trend = self._calculate_trend_statistics(revenues)
        order_trend = self._calculate_trend_statistics(orders)
        
        return {
            "trends": trends,
            "revenue_statistics": revenue_trend,
            "order_statistics": order_trend,
            "forecasting": self._simple_forecast(revenues, periods=7),
            "seasonality": self._detect_seasonality(trends),
            "growth_analysis": self._analyze_growth_patterns(trends)
        }
    
    def get_predictive_insights(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate predictive insights and recommendations"""
        
        # Get historical data for analysis
        sales_data = self.db.query(Sales).filter(
            Sales.date.between(start_date, end_date)
        ).all()
        
        if not sales_data:
            return {"error": "Insufficient data for predictions"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'date': sale.date,
            'revenue': sale.final_value,
            'dealer_code': sale.user_code,
            'customer_code': sale.distributor_code
        } for sale in sales_data])
        
        # Predictive insights
        insights = {
            "churn_prediction": self._predict_customer_churn(df),
            "revenue_forecast": self._forecast_revenue(df),
            "dealer_performance_prediction": self._predict_dealer_performance(df),
            "market_opportunities": self._identify_market_opportunities(df),
            "risk_factors": self._identify_risk_factors(df),
            "recommendations": self._generate_actionable_recommendations(df)
        }
        
        return insights
    
    # Helper methods
    def _empty_kpis(self) -> Dict[str, Any]:
        """Return empty KPI structure"""
        return {
            "total_revenue": 0,
            "total_orders": 0,
            "avg_order_value": 0,
            "unique_customers": 0,
            "unique_dealers": 0,
            "revenue_per_dealer": 0,
            "orders_per_dealer": 0,
            "daily_avg_revenue": 0,
            "daily_avg_orders": 0,
            "growth_rate": 0,
            "peak_sales_day": None,
            "conversion_metrics": {}
        }
    
    def _calculate_daily_metrics(self, sales_data: List[Sales], 
                               start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate daily sales metrics"""
        daily_revenue = defaultdict(float)
        daily_orders = defaultdict(int)
        
        for sale in sales_data:
            day_key = sale.date.date()
            daily_revenue[day_key] += sale.final_value
            daily_orders[day_key] += 1
        
        total_days = (end_date - start_date).days + 1
        avg_daily_revenue = sum(daily_revenue.values()) / total_days
        avg_daily_orders = sum(daily_orders.values()) / total_days
        
        peak_day = max(daily_revenue.items(), key=lambda x: x[1])[0] if daily_revenue else None
        
        return {
            "avg_daily_revenue": avg_daily_revenue,
            "avg_daily_orders": avg_daily_orders,
            "peak_day": peak_day
        }
    
    def _calculate_growth_rate(self, start_date: datetime, end_date: datetime,
                             dealer_code: Optional[str] = None,
                             territory_code: Optional[str] = None) -> float:
        """Calculate growth rate compared to previous period"""
        
        period_length = (end_date - start_date).days
        previous_start = start_date - timedelta(days=period_length)
        previous_end = start_date - timedelta(days=1)
        
        # Current period revenue
        current_query = self.db.query(func.sum(Sales.final_value)).filter(
            Sales.date.between(start_date, end_date)
        )
        
        # Previous period revenue
        previous_query = self.db.query(func.sum(Sales.final_value)).filter(
            Sales.date.between(previous_start, previous_end)
        )
        
        if dealer_code:
            current_query = current_query.filter(Sales.user_code == dealer_code)
            previous_query = previous_query.filter(Sales.user_code == dealer_code)
        
        current_revenue = current_query.scalar() or 0
        previous_revenue = previous_query.scalar() or 0
        
        if previous_revenue == 0:
            return 100.0 if current_revenue > 0 else 0.0
        
        return ((current_revenue - previous_revenue) / previous_revenue) * 100
    
    def _calculate_conversion_metrics(self, sales_data: List[Sales]) -> Dict[str, Any]:
        """Calculate conversion and efficiency metrics"""
        
        dealer_visits = defaultdict(set)
        dealer_sales = defaultdict(int)
        
        for sale in sales_data:
            dealer_visits[sale.user_code].add(sale.distributor_code)
            dealer_sales[sale.user_code] += 1
        
        total_visits = sum(len(visits) for visits in dealer_visits.values())
        total_conversions = sum(dealer_sales.values())
        
        conversion_rate = (total_conversions / total_visits * 100) if total_visits > 0 else 0
        
        return {
            "total_visits": total_visits,
            "total_conversions": total_conversions,
            "conversion_rate": round(conversion_rate, 2),
            "avg_visits_per_dealer": round(total_visits / len(dealer_visits), 2) if dealer_visits else 0
        }
    
    def _get_dealer_gps_activity(self, dealer_code: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get GPS activity metrics for a dealer"""
        
        gps_data = self.db.query(GPSData).filter(
            and_(
                GPSData.user_code == dealer_code,
                GPSData.received_date.between(start_date, end_date)
            )
        ).order_by(GPSData.received_date).all()
        
        if not gps_data:
            return {"total_distance": 0, "active_days": 0, "avg_speed": 0, "efficiency": 0}
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(gps_data)):
            distance = self.geo_utils.calculate_distance(
                gps_data[i-1].latitude, gps_data[i-1].longitude,
                gps_data[i].latitude, gps_data[i].longitude
            )
            total_distance += distance
        
        # Calculate active days
        active_days = len(set(gps.received_date.date() for gps in gps_data))
        
        # Calculate average speed (simplified)
        total_time_hours = (gps_data[-1].received_date - gps_data[0].received_date).total_seconds() / 3600
        avg_speed = total_distance / total_time_hours if total_time_hours > 0 else 0
        
        return {
            "total_distance": round(total_distance, 2),
            "active_days": active_days,
            "avg_speed": round(avg_speed, 2),
            "efficiency": round(total_distance / active_days, 2) if active_days > 0 else 0
        }
    
    def _calculate_dealer_efficiency(self, revenue: float, orders: int, 
                                   distance: float, active_days: int) -> float:
        """Calculate dealer efficiency score"""
        
        if distance == 0 or active_days == 0:
            return 0
        
        revenue_per_km = revenue / distance
        orders_per_day = orders / active_days
        
        # Weighted efficiency score
        efficiency_score = (revenue_per_km * 0.6) + (orders_per_day * 0.4)
        
        return round(efficiency_score, 2)
    
    def _get_performance_rating(self, efficiency_score: float) -> str:
        """Get performance rating based on efficiency score"""
        
        if efficiency_score >= 100:
            return "Excellent"
        elif efficiency_score >= 75:
            return "Good"
        elif efficiency_score >= 50:
            return "Average"
        elif efficiency_score >= 25:
            return "Below Average"
        else:
            return "Poor"
    
    def _segment_customer(self, total_spent: float, frequency: int, recency: int) -> str:
        """Segment customer based on RFM analysis"""
        
        # Simple RFM scoring (in a real implementation, you'd use percentiles)
        if recency <= 7 and frequency >= 10 and total_spent >= 100000:
            return "Champion"
        elif recency <= 14 and frequency >= 5 and total_spent >= 50000:
            return "Loyal"
        elif recency <= 30 and frequency >= 3:
            return "Potential Loyalist"
        elif total_spent >= 75000:
            return "Big Spender"
        elif recency <= 7:
            return "New Customer"
        elif recency > 60:
            return "At Risk"
        elif recency > 90:
            return "Lost"
        else:
            return "Regular"
    
    def _calculate_clv_score(self, customer) -> float:
        """Calculate Customer Lifetime Value score"""
        
        avg_order_value = float(customer.avg_order_value)
        frequency = customer.order_frequency
        
        # Simplified CLV calculation
        clv_score = avg_order_value * frequency * 1.2  # 1.2 is a retention factor
        
        return round(clv_score, 2)
    
    def _analyze_customer_segments(self, customers: List[Dict]) -> Dict[str, Any]:
        """Analyze customer segments"""
        
        segments = defaultdict(list)
        for customer in customers:
            segments[customer["customer_segment"]].append(customer)
        
        segment_analysis = {}
        for segment, segment_customers in segments.items():
            total_spent = sum(c["total_spent"] for c in segment_customers)
            avg_order_value = sum(c["avg_order_value"] for c in segment_customers) / len(segment_customers)
            
            segment_analysis[segment] = {
                "count": len(segment_customers),
                "total_revenue": round(total_spent, 2),
                "avg_order_value": round(avg_order_value, 2),
                "percentage": round((len(segment_customers) / len(customers)) * 100, 2)
            }
        
        return segment_analysis
    
    def _get_customer_geographic_distribution(self) -> List[Dict[str, Any]]:
        """Get geographic distribution of customers"""
        
        # This would need the customer location data
        customer_locations = self.db.query(
            Customer.city,
            func.count(Customer.customer_id).label('customer_count')
        ).group_by(Customer.city).all()
        
        return [
            {
                "city": location.city,
                "customer_count": location.customer_count
            }
            for location in customer_locations
        ]
    
    def _calculate_market_penetration(self, territory_code: str) -> float:
        """Calculate market penetration for a territory"""
        
        # This would need market size data - simplified calculation
        served_customers = self.db.query(func.count(func.distinct(Customer.customer_id))).filter(
            Customer.territory_code == territory_code
        ).scalar() or 0
        
        # Assuming total market size (this would come from external data)
        estimated_market_size = served_customers * 1.5  # Simplified assumption
        
        penetration = (served_customers / estimated_market_size) * 100 if estimated_market_size > 0 else 0
        
        return min(penetration, 100)  # Cap at 100%
    
    def _calculate_trend_statistics(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics"""
        
        if len(values) < 2:
            return {"trend": "insufficient_data", "volatility": 0, "growth_rate": 0}
        
        # Simple linear regression for trend
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        # Volatility (coefficient of variation)
        mean_val = np.mean(values)
        std_val = np.std(values)
        volatility = (std_val / mean_val) * 100 if mean_val != 0 else 0
        
        # Overall growth rate
        growth_rate = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
        
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        
        return {
            "trend": trend_direction,
            "slope": round(slope, 2),
            "volatility": round(volatility, 2),
            "growth_rate": round(growth_rate, 2),
            "correlation": self._calculate_correlation(x, values)
        }
    
    def _calculate_correlation(self, x: List[int], y: List[float]) -> float:
        """Calculate correlation coefficient"""
        
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        return round(np.corrcoef(x, y)[0, 1], 3)
    
    def _simple_forecast(self, values: List[float], periods: int = 7) -> List[Dict[str, Any]]:
        """Simple forecasting using linear regression"""
        
        if len(values) < 3:
            return []
        
        x = np.array(range(len(values)))
        y = np.array(values)
        
        # Linear regression
        coefficients = np.polyfit(x, y, 1)
        
        forecasts = []
        for i in range(periods):
            future_x = len(values) + i
            forecast_value = coefficients[0] * future_x + coefficients[1]
            
            forecasts.append({
                "period": i + 1,
                "forecasted_value": round(max(0, forecast_value), 2)  # Ensure non-negative
            })
        
        return forecasts
    
    def _detect_seasonality(self, trends: List[Dict]) -> Dict[str, Any]:
        """Simple seasonality detection"""
        
        if len(trends) < 7:
            return {"seasonal_pattern": "insufficient_data"}
        
        revenues = [t["revenue"] for t in trends]
        
        # Simple weekly pattern detection
        if len(revenues) >= 7:
            weekly_avg = []
            for i in range(0, len(revenues) - 6, 7):
                weekly_avg.append(np.mean(revenues[i:i+7]))
            
            if len(weekly_avg) > 1:
                weekly_volatility = np.std(weekly_avg) / np.mean(weekly_avg) * 100
                seasonal_strength = "high" if weekly_volatility > 20 else "medium" if weekly_volatility > 10 else "low"
            else:
                seasonal_strength = "unknown"
        else:
            seasonal_strength = "unknown"
        
        return {
            "seasonal_pattern": seasonal_strength,
            "peak_periods": self._identify_peak_periods(trends)
        }
    
    def _identify_peak_periods(self, trends: List[Dict]) -> List[str]:
        """Identify peak sales periods"""
        
        revenues = [t["revenue"] for t in trends]
        if not revenues:
            return []
        
        avg_revenue = np.mean(revenues)
        peak_threshold = avg_revenue * 1.2  # 20% above average
        
        peak_periods = []
        for trend in trends:
            if trend["revenue"] > peak_threshold:
                peak_periods.append(trend["period"])
        
        return peak_periods
    
    def _analyze_growth_patterns(self, trends: List[Dict]) -> Dict[str, Any]:
        """Analyze growth patterns in the data"""
        
        if len(trends) < 3:
            return {"growth_pattern": "insufficient_data"}
        
        revenues = [t["revenue"] for t in trends]
        growth_rates = []
        
        for i in range(1, len(revenues)):
            if revenues[i-1] != 0:
                growth_rate = ((revenues[i] - revenues[i-1]) / revenues[i-1]) * 100
                growth_rates.append(growth_rate)
        
        if not growth_rates:
            return {"growth_pattern": "no_growth_data"}
        
        avg_growth = np.mean(growth_rates)
        growth_volatility = np.std(growth_rates)
        
        if avg_growth > 5:
            pattern = "accelerating"
        elif avg_growth > 0:
            pattern = "growing"
        elif avg_growth > -5:
            pattern = "stable"
        else:
            pattern = "declining"
        
        return {
            "growth_pattern": pattern,
            "avg_growth_rate": round(avg_growth, 2),
            "growth_volatility": round(growth_volatility, 2),
            "consistent_growth": growth_volatility < 10
        }
    
    # Predictive methods (simplified implementations)
    def _predict_customer_churn(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict customer churn risk"""
        
        # Group by customer and calculate recency
        customer_last_purchase = df.groupby('customer_code')['date'].max()
        latest_date = df['date'].max()
        
        churn_predictions = []
        for customer, last_purchase in customer_last_purchase.items():
            days_since_purchase = (latest_date - last_purchase).days
            
            # Simple churn risk scoring
            if days_since_purchase > 60:
                risk_level = "high"
                churn_probability = 0.8
            elif days_since_purchase > 30:
                risk_level = "medium"
                churn_probability = 0.5
            elif days_since_purchase > 14:
                risk_level = "low"
                churn_probability = 0.2
            else:
                risk_level = "very_low"
                churn_probability = 0.1
            
            churn_predictions.append({
                "customer_code": customer,
                "days_since_last_purchase": days_since_purchase,
                "churn_risk": risk_level,
                "churn_probability": churn_probability
            })
        
        return sorted(churn_predictions, key=lambda x: x["churn_probability"], reverse=True)
    
    def _forecast_revenue(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Forecast future revenue"""
        
        daily_revenue = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        daily_revenue = daily_revenue.sort_values('date')
        
        if len(daily_revenue) < 7:
            return {"error": "Insufficient data for forecasting"}
        
        # Simple moving average forecast
        window = min(7, len(daily_revenue) // 2)
        moving_avg = daily_revenue['revenue'].rolling(window=window).mean().iloc[-1]
        
        # Linear trend forecast
        x = np.arange(len(daily_revenue))
        y = daily_revenue['revenue'].values
        slope, intercept = np.polyfit(x, y, 1)
        
        # Generate forecasts for next 30 days
        forecasts = []
        for i in range(1, 31):
            future_index = len(daily_revenue) + i
            trend_forecast = slope * future_index + intercept
            
            # Combine moving average and trend
            forecast = (moving_avg * 0.3) + (trend_forecast * 0.7)
            
            forecasts.append({
                "day": i,
                "forecasted_revenue": max(0, round(forecast, 2))
            })
        
        return {
            "forecasts": forecasts,
            "confidence_level": "medium",
            "methodology": "hybrid_moving_average_trend"
        }
    
    def _predict_dealer_performance(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict dealer performance for next period"""
        
        dealer_stats = df.groupby('dealer_code').agg({
            'revenue': ['sum', 'mean', 'std', 'count'],
            'date': ['min', 'max']
        }).reset_index()
        
        dealer_stats.columns = ['dealer_code', 'total_revenue', 'avg_revenue', 'revenue_std', 
                               'order_count', 'first_sale', 'last_sale']
        
        predictions = []
        for _, dealer in dealer_stats.iterrows():
            # Calculate performance indicators
            consistency = 1 - (dealer['revenue_std'] / dealer['avg_revenue']) if dealer['avg_revenue'] > 0 else 0
            activity_days = (dealer['last_sale'] - dealer['first_sale']).days + 1
            daily_avg = dealer['total_revenue'] / activity_days if activity_days > 0 else 0
            
            # Predict next month performance
            trend_factor = 1.0  # This would be calculated from historical trends
            predicted_revenue = daily_avg * 30 * trend_factor
            
            # Performance rating
            if predicted_revenue > 500000:
                rating = "excellent"
            elif predicted_revenue > 300000:
                rating = "good"
            elif predicted_revenue > 150000:
                rating = "average"
            else:
                rating = "needs_improvement"
            
            predictions.append({
                "dealer_code": dealer['dealer_code'],
                "predicted_monthly_revenue": round(predicted_revenue, 2),
                "consistency_score": round(consistency, 2),
                "performance_rating": rating,
                "confidence": "medium"
            })
        
        return sorted(predictions, key=lambda x: x["predicted_monthly_revenue"], reverse=True)
    
    def _identify_market_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify market opportunities"""
        
        opportunities = []
        
        # Underperforming territories
        territory_performance = df.groupby('dealer_code')['revenue'].sum()
        low_performers = territory_performance[territory_performance < territory_performance.quantile(0.25)]
        
        for dealer_code in low_performers.index:
            opportunities.append({
                "type": "underperforming_territory",
                "dealer_code": dealer_code,
                "current_revenue": round(low_performers[dealer_code], 2),
                "potential_improvement": "50-100%",
                "action": "Training and support needed"
            })
        
        # Customer acquisition opportunities
        active_customers = df['customer_code'].nunique()
        opportunities.append({
            "type": "customer_acquisition",
            "current_customers": active_customers,
            "potential_customers": round(active_customers * 1.3),
            "action": "Expand customer base by 30%"
        })
        
        # Product cross-selling opportunities
        opportunities.append({
            "type": "cross_selling",
            "opportunity": "Increase average order value",
            "potential_impact": "15-25% revenue increase",
            "action": "Implement product bundling strategies"
        })
        
        return opportunities
    
    def _identify_risk_factors(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify business risk factors"""
        
        risks = []
        
        # Customer concentration risk
        customer_revenue = df.groupby('customer_code')['revenue'].sum()
        top_customers_pct = customer_revenue.nlargest(5).sum() / customer_revenue.sum() * 100
        
        if top_customers_pct > 50:
            risks.append({
                "type": "customer_concentration",
                "severity": "high",
                "description": f"Top 5 customers represent {top_customers_pct:.1f}% of revenue",
                "mitigation": "Diversify customer base"
            })
        
        # Dealer dependency risk
        dealer_revenue = df.groupby('dealer_code')['revenue'].sum()
        top_dealers_pct = dealer_revenue.nlargest(3).sum() / dealer_revenue.sum() * 100
        
        if top_dealers_pct > 60:
            risks.append({
                "type": "dealer_dependency",
                "severity": "medium",
                "description": f"Top 3 dealers generate {top_dealers_pct:.1f}% of revenue",
                "mitigation": "Develop more dealers"
            })
        
        # Revenue volatility risk
        daily_revenue = df.groupby(df['date'].dt.date)['revenue'].sum()
        cv = daily_revenue.std() / daily_revenue.mean() * 100
        
        if cv > 50:
            risks.append({
                "type": "revenue_volatility",
                "severity": "medium",
                "description": f"High revenue volatility (CV: {cv:.1f}%)",
                "mitigation": "Stabilize sales processes"
            })
        
        return risks
    
    def _generate_actionable_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate actionable business recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        dealer_performance = df.groupby('dealer_code')['revenue'].sum()
        avg_performance = dealer_performance.mean()
        
        low_performers = dealer_performance[dealer_performance < avg_performance * 0.7]
        if len(low_performers) > 0:
            recommendations.append({
                "category": "dealer_development",
                "priority": "high",
                "title": "Improve Low-Performing Dealers",
                "description": f"{len(low_performers)} dealers performing below 70% of average",
                "action_items": [
                    "Provide additional training",
                    "Review territory assignments",
                    "Implement mentorship programs"
                ],
                "expected_impact": "20-30% revenue increase"
            })
        
        # Customer retention recommendations
        customer_frequency = df.groupby('customer_code').size()
        one_time_customers = len(customer_frequency[customer_frequency == 1])
        total_customers = len(customer_frequency)
        
        if one_time_customers / total_customers > 0.3:
            recommendations.append({
                "category": "customer_retention",
                "priority": "high",
                "title": "Reduce Customer Churn",
                "description": f"{one_time_customers}/{total_customers} customers made only one purchase",
                "action_items": [
                    "Implement follow-up programs",
                    "Create customer loyalty incentives",
                    "Improve after-sales service"
                ],
                "expected_impact": "15-25% revenue increase"
            })
        
        # Territory optimization
        recommendations.append({
            "category": "territory_optimization",
            "priority": "medium",
            "title": "Optimize Territory Coverage",
            "description": "Balance dealer workload and territory coverage",
            "action_items": [
                "Analyze dealer travel patterns",
                "Redistribute territories based on potential",
                "Implement route optimization"
            ],
            "expected_impact": "10-15% efficiency improvement"
        })
        
        # Digital transformation
        recommendations.append({
            "category": "digital_transformation",
            "priority": "medium",
            "title": "Enhance Digital Capabilities",
            "description": "Leverage technology for better insights",
            "action_items": [
                "Implement real-time analytics dashboard",
                "Mobile app for dealers",
                "Automated reporting systems"
            ],
            "expected_impact": "Improved decision making and efficiency"
        })
        
        return recommendations