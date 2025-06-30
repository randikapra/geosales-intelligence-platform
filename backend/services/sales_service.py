from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text, desc, asc
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal
import calendar

from models.sales import Sales
from models.dealer import Dealer
from models.customer import Customer
from repositories.sales_repo import SalesRepository
from utils.date_utils import get_date_range, format_date, get_month_boundaries
from utils.geo_utils import calculate_distance
from core.exceptions import SalesProcessingError, ValidationError


class SalesService:
    def __init__(self, db: Session):
        self.db = db
        self.sales_repo = SalesRepository(db)

    async def process_sales_data(
        self, 
        sales_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process and validate bulk sales data"""
        try:
            processed_count = 0
            error_count = 0
            errors = []
            
            for record in sales_records:
                try:
                    # Validate required fields
                    if not self._validate_sales_record(record):
                        error_count += 1
                        errors.append({
                            "record": record,
                            "error": "Missing required fields"
                        })
                        continue
                    
                    # Create sales record
                    sales_obj = Sales(
                        code=record.get('code'),
                        date=record.get('date'),
                        distributor_code=record.get('distributor_code'),
                        user_code=record.get('user_code'),
                        user_name=record.get('user_name'),
                        final_value=Decimal(str(record.get('final_value', 0))),
                        creation_date=record.get('creation_date'),
                        submitted_date=record.get('submitted_date'),
                        erp_order_number=record.get('erp_order_number')
                    )
                    
                    self.db.add(sales_obj)
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    errors.append({
                        "record": record,
                        "error": str(e)
                    })
            
            self.db.commit()
            
            return {
                "processed_count": processed_count,
                "error_count": error_count,
                "errors": errors[:10],  # Return first 10 errors
                "success_rate": round((processed_count / len(sales_records)) * 100, 2)
            }
            
        except Exception as e:
            self.db.rollback()
            raise SalesProcessingError(f"Sales data processing failed: {str(e)}")

    def _validate_sales_record(self, record: Dict[str, Any]) -> bool:
        """Validate individual sales record"""
        required_fields = ['code', 'date', 'distributor_code', 'user_code', 'final_value']
        return all(field in record and record[field] is not None for field in required_fields)

    async def get_sales_aggregations(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "month",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive sales aggregations"""
        try:
            base_query = self.db.query(Sales).filter(
                and_(
                    Sales.date >= start_date,
                    Sales.date <= end_date
                )
            )
            
            # Apply filters
            if filters:
                base_query = self._apply_sales_filters(base_query, filters)
            
            # Group by different time periods
            if group_by == "day":
                aggregation = await self._aggregate_by_day(base_query, start_date, end_date)
            elif group_by == "week":
                aggregation = await self._aggregate_by_week(base_query, start_date, end_date)
            elif group_by == "month":
                aggregation = await self._aggregate_by_month(base_query, start_date, end_date)
            elif group_by == "dealer":
                aggregation = await self._aggregate_by_dealer(base_query)
            elif group_by == "customer":
                aggregation = await self._aggregate_by_customer(base_query)
            elif group_by == "territory":
                aggregation = await self._aggregate_by_territory(base_query)
            else:
                raise ValidationError(f"Invalid group_by parameter: {group_by}")
            
            # Calculate summary statistics
            summary = await self._calculate_sales_summary(base_query)
            
            return {
                "aggregation_type": group_by,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "data": aggregation,
                "summary": summary,
                "filters_applied": filters or {}
            }
            
        except Exception as e:
            raise SalesProcessingError(f"Sales aggregation failed: {str(e)}")

    def _apply_sales_filters(self, query, filters: Dict[str, Any]):
        """Apply various filters to sales query"""
        if "dealer_ids" in filters and filters["dealer_ids"]:
            query = query.filter(Sales.user_code.in_(filters["dealer_ids"]))
        
        if "customer_ids" in filters and filters["customer_ids"]:
            query = query.filter(Sales.distributor_code.in_(filters["customer_ids"]))
        
        if "min_value" in filters and filters["min_value"]:
            query = query.filter(Sales.final_value >= filters["min_value"])
        
        if "max_value" in filters and filters["max_value"]:
            query = query.filter(Sales.final_value <= filters["max_value"])
        
        if "territory_codes" in filters and filters["territory_codes"]:
            # Join with dealer table to filter by territory
            query = query.join(Dealer, Sales.user_code == Dealer.user_code).filter(
                Dealer.territory_code.in_(filters["territory_codes"])
            )
        
        return query

    async def _aggregate_by_day(self, query, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Aggregate sales data by day"""
        daily_stats = query.with_entities(
            func.date(Sales.date).label('date'),
            func.count(Sales.id).label('order_count'),
            func.sum(Sales.final_value).label('total_revenue'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.count(func.distinct(Sales.distributor_code)).label('unique_customers'),
            func.count(func.distinct(Sales.user_code)).label('active_dealers')
        ).group_by(func.date(Sales.date)).order_by('date').all()
        
        return [
            {
                "date": stat.date.isoformat(),
                "order_count": stat.order_count,
                "total_revenue": float(stat.total_revenue),
                "avg_order_value": float(stat.avg_order_value or 0),
                "unique_customers": stat.unique_customers,
                "active_dealers": stat.active_dealers
            }
            for stat in daily_stats
        ]

    async def _aggregate_by_week(self, query, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Aggregate sales data by week"""
        weekly_stats = query.with_entities(
            func.extract('year', Sales.date).label('year'),
            func.extract('week', Sales.date).label('week'),
            func.count(Sales.id).label('order_count'),
            func.sum(Sales.final_value).label('total_revenue'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.count(func.distinct(Sales.distributor_code)).label('unique_customers'),
            func.count(func.distinct(Sales.user_code)).label('active_dealers')
        ).group_by(
            func.extract('year', Sales.date),
            func.extract('week', Sales.date)
        ).order_by('year', 'week').all()
        
        return [
            {
                "year": int(stat.year),
                "week": int(stat.week),
                "week_label": f"{int(stat.year)}-W{int(stat.week):02d}",
                "order_count": stat.order_count,
                "total_revenue": float(stat.total_revenue),
                "avg_order_value": float(stat.avg_order_value or 0),
                "unique_customers": stat.unique_customers,
                "active_dealers": stat.active_dealers
            }
            for stat in weekly_stats
        ]

    async def _aggregate_by_month(self, query, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Aggregate sales data by month"""
        monthly_stats = query.with_entities(
            func.extract('year', Sales.date).label('year'),
            func.extract('month', Sales.date).label('month'),
            func.count(Sales.id).label('order_count'),
            func.sum(Sales.final_value).label('total_revenue'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.count(func.distinct(Sales.distributor_code)).label('unique_customers'),
            func.count(func.distinct(Sales.user_code)).label('active_dealers')
        ).group_by(
            func.extract('year', Sales.date),
            func.extract('month', Sales.date)
        ).order_by('year', 'month').all()
        
        return [
            {
                "year": int(stat.year),
                "month": int(stat.month),
                "month_name": calendar.month_name[int(stat.month)],
                "month_label": f"{int(stat.year)}-{int(stat.month):02d}",
                "order_count": stat.order_count,
                "total_revenue": float(stat.total_revenue),
                "avg_order_value": float(stat.avg_order_value or 0),
                "unique_customers": stat.unique_customers,
                "active_dealers": stat.active_dealers
            }
            for stat in monthly_stats
        ]

    async def _aggregate_by_dealer(self, query) -> List[Dict[str, Any]]:
        """Aggregate sales data by dealer"""
        dealer_stats = query.with_entities(
            Sales.user_code,
            Sales.user_name,
            func.count(Sales.id).label('order_count'),
            func.sum(Sales.final_value).label('total_revenue'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.count(func.distinct(Sales.distributor_code)).label('unique_customers'),
            func.min(Sales.date).label('first_sale'),
            func.max(Sales.date).label('last_sale')
        ).group_by(
            Sales.user_code, Sales.user_name
        ).order_by(desc('total_revenue')).all()
        
        result = []
        for stat in dealer_stats:
            # Calculate performance metrics
            days_active = (stat.last_sale - stat.first_sale).days + 1
            avg_daily_revenue = float(stat.total_revenue) / days_active if days_active > 0 else 0
            
            result.append({
                "dealer_code": stat.user_code,
                "dealer_name": stat.user_name,
                "order_count": stat.order_count,
                "total_revenue": float(stat.total_revenue),
                "avg_order_value": float(stat.avg_order_value or 0),
                "unique_customers": stat.unique_customers,
                "days_active": days_active,
                "avg_daily_revenue": round(avg_daily_revenue, 2),
                "first_sale_date": stat.first_sale.isoformat(),
                "last_sale_date": stat.last_sale.isoformat(),
                "customer_retention_rate": round((stat.unique_customers / stat.order_count) * 100, 2) if stat.order_count > 0 else 0
            })
        
        return result

    async def _aggregate_by_customer(self, query) -> List[Dict[str, Any]]:
        """Aggregate sales data by customer"""
        customer_stats = query.join(
            Customer, Sales.distributor_code == Customer.no, isouter=True
        ).with_entities(
            Sales.distributor_code,
            Customer.city,
            func.count(Sales.id).label('order_count'),
            func.sum(Sales.final_value).label('total_revenue'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.count(func.distinct(Sales.user_code)).label('dealers_served_by'),
            func.min(Sales.date).label('first_purchase'),
            func.max(Sales.date).label('last_purchase')
        ).group_by(
            Sales.distributor_code, Customer.city
        ).order_by(desc('total_revenue')).all()
        
        result = []
        for stat in customer_stats:
            # Calculate customer lifetime value and frequency
            days_as_customer = (stat.last_purchase - stat.first_purchase).days + 1
            purchase_frequency = stat.order_count / days_as_customer * 30 if days_as_customer > 0 else 0  # Orders per month
            
            result.append({
                "customer_code": stat.distributor_code,
                "city": stat.city,
                "order_count": stat.order_count,
                "total_revenue": float(stat.total_revenue),
                "avg_order_value": float(stat.avg_order_value or 0),
                "dealers_served_by": stat.dealers_served_by,
                "customer_lifetime_days": days_as_customer,
                "purchase_frequency_monthly": round(purchase_frequency, 2),
                "first_purchase_date": stat.first_purchase.isoformat(),
                "last_purchase_date": stat.last_purchase.isoformat(),
                "customer_value_score": round(float(stat.total_revenue) / 1000 + purchase_frequency * 10, 2)
            })
        
        return result

    async def _aggregate_by_territory(self, query) -> List[Dict[str, Any]]:
        """Aggregate sales data by territory"""
        territory_stats = query.join(
            Dealer, Sales.user_code == Dealer.user_code, isouter=True
        ).with_entities(
            Dealer.territory_code,
            func.count(Sales.id).label('order_count'),
            func.sum(Sales.final_value).label('total_revenue'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.count(func.distinct(Sales.user_code)).label('active_dealers'),
            func.count(func.distinct(Sales.distributor_code)).label('unique_customers')
        ).group_by(
            Dealer.territory_code
        ).order_by(desc('total_revenue')).all()
        
        return [
            {
                "territory_code": stat.territory_code,
                "order_count": stat.order_count,
                "total_revenue": float(stat.total_revenue),
                "avg_order_value": float(stat.avg_order_value or 0),
                "active_dealers": stat.active_dealers,
                "unique_customers": stat.unique_customers,
                "revenue_per_dealer": round(float(stat.total_revenue) / stat.active_dealers, 2) if stat.active_dealers > 0 else 0,
                "customers_per_dealer": round(stat.unique_customers / stat.active_dealers, 2) if stat.active_dealers > 0 else 0
            }
            for stat in territory_stats
        ]

    async def _calculate_sales_summary(self, query) -> Dict[str, Any]:
        """Calculate overall sales summary statistics"""
        total_stats = query.with_entities(
            func.count(Sales.id).label('total_orders'),
            func.sum(Sales.final_value).label('total_revenue'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.min(Sales.final_value).label('min_order_value'),
            func.max(Sales.final_value).label('max_order_value'),
            func.count(func.distinct(Sales.user_code)).label('total_dealers'),
            func.count(func.distinct(Sales.distributor_code)).label('total_customers')
        ).first()
        
        return {
            "total_orders": total_stats.total_orders or 0,
            "total_revenue": float(total_stats.total_revenue or 0),
            "avg_order_value": float(total_stats.avg_order_value or 0),
            "min_order_value": float(total_stats.min_order_value or 0),
            "max_order_value": float(total_stats.max_order_value or 0),
            "total_dealers": total_stats.total_dealers or 0,
            "total_customers": total_stats.total_customers or 0,
            "orders_per_dealer": round((total_stats.total_orders or 0) / (total_stats.total_dealers or 1), 2),
            "orders_per_customer": round((total_stats.total_orders or 0) / (total_stats.total_customers or 1), 2),
            "revenue_per_dealer": round(float(total_stats.total_revenue or 0) / (total_stats.total_dealers or 1), 2),
            "revenue_per_customer": round(float(total_stats.total_revenue or 0) / (total_stats.total_customers or 1), 2)
        }

    async def get_sales_trends(
        self,
        start_date: datetime,
        end_date: datetime,
        metric: str = "revenue",
        period: str = "month"
    ) -> Dict[str, Any]:
        """Analyze sales trends and patterns"""
        try:
            # Get time series data
            time_series = await self.get_sales_aggregations(
                start_date, end_date, group_by=period
            )
            
            data_points = time_series["data"]
            
            if not data_points:
                return {
                    "metric": metric,
                    "period": period,
                    "trend": "no_data",
                    "data": []
                }
            
            # Extract values for trend analysis
            if metric == "revenue":
                values = [point["total_revenue"] for point in data_points]
            elif metric == "orders":
                values = [point["order_count"] for point in data_points]
            elif metric == "customers":
                values = [point["unique_customers"] for point in data_points]
            else:
                raise ValidationError(f"Invalid metric: {metric}")
            
            # Calculate trend indicators
            trend_analysis = self._calculate_trend_indicators(values)
            
            # Identify patterns
            patterns = self._identify_sales_patterns(data_points, metric)
            
            # Generate forecasting data
            forecast = self._generate_sales_forecast(values, periods=3)
            
            return {
                "metric": metric,
                "period": period,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "trend_analysis": trend_analysis,
                "patterns": patterns,
                "forecast": forecast,
                "data": data_points
            }
            
        except Exception as e:
            raise SalesProcessingError(f"Trend analysis failed: {str(e)}")

    def _calculate_trend_indicators(self, values: List[float]) -> Dict[str, Any]:
        """Calculate various trend indicators"""
        if len(values) < 2:
            return {"trend": "insufficient_data"}
        
        # Linear trend calculation
        n = len(values)
        x = list(range(n))
        
        # Calculate slope using least squares
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Growth rate calculation
        if values[0] != 0:
            total_growth_rate = ((values[-1] - values[0]) / values[0]) * 100
        else:
            total_growth_rate = 0
        
        # Period-over-period growth rates
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                growth_rate = ((values[i] - values[i-1]) / values[i-1]) * 100
                growth_rates.append(growth_rate)
        
        avg_growth_rate = sum(growth_rates) / len(growth_rates) if growth_rates else 0
        
        # Trend classification
        if slope > 0.1:
            trend_direction = "increasing"
        elif slope < -0.1:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Volatility calculation (coefficient of variation)
        mean_value = sum(values) / len(values)
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        volatility = (std_dev / mean_value) * 100 if mean_value != 0 else 0
        
        return {
            "trend_direction": trend_direction,
            "slope": round(slope, 4),
            "total_growth_rate_percent": round(total_growth_rate, 2),
            "avg_period_growth_rate_percent": round(avg_growth_rate, 2),
            "volatility_percent": round(volatility, 2),
            "trend_strength": "strong" if abs(slope) > 1 else "moderate" if abs(slope) > 0.5 else "weak"
        }

    def _identify_sales_patterns(self, data_points: List[Dict], metric: str) -> Dict[str, Any]:
        """Identify sales patterns and seasonality"""
        if len(data_points) < 4:
            return {"patterns": "insufficient_data"}
        
        values = [point[f"total_{metric}"] if f"total_{metric}" in point else point[metric] for point in data_points]
        
        # Identify peaks and valleys
        peaks = []
        valleys = []
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append(i)
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                valleys.append(i)
        
        # Seasonal pattern detection (if monthly data)
        seasonal_pattern = None
        if len(data_points) >= 12 and "month" in data_points[0]:
            monthly_avg = {}
            for point in data_points:
                month = point.get("month", 1)
                if month not in monthly_avg:
                    monthly_avg[month] = []
                monthly_avg[month].append(values[data_points.index(point)])
            
            # Calculate average for each month
            monthly_averages = {
                month: sum(vals) / len(vals) 
                for month, vals in monthly_avg.items()
            }
            
            # Find peak and low seasons
            peak_month = max(monthly_averages, key=monthly_averages.get)
            low_month = min(monthly_averages, key=monthly_averages.get)
            
            seasonal_pattern = {
                "peak_month": peak_month,
                "peak_month_name": calendar.month_name[peak_month],
                "low_month": low_month,
                "low_month_name": calendar.month_name[low_month],
                "seasonal_variation_percent": round(
                    ((monthly_averages[peak_month] - monthly_averages[low_month]) / 
                     monthly_averages[low_month]) * 100, 2
                ) if monthly_averages[low_month] != 0 else 0
            }
        
        return {
            "peaks_count": len(peaks),
            "valleys_count": len(valleys),
            "peak_positions": peaks,
            "valley_positions": valleys,
            "seasonal_pattern": seasonal_pattern,
            "pattern_regularity": "regular" if len(peaks) > 2 and len(valleys) > 2 else "irregular"
        }

    def _generate_sales_forecast(self, historical_values: List[float], periods: int = 3) -> Dict[str, Any]:
        """Generate simple sales forecast using moving average and trend"""
        if len(historical_values) < 3:
            return {"forecast": "insufficient_data"}
        
        # Simple moving average forecast
        window_size = min(3, len(historical_values))
        moving_avg = sum(historical_values[-window_size:]) / window_size
        
        # Linear trend forecast
        trend = self._calculate_trend_indicators(historical_values)
        slope = trend["slope"]
        
        forecasted_values = []
        last_value = historical_values[-1]
        
        for i in range(1, periods + 1):
            # Combine moving average with trend
            trend_component = last_value + (slope * i)
            ma_component = moving_avg
            
            # Weighted combination (70% trend, 30% moving average)
            forecast_value = (trend_component * 0.7) + (ma_component * 0.3)
            forecasted_values.append(max(0, forecast_value))  # Ensure non-negative
        
        return {
            "method": "trend_adjusted_moving_average",
            "periods_forecasted": periods,
            "forecasted_values": [round(val, 2) for val in forecasted_values],
            "confidence": "medium" if len(historical_values) >= 6 else "low",
            "forecast_accuracy_note": "Simple forecast model - consider using advanced ML models for better accuracy"
        }

    async def get_sales_performance_comparison(
        self,
        current_period_start: datetime,
        current_period_end: datetime,
        comparison_period_start: datetime,
        comparison_period_end: datetime,
        group_by: str = "dealer"
    ) -> Dict[str, Any]:
        """Compare sales performance between two periods"""
        try:
            # Get current period data
            current_data = await self.get_sales_aggregations(
                current_period_start, current_period_end, group_by
            )
            
            # Get comparison period data
            comparison_data = await self.get_sales_aggregations(
                comparison_period_start, comparison_period_end, group_by
            )
            
            # Calculate comparisons
            comparison_results = self._calculate_period_comparisons(
                current_data["data"], 
                comparison_data["data"], 
                group_by
            )
            
            return {
                "current_period": {
                    "start": current_period_start.isoformat(),
                    "end": current_period_end.isoformat(),
                    "summary": current_data["summary"]
                },
                "comparison_period": {
                    "start": comparison_period_start.isoformat(),
                    "end": comparison_period_end.isoformat(),
                    "summary": comparison_data["summary"]
                },
                "comparison_results": comparison_results,
                "group_by": group_by
            }
            
        except Exception as e:
            raise SalesProcessingError(f"Performance comparison failed: {str(e)}")

    def _calculate_period_comparisons(
        self, 
        current_data: List[Dict], 
        comparison_data: List[Dict], 
        group_by: str
    ) -> Dict[str, Any]:
        """Calculate detailed comparisons between periods"""
        # Create lookup dictionaries
        current_lookup = {}
        comparison_lookup = {}
        
        # Determine the key field based on group_by
        if group_by == "dealer":
            key_field = "dealer_code"
        elif group_by == "customer":
            key_field = "customer_code"
        elif group_by == "territory":
            key_field = "territory_code"
        else:
            key_field = "date" if group_by in ["day", "week", "month"] else group_by
        
        # Build lookup dictionaries
        for item in current_data:
            if key_field in item:
                current_lookup[item[key_field]] = item
        
        for item in comparison_data:
            if key_field in item:
                comparison_lookup[item[key_field]] = item
        
        # Calculate comparisons
        comparisons = []
        all_keys = set(current_lookup.keys()) | set(comparison_lookup.keys())
        
        for key in all_keys:
            current_item = current_lookup.get(key, {})
            comparison_item = comparison_lookup.get(key, {})
            
            current_revenue = current_item.get("total_revenue", 0)
            comparison_revenue = comparison_item.get("total_revenue", 0)
            current_orders = current_item.get("order_count", 0)
            comparison_orders = comparison_item.get("order_count", 0)
            
            # Calculate percentage changes
            revenue_change = self._calculate_percentage_change(comparison_revenue, current_revenue)
            orders_change = self._calculate_percentage_change(comparison_orders, current_orders)
            
            comparisons.append({
                key_field: key,
                "current_revenue": current_revenue,
                "comparison_revenue": comparison_revenue,
                "revenue_change_percent": revenue_change,
                "current_orders": current_orders,
                "comparison_orders": comparison_orders,
                "orders_change_percent": orders_change,
                "performance_status": self._get_performance_status(revenue_change, orders_change)
            })
        
        # Sort by revenue change
        comparisons.sort(key=lambda x: x["revenue_change_percent"], reverse=True)
        
        # Calculate overall summary
        total_current_revenue = sum(item["current_revenue"] for item in comparisons)
        total_comparison_revenue = sum(item["comparison_revenue"] for item in comparisons)
        overall_revenue_change = self._calculate_percentage_change(total_comparison_revenue, total_current_revenue)
        
        return {
            "detailed_comparisons": comparisons,
            "summary": {
                "total_entities_compared": len(comparisons),
                "overall_revenue_change_percent": overall_revenue_change,
                "improved_count": len([c for c in comparisons if c["revenue_change_percent"] > 0]),
                "declined_count": len([c for c in comparisons if c["revenue_change_percent"] < 0]),
                "stable_count": len([c for c in comparisons if c["revenue_change_percent"] == 0])
            }
        }

    def _calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 100.0 if new_value > 0 else 0.0
        return round(((new_value - old_value) / old_value) * 100, 2)

    def _get_performance_status(self, revenue_change: float, orders_change: float) -> str:
        """Determine performance status based on changes"""
        if revenue_change > 10 and orders_change > 10:
            return "excellent"
        elif revenue_change > 5 or orders_change > 5:
            return "good"
        elif revenue_change > -5 and orders_change > -5:
            return "stable"
        elif revenue_change > -15 or orders_change > -15:
            return "declining"
        else:
            return "poor"

    async def detect_sales_anomalies(
        self,
        start_date: datetime,
        end_date: datetime,
        sensitivity: str = "medium"
    ) -> Dict[str, Any]:
        """Detect anomalies in sales data"""
        try:
            # Get daily sales data
            daily_data = await self.get_sales_aggregations(
                start_date, end_date, "day"
            )
            
            if not daily_data["data"]:
                return {"anomalies": [], "message": "No data available for anomaly detection"}
            
            revenue_values = [day["total_revenue"] for day in daily_data["data"]]
            order_values = [day["order_count"] for day in daily_data["data"]]
            
            # Set sensitivity thresholds
            sensitivity_multipliers = {
                "low": 3.0,
                "medium": 2.0,
                "high": 1.5
            }
            multiplier = sensitivity_multipliers.get(sensitivity, 2.0)
            
            # Detect revenue anomalies
            revenue_anomalies = self._detect_statistical_anomalies(
                revenue_values, daily_data["data"], "total_revenue", multiplier
            )
            
            # Detect order count anomalies
            order_anomalies = self._detect_statistical_anomalies(
                order_values, daily_data["data"], "order_count", multiplier
            )
            
            all_anomalies = revenue_anomalies + order_anomalies
            all_anomalies.sort(key=lambda x: x["date"])
            
            return {
                "detection_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "sensitivity": sensitivity,
                "anomalies_found": len(all_anomalies),
                "anomalies": all_anomalies,
                "summary": {
                    "revenue_anomalies": len(revenue_anomalies),
                    "order_anomalies": len(order_anomalies),
                    "positive_anomalies": len([a for a in all_anomalies if a["anomaly_type"] == "positive"]),
                    "negative_anomalies": len([a for a in all_anomalies if a["anomaly_type"] == "negative"])
                }
            }
            
        except Exception as e:
            raise SalesProcessingError(f"Anomaly detection failed: {str(e)}")

    def _detect_statistical_anomalies(
        self, 
        values: List[float], 
        data_points: List[Dict], 
        metric: str, 
        multiplier: float
    ) -> List[Dict[str, Any]]:
        """Detect statistical anomalies using z-score method"""
        if len(values) < 3:
            return []
        
        # Calculate mean and standard deviation
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return []
        
        anomalies = []
        
        for i, value in enumerate(values):
            z_score = (value - mean_val) / std_dev
            
            if abs(z_score) > multiplier:
                anomaly_type = "positive" if z_score > 0 else "negative"
                severity = "high" if abs(z_score) > multiplier * 1.5 else "medium"
                
                anomalies.append({
                    "date": data_points[i]["date"],
                    "metric": metric,
                    "value": value,
                    "expected_range": [
                        round(mean_val - multiplier * std_dev, 2),
                        round(mean_val + multiplier * std_dev, 2)
                    ],
                    "z_score": round(z_score, 2),
                    "anomaly_type": anomaly_type,
                    "severity": severity,
                    "deviation_percent": round(((value - mean_val) / mean_val) * 100, 2) if mean_val != 0 else 0
                })
        
        return anomalies