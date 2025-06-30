from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text, desc, asc, extract
from datetime import datetime, timedelta, date
from models.sales import Sales
from models.dealer import Dealer
from models.customer import Customer
from .base import CRUDBase, BaseRepository
from schemas.sales import SalesCreate, SalesUpdate
import math
from decimal import Decimal


class SalesRepository(CRUDBase[Sales, SalesCreate, SalesUpdate]):
    def __init__(self):
        super().__init__(Sales)

    def get_by_date_range(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get sales within date range with pagination"""
        query = db.query(self.model).filter(
            and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            )
        )
        
        total = query.count()
        items = query.offset(skip).limit(limit).all()
        
        return {
            "items": items,
            "total": total,
            "page": (skip // limit) + 1,
            "pages": math.ceil(total / limit),
            "per_page": limit
        }

    def get_by_dealer(
        self,
        db: Session,
        user_code: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Sales]:
        """Get sales by dealer with optional date filtering"""
        query = db.query(self.model).filter(self.model.user_code == user_code)
        
        if start_date:
            query = query.filter(self.model.date >= start_date)
        if end_date:
            query = query.filter(self.model.date <= end_date)
        
        return query.order_by(desc(self.model.date)).all()

    def get_by_distributor(
        self,
        db: Session,
        distributor_code: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Sales]:
        """Get sales by distributor/customer"""
        query = db.query(self.model).filter(self.model.distributor_code == distributor_code)
        
        if start_date:
            query = query.filter(self.model.date >= start_date)
        if end_date:
            query = query.filter(self.model.date <= end_date)
        
        return query.order_by(desc(self.model.date)).all()

    def get_sales_summary(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Get sales summary grouped by time period
        group_by options: 'day', 'week', 'month', 'year'
        """
        if group_by == "day":
            time_field = func.date(self.model.date)
            time_label = "date"
        elif group_by == "week":
            time_field = func.date_trunc('week', self.model.date)
            time_label = "week"
        elif group_by == "month":
            time_field = func.date_trunc('month', self.model.date)
            time_label = "month"
        elif group_by == "year":
            time_field = func.date_trunc('year', self.model.date)
            time_label = "year"
        else:
            time_field = func.date(self.model.date)
            time_label = "date"
        
        results = (
            db.query(
                time_field.label(time_label),
                func.count(self.model.id).label('total_orders'),
                func.sum(self.model.final_value).label('total_sales'),
                func.avg(self.model.final_value).label('avg_order_value'),
                func.min(self.model.final_value).label('min_order'),
                func.max(self.model.final_value).label('max_order'),
                func.count(func.distinct(self.model.user_code)).label('active_dealers'),
                func.count(func.distinct(self.model.distributor_code)).label('unique_customers')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .group_by(time_field)
            .order_by(time_field)
            .all()
        )
        
        return [
            {
                time_label: result[0].isoformat() if hasattr(result[0], 'isoformat') else str(result[0]),
                "total_orders": result.total_orders,
                "total_sales": float(result.total_sales),
                "avg_order_value": float(result.avg_order_value),
                "min_order": float(result.min_order),
                "max_order": float(result.max_order),
                "active_dealers": result.active_dealers,
                "unique_customers": result.unique_customers
            }
            for result in results
        ]

    def get_top_customers(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top customers by sales volume"""
        results = (
            db.query(
                self.model.distributor_code,
                func.count(self.model.id).label('total_orders'),
                func.sum(self.model.final_value).label('total_sales'),
                func.avg(self.model.final_value).label('avg_order_value'),
                func.count(func.distinct(self.model.user_code)).label('dealers_served')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .group_by(self.model.distributor_code)
            .order_by(desc(func.sum(self.model.final_value)))
            .limit(limit)
            .all()
        )
        
        return [
            {
                "distributor_code": result.distributor_code,
                "total_orders": result.total_orders,
                "total_sales": float(result.total_sales),
                "avg_order_value": float(result.avg_order_value),
                "dealers_served": result.dealers_served
            }
            for result in results
        ]

    def get_sales_trends(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime,
        trend_type: str = "daily"
    ) -> Dict[str, Any]:
        """
        Analyze sales trends with growth rates and patterns
        trend_type: 'daily', 'weekly', 'monthly'
        """
        # Get base trend data
        trend_data = self.get_sales_summary(db, start_date, end_date, trend_type.replace('ly', ''))
        
        # Calculate growth rates
        for i in range(1, len(trend_data)):
            prev_sales = trend_data[i-1]['total_sales']
            curr_sales = trend_data[i]['total_sales']
            
            if prev_sales > 0:
                growth_rate = ((curr_sales - prev_sales) / prev_sales) * 100
            else:
                growth_rate = 0
            
            trend_data[i]['growth_rate'] = round(growth_rate, 2)
        
        # Set first period growth rate to 0
        if trend_data:
            trend_data[0]['growth_rate'] = 0
        
        # Calculate overall statistics
        total_sales = sum(period['total_sales'] for period in trend_data)
        total_orders = sum(period['total_orders'] for period in trend_data)
        avg_daily_sales = total_sales / len(trend_data) if trend_data else 0
        
        # Find best and worst performing periods
        best_period = max(trend_data, key=lambda x: x['total_sales']) if trend_data else None
        worst_period = min(trend_data, key=lambda x: x['total_sales']) if trend_data else None
        
        return {
            "trend_data": trend_data,
            "summary": {
                "total_sales": total_sales,
                "total_orders": total_orders,
                "avg_period_sales": round(avg_daily_sales, 2),
                "periods_count": len(trend_data),
                "best_period": best_period,
                "worst_period": worst_period
            }
        }

    def get_dealer_performance_comparison(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime,
        user_codes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Compare dealer performance metrics"""
        query = (
            db.query(
                self.model.user_code,
                self.model.user_name,
                func.count(self.model.id).label('total_orders'),
                func.sum(self.model.final_value).label('total_sales'),
                func.avg(self.model.final_value).label('avg_order_value'),
                func.count(func.distinct(self.model.distributor_code)).label('unique_customers'),
                func.count(func.distinct(func.date(self.model.date))).label('working_days'),
                func.min(self.model.date).label('first_sale'),
                func.max(self.model.date).label('last_sale')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
        )
        
        if user_codes:
            query = query.filter(self.model.user_code.in_(user_codes))
        
        results = (
            query.group_by(self.model.user_code, self.model.user_name)
            .order_by(desc(func.sum(self.model.final_value)))
            .all()
        )
        
        performance_data = []
        for result in results:
            # Calculate efficiency metrics
            orders_per_day = result.total_orders / max(result.working_days, 1)
            sales_per_day = float(result.total_sales) / max(result.working_days, 1)
            customers_per_day = result.unique_customers / max(result.working_days, 1)
            
            performance_data.append({
                "user_code": result.user_code,
                "user_name": result.user_name,
                "total_orders": result.total_orders,
                "total_sales": float(result.total_sales),
                "avg_order_value": float(result.avg_order_value),
                "unique_customers": result.unique_customers,
                "working_days": result.working_days,
                "orders_per_day": round(orders_per_day, 2),
                "sales_per_day": round(sales_per_day, 2),
                "customers_per_day": round(customers_per_day, 2),
                "first_sale": result.first_sale.isoformat(),
                "last_sale": result.last_sale.isoformat()
            })
        
        return performance_data

    def get_sales_by_time_of_day(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Analyze sales patterns by hour of day"""
        results = (
            db.query(
                extract('hour', self.model.date).label('hour'),
                func.count(self.model.id).label('total_orders'),
                func.sum(self.model.final_value).label('total_sales'),
                func.avg(self.model.final_value).label('avg_order_value')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .group_by(extract('hour', self.model.date))
            .order_by(extract('hour', self.model.date))
            .all()
        )
        
        return [
            {
                "hour": int(result.hour),
                "total_orders": result.total_orders,
                "total_sales": float(result.total_sales),
                "avg_order_value": float(result.avg_order_value)
            }
            for result in results
        ]

    def get_sales_forecast_data(
        self,
        db: Session,
        months_back: int = 12
    ) -> List[Dict[str, Any]]:
        """Get historical sales data for forecasting"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        # Monthly aggregation for forecasting
        results = (
            db.query(
                extract('year', self.model.date).label('year'),
                extract('month', self.model.date).label('month'),
                func.count(self.model.id).label('total_orders'),
                func.sum(self.model.final_value).label('total_sales'),
                func.avg(self.model.final_value).label('avg_order_value'),
                func.count(func.distinct(self.model.user_code)).label('active_dealers'),
                func.count(func.distinct(self.model.distributor_code)).label('unique_customers')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .group_by(extract('year', self.model.date), extract('month', self.model.date))
            .order_by(extract('year', self.model.date), extract('month', self.model.date))
            .all()
        )
        
        return [
            {
                "year": int(result.year),
                "month": int(result.month),
                "date": f"{int(result.year)}-{int(result.month):02d}",
                "total_orders": result.total_orders,
                "total_sales": float(result.total_sales),
                "avg_order_value": float(result.avg_order_value),
                "active_dealers": result.active_dealers,
                "unique_customers": result.unique_customers
            }
            for result in results
        ]

    def get_dealer_territory_performance(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Analyze dealer performance by territory/division"""
        results = (
            db.query(
                self.model.user_code,
                self.model.user_name,
                func.count(self.model.id).label('total_orders'),
                func.sum(self.model.final_value).label('total_sales'),
                func.count(func.distinct(self.model.distributor_code)).label('unique_customers'),
                func.count(func.distinct(func.date(self.model.date))).label('working_days')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .group_by(self.model.user_code, self.model.user_name)
            .order_by(desc(func.sum(self.model.final_value)))
            .all()
        )
        
        return [
            {
                "user_code": result.user_code,
                "user_name": result.user_name,
                "total_orders": result.total_orders,
                "total_sales": float(result.total_sales),
                "unique_customers": result.unique_customers,
                "working_days": result.working_days,
                "avg_daily_sales": float(result.total_sales) / max(result.working_days, 1),
                "customer_coverage": result.unique_customers / max(result.working_days, 1)
            }
            for result in results
        ]

    def get_customer_buying_patterns(
        self,
        db: Session,
        distributor_code: str,
        months_back: int = 6
    ) -> Dict[str, Any]:
        """Analyze individual customer buying patterns"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        # Get customer's purchase history
        purchases = (
            db.query(self.model)
            .filter(and_(
                self.model.distributor_code == distributor_code,
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .order_by(self.model.date)
            .all()
        )
        
        if not purchases:
            return {"error": "No purchase history found"}
        
        # Calculate patterns
        total_orders = len(purchases)
        total_spent = sum(p.final_value for p in purchases)
        avg_order_value = total_spent / total_orders
        
        # Calculate purchase frequency (days between orders)
        dates = sorted([p.date for p in purchases])
        intervals = []
        for i in range(1, len(dates)):
            interval = (dates[i] - dates[i-1]).days
            intervals.append(interval)
        
        avg_frequency = sum(intervals) / len(intervals) if intervals else 0
        
        # Monthly spending pattern
        monthly_spending = {}
        for purchase in purchases:
            month_key = f"{purchase.date.year}-{purchase.date.month:02d}"
            if month_key not in monthly_spending:
                monthly_spending[month_key] = {"orders": 0, "total": 0}
            monthly_spending[month_key]["orders"] += 1
            monthly_spending[month_key]["total"] += float(purchase.final_value)
        
        return {
            "distributor_code": distributor_code,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "months": months_back
            },
            "summary": {
                "total_orders": total_orders,
                "total_spent": float(total_spent),
                "avg_order_value": float(avg_order_value),
                "avg_frequency_days": round(avg_frequency, 1),
                "first_purchase": dates[0].isoformat(),
                "last_purchase": dates[-1].isoformat()
            },
            "monthly_pattern": [
                {
                    "month": month,
                    "orders": data["orders"],
                    "total_spent": data["total"],
                    "avg_order_value": data["total"] / data["orders"]
                }
                for month, data in sorted(monthly_spending.items())
            ]
        }

    def get_peak_sales_analysis(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze peak sales periods and patterns"""
        
        # Day of week analysis
        dow_results = (
            db.query(
                extract('dow', self.model.date).label('dow'),
                func.count(self.model.id).label('orders'),
                func.sum(self.model.final_value).label('sales')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .group_by(extract('dow', self.model.date))
            .order_by(extract('dow', self.model.date))
            .all()
        )
        
        # Hour of day analysis
        hour_results = (
            db.query(
                extract('hour', self.model.date).label('hour'),
                func.count(self.model.id).label('orders'),
                func.sum(self.model.final_value).label('sales')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .group_by(extract('hour', self.model.date))
            .order_by(extract('hour', self.model.date))
            .all()
        )
        
        # Weekly patterns
        week_results = (
            db.query(
                extract('week', self.model.date).label('week'),
                func.count(self.model.id).label('orders'),
                func.sum(self.model.final_value).label('sales')
            )
            .filter(and_(
                self.model.date >= start_date,
                self.model.date <= end_date
            ))
            .group_by(extract('week', self.model.date))
            .order_by(extract('week', self.model.date))
            .all()
        )
        
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        return {
            "day_of_week_pattern": [
                {
                    "day": day_names[int(result.dow)],
                    "day_number": int(result.dow),
                    "orders": result.orders,
                    "sales": float(result.sales)
                }
                for result in dow_results
            ],
            "hourly_pattern": [
                {
                    "hour": int(result.hour),
                    "orders": result.orders,
                    "sales": float(result.sales)
                }
                for result in hour_results
            ],
            "weekly_pattern": [
                {
                    "week": int(result.week),
                    "orders": result.orders,
                    "sales": float(result.sales)
                }
                for result in week_results
            ]
        }


# Create instance
sales_repository = SalesRepository()