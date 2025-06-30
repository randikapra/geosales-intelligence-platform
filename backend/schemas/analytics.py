# # backend/schemas/analytics.py
# """
# Analytics schemas
# """
# from pydantic import BaseModel
# from typing import Optional, List
# from datetime import datetime


# class SalesAnalytics(BaseModel):
#     total_sales: float
#     total_orders: int
#     average_order_value: float
#     growth_rate: float
#     period_start: datetime
#     period_end: datetime


# class TerritoryAnalytics(BaseModel):
#     territory_code: str
#     dealer_count: int
#     customer_count: int
#     total_orders: int
#     total_sales: float
#     avg_order_value: float
#     sales_per_dealer: float


# class DealerPerformance(BaseModel):
#     user_code: str
#     user_name: str
#     territory_code: Optional[str]
#     total_orders: int
#     total_sales: float
#     avg_order_value: float
#     unique_customers: int
#     avg_distance_per_day: float
#     active_days: int
#     efficiency_score: float


# class CustomerAnalytics(BaseModel):
#     total_customers: int
#     active_customers: int
#     avg_customer_lifetime_value: float
#     customer_retention_rate: float
#     avg_purchase_frequency: float


# class RouteAnalytics(BaseModel):
#     dealer_code: str
#     date: datetime
#     total_stops: int
#     total_distance: float
#     customers_visited: int
#     sales_generated: float
#     efficiency_score: float


from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class KPIType(str, Enum):
    SALES_REVENUE = "sales_revenue"
    ORDER_COUNT = "order_count"
    CUSTOMER_COUNT = "customer_count"
    DEALER_PERFORMANCE = "dealer_performance"
    CONVERSION_RATE = "conversion_rate"
    GEOGRAPHIC_COVERAGE = "geographic_coverage"

class PeriodType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class KPIMetric(BaseModel):
    name: str
    value: float
    unit: str
    previous_value: Optional[float] = None
    change_percentage: Optional[float] = None
    trend: Optional[TrendDirection] = None
    target: Optional[float] = None
    achievement_rate: Optional[float] = None

class DashboardKPIs(BaseModel):
    period_start: date
    period_end: date
    total_sales: KPIMetric
    total_orders: KPIMetric
    active_dealers: KPIMetric
    active_customers: KPIMetric
    avg_order_value: KPIMetric
    conversion_rate: KPIMetric
    geographic_coverage: KPIMetric
    dealer_productivity: KPIMetric

class SalesTrendData(BaseModel):
    date: date
    sales: Decimal
    orders: int
    customers: int
    dealers: int

class AnalyticsPeriodRequest(BaseModel):
    start_date: date
    end_date: date
    period_type: PeriodType = PeriodType.DAILY
    compare_previous: bool = False

class DealerPerformanceAnalytics(BaseModel):
    dealer_id: int
    user_code: str
    user_name: str
    division_code: str
    territory_code: Optional[str]
    sales_metrics: Dict[str, float]
    activity_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    ranking: Dict[str, int]
    performance_score: float
    recommendations: List[str]

class CustomerSegmentAnalytics(BaseModel):
    segment_name: str
    customer_count: int
    total_sales: Decimal
    avg_order_value: Decimal
    order_frequency: float
    lifetime_value: Decimal
    growth_rate: float
    characteristics: Dict[str, Any]

class GeographicAnalytics(BaseModel):
    region: str
    coordinates: Dict[str, float]  # center lat, lng
    sales_density: float
    dealer_count: int
    customer_count: int
    coverage_percentage: float
    market_potential: float
    competition_level: str

class ProductPerformanceAnalytics(BaseModel):
    product_category: str
    total_sales: Decimal
    order_count: int
    market_share: float
    growth_rate: float
    seasonality_index: float
    top_regions: List[str]
    top_dealers: List[str]

class TimeSeriesAnalytics(BaseModel):
    metric_name: str
    time_series: List[Dict[str, Any]]
    trend_analysis: Dict[str, float]
    seasonality: Dict[str, float]
    forecast: Optional[List[Dict[str, Any]]] = None
    anomalies: List[Dict[str, Any]]

class AdvancedAnalyticsRequest(BaseModel):
    analysis_type: str = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)
    period: AnalyticsPeriodRequest

class PredictiveAnalytics(BaseModel):
    prediction_type: str
    confidence_level: float
    forecast_period: int
    predictions: List[Dict[str, Any]]
    model_accuracy: float
    feature_importance: Dict[str, float]
    recommendations: List[str]

class CompetitiveAnalysis(BaseModel):
    market_position: str
    market_share: float
    competitive_advantages: List[str]
    threats: List[str]
    opportunities: List[str]
    strategic_recommendations: List[str]

class SalesChannelAnalytics(BaseModel):
    channel_name: str
    sales_contribution: float
    order_count: int
    customer_acquisition_cost: float
    customer_lifetime_value: float
    roi: float
    growth_trend: TrendDirection

class CustomerJourneyAnalytics(BaseModel):
    stage: str
    customer_count: int
    conversion_rate: float
    avg_time_in_stage: float
    drop_off_rate: float
    key_actions: List[str]
    optimization_opportunities: List[str]

class AlertConfiguration(BaseModel):
    alert_name: str
    metric: str
    threshold_type: str  # "above", "below", "change"
    threshold_value: float
    notification_channels: List[str]
    is_active: bool

class AnalyticsAlert(BaseModel):
    alert_id: int
    alert_name: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    current_value: float
    threshold_value: float
    triggered_at: datetime
    is_acknowledged: bool

class BusinessIntelligenceReport(BaseModel):
    report_name: str
    generated_at: datetime
    period: AnalyticsPeriodRequest
    executive_summary: str
    key_findings: List[str]
    metrics: List[KPIMetric]
    recommendations: List[str]
    charts_data: Dict[str, Any]
    tables_data: Dict[str, Any]

class RealTimeMetrics(BaseModel):
    timestamp: datetime
    active_dealers: int
    current_orders: int
    today_sales: Decimal
    alerts_count: int
    system_health: Dict[str, str]

class CustomAnalyticsQuery(BaseModel):
    query_name: str
    sql_query: Optional[str] = None
    aggregations: List[Dict[str, str]]
    filters: Dict[str, Any]
    grouping: List[str]
    sorting: List[Dict[str, str]]
    limit: Optional[int] = 1000

class AnalyticsExportRequest(BaseModel):
    export_type: str  # "pdf", "excel", "csv", "json"
    data_source: str
    filters: Dict[str, Any]
    format_options: Dict[str, Any] = Field(default_factory=dict)

class AnalyticsDashboardConfig(BaseModel):
    dashboard_name: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    refresh_interval: int  # seconds
    access_permissions: List[str]
    is_default: bool