

# api/v1/endpoints/reports.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from core.dependencies import get_db, get_current_user
from models.sales import SalesOrder
from models.dealer import Dealer
from models.customer import Customer
from datetime import datetime, date, timedelta
import pandas as pd
import io
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

router = APIRouter()

@router.get("/sales-report")
def generate_sales_report(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    dealer_code: Optional[str] = Query(None),
    territory_code: Optional[str] = Query(None),
    format: str = Query("json", regex="^(json|csv|excel|pdf)$"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Generate comprehensive sales report"""
    if not start_date:
        start_date = datetime.now().date() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now().date()
    
    # Build query
    query = db.query(
        SalesOrder.code,
        SalesOrder.date,
        SalesOrder.user_code,
        SalesOrder.user_name,
        SalesOrder.distributor_code,
        SalesOrder.final_value,
        SalesOrder.creation_date,
        SalesOrder.submitted_date,
        SalesOrder.erp_order_number
    ).filter(
        and_(SalesOrder.date >= start_date, SalesOrder.date <= end_date)
    )
    
    if dealer_code:
        query = query.filter(SalesOrder.user_code == dealer_code)
    
    if territory_code:
        query = query.join(Dealer, SalesOrder.user_code == Dealer.user_code).filter(
            Dealer.territory_code == territory_code
        )
    
    sales_data = query.order_by(desc(SalesOrder.date)).all()
    
    # Prepare data
    report_data = []
    total_sales = 0
    total_orders = len(sales_data)
    
    for sale in sales_data:
        total_sales += float(sale.final_value)
        report_data.append({
            "Order Code": sale.code,
            "Date": str(sale.date),
            "Dealer Code": sale.user_code,
            "Dealer Name": sale.user_name,
            "Distributor Code": sale.distributor_code,
            "Order Value": float(sale.final_value),
            "Creation Date": str(sale.creation_date),
            "Submitted Date": str(sale.submitted_date),
            "ERP Order Number": sale.erp_order_number
        })
    
    # Summary statistics
    summary = {
        "report_period": f"{start_date} to {end_date}",
        "total_orders": total_orders,
        "total_sales_value": round(total_sales, 2),
        "average_order_value": round(total_sales / total_orders if total_orders > 0 else 0, 2),
        "filters_applied": {
            "dealer_code": dealer_code,
            "territory_code": territory_code
        }
    }
    
    if format == "json":
        return {
            "summary": summary,
            "data
            for day in daily_sales
        ]
    }

@router.get("/territory/{territory_code}")
def get_dealers_by_territory(
    territory_code: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get all dealers in a territory"""
    dealers = db.query(Dealer).filter(Dealer.territory_code == territory_code).all()
    return dealers

@router.put("/{dealer_code}/territory")
def update_dealer_territory(
    dealer_code: str,
    territory_code: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Update dealer territory assignment"""
    dealer = db.query(Dealer).filter(Dealer.user_code == dealer_code).first()
    if not dealer:
        raise HTTPException(status_code=404, detail="Dealer not found")
    
    dealer.territory_code = territory_code
    db.commit()
    db.refresh(dealer)
    
    return {"message": "Territory updated successfully", "dealer": dealer}

