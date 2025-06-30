"""
Email utility functions for SFA system.
Handles email templates, sending logic, and notification management.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import base64
import os
import logging
from datetime import datetime, timedelta
from jinja2 import Template, Environment, FileSystemLoader
import asyncio
import aiosmtplib
from pydantic import BaseModel, EmailStr, validator
import json

logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email configuration settings."""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30

class EmailAttachment(BaseModel):
    """Email attachment model."""
    filename: str
    content: Union[bytes, str]
    content_type: str = "application/octet-stream"
    
    @validator('content')
    def validate_content(cls, v):
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except Exception:
                raise ValueError("String content must be base64 encoded")
        return v

class EmailRecipient(BaseModel):
    """Email recipient model."""
    email: EmailStr
    name: Optional[str] = None
    type: str = "to"  # to, cc, bcc
    
    def __str__(self):
        return f"{self.name} <{self.email}>" if self.name else str(self.email)

class EmailTemplate(BaseModel):
    """Email template model."""
    name: str
    subject: str
    html_content: str
    text_content: Optional[str] = None
    variables: List[str] = []
    category: str = "general"

class EmailMessage(BaseModel):
    """Email message model."""
    recipients: List[EmailRecipient]
    subject: str
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    attachments: List[EmailAttachment] = []
    template_name: Optional[str] = None
    template_data: Dict[str, Any] = {}
    priority: str = "normal"  # low, normal, high
    send_at: Optional[datetime] = None

class EmailService:
    """Comprehensive email service for SFA system."""
    
    def __init__(self, config: EmailConfig, template_dir: Optional[str] = None):
        self.config = config
        self.template_env = None
        if template_dir and os.path.exists(template_dir):
            self.template_env = Environment(loader=FileSystemLoader(template_dir))
        
        # Email templates registry
        self.templates = self._load_default_templates()
        
        # Email queue for batch processing
        self.email_queue = []
        
    def _load_default_templates(self) -> Dict[str, EmailTemplate]:
        """Load default email templates for SFA operations."""
        return {
            "dealer_welcome": EmailTemplate(
                name="dealer_welcome",
                subject="Welcome to SFA System - {{ dealer_name }}",
                html_content=self._get_dealer_welcome_template(),
                text_content="Welcome to SFA System, {{ dealer_name }}! Your account has been created successfully.",
                variables=["dealer_name", "username", "territory", "manager_name", "login_url"],
                category="onboarding"
            ),
            "sales_report": EmailTemplate(
                name="sales_report",
                subject="Sales Report - {{ period }} - {{ dealer_name }}",
                html_content=self._get_sales_report_template(),
                text_content="Sales report for {{ period }} attached.",
                variables=["dealer_name", "period", "total_sales", "target", "achievement_percentage", "top_products"],
                category="reports"
            ),
            "target_achievement": EmailTemplate(
                name="target_achievement",
                subject="🎉 Target Achievement Alert - {{ dealer_name }}",
                html_content=self._get_target_achievement_template(),
                text_content="Congratulations! You have achieved {{ achievement_percentage }}% of your target.",
                variables=["dealer_name", "achievement_percentage", "target_amount", "actual_amount", "period"],
                category="notifications"
            ),
            "low_performance_alert": EmailTemplate(
                name="low_performance_alert",
                subject="⚠️ Performance Alert - {{ dealer_name }}",
                html_content=self._get_low_performance_template(),
                text_content="Performance alert: Current achievement is {{ achievement_percentage }}%.",
                variables=["dealer_name", "achievement_percentage", "target_amount", "actual_amount", "manager_name"],
                category="alerts"
            ),
            "route_optimization": EmailTemplate(
                name="route_optimization",
                subject="📍 Optimized Route Plan - {{ date }}",
                html_content=self._get_route_optimization_template(),
                text_content="Your optimized route for {{ date }} is ready.",
                variables=["dealer_name", "date", "total_customers", "estimated_distance", "estimated_time", "route_map_url"],
                category="operations"
            ),
            "customer_visit_reminder": EmailTemplate(
                name="customer_visit_reminder",
                subject="📅 Customer Visit Reminder - {{ customer_name }}",
                html_content=self._get_visit_reminder_template(),
                text_content="Reminder: Visit {{ customer_name }} at {{ visit_time }}.",
                variables=["dealer_name", "customer_name", "visit_time", "customer_address", "last_visit_date", "notes"],
                category="reminders"
            ),
            "system_maintenance": EmailTemplate(
                name="system_maintenance",
                subject="🔧 System Maintenance Notification",
                html_content=self._get_maintenance_template(),
                text_content="System maintenance scheduled from {{ start_time }} to {{ end_time }}.",
                variables=["start_time", "end_time", "description", "impact"],
                category="system"
            ),
            "password_reset": EmailTemplate(
                name="password_reset",
                subject="🔒 Password Reset Request - SFA System",
                html_content=self._get_password_reset_template(),
                text_content="Password reset link: {{ reset_url }}",
                variables=["user_name", "reset_url", "expiry_time"],
                category="security"
            )
        }
    
    def register_template(self, template: EmailTemplate):
        """Register a new email template."""
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[EmailTemplate]:
        """Get email template by name."""
        return self.templates.get(name)
    
    def render_template(self, template_name: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Render email template with data."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Use Jinja2 if available, otherwise simple string replacement
        if self.template_env:
            try:
                html_template = self.template_env.from_string(template.html_content)
                html_content = html_template.render(**data)
                
                if template.text_content:
                    text_template = self.template_env.from_string(template.text_content)
                    text_content = text_template.render(**data)
                else:
                    text_content = None
            except Exception as e:
                logger.error(f"Template rendering error: {e}")
                # Fallback to simple replacement
                html_content = self._simple_template_render(template.html_content, data)
                text_content = self._simple_template_render(template.text_content, data) if template.text_content else None
        else:
            html_content = self._simple_template_render(template.html_content, data)
            text_content = self._simple_template_render(template.text_content, data) if template.text_content else None
        
        # Render subject
        subject = self._simple_template_render(template.subject, data)
        
        return {
            "subject": subject,
            "html_content": html_content,
            "text_content": text_content
        }
    
    def _simple_template_render(self, template: str, data: Dict[str, Any]) -> str:
        """Simple template rendering using string replacement."""
        if not template:
            return ""
        
        rendered = template
        for key, value in data.items():
            rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
        return rendered
    
    def send_email(self, message: EmailMessage) -> bool:
        """Send single email."""
        try:
            # Render template if specified
            if message.template_name:
                rendered = self.render_template(message.template_name, message.template_data)
                message.subject = rendered["subject"]
                message.html_content = rendered["html_content"]
                message.text_content = rendered["text_content"]
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.username
            msg['Subject'] = message.subject
            
            # Add recipients
            to_recipients = [str(r) for r in message.recipients if r.type == "to"]
            cc_recipients = [str(r) for r in message.recipients if r.type == "cc"]
            bcc_recipients = [str(r) for r in message.recipients if r.type == "bcc"]
            
            msg['To'] = ", ".join(to_recipients)
            if cc_recipients:
                msg['Cc'] = ", ".join(cc_recipients)
            
            # Add content
            if message.text_content:
                msg.attach(MIMEText(message.text_content, 'plain'))
            if message.html_content:
                msg.attach(MIMEText(message.html_content, 'html'))
            
            # Add attachments
            for attachment in message.attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.content)
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment.filename}'
                )
                msg.attach(part)
            
            # Send email
            all_recipients = [r.email for r in message.recipients]
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls(context=context)
                elif self.config.use_ssl:
                    server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port, context=context)
                
                server.login(self.config.username, self.config.password)
                server.sendmail(self.config.username, all_recipients, msg.as_string())
            
            logger.info(f"Email sent successfully to {len(all_recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    async def send_email_async(self, message: EmailMessage) -> bool:
        """Send email asynchronously."""
        try:
            # Render template if specified
            if message.template_name:
                rendered = self.render_template(message.template_name, message.template_data)
                message.subject = rendered["subject"]
                message.html_content = rendered["html_content"]
                message.text_content = rendered["text_content"]
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.username
            msg['Subject'] = message.subject
            
            # Add recipients
            to_recipients = [str(r) for r in message.recipients if r.type == "to"]
            cc_recipients = [str(r) for r in message.recipients if r.type == "cc"]
            
            msg['To'] = ", ".join(to_recipients)
            if cc_recipients:
                msg['Cc'] = ", ".join(cc_recipients)
            
            # Add content
            if message.text_content:
                msg.attach(MIMEText(message.text_content, 'plain'))
            if message.html_content:
                msg.attach(MIMEText(message.html_content, 'html'))
            
            # Add attachments
            for attachment in message.attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.content)
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment.filename}'
                )
                msg.attach(part)
            
            # Send email asynchronously
            all_recipients = [r.email for r in message.recipients]
            
            await aiosmtplib.send(
                msg,
                hostname=self.config.smtp_server,
                port=self.config.smtp_port,
                start_tls=self.config.use_tls,
                username=self.config.username,
                password=self.config.password,
                recipients=all_recipients
            )
            
            logger.info(f"Email sent successfully (async) to {len(all_recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email (async): {e}")
            return False
    
    def send_bulk_emails(self, messages: List[EmailMessage]) -> Dict[str, int]:
        """Send multiple emails in batch."""
        results = {"success": 0, "failed": 0}
        
        for message in messages:
            if self.send_email(message):
                results["success"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    async def send_bulk_emails_async(self, messages: List[EmailMessage]) -> Dict[str, int]:
        """Send multiple emails asynchronously."""
        tasks = [self.send_email_async(message) for message in messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        failed_count = len(results) - success_count
        
        return {"success": success_count, "failed": failed_count}
    
    def add_to_queue(self, message: EmailMessage):
        """Add email to sending queue."""
        self.email_queue.append(message)
    
    def process_queue(self) -> Dict[str, int]:
        """Process all emails in queue."""
        if not self.email_queue:
            return {"success": 0, "failed": 0}
        
        results = self.send_bulk_emails(self.email_queue)
        self.email_queue.clear()
        return results
    
    def create_sales_report_email(self, dealer_data: Dict, sales_data: Dict, 
                                 recipient_email: str) -> EmailMessage:
        """Create sales report email."""
        return EmailMessage(
            recipients=[EmailRecipient(email=recipient_email, name=dealer_data.get('name'))],
            template_name="sales_report",
            template_data={
                "dealer_name": dealer_data.get('name', 'Dealer'),
                "period": sales_data.get('period', 'This Month'),
                "total_sales": f"Rs. {sales_data.get('total_sales', 0):,.2f}",
                "target": f"Rs. {sales_data.get('target', 0):,.2f}",
                "achievement_percentage": f"{sales_data.get('achievement_percentage', 0):.1f}",
                "top_products": sales_data.get('top_products', [])
            }
        )
    
    def create_target_achievement_email(self, dealer_data: Dict, achievement_data: Dict,
                                      recipient_email: str) -> EmailMessage:
        """Create target achievement congratulations email."""
        return EmailMessage(
            recipients=[EmailRecipient(email=recipient_email, name=dealer_data.get('name'))],
            template_name="target_achievement",
            template_data={
                "dealer_name": dealer_data.get('name', 'Dealer'),
                "achievement_percentage": f"{achievement_data.get('percentage', 0):.1f}",
                "target_amount": f"Rs. {achievement_data.get('target', 0):,.2f}",
                "actual_amount": f"Rs. {achievement_data.get('actual', 0):,.2f}",
                "period": achievement_data.get('period', 'This Month')
            }
        )
    
    def create_route_optimization_email(self, dealer_data: Dict, route_data: Dict,
                                      recipient_email: str) -> EmailMessage:
        """Create route optimization email."""
        return EmailMessage(
            recipients=[EmailRecipient(email=recipient_email, name=dealer_data.get('name'))],
            template_name="route_optimization",
            template_data={
                "dealer_name": dealer_data.get('name', 'Dealer'),
                "date": route_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                "total_customers": route_data.get('customer_count', 0),
                "estimated_distance": f"{route_data.get('distance_km', 0):.1f} km",
                "estimated_time": f"{route_data.get('time_hours', 0):.1f} hours",
                "route_map_url": route_data.get('map_url', '#')
            }
        )
    
    # HTML Template Methods
    def _get_dealer_welcome_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: #007bff; color: white; padding: 20px; text-align: center; }
                .content { padding: 30px 20px; background: #f8f9fa; }
                .footer { padding: 20px; text-align: center; color: #666; font-size: 12px; }
                .btn { background: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to SFA System</h1>
                </div>
                <div class="content">
                    <h2>Hello {{ dealer_name }}!</h2>
                    <p>Welcome to our Sales Force Automation system. Your account has been successfully created.</p>
                    
                    <h3>Your Account Details:</h3>
                    <ul>
                        <li><strong>Username:</strong> {{ username }}</li>
                        <li><strong>Territory:</strong> {{ territory }}</li>
                        <li><strong>Manager:</strong> {{ manager_name }}</li>
                    </ul>
                    
                    <p>You can now login to the system and start managing your sales activities.</p>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{{ login_url }}" class="btn">Login to SFA System</a>
                    </div>
                    
                    <p>If you have any questions, please contact your manager or our support team.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2025 SFA System. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_sales_report_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: #28a745; color: white; padding: 20px; text-align: center; }
                .content { padding: 30px 20px; background: #f8f9fa; }
                .stats { display: flex; justify-content: space-around; margin: 20px 0; }
                .stat-box { background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .stat-value { font-size: 24px; font-weight: bold; color: #007bff; }
                .stat-label { color: #666; font-size: 14px; }
                .progress-bar { background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }
                .progress-fill { background: #28a745; height: 100%; border-radius: 10px; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background: #f1f3f4; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Sales Report</h1>
                    <p>{{ period }} - {{ dealer_name }}</p>
                </div>
                <div class="content">
                    <div class="stats">
                        <div class="stat-box">
                            <div class="stat-value">{{ total_sales }}</div>
                            <div class="stat-label">Total Sales</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{{ target }}</div>
                            <div class="stat-label">Target</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{{ achievement_percentage }}%</div>
                            <div class="stat-label">Achievement</div>
                        </div>
                    </div>
                    
                    <h3>Target Progress</h3>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ achievement_percentage }}%;"></div>
                    </div>
                    
                    <h3>Top Products</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Product</th>
                                <th>Quantity</th>
                                <th>Revenue</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for product in top_products %}
                            <tr>
                                <td>{{ product.name }}</td>
                                <td>{{ product.quantity }}</td>
                                <td>{{ product.revenue }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_target_achievement_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #ffd700, #ffed4e); color: #333; padding: 30px; text-align: center; }
                .trophy { font-size: 48px; margin-bottom: 10px; }
                .content { padding: 30px 20px; background: #f8f9fa; }
                .achievement-box { background: white; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
                .percentage { font-size: 48px; font-weight: bold; color: #28a745; }
                .congratulations { color: #28a745; font-size: 24px; font-weight: bold; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="trophy">🏆</div>
                    <h1>Congratulations!</h1>
                    <p>Target Achievement Alert</p>
                </div>
                <div class="content">
                    <div class="achievement-box">
                        <h2>{{ dealer_name }}</h2>
                        <div class="percentage">{{ achievement_percentage }}%</div>
                        <div class="congratulations">Target Achieved!</div>
                        
                        <p><strong>Target:</strong> {{ target_amount }}</p>
                        <p><strong>Achieved:</strong> {{ actual_amount }}</p>
                        <p><strong>Period:</strong> {{ period }}</p>
                    </div>
                    
                    <p>Excellent work! You have successfully achieved your sales target. Keep up the great performance!</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_low_performance_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: #dc3545; color: white; padding: 20px; text-align: center; }
                .content { padding: 30px 20px; background: #f8f9fa; }
                .alert-box { background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 10px; margin: 20px 0; }
                .performance-meter { background: #e9ecef; height: 30px; border-radius: 15px; overflow: hidden; margin: 15px 0; }
                .performance-fill { background: #dc3545; height: 100%; border-radius: 15px; }
                .tips { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚠️ Performance Alert</h1>
                    <p>{{ dealer_name }}</p>
                </div>
                <div class="content">
                    <div class="alert-box">
                        <h3>Current Performance: {{ achievement_percentage }}%</h3>
                        <div class="performance-meter">
                            <div class="performance-fill" style="width: {{ achievement_percentage }}%;"></div>
                        </div>
                        <p><strong>Target:</strong> {{ target_amount }}</p>
                        <p><strong>Current:</strong> {{ actual_amount }}</p>
                    </div>
                    
                    <div class="tips">
                        <h3>💡 Improvement Suggestions:</h3>
                        <ul>
                            <li>Focus on high-value customers</li>
                            <li>Increase visit frequency</li>
                            <li>Promote premium products</li>
                            <li>Contact your manager for support: {{ manager_name }}</li>
                        </ul>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_route_optimization_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: #17a2b8; color: white; padding: 20px; text-align: center; }
                .content { padding: 30px 20px; background: #f8f9fa; }
                .route-info { display: flex; justify-content: space-around; margin: 20px 0; }
                .info-box { background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .info-value { font-size: 20px; font-weight: bold; color: #17a2b8; }
                .map-section { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; }
                .btn { background: #17a2b8; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📍 Route Optimization</h1>
                    <p>{{ dealer_name }} - {{ date }}</p>
                </div>
                <div class="content">
                    <div class="route-info">
                        <div class="info-box">
                            <div class="info-value">{{ total_customers }}</div>
                            <div>Customers</div>
                        </div>
                        <div class="info-box">
                            <div class="info-value">{{ estimated_distance }}</div>
                            <div>Distance</div>
                        </div>
                        <div class="info-box">
                            <div class="info-value">{{ estimated_time }}</div>
                            <div>Time</div>
                        </div>
                    </div>
                    
                    <div class="map-section">
                        <h3>🗺️ Your Optimized Route</h3>
                        <p>We've calculated the most efficient route for your customer visits today.</p>
                        <a href="{{ route_map_url }}" class="btn">View Route Map</a>
                    </div>
                    
                    <p><strong>Tips for efficient route execution:</strong></p>
                    <ul>
                        <li>Start early to avoid traffic</li>
                        <li>Call customers to confirm appointments</li>
                        <li>Keep track of your actual vs planned time</li>
                        <li>Update customer information during visits</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_visit_reminder_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: #6f42c1; color: white; padding: 20px; text-align: center; }
                .content { padding: 30px 20px; background: #f8f9fa; }
                .reminder-box { background: white; padding: 25px; border-radius: 10px; border-left: 5px solid #6f42c1; margin: 20px 0; }
                .customer-info { background: #e7f3ff; padding: 15px; border-radius: 8px; margin: 15px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📅 Visit Reminder</h1>
                    <p>{{ dealer_name }}</p>
                </div>
                <div class="content">
                    <div class="reminder-box">
                        <h2>{{ customer_name }}</h2>
                        <p><strong>📍 Address:</strong> {{ customer_address }}</p>
                        <p><strong>⏰ Visit Time:</strong> {{ visit_time }}</p>
                        <p><strong>📅 Last Visit:</strong> {{ last_visit_date }}</p>
                    </div>
                    
                    {% if notes %}
                    <div class="customer-info">
                        <h3>📝 Notes:</h3>
                        <p>{{ notes }}</p>
                    </div>
                    {% endif %}
                    
                    <h3>✅ Visit Checklist:</h3>
                    <ul>
                        <li>Review customer's purchase history</li>
                        <li>Prepare product samples/catalogs</li>
                        <li>Check for any pending orders</li>
                        <li>Update customer contact information</li>
                        <li>Record visit notes in the system</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_maintenance_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: #fd7e14; color: white; padding: 20px; text-align: center; }
                .content { padding: 30px 20px; background: #f8f9fa; }
                .maintenance-box { background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 10px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔧 System Maintenance</h1>
                </div>
                <div class="content">
                    <div class="maintenance-box">
                        <h3>Scheduled Maintenance Window</h3>
                        <p><strong>Start:</strong> {{ start_time }}</p>
                        <p><strong>End:</strong> {{ end_time }}</p>
                        <p><strong>Description:</strong> {{ description }}</p>
                        <p><strong>Expected Impact:</strong> {{ impact }}</p>
                    </div>
                    
                    <p>We apologize for any inconvenience this may cause. The system will be back online as soon as maintenance is completed.</p>
                    
                    <h3>📱 What you can do during maintenance:</h3>
                    <ul>
                        <li>Continue with your scheduled customer visits</li>
                        <li>Use offline mode features if available</li>
                        <li>Plan your next day's activities</li>
                        <li>Update your customer notes manually</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_password_reset_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: #dc3545; color: white; padding: 20px; text-align: center; }
                .content { padding: 30px 20px; background: #f8f9fa; }
                .security-box { background: white; padding: 25px; border-radius: 10px; border-left: 5px solid #dc3545; margin: 20px 0; }
                .btn { background: #dc3545; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; }
                .warning { background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; margin: 15px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔒 Password Reset</h1>
                </div>
                <div class="content">
                    <div class="security-box">
                        <h2>Hello {{ user_name }},</h2>
                        <p>We received a request to reset your password for the SFA System.</p>
                        
                        <div style="text-align: center; margin: 30px 0;">
                            <a href="{{ reset_url }}" class="btn">Reset Password</a>
                        </div>
                        
                        <p><strong>This link expires at:</strong> {{ expiry_time }}</p>
                    </div>
                    
                    <div class="warning">
                        <strong>Security Notice:</strong> If you didn't request this password reset, please ignore this email or contact our support team immediately.
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

# Email notification service for SFA events
class SFAEmailNotifications:
    """High-level email notification service for SFA system events."""
    
    def __init__(self, email_service: EmailService):
        self.email_service = email_service
    
    def send_daily_sales_summary(self, dealers: List[Dict], date: str) -> Dict[str, int]:
        """Send daily sales summary to all dealers."""
        messages = []
        
        for dealer in dealers:
            if dealer.get('email'):
                message = self.email_service.create_sales_report_email(
                    dealer_data=dealer,
                    sales_data={
                        'period': f'Daily Summary - {date}',
                        'total_sales': dealer.get('daily_sales', 0),
                        'target': dealer.get('daily_target', 0),
                        'achievement_percentage': dealer.get('daily_achievement', 0),
                        'top_products': dealer.get('top_products', [])
                    },
                    recipient_email=dealer['email']
                )
                messages.append(message)
        
        return self.email_service.send_bulk_emails(messages)
    
    def send_target_achievement_alerts(self, dealers: List[Dict]) -> Dict[str, int]:
        """Send target achievement alerts to qualifying dealers."""
        messages = []
        
        for dealer in dealers:
            achievement = dealer.get('achievement_percentage', 0)
            if achievement >= 100 and dealer.get('email'):
                message = self.email_service.create_target_achievement_email(
                    dealer_data=dealer,
                    achievement_data={
                        'percentage': achievement,
                        'target': dealer.get('target', 0),
                        'actual': dealer.get('actual_sales', 0),
                        'period': dealer.get('period', 'This Month')
                    },
                    recipient_email=dealer['email']
                )
                messages.append(message)
        
        return self.email_service.send_bulk_emails(messages)
    
    def send_route_plans(self, route_plans: List[Dict]) -> Dict[str, int]:
        """Send optimized route plans to dealers."""
        messages = []
        
        for plan in route_plans:
            dealer = plan.get('dealer', {})
            if dealer.get('email'):
                message = self.email_service.create_route_optimization_email(
                    dealer_data=dealer,
                    route_data=plan.get('route_data', {}),
                    recipient_email=dealer['email']
                )
                messages.append(message)
        
        return self.email_service.send_bulk_emails(messages)

# Email analytics and tracking
class EmailAnalytics:
    """Email analytics and tracking functionality."""
    
    def __init__(self):
        self.email_logs = []
    
    def log_email_sent(self, recipient: str, template: str, status: str, timestamp: datetime = None):
        """Log email sending activity."""
        self.email_logs.append({
            'recipient': recipient,
            'template': template,
            'status': status,
            'timestamp': timestamp or datetime.now(),
            'id': len(self.email_logs) + 1
        })
    
    def get_email_statistics(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """Get email sending statistics."""
        filtered_logs = self.email_logs
        
        if start_date or end_date:
            filtered_logs = [
                log for log in self.email_logs
                if (not start_date or log['timestamp'] >= start_date) and
                   (not end_date or log['timestamp'] <= end_date)
            ]
        
        total_emails = len(filtered_logs)
        successful = len([log for log in filtered_logs if log['status'] == 'success'])
        failed = total_emails - successful
        
        # Template usage stats
        template_stats = {}
        for log in filtered_logs:
            template = log['template']
            template_stats[template] = template_stats.get(template, 0) + 1
        
        return {
            'total_emails': total_emails,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total_emails * 100) if total_emails > 0 else 0,
            'template_usage': template_stats,
            'period': {
                'start': min(log['timestamp'] for log in filtered_logs) if filtered_logs else None,
                'end': max(log['timestamp'] for log in filtered_logs) if filtered_logs else None
            }
        }
    
    def get_recipient_engagement(self, recipient: str) -> Dict:
        """Get engagement statistics for a specific recipient."""
        recipient_logs = [log for log in self.email_logs if log['recipient'] == recipient]
        
        if not recipient_logs:
            return {}
        
        return {
            'total_emails_received': len(recipient_logs),
            'successful_deliveries': len([log for log in recipient_logs if log['status'] == 'success']),
            'failed_deliveries': len([log for log in recipient_logs if log['status'] == 'failed']),
            'first_email': min(log['timestamp'] for log in recipient_logs),
            'last_email': max(log['timestamp'] for log in recipient_logs),
            'templates_received': list(set(log['template'] for log in recipient_logs))
        }