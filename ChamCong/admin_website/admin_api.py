"""
Admin API Backend for AI Attendance System
Provides REST API endpoints for admin website functionality
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, asdict
import os
import sys

# Add the parent directory to the path to import AI services
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ai_services.gps_antifraud_service import GPSAntiFraudService, LocationData
    from ai_services.anomaly_detection_service import AnomalyDetectionService, AttendanceRecord
    from ai_services.predictive_analytics_service import PredictiveAnalyticsService, EmployeeProfile, AttendanceMetrics
except ImportError as e:
    print(f"Warning: Could not import AI services: {e}")
    # Create mock services for development
    class MockService:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: {}
    
    GPSAntiFraudService = MockService
    AnomalyDetectionService = MockService
    PredictiveAnalyticsService = MockService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Attendance Admin API",
    description="Admin API for AI-powered attendance management system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="admin_website"), name="static")

# Initialize AI services
gps_service = GPSAntiFraudService()
anomaly_service = AnomalyDetectionService()
predictive_service = PredictiveAnalyticsService()

# In-memory storage (in production, use a database)
employees_db = {}
checkins_db = {}
notifications_db = []
system_settings = {
    "face_threshold": 0.6,
    "liveness_sensitivity": 0.8,
    "anomaly_sensitivity": 0.6,
    "notifications": {
        "email": True,
        "sms": True,
        "slack": False
    }
}

# Pydantic models
class EmployeeBase(BaseModel):
    employee_id: str
    name: str
    department: str
    position: str
    age: int
    tenure_months: int
    salary_level: str
    education_level: str
    performance_rating: float
    manager_id: str
    team_size: int

class Employee(EmployeeBase):
    photo_url: Optional[str] = None
    status: str = "absent"
    last_checkin: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class CheckInBase(BaseModel):
    employee_id: str
    latitude: float
    longitude: float
    accuracy: float
    device_type: str
    wifi_data: Optional[List[Dict]] = None
    nfc_data: Optional[Dict] = None

class CheckIn(CheckInBase):
    checkin_id: str
    checkin_time: datetime = Field(default_factory=datetime.now)
    confidence_score: float
    is_fraud: bool
    geofence_valid: bool
    valid_geofences: List[str]
    liveness_passed: bool
    face_recognized: bool
    anomaly_detected: bool
    anomaly_details: Optional[Dict] = None

class Notification(BaseModel):
    id: str
    type: str  # success, error, warning, info
    title: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    read: bool = False

class DashboardStats(BaseModel):
    total_employees: int
    today_checkins: int
    anomalies_detected: int
    attendance_rate: float
    present_count: int
    late_count: int
    absent_count: int
    remote_count: int

class ReportRequest(BaseModel):
    report_type: str
    start_date: datetime
    end_date: datetime
    employee_ids: Optional[List[str]] = None
    departments: Optional[List[str]] = None

class SystemSettings(BaseModel):
    face_threshold: float
    liveness_sensitivity: float
    anomaly_sensitivity: float
    notifications: Dict[str, bool]

# Authentication (simplified for demo)
def get_current_admin():
    # In production, implement proper JWT authentication
    return {
        "id": "admin",
        "name": "Admin User",
        "role": "System Administrator"
    }

# Utility functions
def generate_id(prefix: str) -> str:
    return f"{prefix}{datetime.now().strftime('%Y%m%d%H%M%S')}{hash(datetime.now()) % 10000:04d}"

def load_sample_data():
    """Load sample data for demonstration"""
    global employees_db, checkins_db, notifications_db
    
    # Sample employees
    sample_employees = [
        {
            "employee_id": "EMP001",
            "name": "John Doe",
            "department": "Engineering",
            "position": "Senior Developer",
            "age": 32,
            "tenure_months": 36,
            "salary_level": "senior",
            "education_level": "bachelor",
            "performance_rating": 4.5,
            "manager_id": "MGR001",
            "team_size": 8,
            "photo_url": "https://via.placeholder.com/150x150",
            "status": "present",
            "last_checkin": datetime.now() - timedelta(hours=2),
            "created_at": datetime.now() - timedelta(days=365),
            "updated_at": datetime.now() - timedelta(days=30)
        },
        {
            "employee_id": "EMP002",
            "name": "Jane Smith",
            "department": "Sales",
            "position": "Sales Manager",
            "age": 28,
            "tenure_months": 24,
            "salary_level": "mid",
            "education_level": "bachelor",
            "performance_rating": 4.2,
            "manager_id": "MGR002",
            "team_size": 5,
            "photo_url": "https://via.placeholder.com/150x150",
            "status": "late",
            "last_checkin": datetime.now() - timedelta(hours=1),
            "created_at": datetime.now() - timedelta(days=300),
            "updated_at": datetime.now() - timedelta(days=15)
        },
        {
            "employee_id": "EMP003",
            "name": "Mike Johnson",
            "department": "HR",
            "position": "HR Specialist",
            "age": 35,
            "tenure_months": 48,
            "salary_level": "mid",
            "education_level": "master",
            "performance_rating": 4.0,
            "manager_id": "MGR003",
            "team_size": 3,
            "photo_url": "https://via.placeholder.com/150x150",
            "status": "remote",
            "last_checkin": datetime.now() - timedelta(minutes=30),
            "created_at": datetime.now() - timedelta(days=400),
            "updated_at": datetime.now() - timedelta(days=7)
        },
        {
            "employee_id": "EMP004",
            "name": "Sarah Wilson",
            "department": "Finance",
            "position": "Financial Analyst",
            "age": 29,
            "tenure_months": 18,
            "salary_level": "mid",
            "education_level": "master",
            "performance_rating": 4.3,
            "manager_id": "MGR004",
            "team_size": 4,
            "photo_url": "https://via.placeholder.com/150x150",
            "status": "absent",
            "last_checkin": datetime.now() - timedelta(days=1),
            "created_at": datetime.now() - timedelta(days=200),
            "updated_at": datetime.now() - timedelta(days=1)
        }
    ]
    
    for emp in sample_employees:
        employees_db[emp["employee_id"]] = Employee(**emp)
    
    # Sample check-ins
    sample_checkins = [
        {
            "checkin_id": "CHK001",
            "employee_id": "EMP001",
            "latitude": 10.762622,
            "longitude": 106.660172,
            "accuracy": 5.0,
            "device_type": "camera",
            "checkin_time": datetime.now() - timedelta(hours=2),
            "confidence_score": 0.95,
            "is_fraud": False,
            "geofence_valid": True,
            "valid_geofences": ["Main Office"],
            "liveness_passed": True,
            "face_recognized": True,
            "anomaly_detected": False
        },
        {
            "checkin_id": "CHK002",
            "employee_id": "EMP002",
            "latitude": 10.762622,
            "longitude": 106.660172,
            "accuracy": 8.0,
            "device_type": "mobile",
            "checkin_time": datetime.now() - timedelta(hours=1),
            "confidence_score": 0.87,
            "is_fraud": False,
            "geofence_valid": True,
            "valid_geofences": ["Main Office"],
            "liveness_passed": True,
            "face_recognized": True,
            "anomaly_detected": False
        },
        {
            "checkin_id": "CHK003",
            "employee_id": "EMP003",
            "latitude": 10.8231,
            "longitude": 106.6297,
            "accuracy": 15.0,
            "device_type": "mobile",
            "checkin_time": datetime.now() - timedelta(minutes=30),
            "confidence_score": 0.92,
            "is_fraud": False,
            "geofence_valid": True,
            "valid_geofences": ["Remote Work Zone"],
            "liveness_passed": True,
            "face_recognized": True,
            "anomaly_detected": False
        }
    ]
    
    for chk in sample_checkins:
        checkins_db[chk["checkin_id"]] = CheckIn(**chk)
    
    # Sample notifications
    sample_notifications = [
        {
            "id": "NOT001",
            "type": "warning",
            "title": "Anomaly Detected",
            "message": "Unusual check-in pattern detected for EMP004",
            "timestamp": datetime.now() - timedelta(minutes=15),
            "read": False
        },
        {
            "id": "NOT002",
            "type": "info",
            "title": "New Employee Registered",
            "message": "Sarah Wilson has been successfully registered",
            "timestamp": datetime.now() - timedelta(minutes=45),
            "read": False
        },
        {
            "id": "NOT003",
            "type": "success",
            "title": "System Update",
            "message": "AI models have been successfully retrained",
            "timestamp": datetime.now() - timedelta(hours=2),
            "read": True
        }
    ]
    
    notifications_db.extend([Notification(**notif) for notif in sample_notifications])

# API Endpoints

@app.get("/")
async def root():
    """Serve the admin website"""
    return {"message": "AI Attendance Admin API", "docs": "/docs"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gps_antifraud": "healthy",
            "anomaly_detection": "healthy",
            "predictive_analytics": "healthy"
        }
    }

# Dashboard endpoints
@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(admin: dict = Depends(get_current_admin)):
    """Get dashboard statistics"""
    try:
        today = datetime.now().date()
        today_checkins = [
            chk for chk in checkins_db.values() 
            if chk.checkin_time.date() == today
        ]
        
        anomalies = [chk for chk in checkins_db.values() if chk.anomaly_detected]
        
        status_counts = {"present": 0, "late": 0, "absent": 0, "remote": 0}
        for emp in employees_db.values():
            if emp.status in status_counts:
                status_counts[emp.status] += 1
        
        attendance_rate = (len(today_checkins) / len(employees_db) * 100) if employees_db else 0
        
        return DashboardStats(
            total_employees=len(employees_db),
            today_checkins=len(today_checkins),
            anomalies_detected=len(anomalies),
            attendance_rate=round(attendance_rate, 1),
            present_count=status_counts["present"],
            late_count=status_counts["late"],
            absent_count=status_counts["absent"],
            remote_count=status_counts["remote"]
        )
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/recent-checkins")
async def get_recent_checkins(limit: int = Query(10, ge=1, le=50)):
    """Get recent check-ins for dashboard"""
    try:
        recent_checkins = sorted(
            checkins_db.values(),
            key=lambda x: x.checkin_time,
            reverse=True
        )[:limit]
        
        return [
            {
                "checkin_id": chk.checkin_id,
                "employee_id": chk.employee_id,
                "employee_name": employees_db.get(chk.employee_id, {}).get("name", "Unknown"),
                "checkin_time": chk.checkin_time,
                "location": chk.valid_geofences[0] if chk.valid_geofences else "Unknown",
                "device": chk.device_type,
                "confidence": chk.confidence_score,
                "status": "present" if chk.geofence_valid else "invalid"
            }
            for chk in recent_checkins
        ]
    except Exception as e:
        logger.error(f"Error getting recent check-ins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Employee management endpoints
@app.get("/api/employees")
async def get_employees(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    department: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """Get list of employees with filtering and pagination"""
    try:
        employees_list = list(employees_db.values())
        
        # Apply filters
        if department:
            employees_list = [emp for emp in employees_list if emp.department == department]
        if status:
            employees_list = [emp for emp in employees_list if emp.status == status]
        
        # Apply pagination
        total = len(employees_list)
        employees_list = employees_list[skip:skip + limit]
        
        return {
            "employees": [asdict(emp) for emp in employees_list],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/employees")
async def create_employee(
    employee_id: str = Form(...),
    name: str = Form(...),
    department: str = Form(...),
    position: str = Form(...),
    age: int = Form(...),
    tenure_months: int = Form(0),
    salary_level: str = Form(...),
    education_level: str = Form(...),
    performance_rating: float = Form(4.0),
    manager_id: str = Form(""),
    team_size: int = Form(1),
    face_image: UploadFile = File(...),
    admin: dict = Depends(get_current_admin)
):
    """Create a new employee"""
    try:
        if employee_id in employees_db:
            raise HTTPException(status_code=400, detail="Employee ID already exists")
        
        # Save uploaded image (simplified)
        photo_url = f"/static/photos/{employee_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        
        # Create employee record
        employee_data = {
            "employee_id": employee_id,
            "name": name,
            "department": department,
            "position": position,
            "age": age,
            "tenure_months": tenure_months,
            "salary_level": salary_level,
            "education_level": education_level,
            "performance_rating": performance_rating,
            "manager_id": manager_id,
            "team_size": team_size,
            "photo_url": photo_url,
            "status": "absent",
            "last_checkin": None
        }
        
        employee = Employee(**employee_data)
        employees_db[employee_id] = employee
        
        # Add notification
        notification = Notification(
            id=generate_id("NOT"),
            type="success",
            title="New Employee Added",
            message=f"{name} has been successfully added to the system",
            timestamp=datetime.now()
        )
        notifications_db.append(notification)
        
        logger.info(f"Created employee: {employee_id}")
        return {"message": "Employee created successfully", "employee_id": employee_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating employee: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/employees/{employee_id}")
async def update_employee(
    employee_id: str,
    employee_data: EmployeeBase,
    admin: dict = Depends(get_current_admin)
):
    """Update an employee"""
    try:
        if employee_id not in employees_db:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        existing_employee = employees_db[employee_id]
        
        # Update fields
        update_data = employee_data.dict()
        update_data["updated_at"] = datetime.now()
        update_data["photo_url"] = existing_employee.photo_url
        update_data["status"] = existing_employee.status
        update_data["last_checkin"] = existing_employee.last_checkin
        update_data["created_at"] = existing_employee.created_at
        
        employees_db[employee_id] = Employee(**update_data)
        
        logger.info(f"Updated employee: {employee_id}")
        return {"message": "Employee updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating employee: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/employees/{employee_id}")
async def delete_employee(
    employee_id: str,
    admin: dict = Depends(get_current_admin)
):
    """Delete an employee"""
    try:
        if employee_id not in employees_db:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        employee_name = employees_db[employee_id].name
        del employees_db[employee_id]
        
        # Add notification
        notification = Notification(
            id=generate_id("NOT"),
            type="warning",
            title="Employee Deleted",
            message=f"{employee_name} has been removed from the system",
            timestamp=datetime.now()
        )
        notifications_db.append(notification)
        
        logger.info(f"Deleted employee: {employee_id}")
        return {"message": "Employee deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting employee: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Check-in endpoints
@app.get("/api/checkins")
async def get_checkins(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    employee_id: Optional[str] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    anomalies_only: bool = Query(False)
):
    """Get check-in records with filtering and pagination"""
    try:
        checkins_list = list(checkins_db.values())
        
        # Apply filters
        if employee_id:
            checkins_list = [chk for chk in checkins_list if chk.employee_id == employee_id]
        if date_from:
            checkins_list = [chk for chk in checkins_list if chk.checkin_time >= date_from]
        if date_to:
            checkins_list = [chk for chk in checkins_list if chk.checkin_time <= date_to]
        if anomalies_only:
            checkins_list = [chk for chk in checkins_list if chk.anomaly_detected]
        
        # Sort by check-in time (newest first)
        checkins_list.sort(key=lambda x: x.checkin_time, reverse=True)
        
        # Apply pagination
        total = len(checkins_list)
        checkins_list = checkins_list[skip:skip + limit]
        
        # Add employee information
        result = []
        for chk in checkins_list:
            employee = employees_db.get(chk.employee_id)
            result.append({
                "checkin_id": chk.checkin_id,
                "employee_id": chk.employee_id,
                "employee_name": employee.name if employee else "Unknown",
                "checkin_time": chk.checkin_time,
                "location": chk.valid_geofences[0] if chk.valid_geofences else "Unknown",
                "device_type": chk.device_type,
                "confidence_score": chk.confidence_score,
                "is_fraud": chk.is_fraud,
                "geofence_valid": chk.geofence_valid,
                "liveness_passed": chk.liveness_passed,
                "face_recognized": chk.face_recognized,
                "anomaly_detected": chk.anomaly_detected,
                "anomaly_details": chk.anomaly_details
            })
        
        return {
            "checkins": result,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting check-ins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/checkins")
async def create_checkin(
    checkin_data: CheckInBase,
    admin: dict = Depends(get_current_admin)
):
    """Create a new check-in record"""
    try:
        if checkin_data.employee_id not in employees_db:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Create location data for GPS validation
        location = LocationData(
            latitude=checkin_data.latitude,
            longitude=checkin_data.longitude,
            accuracy=checkin_data.accuracy,
            timestamp=datetime.now()
        )
        
        # Perform GPS validation
        gps_validation = gps_service.comprehensive_location_validation(
            checkin_data.employee_id, location
        )
        
        # Create check-in record
        checkin_id = generate_id("CHK")
        checkin = CheckIn(
            checkin_id=checkin_id,
            **checkin_data.dict(),
            confidence_score=gps_validation.get('fraud_score', 0.8),
            is_fraud=gps_validation.get('is_fraud', False),
            geofence_valid=gps_validation.get('geofence_valid', False),
            valid_geofences=gps_validation.get('valid_geofences', []),
            liveness_passed=True,  # Would be determined by liveness detection
            face_recognized=True,  # Would be determined by face recognition
            anomaly_detected=False  # Would be determined by anomaly detection
        )
        
        checkins_db[checkin_id] = checkin
        
        # Update employee status
        employee = employees_db[checkin_data.employee_id]
        employee.last_checkin = checkin.checkin_time
        employee.status = "present"
        employee.updated_at = datetime.now()
        
        logger.info(f"Created check-in: {checkin_id}")
        return {"message": "Check-in recorded successfully", "checkin_id": checkin_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating check-in: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Notification endpoints
@app.get("/api/notifications")
async def get_notifications(
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200)
):
    """Get notifications"""
    try:
        notifications_list = notifications_db.copy()
        
        if unread_only:
            notifications_list = [n for n in notifications_list if not n.read]
        
        # Sort by timestamp (newest first)
        notifications_list.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        notifications_list = notifications_list[:limit]
        
        return [asdict(n) for n in notifications_list]
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """Mark a notification as read"""
    try:
        for notification in notifications_db:
            if notification.id == notification_id:
                notification.read = True
                return {"message": "Notification marked as read"}
        
        raise HTTPException(status_code=404, detail="Notification not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/notifications/read-all")
async def mark_all_notifications_read():
    """Mark all notifications as read"""
    try:
        for notification in notifications_db:
            notification.read = True
        
        return {"message": "All notifications marked as read"}
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Reports endpoints
@app.post("/api/reports")
async def generate_report(
    report_request: ReportRequest,
    admin: dict = Depends(get_current_admin)
):
    """Generate attendance report"""
    try:
        # Filter check-ins based on request
        filtered_checkins = []
        for chk in checkins_db.values():
            if (report_request.start_date <= chk.checkin_time <= report_request.end_date):
                if report_request.employee_ids and chk.employee_id not in report_request.employee_ids:
                    continue
                if report_request.departments:
                    employee = employees_db.get(chk.employee_id)
                    if not employee or employee.department not in report_request.departments:
                        continue
                filtered_checkins.append(chk)
        
        # Generate report data
        report_data = {
            "report_type": report_request.report_type,
            "period": {
                "start_date": report_request.start_date,
                "end_date": report_request.end_date
            },
            "total_records": len(filtered_checkins),
            "summary": {
                "total_employees": len(set(chk.employee_id for chk in filtered_checkins)),
                "anomalies_detected": len([chk for chk in filtered_checkins if chk.anomaly_detected]),
                "fraud_attempts": len([chk for chk in filtered_checkins if chk.is_fraud]),
                "average_confidence": sum(chk.confidence_score for chk in filtered_checkins) / len(filtered_checkins) if filtered_checkins else 0
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return report_data
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Settings endpoints
@app.get("/api/settings")
async def get_settings(admin: dict = Depends(get_current_admin)):
    """Get system settings"""
    return system_settings

@app.put("/api/settings")
async def update_settings(
    settings: SystemSettings,
    admin: dict = Depends(get_current_admin)
):
    """Update system settings"""
    try:
        global system_settings
        system_settings.update(settings.dict())
        
        logger.info("System settings updated")
        return {"message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/api/analytics/turnover-risk")
async def get_turnover_risk_analysis():
    """Get turnover risk analysis"""
    try:
        risk_analysis = []
        
        for employee in employees_db.values():
            # Create attendance metrics
            employee_checkins = [
                chk for chk in checkins_db.values() 
                if chk.employee_id == employee.employee_id
            ]
            
            if len(employee_checkins) < 10:
                continue
            
            metrics = AttendanceMetrics(
                employee_id=employee.employee_id,
                total_work_days=len(employee_checkins),
                total_late_days=len([chk for chk in employee_checkins if chk.checkin_time.hour > 9]),
                total_absent_days=0,  # Would be calculated from schedule
                avg_work_hours=8.0,
                avg_overtime_hours=1.0,
                punctuality_score=0.85,
                attendance_consistency=0.90,
                last_30_days_attendance=0.88,
                last_30_days_lateness=5.0
            )
            
            # Create employee profile
            profile = EmployeeProfile(
                employee_id=employee.employee_id,
                age=employee.age,
                department=employee.department,
                position=employee.position,
                tenure_months=employee.tenure_months,
                salary_level=employee.salary_level,
                education_level=employee.education_level,
                performance_rating=employee.performance_rating,
                manager_id=employee.manager_id,
                team_size=employee.team_size
            )
            
            # Predict turnover risk
            turnover_prediction = predictive_service.predict_turnover(profile, metrics)
            
            risk_analysis.append({
                "employee_id": employee.employee_id,
                "employee_name": employee.name,
                "department": employee.department,
                "turnover_probability": turnover_prediction.turnover_probability,
                "risk_level": turnover_prediction.risk_level,
                "key_factors": turnover_prediction.key_factors,
                "recommendations": turnover_prediction.recommendations
            })
        
        return {"turnover_risk_analysis": risk_analysis}
    except Exception as e:
        logger.error(f"Error getting turnover risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/anomalies")
async def get_anomaly_analysis():
    """Get anomaly analysis"""
    try:
        anomaly_analysis = []
        
        for employee in employees_db.values():
            employee_checkins = [
                chk for chk in checkins_db.values() 
                if chk.employee_id == employee.employee_id
            ]
            
            if len(employee_checkins) < 5:
                continue
            
            # Analyze recent check-ins for anomalies
            recent_checkins = sorted(employee_checkins, key=lambda x: x.checkin_time, reverse=True)[:10]
            
            anomalies = []
            for chk in recent_checkins:
                if chk.anomaly_detected:
                    anomalies.append({
                        "checkin_id": chk.checkin_id,
                        "checkin_time": chk.checkin_time,
                        "anomaly_details": chk.anomaly_details
                    })
            
            if anomalies:
                anomaly_analysis.append({
                    "employee_id": employee.employee_id,
                    "employee_name": employee.name,
                    "department": employee.department,
                    "anomaly_count": len(anomalies),
                    "recent_anomalies": anomalies
                })
        
        return {"anomaly_analysis": anomaly_analysis}
    except Exception as e:
        logger.error(f"Error getting anomaly analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize sample data
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting AI Attendance Admin API...")
    load_sample_data()
    logger.info(f"Loaded {len(employees_db)} employees, {len(checkins_db)} check-ins, {len(notifications_db)} notifications")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
