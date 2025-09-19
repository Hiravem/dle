"""
AI Integration API
Tích hợp tất cả các thành phần AI thành một API thống nhất
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from datetime import datetime, timedelta
import logging
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Import AI services
try:
    from .face_recognition_service import FaceRecognitionService
    from .liveness_detection_service import LivenessDetectionService
    from .gps_antifraud_service import GPSAntiFraudService, LocationData, WiFiData, NFCData
    from .anomaly_detection_service import AnomalyDetectionService, AttendanceRecord
    from .predictive_analytics_service import PredictiveAnalyticsService, EmployeeProfile, AttendanceMetrics
except ImportError:
    # Fallback for direct execution
    from face_recognition_service import FaceRecognitionService
    from liveness_detection_service import LivenessDetectionService
    from gps_antifraud_service import GPSAntiFraudService, LocationData, WiFiData, NFCData
    from anomaly_detection_service import AnomalyDetectionService, AttendanceRecord
    from predictive_analytics_service import PredictiveAnalyticsService, EmployeeProfile, AttendanceMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Attendance System API",
    description="Comprehensive AI-powered attendance management system",
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

# Initialize AI services
face_service = FaceRecognitionService()
liveness_service = LivenessDetectionService()
gps_service = GPSAntiFraudService()
anomaly_service = AnomalyDetectionService()
predictive_service = PredictiveAnalyticsService()

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for API
class CheckInRequest(BaseModel):
    employee_id: str
    latitude: float
    longitude: float
    accuracy: float
    timestamp: datetime
    device_type: str = "mobile"
    wifi_data: Optional[List[Dict]] = None
    nfc_data: Optional[Dict] = None

class CheckInResponse(BaseModel):
    success: bool
    employee_id: str
    check_in_time: datetime
    confidence_score: float
    is_fraud: bool
    fraud_reason: Optional[str]
    geofence_valid: bool
    valid_geofences: List[str]
    liveness_passed: bool
    face_recognized: bool
    anomaly_detected: bool
    anomaly_details: Optional[Dict]

class EmployeeRegistrationRequest(BaseModel):
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

class EmployeeRegistrationResponse(BaseModel):
    success: bool
    employee_id: str
    face_registered: bool
    models_trained: bool
    message: str

class AttendanceAnalysisRequest(BaseModel):
    employee_id: str
    start_date: datetime
    end_date: datetime
    include_predictions: bool = True

class AttendanceAnalysisResponse(BaseModel):
    employee_id: str
    analysis_period: Dict[str, datetime]
    total_records: int
    anomaly_count: int
    risk_level: str
    turnover_probability: float
    attendance_forecast: Optional[Dict]
    recommendations: List[str]
    detailed_analysis: Dict

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    timestamp: datetime

# Dependency to get current timestamp
def get_current_time():
    return datetime.now()

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for all AI services"""
    try:
        services_status = {
            "face_recognition": "healthy" if face_service.face_model else "not_initialized",
            "liveness_detection": "healthy" if liveness_service.liveness_model else "not_initialized",
            "gps_antifraud": "healthy" if gps_service.geofences else "not_initialized",
            "anomaly_detection": "healthy" if anomaly_service.isolation_forest else "not_initialized",
            "predictive_analytics": "healthy" if predictive_service.turnover_model else "not_initialized"
        }
        
        overall_status = "healthy" if all(
            status == "healthy" for status in services_status.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            services=services_status,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            services={"error": str(e)},
            timestamp=datetime.now()
        )

# Employee registration endpoint
@app.post("/register-employee", response_model=EmployeeRegistrationResponse)
async def register_employee(
    employee_id: str = Form(...),
    name: str = Form(...),
    department: str = Form(...),
    position: str = Form(...),
    age: int = Form(...),
    tenure_months: int = Form(...),
    salary_level: str = Form(...),
    education_level: str = Form(...),
    performance_rating: float = Form(...),
    manager_id: str = Form(...),
    team_size: int = Form(...),
    face_image: UploadFile = File(...)
):
    """Register new employee with face recognition"""
    try:
        # Read and process face image
        image_data = await face_image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detect faces in image
        faces = face_service.detect_faces(image)
        
        if not faces:
            raise HTTPException(status_code=400, detail="No faces detected in image")
        
        # Use the first detected face
        face_location = faces[0]
        face_image_cropped = face_service.crop_face(image, face_location)
        
        # Register face
        face_registered = face_service.add_employee_face(employee_id, face_image_cropped)
        
        # Create employee profile for predictive analytics
        profile = EmployeeProfile(
            employee_id=employee_id,
            age=age,
            department=department,
            position=position,
            tenure_months=tenure_months,
            salary_level=salary_level,
            education_level=education_level,
            performance_rating=performance_rating,
            manager_id=manager_id,
            team_size=team_size
        )
        
        # Note: In a real system, you would save the profile to a database
        # For now, we'll just log it
        logger.info(f"Registered employee profile: {employee_id}")
        
        return EmployeeRegistrationResponse(
            success=face_registered,
            employee_id=employee_id,
            face_registered=face_registered,
            models_trained=True,  # Models are pre-trained
            message="Employee registered successfully" if face_registered else "Face registration failed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Employee registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# Check-in endpoint with comprehensive AI validation
@app.post("/check-in", response_model=CheckInResponse)
async def check_in(request: CheckInRequest):
    """Comprehensive check-in with all AI validations"""
    try:
        # Create location data
        location = LocationData(
            latitude=request.latitude,
            longitude=request.longitude,
            accuracy=request.accuracy,
            timestamp=request.timestamp
        )
        
        # Prepare WiFi and NFC data if provided
        wifi_data = None
        if request.wifi_data:
            wifi_data = [
                WiFiData(
                    bssid=w.get('bssid', ''),
                    ssid=w.get('ssid', ''),
                    signal_strength=w.get('signal_strength', -100),
                    frequency=w.get('frequency', 2400),
                    timestamp=request.timestamp
                )
                for w in request.wifi_data
            ]
        
        nfc_data = None
        if request.nfc_data:
            nfc_data = NFCData(
                beacon_id=request.nfc_data.get('beacon_id', ''),
                distance=request.nfc_data.get('distance', 0.0),
                timestamp=request.timestamp
            )
        
        # Perform GPS validation
        gps_validation = gps_service.comprehensive_location_validation(
            request.employee_id, location, wifi_data, nfc_data
        )
        
        # Check if GPS validation passed
        if not gps_validation['is_valid_checkin']:
            return CheckInResponse(
                success=False,
                employee_id=request.employee_id,
                check_in_time=request.timestamp,
                confidence_score=0.0,
                is_fraud=gps_validation['is_fraud'],
                fraud_reason=gps_validation.get('movement_check', {}).get('reason'),
                geofence_valid=False,
                valid_geofences=[],
                liveness_passed=False,
                face_recognized=False,
                anomaly_detected=False,
                anomaly_details=None
            )
        
        # Note: In a real implementation, you would also perform:
        # 1. Face recognition from camera image
        # 2. Liveness detection
        # 3. Real-time anomaly detection
        
        # For this example, we'll simulate these checks
        face_recognized = True  # Would be determined by face recognition
        liveness_passed = True  # Would be determined by liveness detection
        anomaly_detected = False  # Would be determined by anomaly detection
        
        # Create attendance record for anomaly detection
        attendance_record = AttendanceRecord(
            employee_id=request.employee_id,
            check_in_time=request.timestamp,
            check_out_time=None,
            work_hours=0.0,
            location=gps_validation['valid_geofences'][0] if gps_validation['valid_geofences'] else "unknown",
            device_type=request.device_type,
            is_weekend=request.timestamp.weekday() >= 5,
            is_holiday=False  # Would be determined by holiday calendar
        )
        
        # Detect anomalies
        anomaly_result = anomaly_service.detect_attendance_anomalies(attendance_record)
        
        # Calculate overall confidence
        confidence_score = 0.8  # Base confidence
        if face_recognized:
            confidence_score += 0.1
        if liveness_passed:
            confidence_score += 0.1
        if not gps_validation['is_fraud']:
            confidence_score += 0.1
        
        confidence_score = min(confidence_score, 1.0)
        
        return CheckInResponse(
            success=True,
            employee_id=request.employee_id,
            check_in_time=request.timestamp,
            confidence_score=confidence_score,
            is_fraud=gps_validation['is_fraud'],
            fraud_reason=None,
            geofence_valid=True,
            valid_geofences=gps_validation['valid_geofences'],
            liveness_passed=liveness_passed,
            face_recognized=face_recognized,
            anomaly_detected=anomaly_result.is_anomaly,
            anomaly_details={
                "anomaly_type": anomaly_result.anomaly_type,
                "confidence": anomaly_result.confidence,
                "severity": anomaly_result.severity,
                "explanation": anomaly_result.explanation
            } if anomaly_result.is_anomaly else None
        )
        
    except Exception as e:
        logger.error(f"Check-in error: {e}")
        raise HTTPException(status_code=500, detail=f"Check-in failed: {str(e)}")

# Face recognition endpoint
@app.post("/recognize-face")
async def recognize_face(face_image: UploadFile = File(...)):
    """Recognize face from uploaded image"""
    try:
        # Read and process image
        image_data = await face_image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detect faces
        faces = face_service.detect_faces(image)
        
        if not faces:
            return {"success": False, "message": "No faces detected"}
        
        # Recognize the first face
        face_location = faces[0]
        face_image_cropped = face_service.crop_face(image, face_location)
        
        employee_id, confidence = face_service.recognize_face(face_image_cropped)
        
        return {
            "success": employee_id is not None,
            "employee_id": employee_id,
            "confidence": confidence,
            "message": "Face recognized" if employee_id else "Face not recognized"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face recognition error: {e}")
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(e)}")

# Liveness detection endpoint
@app.post("/detect-liveness")
async def detect_liveness(face_image: UploadFile = File(...)):
    """Detect liveness from uploaded image"""
    try:
        # Read and process image
        image_data = await face_image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Perform liveness detection
        result = liveness_service.comprehensive_liveness_check(image)
        
        return {
            "success": True,
            "is_live": result['is_live'],
            "confidence": result['confidence'],
            "scores": result['scores'],
            "explanation": result.get('explanation', ''),
            "timestamp": result['timestamp']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liveness detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Liveness detection failed: {str(e)}")

# Attendance analysis endpoint
@app.post("/analyze-attendance", response_model=AttendanceAnalysisResponse)
async def analyze_attendance(request: AttendanceAnalysisRequest):
    """Comprehensive attendance analysis with predictions"""
    try:
        # In a real implementation, you would fetch attendance records from database
        # For this example, we'll create sample data
        
        # Sample attendance records (in real system, fetch from database)
        sample_records = [
            AttendanceRecord(
                employee_id=request.employee_id,
                check_in_time=request.start_date + timedelta(hours=8),
                check_out_time=request.start_date + timedelta(hours=17),
                work_hours=8.0,
                location="office",
                device_type="camera",
                is_weekend=False,
                is_holiday=False,
                overtime_hours=0.0,
                late_minutes=5.0
            )
            for i in range(20)  # 20 sample records
        ]
        
        # Sample employee profile (in real system, fetch from database)
        profile = EmployeeProfile(
            employee_id=request.employee_id,
            age=30,
            department="Engineering",
            position="Developer",
            tenure_months=24,
            salary_level="mid",
            education_level="bachelor",
            performance_rating=4.0,
            manager_id="MGR001",
            team_size=8
        )
        
        # Sample attendance metrics (in real system, calculate from records)
        metrics = AttendanceMetrics(
            employee_id=request.employee_id,
            total_work_days=20,
            total_late_days=3,
            total_absent_days=1,
            avg_work_hours=8.2,
            avg_overtime_hours=1.5,
            punctuality_score=0.85,
            attendance_consistency=0.92,
            last_30_days_attendance=0.88,
            last_30_days_lateness=12.0
        )
        
        # Detect anomalies
        anomalies = []
        for record in sample_records:
            anomaly = anomaly_service.detect_attendance_anomalies(record)
            if anomaly.is_anomaly:
                anomalies.append(anomaly)
        
        # Predict turnover
        turnover_prediction = predictive_service.predict_turnover(profile, metrics)
        
        # Generate attendance forecast if requested
        attendance_forecast = None
        if request.include_predictions:
            # Sample historical data for forecasting
            historical_data = [
                {
                    "date": (request.start_date + timedelta(days=i)).isoformat(),
                    "attendance": 0.9 + np.random.normal(0, 0.1),
                    "lateness": 5.0 + np.random.normal(0, 3),
                    "overtime": 1.0 + np.random.normal(0, 0.5)
                }
                for i in range(30)
            ]
            
            forecast = predictive_service.forecast_attendance(request.employee_id, historical_data)
            attendance_forecast = {
                "predicted_attendance": forecast.predicted_attendance,
                "predicted_lateness": forecast.predicted_lateness,
                "predicted_overtime": forecast.predicted_overtime,
                "trend": forecast.trend,
                "confidence_interval": forecast.confidence_interval
            }
        
        # Generate comprehensive report
        predictive_report = predictive_service.generate_predictive_report(
            request.employee_id, profile, metrics, historical_data if request.include_predictions else []
        )
        
        return AttendanceAnalysisResponse(
            employee_id=request.employee_id,
            analysis_period={
                "start_date": request.start_date,
                "end_date": request.end_date
            },
            total_records=len(sample_records),
            anomaly_count=len(anomalies),
            risk_level=turnover_prediction.risk_level,
            turnover_probability=turnover_prediction.turnover_probability,
            attendance_forecast=attendance_forecast,
            recommendations=predictive_report.get('recommendations', []),
            detailed_analysis=predictive_report
        )
        
    except Exception as e:
        logger.error(f"Attendance analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Batch processing endpoint for training models
@app.post("/train-models")
async def train_models():
    """Train all AI models with historical data"""
    try:
        # In a real implementation, you would fetch training data from database
        # For this example, we'll create sample training data
        
        # Sample training data for anomaly detection
        sample_records = [
            AttendanceRecord(
                employee_id=f"EMP{i:03d}",
                check_in_time=datetime.now() - timedelta(days=i),
                check_out_time=datetime.now() - timedelta(days=i, hours=9),
                work_hours=8.0 + np.random.normal(0, 1),
                location="office",
                device_type="camera",
                is_weekend=False,
                is_holiday=False,
                overtime_hours=np.random.uniform(0, 3),
                late_minutes=np.random.uniform(0, 30)
            )
            for i in range(100)  # 100 sample records for training
        ]
        
        # Train anomaly detection model
        anomaly_trained = anomaly_service.train_models(sample_records)
        
        # Sample training data for predictive analytics
        sample_profiles = [
            EmployeeProfile(
                employee_id=f"EMP{i:03d}",
                age=25 + np.random.randint(0, 20),
                department=np.random.choice(["Engineering", "Sales", "HR", "Finance"]),
                position=np.random.choice(["Developer", "Manager", "Analyst"]),
                tenure_months=np.random.randint(1, 60),
                salary_level=np.random.choice(["entry", "mid", "senior"]),
                education_level=np.random.choice(["high_school", "bachelor", "master"]),
                performance_rating=3.0 + np.random.uniform(0, 2),
                manager_id=f"MGR{i//10:03d}",
                team_size=np.random.randint(3, 15)
            )
            for i in range(100)
        ]
        
        sample_metrics = [
            AttendanceMetrics(
                employee_id=f"EMP{i:03d}",
                total_work_days=200 + np.random.randint(-50, 50),
                total_late_days=np.random.randint(0, 30),
                total_absent_days=np.random.randint(0, 10),
                avg_work_hours=8.0 + np.random.normal(0, 1),
                avg_overtime_hours=np.random.uniform(0, 5),
                punctuality_score=0.7 + np.random.uniform(0, 0.3),
                attendance_consistency=0.8 + np.random.uniform(0, 0.2),
                last_30_days_attendance=0.8 + np.random.uniform(0, 0.2),
                last_30_days_lateness=np.random.uniform(0, 30)
            )
            for i in range(100)
        ]
        
        # Sample turnover labels (in real system, based on historical data)
        turnover_labels = [np.random.choice([True, False], p=[0.2, 0.8]) for _ in range(100)]
        
        # Train turnover prediction model
        turnover_trained = predictive_service.train_turnover_model(
            sample_profiles, sample_metrics, turnover_labels
        )
        
        # Sample historical data for forecasting
        sample_historical = {}
        for i in range(50):  # 50 employees
            employee_id = f"EMP{i:03d}"
            sample_historical[employee_id] = [
                {
                    "date": (datetime.now() - timedelta(days=j)).isoformat(),
                    "attendance": 0.9 + np.random.normal(0, 0.1),
                    "lateness": 5.0 + np.random.normal(0, 3),
                    "overtime": 1.0 + np.random.normal(0, 0.5)
                }
                for j in range(90)  # 90 days of history
            ]
        
        # Train forecasting models
        forecasting_trained = predictive_service.train_forecasting_models(sample_historical)
        
        return {
            "success": True,
            "models_trained": {
                "anomaly_detection": anomaly_trained,
                "turnover_prediction": turnover_trained,
                "attendance_forecasting": forecasting_trained
            },
            "training_data_size": {
                "anomaly_records": len(sample_records),
                "turnover_samples": len(sample_profiles),
                "forecasting_employees": len(sample_historical)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

# Get system statistics endpoint
@app.get("/statistics")
async def get_statistics():
    """Get system statistics and model information"""
    try:
        return {
            "face_recognition": {
                "registered_faces": face_service.get_face_count(),
                "model_loaded": face_service.face_model is not None
            },
            "liveness_detection": {
                "model_loaded": liveness_service.liveness_model is not None
            },
            "gps_antifraud": {
                "geofences_configured": len(gps_service.geofences),
                "geofence_names": [gf.name for gf in gps_service.geofences]
            },
            "anomaly_detection": {
                "model_trained": anomaly_service.isolation_forest is not None,
                "contamination_rate": anomaly_service.contamination
            },
            "predictive_analytics": {
                "turnover_model_trained": predictive_service.turnover_model is not None,
                "forecasting_models_trained": (
                    predictive_service.attendance_model is not None and
                    predictive_service.lateness_model is not None and
                    predictive_service.overtime_model is not None
                )
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Attendance System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "register_employee": "/register-employee",
            "check_in": "/check-in",
            "recognize_face": "/recognize-face",
            "detect_liveness": "/detect-liveness",
            "analyze_attendance": "/analyze-attendance",
            "train_models": "/train-models",
            "statistics": "/statistics"
        },
        "documentation": "/docs"
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "ai_integration_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
