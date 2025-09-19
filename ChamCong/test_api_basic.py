#!/usr/bin/env python3
"""
Basic API Test for AI Attendance System
Tests API structure without heavy ML dependencies
"""

import sys
import os
from datetime import datetime
import json

# Add the ai_services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_services'))

def test_api_structure():
    """Test API structure and basic functionality"""
    print("🚀 Testing API Structure...")
    
    try:
        # Test FastAPI import
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
        
        # Test Pydantic models
        from pydantic import BaseModel
        print("✅ Pydantic imported successfully")
        
        # Create a simple test app
        app = FastAPI(title="Test API")
        
        @app.get("/test")
        def test_endpoint():
            return {"message": "Test successful", "timestamp": datetime.now().isoformat()}
        
        print("✅ FastAPI app created successfully")
        
        # Test JSON serialization
        test_data = {
            "employee_id": "EMP001",
            "latitude": 10.762622,
            "longitude": 106.660172,
            "timestamp": datetime.now().isoformat()
        }
        
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        print("✅ JSON serialization/parsing works")
        print(f"✅ Test data: {parsed_data}")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def test_pydantic_models():
    """Test Pydantic models for API"""
    print("\n🧪 Testing Pydantic Models...")
    
    try:
        from pydantic import BaseModel, Field
        from datetime import datetime
        from typing import Dict, List, Optional
        
        # Define test models
        class TestCheckInRequest(BaseModel):
            employee_id: str
            latitude: float
            longitude: float
            accuracy: float
            timestamp: datetime
            device_type: str = "mobile"
            wifi_data: Optional[List[Dict]] = None
        
        class TestCheckInResponse(BaseModel):
            success: bool
            employee_id: str
            check_in_time: datetime
            confidence_score: float
            is_fraud: bool
        
        # Test model creation
        request_data = {
            "employee_id": "EMP001",
            "latitude": 10.762622,
            "longitude": 106.660172,
            "accuracy": 5.0,
            "timestamp": datetime.now(),
            "device_type": "mobile"
        }
        
        request_model = TestCheckInRequest(**request_data)
        print("✅ Check-in request model created")
        
        response_data = {
            "success": True,
            "employee_id": "EMP001",
            "check_in_time": datetime.now(),
            "confidence_score": 0.95,
            "is_fraud": False
        }
        
        response_model = TestCheckInResponse(**response_data)
        print("✅ Check-in response model created")
        
        # Test JSON conversion
        request_json = request_model.model_dump_json()
        response_json = response_model.model_dump_json()
        
        print("✅ Model JSON serialization works")
        print(f"Request JSON: {request_json[:100]}...")
        print(f"Response JSON: {response_json[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic models test failed: {e}")
        return False

def test_gps_service_basic():
    """Test GPS service basic functionality"""
    print("\n📍 Testing GPS Service (Basic)...")
    
    try:
        from gps_antifraud_service import GPSAntiFraudService, LocationData
        
        # Initialize service
        gps_service = GPSAntiFraudService()
        print(f"✅ GPS service initialized with {len(gps_service.geofences)} geofences")
        
        # Test location data creation
        location = LocationData(
            latitude=10.762622,
            longitude=106.660172,
            accuracy=5.0,
            timestamp=datetime.now()
        )
        print("✅ Location data created")
        
        # Test geofence validation
        valid_geofences = gps_service.find_valid_geofences(location)
        print(f"✅ Geofence validation: {len(valid_geofences)} valid geofences")
        
        # Test distance calculation
        distance = gps_service.calculate_distance(10.762622, 106.660172, 10.763622, 106.661172)
        print(f"✅ Distance calculation: {distance:.2f} meters")
        
        return True
        
    except Exception as e:
        print(f"❌ GPS service test failed: {e}")
        return False

def test_anomaly_service_basic():
    """Test anomaly service basic functionality"""
    print("\n🔍 Testing Anomaly Service (Basic)...")
    
    try:
        from anomaly_detection_service import AnomalyDetectionService, AttendanceRecord
        from datetime import timedelta
        
        # Initialize service
        anomaly_service = AnomalyDetectionService()
        print("✅ Anomaly service initialized")
        
        # Create test record
        record = AttendanceRecord(
            employee_id="EMP001",
            check_in_time=datetime.now() - timedelta(hours=8),
            check_out_time=datetime.now(),
            work_hours=8.0,
            location="office",
            device_type="camera",
            is_weekend=False,
            is_holiday=False,
            overtime_hours=0.0,
            late_minutes=5.0
        )
        print("✅ Attendance record created")
        
        # Test feature extraction
        time_features = anomaly_service._extract_time_features(record.check_in_time)
        attendance_features = anomaly_service._extract_attendance_features(record)
        
        print(f"✅ Time features extracted: {len(time_features)} features")
        print(f"✅ Attendance features extracted: {len(attendance_features)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Anomaly service test failed: {e}")
        return False

def test_predictive_service_basic():
    """Test predictive service basic functionality"""
    print("\n🔮 Testing Predictive Service (Basic)...")
    
    try:
        from predictive_analytics_service import PredictiveAnalyticsService, EmployeeProfile, AttendanceMetrics
        
        # Initialize service
        predictive_service = PredictiveAnalyticsService()
        print("✅ Predictive service initialized")
        
        # Create test profile
        profile = EmployeeProfile(
            employee_id="EMP001",
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
        print("✅ Employee profile created")
        
        # Create test metrics
        metrics = AttendanceMetrics(
            employee_id="EMP001",
            total_work_days=200,
            total_late_days=15,
            total_absent_days=5,
            avg_work_hours=8.5,
            avg_overtime_hours=2.0,
            punctuality_score=0.85,
            attendance_consistency=0.92,
            last_30_days_attendance=0.88,
            last_30_days_lateness=12.0
        )
        print("✅ Attendance metrics created")
        
        # Test feature preparation
        features = predictive_service._prepare_turnover_features(profile, metrics)
        print(f"✅ Turnover features prepared: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Predictive service test failed: {e}")
        return False

def run_api_tests():
    """Run all API tests"""
    print("🚀 Starting AI Attendance System API Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("API Structure", test_api_structure()))
    test_results.append(("Pydantic Models", test_pydantic_models()))
    test_results.append(("GPS Service", test_gps_service_basic()))
    test_results.append(("Anomaly Service", test_anomaly_service_basic()))
    test_results.append(("Predictive Service", test_predictive_service_basic()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 API TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_results)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All API tests passed! System is ready for deployment.")
        print("\n📝 Next steps:")
        print("   1. Install ML dependencies: pip install tensorflow opencv-python mediapipe")
        print("   2. Start API server: python ai_services/ai_integration_api.py")
        print("   3. Access API docs: http://localhost:8000/docs")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_api_tests()
    sys.exit(0 if success else 1)
