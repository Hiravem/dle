#!/usr/bin/env python3
"""
Simplified test for AI Attendance System Components
Tests basic functionality without heavy dependencies
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the ai_services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_services'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports"""
    logger.info("Testing basic imports...")
    
    try:
        import numpy as np
        import pandas as pd
        from datetime import datetime
        logger.info("✅ Basic imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Basic imports failed: {e}")
        return False

def test_gps_antifraud_basic():
    """Test GPS anti-fraud service without dependencies"""
    logger.info("Testing GPS Anti-Fraud Service (basic)...")
    
    try:
        # Import only the GPS service
        from gps_antifraud_service import GPSAntiFraudService, LocationData
        
        # Initialize service
        gps_service = GPSAntiFraudService()
        
        # Create test location data
        test_location = LocationData(
            latitude=10.762622,
            longitude=106.660172,
            accuracy=5.0,
            timestamp=datetime.now()
        )
        
        # Test basic functionality
        logger.info(f"GPS service initialized with {len(gps_service.geofences)} geofences")
        
        # Test distance calculation
        distance = gps_service.calculate_distance(10.762622, 106.660172, 10.763622, 106.661172)
        logger.info(f"Distance calculation: {distance:.2f} meters")
        
        logger.info("✅ GPS Anti-Fraud Service basic test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPS Anti-Fraud Service basic test failed: {e}")
        return False

def test_anomaly_detection_basic():
    """Test anomaly detection service without ML dependencies"""
    logger.info("Testing Anomaly Detection Service (basic)...")
    
    try:
        from anomaly_detection_service import AnomalyDetectionService, AttendanceRecord
        
        # Initialize service
        anomaly_service = AnomalyDetectionService()
        
        # Create sample attendance record
        record = AttendanceRecord(
            employee_id="TEST_EMP_001",
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
        
        logger.info(f"Created attendance record for employee: {record.employee_id}")
        logger.info(f"Work hours: {record.work_hours}, Late minutes: {record.late_minutes}")
        
        # Test feature extraction (without ML models)
        time_features = anomaly_service._extract_time_features(record.check_in_time)
        attendance_features = anomaly_service._extract_attendance_features(record)
        
        logger.info(f"Time features extracted: {len(time_features)} features")
        logger.info(f"Attendance features extracted: {len(attendance_features)} features")
        
        logger.info("✅ Anomaly Detection Service basic test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Anomaly Detection Service basic test failed: {e}")
        return False

def test_predictive_analytics_basic():
    """Test predictive analytics service without ML dependencies"""
    logger.info("Testing Predictive Analytics Service (basic)...")
    
    try:
        from predictive_analytics_service import PredictiveAnalyticsService, EmployeeProfile, AttendanceMetrics
        
        # Initialize service
        predictive_service = PredictiveAnalyticsService()
        
        # Create sample employee profile
        profile = EmployeeProfile(
            employee_id="TEST_EMP_001",
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
        
        # Create sample attendance metrics
        metrics = AttendanceMetrics(
            employee_id="TEST_EMP_001",
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
        
        logger.info(f"Created employee profile: {profile.employee_id}")
        logger.info(f"Department: {profile.department}, Position: {profile.position}")
        logger.info(f"Attendance metrics - Work days: {metrics.total_work_days}")
        logger.info(f"Punctuality score: {metrics.punctuality_score}")
        
        # Test feature preparation (without ML models)
        features = predictive_service._prepare_turnover_features(profile, metrics)
        if features.size > 0:
            logger.info(f"Turnover features prepared: {features.shape}")
        
        logger.info("✅ Predictive Analytics Service basic test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Predictive Analytics Service basic test failed: {e}")
        return False

def test_config_files():
    """Test configuration files"""
    logger.info("Testing configuration files...")
    
    try:
        import json
        
        # Test geofences config
        if os.path.exists("config/geofences.json"):
            with open("config/geofences.json", 'r') as f:
                geofences = json.load(f)
            logger.info(f"✅ Geofences config loaded: {len(geofences['geofences'])} geofences")
        else:
            logger.warning("⚠️ Geofences config file not found")
        
        # Test requirements
        if os.path.exists("requirements.txt"):
            with open("requirements.txt", 'r') as f:
                requirements = f.readlines()
            logger.info(f"✅ Requirements file loaded: {len(requirements)} packages")
        else:
            logger.warning("⚠️ Requirements file not found")
        
        # Test README
        if os.path.exists("README.md"):
            with open("README.md", 'r', encoding='utf-8') as f:
                readme = f.read()
            logger.info(f"✅ README file loaded: {len(readme)} characters")
        else:
            logger.warning("⚠️ README file not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration files test failed: {e}")
        return False

def test_file_structure():
    """Test file structure"""
    logger.info("Testing file structure...")
    
    try:
        required_files = [
            "ai_services/face_recognition_service.py",
            "ai_services/liveness_detection_service.py", 
            "ai_services/gps_antifraud_service.py",
            "ai_services/anomaly_detection_service.py",
            "ai_services/predictive_analytics_service.py",
            "ai_services/ai_integration_api.py",
            "requirements.txt",
            "config/geofences.json",
            "README.md",
            "Dockerfile",
            "docker-compose.yml"
        ]
        
        missing_files = []
        for file_path in required_files:
            if os.path.exists(file_path):
                logger.info(f"✅ {file_path} exists")
            else:
                logger.error(f"❌ {file_path} missing")
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
            return False
        else:
            logger.info("✅ All required files present")
            return True
        
    except Exception as e:
        logger.error(f"❌ File structure test failed: {e}")
        return False

def run_basic_tests():
    """Run basic tests without heavy dependencies"""
    logger.info("🚀 Starting Basic AI Attendance System Tests")
    logger.info("=" * 60)
    
    test_results = []
    
    # Run basic tests
    test_results.append(("Basic Imports", test_basic_imports()))
    test_results.append(("File Structure", test_file_structure()))
    test_results.append(("Configuration Files", test_config_files()))
    test_results.append(("GPS Anti-Fraud (Basic)", test_gps_antifraud_basic()))
    test_results.append(("Anomaly Detection (Basic)", test_anomaly_detection_basic()))
    test_results.append(("Predictive Analytics (Basic)", test_predictive_analytics_basic()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 BASIC TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:25} : {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"Total Tests: {len(test_results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(test_results)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 All basic tests passed! Core system structure is ready.")
        logger.info("📝 Note: For full AI functionality, install ML dependencies:")
        logger.info("   pip install tensorflow opencv-python mediapipe face-recognition")
        return True
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
