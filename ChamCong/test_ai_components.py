#!/usr/bin/env python3
"""
Test script for AI Attendance System Components
Tests all AI services and validates functionality
"""

import sys
import os
import numpy as np
import cv2
from datetime import datetime, timedelta
import logging

# Add the ai_services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_services'))

# Import AI services
from face_recognition_service import FaceRecognitionService
from liveness_detection_service import LivenessDetectionService
from gps_antifraud_service import GPSAntiFraudService, LocationData
from anomaly_detection_service import AnomalyDetectionService, AttendanceRecord
from predictive_analytics_service import PredictiveAnalyticsService, EmployeeProfile, AttendanceMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_face_image():
    """Create a sample face image for testing"""
    # Create a simple test image with a face-like pattern
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Draw a simple face
    # Face outline
    cv2.circle(image, (112, 112), 80, (255, 220, 177), -1)
    
    # Eyes
    cv2.circle(image, (95, 90), 8, (0, 0, 0), -1)
    cv2.circle(image, (129, 90), 8, (0, 0, 0), -1)
    
    # Nose
    cv2.circle(image, (112, 110), 3, (200, 150, 100), -1)
    
    # Mouth
    cv2.ellipse(image, (112, 130), (15, 8), 0, 0, 180, (150, 100, 100), 2)
    
    return image

def test_face_recognition_service():
    """Test face recognition service"""
    logger.info("Testing Face Recognition Service...")
    
    try:
        # Initialize service
        face_service = FaceRecognitionService()
        
        # Create sample face image
        test_image = create_sample_face_image()
        
        # Test face detection
        faces = face_service.detect_faces(test_image)
        logger.info(f"Detected {len(faces)} faces")
        
        if faces:
            # Test face cropping
            face_location = faces[0]
            cropped_face = face_service.crop_face(test_image, face_location)
            logger.info(f"Cropped face shape: {cropped_face.shape}")
            
            # Test face registration
            employee_id = "TEST_EMP_001"
            registration_success = face_service.add_employee_face(employee_id, cropped_face)
            logger.info(f"Face registration success: {registration_success}")
            
            # Test face recognition
            employee_id_result, confidence = face_service.recognize_face(cropped_face)
            logger.info(f"Face recognition result: {employee_id_result}, confidence: {confidence:.3f}")
        
        logger.info("✅ Face Recognition Service test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Face Recognition Service test failed: {e}")
        return False

def test_liveness_detection_service():
    """Test liveness detection service"""
    logger.info("Testing Liveness Detection Service...")
    
    try:
        # Initialize service
        liveness_service = LivenessDetectionService()
        
        # Create sample face image
        test_image = create_sample_face_image()
        
        # Test comprehensive liveness check
        result = liveness_service.comprehensive_liveness_check(test_image)
        
        logger.info(f"Liveness check result:")
        logger.info(f"  - Is live: {result['is_live']}")
        logger.info(f"  - Confidence: {result['confidence']:.3f}")
        logger.info(f"  - Blink detected: {result.get('blink_detected', False)}")
        logger.info(f"  - Mouth movement: {result.get('mouth_movement', False)}")
        logger.info(f"  - 3D structure: {result.get('has_3d_structure', False)}")
        
        logger.info("✅ Liveness Detection Service test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Liveness Detection Service test failed: {e}")
        return False

def test_gps_antifraud_service():
    """Test GPS anti-fraud service"""
    logger.info("Testing GPS Anti-Fraud Service...")
    
    try:
        # Initialize service
        gps_service = GPSAntiFraudService()
        
        # Create test location data
        test_location = LocationData(
            latitude=10.762622,
            longitude=106.660172,
            accuracy=5.0,
            timestamp=datetime.now()
        )
        
        # Test geofence validation
        valid_geofences = gps_service.find_valid_geofences(test_location)
        logger.info(f"Valid geofences: {[gf.name for gf, _ in valid_geofences]}")
        
        # Test comprehensive location validation
        validation_result = gps_service.comprehensive_location_validation(
            "TEST_EMP_001", test_location
        )
        
        logger.info(f"Location validation result:")
        logger.info(f"  - Valid check-in: {validation_result['is_valid_checkin']}")
        logger.info(f"  - Is fraud: {validation_result['is_fraud']}")
        logger.info(f"  - Fraud score: {validation_result['fraud_score']:.3f}")
        logger.info(f"  - Geofence valid: {validation_result['geofence_valid']}")
        
        logger.info("✅ GPS Anti-Fraud Service test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPS Anti-Fraud Service test failed: {e}")
        return False

def test_anomaly_detection_service():
    """Test anomaly detection service"""
    logger.info("Testing Anomaly Detection Service...")
    
    try:
        # Initialize service
        anomaly_service = AnomalyDetectionService()
        
        # Create sample attendance records for training
        sample_records = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(50):
            record = AttendanceRecord(
                employee_id="TEST_EMP_001",
                check_in_time=base_time + timedelta(days=i, hours=8),
                check_out_time=base_time + timedelta(days=i, hours=17),
                work_hours=8.0 + np.random.normal(0, 0.5),
                location="office",
                device_type="camera",
                is_weekend=False,
                is_holiday=False,
                overtime_hours=np.random.uniform(0, 2),
                late_minutes=np.random.uniform(0, 15)
            )
            sample_records.append(record)
        
        # Train models
        training_success = anomaly_service.train_models(sample_records)
        logger.info(f"Model training success: {training_success}")
        
        # Test anomaly detection on a sample record
        test_record = sample_records[0]
        anomaly_result = anomaly_service.detect_attendance_anomalies(test_record)
        
        logger.info(f"Anomaly detection result:")
        logger.info(f"  - Is anomaly: {anomaly_result.is_anomaly}")
        logger.info(f"  - Anomaly type: {anomaly_result.anomaly_type}")
        logger.info(f"  - Confidence: {anomaly_result.confidence:.3f}")
        logger.info(f"  - Severity: {anomaly_result.severity}")
        logger.info(f"  - Explanation: {anomaly_result.explanation}")
        
        logger.info("✅ Anomaly Detection Service test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Anomaly Detection Service test failed: {e}")
        return False

def test_predictive_analytics_service():
    """Test predictive analytics service"""
    logger.info("Testing Predictive Analytics Service...")
    
    try:
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
        
        # Create sample historical data
        historical_data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            data_point = {
                "date": (base_date + timedelta(days=i)).isoformat(),
                "attendance": 0.9 + np.random.normal(0, 0.1),
                "lateness": 5.0 + np.random.normal(0, 3),
                "overtime": 1.0 + np.random.normal(0, 0.5)
            }
            historical_data.append(data_point)
        
        # Test turnover prediction
        turnover_result = predictive_service.predict_turnover(profile, metrics)
        
        logger.info(f"Turnover prediction result:")
        logger.info(f"  - Turnover probability: {turnover_result.turnover_probability:.3f}")
        logger.info(f"  - Risk level: {turnover_result.risk_level}")
        logger.info(f"  - Predicted timeline: {turnover_result.predicted_timeline}")
        logger.info(f"  - Confidence: {turnover_result.confidence:.3f}")
        logger.info(f"  - Key factors: {turnover_result.key_factors}")
        
        # Test attendance forecasting
        forecast_result = predictive_service.forecast_attendance("TEST_EMP_001", historical_data)
        
        logger.info(f"Attendance forecast result:")
        logger.info(f"  - Predicted attendance: {forecast_result.predicted_attendance:.3f}")
        logger.info(f"  - Predicted lateness: {forecast_result.predicted_lateness:.3f}")
        logger.info(f"  - Predicted overtime: {forecast_result.predicted_overtime:.3f}")
        logger.info(f"  - Trend: {forecast_result.trend}")
        
        logger.info("✅ Predictive Analytics Service test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Predictive Analytics Service test failed: {e}")
        return False

def run_all_tests():
    """Run all AI component tests"""
    logger.info("🚀 Starting AI Attendance System Component Tests")
    logger.info("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Face Recognition", test_face_recognition_service()))
    test_results.append(("Liveness Detection", test_liveness_detection_service()))
    test_results.append(("GPS Anti-Fraud", test_gps_antifraud_service()))
    test_results.append(("Anomaly Detection", test_anomaly_detection_service()))
    test_results.append(("Predictive Analytics", test_predictive_analytics_service()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:20} : {status}")
        
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
        logger.info("🎉 All tests passed! AI system is ready for deployment.")
        return True
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
