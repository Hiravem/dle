"""
AI Anomaly Detection Service
Phát hiện bất thường trong dữ liệu chấm công sử dụng ML algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttendanceRecord:
    """Attendance record data structure"""
    employee_id: str
    check_in_time: datetime
    check_out_time: Optional[datetime]
    work_hours: float
    location: str
    device_type: str
    is_weekend: bool
    is_holiday: bool
    overtime_hours: float = 0.0
    late_minutes: float = 0.0

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    is_anomaly: bool
    anomaly_type: str
    confidence: float
    features: Dict[str, float]
    explanation: str
    severity: str  # low, medium, high, critical
    timestamp: datetime

class AnomalyDetectionService:
    def __init__(self, model_path: str = "models/anomaly_models.joblib"):
        """
        Initialize Anomaly Detection Service
        
        Args:
            model_path: Path to saved anomaly detection models
        """
        self.model_path = model_path
        self.isolation_forest = None
        self.one_class_svm = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Anomaly detection parameters
        self.contamination = 0.1  # Expected proportion of anomalies
        self.min_samples = 50  # Minimum samples for training
        
        # Feature engineering parameters
        self.time_features = ['hour', 'day_of_week', 'day_of_month', 'month']
        self.attendance_features = ['work_hours', 'overtime_hours', 'late_minutes']
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained anomaly detection models"""
        try:
            if joblib.os.path.exists(self.model_path):
                models_data = joblib.load(self.model_path)
                self.isolation_forest = models_data.get('isolation_forest')
                self.one_class_svm = models_data.get('one_class_svm')
                self.scaler = models_data.get('scaler', StandardScaler())
                self.label_encoders = models_data.get('label_encoders', {})
                logger.info("Loaded pre-trained anomaly detection models")
            else:
                logger.info("No pre-trained models found, will train new models")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            models_data = {
                'isolation_forest': self.isolation_forest,
                'one_class_svm': self.one_class_svm,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders
            }
            joblib.dump(models_data, self.model_path)
            logger.info(f"Saved models to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _extract_time_features(self, timestamp: datetime) -> Dict[str, float]:
        """Extract time-based features from timestamp"""
        return {
            'hour': float(timestamp.hour),
            'day_of_week': float(timestamp.weekday()),
            'day_of_month': float(timestamp.day),
            'month': float(timestamp.month),
            'is_weekend': float(timestamp.weekday() >= 5),
            'is_holiday': 0.0  # This would be determined by holiday calendar
        }
    
    def _extract_attendance_features(self, record: AttendanceRecord) -> Dict[str, float]:
        """Extract attendance-related features"""
        return {
            'work_hours': record.work_hours,
            'overtime_hours': record.overtime_hours,
            'late_minutes': record.late_minutes,
            'has_checkout': 1.0 if record.check_out_time else 0.0
        }
    
    def _encode_categorical_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Encode categorical features using label encoders"""
        encoded_features = features.copy()
        
        for feature_name, value in features.items():
            if isinstance(value, str):
                if feature_name not in self.label_encoders:
                    self.label_encoders[feature_name] = LabelEncoder()
                    # Fit encoder with known categories
                    self.label_encoders[feature_name].fit([value])
                
                try:
                    encoded_features[feature_name] = float(
                        self.label_encoders[feature_name].transform([value])[0]
                    )
                except ValueError:
                    # Handle unknown categories
                    encoded_features[feature_name] = -1.0
        
        return encoded_features
    
    def _prepare_features(self, records: List[AttendanceRecord]) -> np.ndarray:
        """Prepare feature matrix from attendance records"""
        try:
            feature_matrix = []
            
            for record in records:
                # Extract time features
                time_features = self._extract_time_features(record.check_in_time)
                
                # Extract attendance features
                attendance_features = self._extract_attendance_features(record)
                
                # Combine all features
                combined_features = {
                    **time_features,
                    **attendance_features,
                    'location': record.location,
                    'device_type': record.device_type
                }
                
                # Encode categorical features
                encoded_features = self._encode_categorical_features(combined_features)
                
                # Convert to array
                feature_vector = np.array(list(encoded_features.values()))
                feature_matrix.append(feature_vector)
            
            return np.array(feature_matrix)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([])
    
    def train_models(self, records: List[AttendanceRecord]) -> bool:
        """
        Train anomaly detection models on historical data
        
        Args:
            records: Historical attendance records
            
        Returns:
            Training success status
        """
        try:
            if len(records) < self.min_samples:
                logger.warning(f"Insufficient data for training: {len(records)} < {self.min_samples}")
                return False
            
            # Prepare features
            feature_matrix = self._prepare_features(records)
            
            if feature_matrix.size == 0:
                logger.error("Failed to prepare features for training")
                return False
            
            # Fit scaler
            self.scaler.fit(feature_matrix)
            scaled_features = self.scaler.transform(feature_matrix)
            
            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.isolation_forest.fit(scaled_features)
            
            # Train One-Class SVM
            self.one_class_svm = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
            self.one_class_svm.fit(scaled_features)
            
            # Save models
            self._save_models()
            
            logger.info(f"Successfully trained anomaly detection models on {len(records)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def detect_attendance_anomalies(self, record: AttendanceRecord) -> AnomalyResult:
        """
        Detect anomalies in a single attendance record
        
        Args:
            record: Attendance record to analyze
            
        Returns:
            Anomaly detection result
        """
        try:
            if self.isolation_forest is None or self.one_class_svm is None:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type="model_not_trained",
                    confidence=0.0,
                    features={},
                    explanation="Models not trained yet",
                    severity="low",
                    timestamp=datetime.now()
                )
            
            # Prepare features for the record
            feature_matrix = self._prepare_features([record])
            
            if feature_matrix.size == 0:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type="feature_error",
                    confidence=0.0,
                    features={},
                    explanation="Failed to prepare features",
                    severity="low",
                    timestamp=datetime.now()
                )
            
            # Scale features
            scaled_features = self.scaler.transform(feature_matrix)
            
            # Predict with both models
            if_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
            if_score = self.isolation_forest.decision_function(scaled_features)[0]
            
            svm_anomaly = self.one_class_svm.predict(scaled_features)[0] == -1
            svm_score = self.one_class_svm.decision_function(scaled_features)[0]
            
            # Combine predictions
            is_anomaly = if_anomaly or svm_anomaly
            confidence = max(abs(if_score), abs(svm_score))
            
            # Determine anomaly type and explanation
            anomaly_type, explanation, severity = self._classify_anomaly(record, if_anomaly, svm_anomaly)
            
            # Extract feature importance (simplified)
            features = self._extract_feature_importance(record)
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_type=anomaly_type,
                confidence=confidence,
                features=features,
                explanation=explanation,
                severity=severity,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return AnomalyResult(
                is_anomaly=True,
                anomaly_type="detection_error",
                confidence=1.0,
                features={},
                explanation=f"Detection error: {str(e)}",
                severity="medium",
                timestamp=datetime.now()
            )
    
    def _classify_anomaly(self, record: AttendanceRecord, 
                         if_anomaly: bool, svm_anomaly: bool) -> Tuple[str, str, str]:
        """Classify the type of anomaly and determine severity"""
        
        # Check for specific anomaly patterns
        if record.work_hours > 16:  # Excessive work hours
            return "excessive_hours", "Work hours exceed normal limits (>16 hours)", "high"
        
        if record.late_minutes > 60:  # Very late arrival
            return "excessive_lateness", f"Arrived {record.late_minutes:.1f} minutes late", "medium"
        
        if record.overtime_hours > 8:  # Excessive overtime
            return "excessive_overtime", f"Overtime hours: {record.overtime_hours:.1f}", "medium"
        
        if record.check_in_time.hour < 4:  # Very early check-in
            return "early_checkin", f"Checked in at {record.check_in_time.hour}:{record.check_in_time.minute:02d}", "low"
        
        if record.check_in_time.hour > 22:  # Very late check-in
            return "late_checkin", f"Checked in at {record.check_in_time.hour}:{record.check_in_time.minute:02d}", "medium"
        
        if record.is_weekend and record.work_hours > 8:  # Long weekend work
            return "weekend_overtime", "Working long hours on weekend", "medium"
        
        if record.device_type == "mobile" and record.location != "office":  # Remote work pattern
            return "remote_work", "Working remotely via mobile device", "low"
        
        # General anomaly
        if if_anomaly and svm_anomaly:
            return "general_anomaly", "Multiple anomaly patterns detected", "medium"
        elif if_anomaly or svm_anomaly:
            return "mild_anomaly", "Single anomaly pattern detected", "low"
        
        return "normal", "No anomalies detected", "low"
    
    def _extract_feature_importance(self, record: AttendanceRecord) -> Dict[str, float]:
        """Extract feature importance for the record"""
        return {
            'work_hours': record.work_hours,
            'late_minutes': record.late_minutes,
            'overtime_hours': record.overtime_hours,
            'check_in_hour': record.check_in_time.hour,
            'day_of_week': record.check_in_time.weekday(),
            'is_weekend': 1.0 if record.is_weekend else 0.0
        }
    
    def detect_pattern_anomalies(self, employee_records: List[AttendanceRecord], 
                                window_days: int = 30) -> List[AnomalyResult]:
        """
        Detect pattern-based anomalies across multiple records
        
        Args:
            employee_records: Employee's attendance records
            window_days: Time window for pattern analysis
            
        Returns:
            List of pattern anomaly results
        """
        try:
            if len(employee_records) < 10:
                return []
            
            # Filter recent records
            cutoff_date = datetime.now() - timedelta(days=window_days)
            recent_records = [r for r in employee_records if r.check_in_time >= cutoff_date]
            
            if len(recent_records) < 5:
                return []
            
            anomalies = []
            
            # Check for attendance frequency anomalies
            attendance_frequency = self._analyze_attendance_frequency(recent_records)
            if attendance_frequency['is_anomaly']:
                anomalies.append(attendance_frequency)
            
            # Check for work hour pattern anomalies
            work_hour_pattern = self._analyze_work_hour_patterns(recent_records)
            if work_hour_pattern['is_anomaly']:
                anomalies.append(work_hour_pattern)
            
            # Check for punctuality pattern anomalies
            punctuality_pattern = self._analyze_punctuality_patterns(recent_records)
            if punctuality_pattern['is_anomaly']:
                anomalies.append(punctuality_pattern)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting pattern anomalies: {e}")
            return []
    
    def _analyze_attendance_frequency(self, records: List[AttendanceRecord]) -> AnomalyResult:
        """Analyze attendance frequency patterns"""
        try:
            total_days = len(set(r.check_in_time.date() for r in records))
            expected_days = 22  # Average working days per month
            
            attendance_rate = total_days / expected_days
            
            if attendance_rate < 0.5:  # Less than 50% attendance
                return AnomalyResult(
                    is_anomaly=True,
                    anomaly_type="low_attendance",
                    confidence=1.0 - attendance_rate,
                    features={'attendance_rate': attendance_rate},
                    explanation=f"Low attendance rate: {attendance_rate:.1%}",
                    severity="high",
                    timestamp=datetime.now()
                )
            
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type="normal_attendance",
                confidence=0.0,
                features={'attendance_rate': attendance_rate},
                explanation="Normal attendance pattern",
                severity="low",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing attendance frequency: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type="analysis_error",
                confidence=0.0,
                features={},
                explanation=f"Analysis error: {str(e)}",
                severity="low",
                timestamp=datetime.now()
            )
    
    def _analyze_work_hour_patterns(self, records: List[AttendanceRecord]) -> AnomalyResult:
        """Analyze work hour patterns"""
        try:
            work_hours = [r.work_hours for r in records if r.work_hours > 0]
            
            if len(work_hours) < 3:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type="insufficient_data",
                    confidence=0.0,
                    features={},
                    explanation="Insufficient work hour data",
                    severity="low",
                    timestamp=datetime.now()
                )
            
            avg_hours = np.mean(work_hours)
            std_hours = np.std(work_hours)
            
            # Check for excessive work hours
            if avg_hours > 12:
                return AnomalyResult(
                    is_anomaly=True,
                    anomaly_type="excessive_work_hours",
                    confidence=min((avg_hours - 8) / 4, 1.0),
                    features={'avg_work_hours': avg_hours, 'std_hours': std_hours},
                    explanation=f"Average work hours: {avg_hours:.1f} (excessive)",
                    severity="medium",
                    timestamp=datetime.now()
                )
            
            # Check for inconsistent work hours
            if std_hours > 4:  # High variability
                return AnomalyResult(
                    is_anomaly=True,
                    anomaly_type="inconsistent_hours",
                    confidence=min(std_hours / 4, 1.0),
                    features={'avg_work_hours': avg_hours, 'std_hours': std_hours},
                    explanation=f"Inconsistent work hours (std: {std_hours:.1f})",
                    severity="low",
                    timestamp=datetime.now()
                )
            
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type="normal_work_hours",
                confidence=0.0,
                features={'avg_work_hours': avg_hours, 'std_hours': std_hours},
                explanation="Normal work hour pattern",
                severity="low",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing work hour patterns: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type="analysis_error",
                confidence=0.0,
                features={},
                explanation=f"Analysis error: {str(e)}",
                severity="low",
                timestamp=datetime.now()
            )
    
    def _analyze_punctuality_patterns(self, records: List[AttendanceRecord]) -> AnomalyResult:
        """Analyze punctuality patterns"""
        try:
            late_records = [r for r in records if r.late_minutes > 0]
            
            if len(late_records) == 0:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type="perfect_punctuality",
                    confidence=0.0,
                    features={'late_percentage': 0.0, 'avg_late_minutes': 0.0},
                    explanation="Perfect punctuality",
                    severity="low",
                    timestamp=datetime.now()
                )
            
            late_percentage = len(late_records) / len(records)
            avg_late_minutes = np.mean([r.late_minutes for r in late_records])
            
            # Check for frequent lateness
            if late_percentage > 0.3:  # More than 30% late
                return AnomalyResult(
                    is_anomaly=True,
                    anomaly_type="frequent_lateness",
                    confidence=late_percentage,
                    features={'late_percentage': late_percentage, 'avg_late_minutes': avg_late_minutes},
                    explanation=f"Frequently late: {late_percentage:.1%} of days",
                    severity="medium",
                    timestamp=datetime.now()
                )
            
            # Check for excessive lateness
            if avg_late_minutes > 30:
                return AnomalyResult(
                    is_anomaly=True,
                    anomaly_type="excessive_lateness",
                    confidence=min(avg_late_minutes / 60, 1.0),
                    features={'late_percentage': late_percentage, 'avg_late_minutes': avg_late_minutes},
                    explanation=f"Excessive lateness: {avg_late_minutes:.1f} minutes average",
                    severity="medium",
                    timestamp=datetime.now()
                )
            
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type="normal_punctuality",
                confidence=0.0,
                features={'late_percentage': late_percentage, 'avg_late_minutes': avg_late_minutes},
                explanation="Normal punctuality pattern",
                severity="low",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing punctuality patterns: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type="analysis_error",
                confidence=0.0,
                features={},
                explanation=f"Analysis error: {str(e)}",
                severity="low",
                timestamp=datetime.now()
            )
    
    def generate_anomaly_report(self, employee_id: str, 
                               records: List[AttendanceRecord]) -> Dict:
        """
        Generate comprehensive anomaly report for employee
        
        Args:
            employee_id: Employee identifier
            records: Employee's attendance records
            
        Returns:
            Comprehensive anomaly report
        """
        try:
            # Detect individual record anomalies
            individual_anomalies = []
            for record in records[-30:]:  # Last 30 records
                anomaly = self.detect_attendance_anomalies(record)
                if anomaly.is_anomaly:
                    individual_anomalies.append(anomaly)
            
            # Detect pattern anomalies
            pattern_anomalies = self.detect_pattern_anomalies(records)
            
            # Calculate anomaly statistics
            total_anomalies = len(individual_anomalies) + len(pattern_anomalies)
            critical_anomalies = [a for a in individual_anomalies + pattern_anomalies 
                                if a.severity == "critical"]
            high_anomalies = [a for a in individual_anomalies + pattern_anomalies 
                            if a.severity == "high"]
            
            # Risk assessment
            risk_score = self._calculate_risk_score(individual_anomalies, pattern_anomalies)
            risk_level = self._determine_risk_level(risk_score)
            
            return {
                'employee_id': employee_id,
                'report_date': datetime.now().isoformat(),
                'total_anomalies': total_anomalies,
                'critical_anomalies': len(critical_anomalies),
                'high_anomalies': len(high_anomalies),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'individual_anomalies': [asdict(a) for a in individual_anomalies],
                'pattern_anomalies': [asdict(a) for a in pattern_anomalies],
                'recommendations': self._generate_recommendations(individual_anomalies, pattern_anomalies)
            }
            
        except Exception as e:
            logger.error(f"Error generating anomaly report: {e}")
            return {
                'employee_id': employee_id,
                'error': str(e),
                'report_date': datetime.now().isoformat()
            }
    
    def _calculate_risk_score(self, individual_anomalies: List[AnomalyResult], 
                             pattern_anomalies: List[AnomalyResult]) -> float:
        """Calculate overall risk score"""
        try:
            severity_weights = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            
            total_score = 0.0
            for anomaly in individual_anomalies + pattern_anomalies:
                weight = severity_weights.get(anomaly.severity, 1)
                total_score += anomaly.confidence * weight
            
            # Normalize to 0-1 scale
            max_possible_score = len(individual_anomalies + pattern_anomalies) * 4
            if max_possible_score > 0:
                return min(total_score / max_possible_score, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.0
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _generate_recommendations(self, individual_anomalies: List[AnomalyResult], 
                                pattern_anomalies: List[AnomalyResult]) -> List[str]:
        """Generate recommendations based on anomalies"""
        recommendations = []
        
        # Analyze anomaly types
        anomaly_types = [a.anomaly_type for a in individual_anomalies + pattern_anomalies]
        
        if 'excessive_hours' in anomaly_types:
            recommendations.append("Review work schedule to prevent burnout")
        
        if 'frequent_lateness' in anomaly_types:
            recommendations.append("Discuss punctuality issues with employee")
        
        if 'low_attendance' in anomaly_types:
            recommendations.append("Investigate attendance issues and provide support")
        
        if 'excessive_overtime' in anomaly_types:
            recommendations.append("Monitor overtime patterns and workload distribution")
        
        if not recommendations:
            recommendations.append("Continue monitoring attendance patterns")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Initialize service
    anomaly_service = AnomalyDetectionService()
    
    # Example: Create sample attendance records
    # records = [
    #     AttendanceRecord(
    #         employee_id="EMP001",
    #         check_in_time=datetime.now() - timedelta(hours=8),
    #         check_out_time=datetime.now(),
    #         work_hours=8.0,
    #         location="office",
    #         device_type="camera",
    #         is_weekend=False,
    #         is_holiday=False
    #     )
    # ]
    # 
    # # Train models
    # anomaly_service.train_models(records)
    # 
    # # Detect anomalies
    # result = anomaly_service.detect_attendance_anomalies(records[0])
    # print(f"Anomaly detection result: {result}")
