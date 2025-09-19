"""
AI Predictive Analytics Service
Dự đoán xu hướng chấm công và khả năng nghỉ việc sử dụng ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmployeeProfile:
    """Employee profile for predictive modeling"""
    employee_id: str
    age: int
    department: str
    position: str
    tenure_months: int
    salary_level: str
    education_level: str
    performance_rating: float
    manager_id: str
    team_size: int

@dataclass
class AttendanceMetrics:
    """Attendance metrics for prediction"""
    employee_id: str
    total_work_days: int
    total_late_days: int
    total_absent_days: int
    avg_work_hours: float
    avg_overtime_hours: float
    punctuality_score: float
    attendance_consistency: float
    last_30_days_attendance: float
    last_30_days_lateness: float

@dataclass
class TurnoverPrediction:
    """Turnover prediction result"""
    employee_id: str
    turnover_probability: float
    risk_level: str  # low, medium, high, critical
    key_factors: List[str]
    predicted_timeline: str  # "within_1_month", "within_3_months", "within_6_months", "low_risk"
    confidence: float
    recommendations: List[str]
    timestamp: datetime

@dataclass
class AttendanceForecast:
    """Attendance pattern forecast"""
    employee_id: str
    forecast_date: datetime
    predicted_attendance: float
    predicted_lateness: float
    predicted_overtime: float
    confidence_interval: Tuple[float, float]
    trend: str  # "improving", "declining", "stable"
    seasonal_patterns: Dict[str, float]
    timestamp: datetime

class PredictiveAnalyticsService:
    def __init__(self, model_path: str = "models/predictive_models.joblib"):
        """
        Initialize Predictive Analytics Service
        
        Args:
            model_path: Path to saved predictive models
        """
        self.model_path = model_path
        
        # Turnover prediction models
        self.turnover_model = None
        self.turnover_scaler = StandardScaler()
        
        # Attendance forecasting models
        self.attendance_model = None
        self.lateness_model = None
        self.overtime_model = None
        self.forecast_scaler = StandardScaler()
        
        # Feature engineering
        self.label_encoders = {}
        self.feature_importance = {}
        
        # Model parameters
        self.min_training_samples = 100
        self.forecast_horizon_days = 30
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained predictive models"""
        try:
            if joblib.os.path.exists(self.model_path):
                models_data = joblib.load(self.model_path)
                
                # Load turnover models
                self.turnover_model = models_data.get('turnover_model')
                self.turnover_scaler = models_data.get('turnover_scaler', StandardScaler())
                
                # Load forecasting models
                self.attendance_model = models_data.get('attendance_model')
                self.lateness_model = models_data.get('lateness_model')
                self.overtime_model = models_data.get('overtime_model')
                self.forecast_scaler = models_data.get('forecast_scaler', StandardScaler())
                
                # Load feature engineering components
                self.label_encoders = models_data.get('label_encoders', {})
                self.feature_importance = models_data.get('feature_importance', {})
                
                logger.info("Loaded pre-trained predictive models")
            else:
                logger.info("No pre-trained models found, will train new models")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            models_data = {
                'turnover_model': self.turnover_model,
                'turnover_scaler': self.turnover_scaler,
                'attendance_model': self.attendance_model,
                'lateness_model': self.lateness_model,
                'overtime_model': self.overtime_model,
                'forecast_scaler': self.forecast_scaler,
                'label_encoders': self.label_encoders,
                'feature_importance': self.feature_importance
            }
            joblib.dump(models_data, self.model_path)
            logger.info(f"Saved predictive models to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _prepare_turnover_features(self, profile: EmployeeProfile, 
                                  metrics: AttendanceMetrics) -> np.ndarray:
        """Prepare features for turnover prediction"""
        try:
            # Encode categorical features
            features = {
                'age': float(profile.age),
                'tenure_months': float(profile.tenure_months),
                'performance_rating': profile.performance_rating,
                'team_size': float(profile.team_size),
                'punctuality_score': metrics.punctuality_score,
                'attendance_consistency': metrics.attendance_consistency,
                'last_30_days_attendance': metrics.last_30_days_attendance,
                'last_30_days_lateness': metrics.last_30_days_lateness,
                'avg_work_hours': metrics.avg_work_hours,
                'avg_overtime_hours': metrics.avg_overtime_hours
            }
            
            # Encode categorical variables
            categorical_features = ['department', 'position', 'salary_level', 'education_level']
            for feature_name in categorical_features:
                feature_value = getattr(profile, feature_name)
                
                if feature_name not in self.label_encoders:
                    self.label_encoders[feature_name] = LabelEncoder()
                    self.label_encoders[feature_name].fit([feature_value])
                
                try:
                    encoded_value = self.label_encoders[feature_name].transform([feature_value])[0]
                    features[f'{feature_name}_encoded'] = float(encoded_value)
                except ValueError:
                    features[f'{feature_name}_encoded'] = -1.0
            
            # Convert to array
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error preparing turnover features: {e}")
            return np.array([])
    
    def _prepare_forecast_features(self, employee_id: str, 
                                  historical_data: List[Dict]) -> np.ndarray:
        """Prepare features for attendance forecasting"""
        try:
            if len(historical_data) < 30:
                return np.array([])
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate rolling statistics
            window_sizes = [7, 14, 30]
            features = {}
            
            for window in window_sizes:
                if len(df) >= window:
                    features[f'attendance_avg_{window}d'] = df['attendance'].rolling(window).mean().iloc[-1]
                    features[f'lateness_avg_{window}d'] = df['lateness'].rolling(window).mean().iloc[-1]
                    features[f'overtime_avg_{window}d'] = df['overtime'].rolling(window).mean().iloc[-1]
                    
                    features[f'attendance_std_{window}d'] = df['attendance'].rolling(window).std().iloc[-1]
                    features[f'lateness_std_{window}d'] = df['lateness'].rolling(window).std().iloc[-1]
                    features[f'overtime_std_{window}d'] = df['overtime'].rolling(window).std().iloc[-1]
            
            # Time-based features
            last_date = df['date'].iloc[-1]
            features['day_of_week'] = float(last_date.weekday())
            features['day_of_month'] = float(last_date.day)
            features['month'] = float(last_date.month)
            features['is_weekend'] = 1.0 if last_date.weekday() >= 5 else 0.0
            
            # Trend features
            if len(df) >= 14:
                recent_trend = df['attendance'].tail(14).mean() - df['attendance'].head(14).mean()
                features['attendance_trend'] = recent_trend
            
            # Convert to array
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error preparing forecast features: {e}")
            return np.array([])
    
    def train_turnover_model(self, profiles: List[EmployeeProfile], 
                           metrics: List[AttendanceMetrics],
                           turnover_labels: List[bool]) -> bool:
        """
        Train turnover prediction model
        
        Args:
            profiles: Employee profiles
            metrics: Attendance metrics
            turnover_labels: True if employee left within 6 months
            
        Returns:
            Training success status
        """
        try:
            if len(profiles) < self.min_training_samples:
                logger.warning(f"Insufficient data for training: {len(profiles)} < {self.min_training_samples}")
                return False
            
            # Prepare features
            feature_matrix = []
            labels = []
            
            for i, (profile, metric) in enumerate(zip(profiles, metrics)):
                features = self._prepare_turnover_features(profile, metric)
                if features.size > 0:
                    feature_matrix.append(features.flatten())
                    labels.append(turnover_labels[i])
            
            if len(feature_matrix) == 0:
                logger.error("Failed to prepare features for turnover training")
                return False
            
            feature_matrix = np.array(feature_matrix)
            labels = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                feature_matrix, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Scale features
            X_train_scaled = self.turnover_scaler.fit_transform(X_train)
            X_test_scaled = self.turnover_scaler.transform(X_test)
            
            # Train XGBoost model
            self.turnover_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            self.turnover_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.turnover_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store feature importance
            feature_names = list(range(feature_matrix.shape[1]))
            importance_scores = self.turnover_model.feature_importances_
            self.feature_importance['turnover'] = dict(zip(feature_names, importance_scores))
            
            # Save models
            self._save_models()
            
            logger.info(f"Turnover model trained with accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training turnover model: {e}")
            return False
    
    def predict_turnover(self, profile: EmployeeProfile, 
                        metrics: AttendanceMetrics) -> TurnoverPrediction:
        """
        Predict employee turnover risk
        
        Args:
            profile: Employee profile
            metrics: Attendance metrics
            
        Returns:
            Turnover prediction result
        """
        try:
            if self.turnover_model is None:
                return TurnoverPrediction(
                    employee_id=profile.employee_id,
                    turnover_probability=0.0,
                    risk_level="unknown",
                    key_factors=[],
                    predicted_timeline="model_not_trained",
                    confidence=0.0,
                    recommendations=["Model not trained yet"],
                    timestamp=datetime.now()
                )
            
            # Prepare features
            features = self._prepare_turnover_features(profile, metrics)
            
            if features.size == 0:
                return TurnoverPrediction(
                    employee_id=profile.employee_id,
                    turnover_probability=0.0,
                    risk_level="unknown",
                    key_factors=[],
                    predicted_timeline="feature_error",
                    confidence=0.0,
                    recommendations=["Failed to prepare features"],
                    timestamp=datetime.now()
                )
            
            # Scale features
            features_scaled = self.turnover_scaler.transform(features)
            
            # Predict probability
            turnover_prob = self.turnover_model.predict_proba(features_scaled)[0][1]
            
            # Determine risk level
            if turnover_prob >= 0.8:
                risk_level = "critical"
                timeline = "within_1_month"
            elif turnover_prob >= 0.6:
                risk_level = "high"
                timeline = "within_3_months"
            elif turnover_prob >= 0.4:
                risk_level = "medium"
                timeline = "within_6_months"
            else:
                risk_level = "low"
                timeline = "low_risk"
            
            # Identify key factors
            key_factors = self._identify_key_factors(features.flatten(), profile, metrics)
            
            # Generate recommendations
            recommendations = self._generate_turnover_recommendations(risk_level, key_factors)
            
            return TurnoverPrediction(
                employee_id=profile.employee_id,
                turnover_probability=turnover_prob,
                risk_level=risk_level,
                key_factors=key_factors,
                predicted_timeline=timeline,
                confidence=min(turnover_prob * 2, 1.0),
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting turnover: {e}")
            return TurnoverPrediction(
                employee_id=profile.employee_id,
                turnover_probability=1.0,
                risk_level="error",
                key_factors=[],
                predicted_timeline="prediction_error",
                confidence=0.0,
                recommendations=[f"Prediction error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    def _identify_key_factors(self, features: np.ndarray, 
                             profile: EmployeeProfile, 
                             metrics: AttendanceMetrics) -> List[str]:
        """Identify key factors contributing to turnover risk"""
        factors = []
        
        # Check attendance factors
        if metrics.punctuality_score < 0.7:
            factors.append("Poor punctuality")
        
        if metrics.attendance_consistency < 0.8:
            factors.append("Inconsistent attendance")
        
        if metrics.last_30_days_lateness > 30:
            factors.append("Recent frequent lateness")
        
        # Check performance factors
        if profile.performance_rating < 3.0:
            factors.append("Low performance rating")
        
        # Check tenure factors
        if profile.tenure_months < 6:
            factors.append("Short tenure")
        elif profile.tenure_months > 60:
            factors.append("Long tenure (potential stagnation)")
        
        # Check workload factors
        if metrics.avg_overtime_hours > 10:
            factors.append("Excessive overtime")
        
        if metrics.avg_work_hours > 50:
            factors.append("Overworked")
        
        return factors
    
    def _generate_turnover_recommendations(self, risk_level: str, 
                                         key_factors: List[str]) -> List[str]:
        """Generate recommendations based on turnover risk"""
        recommendations = []
        
        if risk_level == "critical":
            recommendations.extend([
                "Immediate intervention required",
                "Schedule one-on-one meeting with employee",
                "Review workload and work-life balance"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "Monitor employee closely",
                "Discuss career development opportunities",
                "Address identified risk factors"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Regular check-ins recommended",
                "Focus on employee engagement",
                "Address performance concerns"
            ])
        else:
            recommendations.append("Continue current management approach")
        
        # Add specific recommendations based on key factors
        if "Poor punctuality" in key_factors:
            recommendations.append("Implement punctuality improvement plan")
        
        if "Excessive overtime" in key_factors:
            recommendations.append("Review and redistribute workload")
        
        if "Low performance rating" in key_factors:
            recommendations.append("Develop performance improvement plan")
        
        return recommendations
    
    def train_forecasting_models(self, historical_data: Dict[str, List[Dict]]) -> bool:
        """
        Train attendance forecasting models
        
        Args:
            historical_data: Dictionary of employee_id -> list of historical records
            
        Returns:
            Training success status
        """
        try:
            # Prepare training data
            all_features = []
            attendance_targets = []
            lateness_targets = []
            overtime_targets = []
            
            for employee_id, records in historical_data.items():
                if len(records) < 30:
                    continue
                
                # Prepare features for each time point
                for i in range(30, len(records)):
                    window_data = records[i-30:i]
                    features = self._prepare_forecast_features(employee_id, window_data)
                    
                    if features.size > 0:
                        all_features.append(features.flatten())
                        attendance_targets.append(records[i]['attendance'])
                        lateness_targets.append(records[i]['lateness'])
                        overtime_targets.append(records[i]['overtime'])
            
            if len(all_features) < self.min_training_samples:
                logger.warning("Insufficient data for forecasting training")
                return False
            
            # Convert to arrays
            X = np.array(all_features)
            y_attendance = np.array(attendance_targets)
            y_lateness = np.array(lateness_targets)
            y_overtime = np.array(overtime_targets)
            
            # Scale features
            X_scaled = self.forecast_scaler.fit_transform(X)
            
            # Train models
            self.attendance_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.lateness_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.overtime_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.attendance_model.fit(X_scaled, y_attendance)
            self.lateness_model.fit(X_scaled, y_lateness)
            self.overtime_model.fit(X_scaled, y_overtime)
            
            # Store feature importance
            self.feature_importance['attendance'] = dict(zip(
                range(X.shape[1]), self.attendance_model.feature_importances_
            ))
            self.feature_importance['lateness'] = dict(zip(
                range(X.shape[1]), self.lateness_model.feature_importances_
            ))
            self.feature_importance['overtime'] = dict(zip(
                range(X.shape[1]), self.overtime_model.feature_importances_
            ))
            
            # Save models
            self._save_models()
            
            logger.info("Forecasting models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training forecasting models: {e}")
            return False
    
    def forecast_attendance(self, employee_id: str, 
                          historical_data: List[Dict]) -> AttendanceForecast:
        """
        Forecast attendance patterns for employee
        
        Args:
            employee_id: Employee identifier
            historical_data: Historical attendance data
            
        Returns:
            Attendance forecast result
        """
        try:
            if (self.attendance_model is None or 
                self.lateness_model is None or 
                self.overtime_model is None):
                return AttendanceForecast(
                    employee_id=employee_id,
                    forecast_date=datetime.now() + timedelta(days=7),
                    predicted_attendance=0.0,
                    predicted_lateness=0.0,
                    predicted_overtime=0.0,
                    confidence_interval=(0.0, 0.0),
                    trend="unknown",
                    seasonal_patterns={},
                    timestamp=datetime.now()
                )
            
            # Prepare features
            features = self._prepare_forecast_features(employee_id, historical_data)
            
            if features.size == 0:
                return AttendanceForecast(
                    employee_id=employee_id,
                    forecast_date=datetime.now() + timedelta(days=7),
                    predicted_attendance=0.0,
                    predicted_lateness=0.0,
                    predicted_overtime=0.0,
                    confidence_interval=(0.0, 0.0),
                    trend="insufficient_data",
                    seasonal_patterns={},
                    timestamp=datetime.now()
                )
            
            # Scale features
            features_scaled = self.forecast_scaler.transform(features)
            
            # Make predictions
            pred_attendance = self.attendance_model.predict(features_scaled)[0]
            pred_lateness = self.lateness_model.predict(features_scaled)[0]
            pred_overtime = self.overtime_model.predict(features_scaled)[0]
            
            # Calculate confidence interval (simplified)
            std_attendance = np.std([r['attendance'] for r in historical_data[-30:]])
            confidence_lower = max(0, pred_attendance - 1.96 * std_attendance)
            confidence_upper = min(1, pred_attendance + 1.96 * std_attendance)
            
            # Determine trend
            if len(historical_data) >= 14:
                recent_avg = np.mean([r['attendance'] for r in historical_data[-14:]])
                older_avg = np.mean([r['attendance'] for r in historical_data[-28:-14]])
                
                if recent_avg > older_avg + 0.05:
                    trend = "improving"
                elif recent_avg < older_avg - 0.05:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Calculate seasonal patterns
            seasonal_patterns = self._calculate_seasonal_patterns(historical_data)
            
            return AttendanceForecast(
                employee_id=employee_id,
                forecast_date=datetime.now() + timedelta(days=7),
                predicted_attendance=pred_attendance,
                predicted_lateness=pred_lateness,
                predicted_overtime=pred_overtime,
                confidence_interval=(confidence_lower, confidence_upper),
                trend=trend,
                seasonal_patterns=seasonal_patterns,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error forecasting attendance: {e}")
            return AttendanceForecast(
                employee_id=employee_id,
                forecast_date=datetime.now() + timedelta(days=7),
                predicted_attendance=0.0,
                predicted_lateness=0.0,
                predicted_overtime=0.0,
                confidence_interval=(0.0, 0.0),
                trend="prediction_error",
                seasonal_patterns={},
                timestamp=datetime.now()
            )
    
    def _calculate_seasonal_patterns(self, historical_data: List[Dict]) -> Dict[str, float]:
        """Calculate seasonal patterns in attendance data"""
        try:
            if len(historical_data) < 30:
                return {}
            
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            
            patterns = {}
            
            # Day of week patterns
            dow_patterns = df.groupby('day_of_week')['attendance'].mean()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for i, day in enumerate(day_names):
                if i in dow_patterns.index:
                    patterns[f'{day}_attendance'] = float(dow_patterns[i])
            
            # Monthly patterns
            monthly_patterns = df.groupby('month')['attendance'].mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i, month in enumerate(month_names, 1):
                if i in monthly_patterns.index:
                    patterns[f'{month}_attendance'] = float(monthly_patterns[i])
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error calculating seasonal patterns: {e}")
            return {}
    
    def generate_predictive_report(self, employee_id: str,
                                 profile: EmployeeProfile,
                                 metrics: AttendanceMetrics,
                                 historical_data: List[Dict]) -> Dict:
        """
        Generate comprehensive predictive analytics report
        
        Args:
            employee_id: Employee identifier
            profile: Employee profile
            metrics: Attendance metrics
            historical_data: Historical attendance data
            
        Returns:
            Comprehensive predictive report
        """
        try:
            # Predict turnover
            turnover_prediction = self.predict_turnover(profile, metrics)
            
            # Forecast attendance
            attendance_forecast = self.forecast_attendance(employee_id, historical_data)
            
            # Generate insights
            insights = self._generate_predictive_insights(
                turnover_prediction, attendance_forecast, profile, metrics
            )
            
            return {
                'employee_id': employee_id,
                'report_date': datetime.now().isoformat(),
                'turnover_prediction': asdict(turnover_prediction),
                'attendance_forecast': asdict(attendance_forecast),
                'insights': insights,
                'recommendations': self._generate_comprehensive_recommendations(
                    turnover_prediction, attendance_forecast, profile, metrics
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating predictive report: {e}")
            return {
                'employee_id': employee_id,
                'error': str(e),
                'report_date': datetime.now().isoformat()
            }
    
    def _generate_predictive_insights(self, turnover_pred: TurnoverPrediction,
                                    attendance_forecast: AttendanceForecast,
                                    profile: EmployeeProfile,
                                    metrics: AttendanceMetrics) -> List[str]:
        """Generate predictive insights"""
        insights = []
        
        # Turnover insights
        if turnover_pred.risk_level == "critical":
            insights.append("High risk of turnover - immediate action required")
        elif turnover_pred.risk_level == "high":
            insights.append("Elevated turnover risk - monitor closely")
        
        # Attendance insights
        if attendance_forecast.trend == "declining":
            insights.append("Attendance trend is declining")
        elif attendance_forecast.trend == "improving":
            insights.append("Attendance trend is improving")
        
        # Performance insights
        if profile.performance_rating < 3.0 and turnover_pred.risk_level in ["high", "critical"]:
            insights.append("Performance and retention concerns align")
        
        # Workload insights
        if metrics.avg_overtime_hours > 10 and attendance_forecast.predicted_overtime > 8:
            insights.append("Overtime pattern suggests potential burnout risk")
        
        return insights
    
    def _generate_comprehensive_recommendations(self, turnover_pred: TurnoverPrediction,
                                              attendance_forecast: AttendanceForecast,
                                              profile: EmployeeProfile,
                                              metrics: AttendanceMetrics) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Combine turnover and attendance recommendations
        recommendations.extend(turnover_pred.recommendations)
        
        # Add attendance-specific recommendations
        if attendance_forecast.trend == "declining":
            recommendations.append("Implement attendance improvement plan")
        
        if attendance_forecast.predicted_lateness > 20:
            recommendations.append("Address punctuality concerns proactively")
        
        if attendance_forecast.predicted_overtime > 8:
            recommendations.append("Review workload distribution to prevent burnout")
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Initialize service
    predictive_service = PredictiveAnalyticsService()
    
    # Example: Create sample data
    # profile = EmployeeProfile(
    #     employee_id="EMP001",
    #     age=30,
    #     department="Engineering",
    #     position="Developer",
    #     tenure_months=24,
    #     salary_level="mid",
    #     education_level="bachelor",
    #     performance_rating=4.0,
    #     manager_id="MGR001",
    #     team_size=8
    # )
    # 
    # metrics = AttendanceMetrics(
    #     employee_id="EMP001",
    #     total_work_days=200,
    #     total_late_days=15,
    #     total_absent_days=5,
    #     avg_work_hours=8.5,
    #     avg_overtime_hours=2.0,
    #     punctuality_score=0.85,
    #     attendance_consistency=0.92,
    #     last_30_days_attendance=0.88,
    #     last_30_days_lateness=12.0
    # )
    # 
    # # Predict turnover
    # turnover_result = predictive_service.predict_turnover(profile, metrics)
    # print(f"Turnover prediction: {turnover_result}")
