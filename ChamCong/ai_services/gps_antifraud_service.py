"""
AI GPS Anti-Fraud Service
Chống gian lận GPS với geofencing và phân tích multi-sensor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LocationData:
    """Location data structure"""
    latitude: float
    longitude: float
    accuracy: float
    timestamp: datetime
    altitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None

@dataclass
class WiFiData:
    """WiFi access point data"""
    bssid: str
    ssid: str
    signal_strength: int
    frequency: int
    timestamp: datetime

@dataclass
class NFCData:
    """NFC beacon data"""
    beacon_id: str
    distance: float
    timestamp: datetime

@dataclass
class Geofence:
    """Geofence definition"""
    name: str
    center_lat: float
    center_lon: float
    radius_meters: float
    allowed_deviation: float = 50.0  # meters

class GPSAntiFraudService:
    def __init__(self, geofences_file: str = "config/geofences.json"):
        """
        Initialize GPS Anti-Fraud Service
        
        Args:
            geofences_file: Path to geofences configuration file
        """
        self.geofences_file = geofences_file
        self.geofences: List[Geofence] = []
        self.location_history: Dict[str, List[LocationData]] = {}
        self.wifi_fingerprints: Dict[str, List[WiFiData]] = {}
        self.nfc_beacons: Dict[str, List[NFCData]] = {}
        
        # Fraud detection parameters
        self.max_location_jump = 1000  # meters (impossible movement)
        self.min_checkin_distance = 10  # meters (minimum distance from geofence center)
        self.suspicious_speed_threshold = 200  # km/h (impossible speed)
        self.time_window_minutes = 30  # Time window for location analysis
        
        self._load_geofences()
    
    def _load_geofences(self):
        """Load geofences from configuration file"""
        try:
            with open(self.geofences_file, 'r') as f:
                geofences_data = json.load(f)
            
            for gf_data in geofences_data.get('geofences', []):
                geofence = Geofence(
                    name=gf_data['name'],
                    center_lat=gf_data['center_lat'],
                    center_lon=gf_data['center_lon'],
                    radius_meters=gf_data['radius_meters'],
                    allowed_deviation=gf_data.get('allowed_deviation', 50.0)
                )
                self.geofences.append(geofence)
            
            logger.info(f"Loaded {len(self.geofences)} geofences")
            
        except FileNotFoundError:
            logger.warning(f"Geofences file not found: {self.geofences_file}")
            # Create default geofence
            self._create_default_geofence()
        except Exception as e:
            logger.error(f"Error loading geofences: {e}")
            self._create_default_geofence()
    
    def _create_default_geofence(self):
        """Create default geofence for testing"""
        default_geofence = Geofence(
            name="Default Office",
            center_lat=10.762622,  # Ho Chi Minh City coordinates
            center_lon=106.660172,
            radius_meters=100,
            allowed_deviation=50.0
        )
        self.geofences.append(default_geofence)
        logger.info("Created default geofence")
    
    def calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates in meters
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
            
        Returns:
            Distance in meters
        """
        return geodesic((lat1, lon1), (lat2, lon2)).meters
    
    def is_within_geofence(self, location: LocationData, 
                          geofence: Geofence) -> Tuple[bool, float]:
        """
        Check if location is within geofence
        
        Args:
            location: Location data to check
            geofence: Geofence definition
            
        Returns:
            Tuple of (is_within, distance_from_center)
        """
        distance = self.calculate_distance(
            location.latitude, location.longitude,
            geofence.center_lat, geofence.center_lon
        )
        
        is_within = distance <= geofence.radius_meters
        return is_within, distance
    
    def find_valid_geofences(self, location: LocationData) -> List[Tuple[Geofence, float]]:
        """
        Find all geofences that contain the given location
        
        Args:
            location: Location to check
            
        Returns:
            List of (geofence, distance) tuples
        """
        valid_geofences = []
        
        for geofence in self.geofences:
            is_within, distance = self.is_within_geofence(location, geofence)
            if is_within:
                valid_geofences.append((geofence, distance))
        
        return valid_geofences
    
    def add_location_data(self, employee_id: str, location: LocationData):
        """
        Add location data to employee history
        
        Args:
            employee_id: Employee identifier
            location: Location data
        """
        if employee_id not in self.location_history:
            self.location_history[employee_id] = []
        
        self.location_history[employee_id].append(location)
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.location_history[employee_id] = [
            loc for loc in self.location_history[employee_id] 
            if loc.timestamp > cutoff_time
        ]
    
    def add_wifi_data(self, employee_id: str, wifi_data: List[WiFiData]):
        """
        Add WiFi fingerprint data
        
        Args:
            employee_id: Employee identifier
            wifi_data: List of WiFi access point data
        """
        if employee_id not in self.wifi_fingerprints:
            self.wifi_fingerprints[employee_id] = []
        
        self.wifi_fingerprints[employee_id].extend(wifi_data)
    
    def add_nfc_data(self, employee_id: str, nfc_data: NFCData):
        """
        Add NFC beacon data
        
        Args:
            employee_id: Employee identifier
            nfc_data: NFC beacon data
        """
        if employee_id not in self.nfc_beacons:
            self.nfc_beacons[employee_id] = []
        
        self.nfc_beacons[employee_id].append(nfc_data)
    
    def detect_impossible_movement(self, employee_id: str, 
                                  new_location: LocationData) -> Dict:
        """
        Detect impossible movement patterns
        
        Args:
            employee_id: Employee identifier
            new_location: New location to check
            
        Returns:
            Dictionary with fraud detection results
        """
        try:
            if employee_id not in self.location_history or len(self.location_history[employee_id]) == 0:
                return {
                    'is_fraud': False,
                    'reason': 'No location history',
                    'confidence': 0.0
                }
            
            # Get recent location history
            recent_locations = [
                loc for loc in self.location_history[employee_id]
                if (new_location.timestamp - loc.timestamp).total_seconds() <= 3600  # Last hour
            ]
            
            if not recent_locations:
                return {
                    'is_fraud': False,
                    'reason': 'No recent location history',
                    'confidence': 0.0
                }
            
            # Find most recent location
            last_location = max(recent_locations, key=lambda x: x.timestamp)
            
            # Calculate distance and time difference
            distance = self.calculate_distance(
                last_location.latitude, last_location.longitude,
                new_location.latitude, new_location.longitude
            )
            
            time_diff = (new_location.timestamp - last_location.timestamp).total_seconds()
            
            # Calculate required speed
            if time_diff > 0:
                speed_kmh = (distance / 1000) / (time_diff / 3600)
                
                # Check for impossible speed
                if speed_kmh > self.suspicious_speed_threshold:
                    return {
                        'is_fraud': True,
                        'reason': f'Impossible speed: {speed_kmh:.1f} km/h',
                        'confidence': min(speed_kmh / self.suspicious_speed_threshold, 1.0),
                        'speed_kmh': speed_kmh,
                        'distance_meters': distance,
                        'time_seconds': time_diff
                    }
            
            # Check for impossible location jump
            if distance > self.max_location_jump:
                return {
                    'is_fraud': True,
                    'reason': f'Impossible location jump: {distance:.1f}m',
                    'confidence': min(distance / self.max_location_jump, 1.0),
                    'distance_meters': distance,
                    'time_seconds': time_diff
                }
            
            return {
                'is_fraud': False,
                'reason': 'Movement appears normal',
                'confidence': 0.0,
                'distance_meters': distance,
                'time_seconds': time_diff
            }
            
        except Exception as e:
            logger.error(f"Error in impossible movement detection: {e}")
            return {
                'is_fraud': False,
                'reason': f'Error: {str(e)}',
                'confidence': 0.0
            }
    
    def detect_gps_spoofing(self, location: LocationData, 
                           wifi_data: List[WiFiData] = None,
                           nfc_data: NFCData = None) -> Dict:
        """
        Detect GPS spoofing using multi-sensor analysis
        
        Args:
            location: GPS location data
            wifi_data: WiFi fingerprint data
            nfc_data: NFC beacon data
            
        Returns:
            Dictionary with spoofing detection results
        """
        try:
            spoofing_indicators = []
            confidence_scores = []
            
            # Check GPS accuracy
            if location.accuracy > 100:  # Poor GPS accuracy
                spoofing_indicators.append("Poor GPS accuracy")
                confidence_scores.append(0.3)
            
            # Check for suspiciously perfect coordinates
            lat_precision = len(str(location.latitude).split('.')[-1]) if '.' in str(location.latitude) else 0
            lon_precision = len(str(location.longitude).split('.')[-1]) if '.' in str(location.longitude) else 0
            
            if lat_precision > 6 or lon_precision > 6:  # Too precise
                spoofing_indicators.append("Suspicious coordinate precision")
                confidence_scores.append(0.4)
            
            # WiFi fingerprint validation
            if wifi_data:
                wifi_confidence = self._validate_wifi_fingerprint(location, wifi_data)
                if wifi_confidence < 0.5:
                    spoofing_indicators.append("WiFi fingerprint mismatch")
                    confidence_scores.append(1.0 - wifi_confidence)
            
            # NFC beacon validation
            if nfc_data:
                nfc_confidence = self._validate_nfc_beacon(location, nfc_data)
                if nfc_confidence < 0.5:
                    spoofing_indicators.append("NFC beacon mismatch")
                    confidence_scores.append(1.0 - nfc_confidence)
            
            # Calculate overall spoofing confidence
            if confidence_scores:
                overall_confidence = max(confidence_scores)
                is_spoofed = overall_confidence > 0.6
            else:
                overall_confidence = 0.0
                is_spoofed = False
            
            return {
                'is_spoofed': is_spoofed,
                'confidence': overall_confidence,
                'indicators': spoofing_indicators,
                'gps_accuracy': location.accuracy,
                'wifi_validation': wifi_confidence if wifi_data else None,
                'nfc_validation': nfc_confidence if nfc_data else None
            }
            
        except Exception as e:
            logger.error(f"Error in GPS spoofing detection: {e}")
            return {
                'is_spoofed': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _validate_wifi_fingerprint(self, location: LocationData, 
                                  wifi_data: List[WiFiData]) -> float:
        """
        Validate WiFi fingerprint against location
        
        Args:
            location: GPS location
            wifi_data: WiFi access points data
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # This is a simplified implementation
            # In practice, you'd compare against a WiFi fingerprint database
            
            # Check if WiFi data is consistent with location
            # For now, return a basic validation score
            if len(wifi_data) < 2:
                return 0.3  # Low confidence with few APs
            
            # Check signal strength consistency
            avg_signal = np.mean([ap.signal_strength for ap in wifi_data])
            if avg_signal > -30:  # Too strong signal
                return 0.4
            
            return 0.8  # Default high confidence
            
        except Exception as e:
            logger.error(f"Error in WiFi fingerprint validation: {e}")
            return 0.0
    
    def _validate_nfc_beacon(self, location: LocationData, 
                            nfc_data: NFCData) -> float:
        """
        Validate NFC beacon data against location
        
        Args:
            location: GPS location
            nfc_data: NFC beacon data
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Check if NFC beacon distance is consistent with location
            # This is a simplified implementation
            
            if nfc_data.distance > 10:  # Too far from beacon
                return 0.2
            
            return 0.9  # High confidence for close NFC beacon
            
        except Exception as e:
            logger.error(f"Error in NFC beacon validation: {e}")
            return 0.0
    
    def detect_location_clustering_anomaly(self, employee_id: str) -> Dict:
        """
        Detect anomalies in location clustering patterns
        
        Args:
            employee_id: Employee identifier
            
        Returns:
            Dictionary with clustering analysis results
        """
        try:
            if employee_id not in self.location_history or len(self.location_history[employee_id]) < 10:
                return {
                    'is_anomaly': False,
                    'reason': 'Insufficient location history',
                    'confidence': 0.0
                }
            
            # Get recent locations
            recent_locations = self.location_history[employee_id][-50:]  # Last 50 locations
            
            # Prepare data for clustering
            coordinates = np.array([
                [loc.latitude, loc.longitude] for loc in recent_locations
            ])
            
            # Standardize coordinates
            scaler = StandardScaler()
            coordinates_scaled = scaler.fit_transform(coordinates)
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=0.3, min_samples=3).fit(coordinates_scaled)
            
            # Analyze clustering results
            unique_labels = set(clustering.labels_)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(clustering.labels_).count(-1)
            
            # Calculate anomaly score
            noise_ratio = n_noise / len(recent_locations)
            
            # High noise ratio indicates anomalous movement patterns
            is_anomaly = noise_ratio > 0.3
            confidence = noise_ratio
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'n_clusters': n_clusters,
                'noise_points': n_noise,
                'noise_ratio': noise_ratio,
                'total_points': len(recent_locations)
            }
            
        except Exception as e:
            logger.error(f"Error in location clustering analysis: {e}")
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def comprehensive_location_validation(self, employee_id: str, 
                                        location: LocationData,
                                        wifi_data: List[WiFiData] = None,
                                        nfc_data: NFCData = None) -> Dict:
        """
        Comprehensive location validation combining all fraud detection methods
        
        Args:
            employee_id: Employee identifier
            location: Location data to validate
            wifi_data: WiFi fingerprint data
            nfc_data: NFC beacon data
            
        Returns:
            Dictionary with comprehensive validation results
        """
        try:
            # Add location to history
            self.add_location_data(employee_id, location)
            
            # Perform various fraud detection checks
            movement_check = self.detect_impossible_movement(employee_id, location)
            spoofing_check = self.detect_gps_spoofing(location, wifi_data, nfc_data)
            clustering_check = self.detect_location_clustering_anomaly(employee_id)
            
            # Check geofence validation
            valid_geofences = self.find_valid_geofences(location)
            geofence_valid = len(valid_geofences) > 0
            
            # Calculate overall fraud score
            fraud_scores = []
            
            if movement_check['is_fraud']:
                fraud_scores.append(movement_check['confidence'])
            
            if spoofing_check['is_spoofed']:
                fraud_scores.append(spoofing_check['confidence'])
            
            if clustering_check['is_anomaly']:
                fraud_scores.append(clustering_check['confidence'])
            
            # Overall fraud assessment
            overall_fraud_score = max(fraud_scores) if fraud_scores else 0.0
            is_fraud = overall_fraud_score > 0.6
            
            # Determine check-in validity
            is_valid_checkin = geofence_valid and not is_fraud
            
            result = {
                'is_valid_checkin': is_valid_checkin,
                'is_fraud': is_fraud,
                'fraud_score': overall_fraud_score,
                'geofence_valid': geofence_valid,
                'valid_geofences': [gf.name for gf, _ in valid_geofences],
                'movement_check': movement_check,
                'spoofing_check': spoofing_check,
                'clustering_check': clustering_check,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Location validation for {employee_id}: "
                       f"Valid={is_valid_checkin}, Fraud={is_fraud}, Score={overall_fraud_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive location validation: {e}")
            return {
                'is_valid_checkin': False,
                'is_fraud': True,
                'fraud_score': 1.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Example usage
if __name__ == "__main__":
    # Initialize service
    gps_service = GPSAntiFraudService()
    
    # Example: Validate check-in location
    # location = LocationData(
    #     latitude=10.762622,
    #     longitude=106.660172,
    #     accuracy=5.0,
    #     timestamp=datetime.now()
    # )
    # 
    # wifi_data = [
    #     WiFiData("aa:bb:cc:dd:ee:ff", "Office_WiFi", -45, 2437, datetime.now())
    # ]
    # 
    # result = gps_service.comprehensive_location_validation(
    #     "EMP001", location, wifi_data
    # )
    # print(f"Validation result: {result}")
