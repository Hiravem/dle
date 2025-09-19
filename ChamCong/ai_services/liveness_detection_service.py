"""
AI Liveness Detection Service
Phát hiện "người thật" để chống tấn công giả mạo (ảnh, video, mask)
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mediapipe as mp
from typing import Tuple, List, Dict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LivenessDetectionService:
    def __init__(self, model_path: str = "models/liveness_model.h5"):
        """
        Initialize Liveness Detection Service
        
        Args:
            model_path: Path to saved liveness detection model
        """
        self.model_path = model_path
        self.liveness_model = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Liveness detection parameters
        self.eye_ar_threshold = 0.25  # Eye aspect ratio threshold
        self.mouth_ar_threshold = 0.3  # Mouth aspect ratio threshold
        self.frame_count_threshold = 3  # Minimum frames for detection
        
        self._load_model()
    
    def _build_liveness_model(self) -> Model:
        """Build CNN model for liveness detection"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')  # Real vs Fake
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _load_model(self):
        """Load or create liveness detection model"""
        try:
            if tf.io.gfile.exists(self.model_path):
                self.liveness_model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded liveness model from {self.model_path}")
            else:
                self.liveness_model = self._build_liveness_model()
                logger.info("Created new liveness detection model")
        except Exception as e:
            logger.error(f"Error loading liveness model: {e}")
            self.liveness_model = self._build_liveness_model()
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        
        Args:
            eye_landmarks: Eye landmark points
            
        Returns:
            Eye aspect ratio value
        """
        # Calculate vertical eye landmarks distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Calculate horizontal eye landmarks distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) for mouth movement detection
        
        Args:
            mouth_landmarks: Mouth landmark points
            
        Returns:
            Mouth aspect ratio value
        """
        # Calculate vertical mouth distances
        A = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[7])
        B = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])
        C = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[5])
        
        # Calculate horizontal mouth distance
        D = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])
        
        # Calculate MAR
        mar = (A + B + C) / (3.0 * D)
        return mar
    
    def extract_face_landmarks(self, image: np.ndarray) -> Dict:
        """
        Extract facial landmarks using MediaPipe
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing facial landmarks and metrics
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert landmarks to numpy array
                landmarks = np.array([
                    [lm.x * image.shape[1], lm.y * image.shape[0]] 
                    for lm in face_landmarks.landmark
                ])
                
                # Extract eye landmarks (simplified indices)
                left_eye = landmarks[33:42]  # Approximate left eye indices
                right_eye = landmarks[263:272]  # Approximate right eye indices
                
                # Extract mouth landmarks
                mouth = landmarks[61:68]  # Approximate mouth indices
                
                # Calculate metrics
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                mar = self.calculate_mouth_aspect_ratio(mouth)
                
                return {
                    'landmarks': landmarks,
                    'left_ear': left_ear,
                    'right_ear': right_ear,
                    'mar': mar,
                    'average_ear': (left_ear + right_ear) / 2.0
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def detect_blink(self, landmarks_data: Dict) -> bool:
        """
        Detect blink based on eye aspect ratio
        
        Args:
            landmarks_data: Facial landmarks data
            
        Returns:
            True if blink detected
        """
        if landmarks_data is None:
            return False
        
        return landmarks_data['average_ear'] < self.eye_ar_threshold
    
    def detect_mouth_movement(self, landmarks_data: Dict) -> bool:
        """
        Detect mouth movement based on mouth aspect ratio
        
        Args:
            landmarks_data: Facial landmarks data
            
        Returns:
            True if mouth movement detected
        """
        if landmarks_data is None:
            return False
        
        return landmarks_data['mar'] > self.mouth_ar_threshold
    
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CNN liveness model
        
        Args:
            image: Input face image
            
        Returns:
            Preprocessed image
        """
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_liveness_cnn(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Predict liveness using CNN model
        
        Args:
            image: Preprocessed face image
            
        Returns:
            Tuple of (is_live, confidence)
        """
        try:
            processed_image = self.preprocess_for_model(image)
            prediction = self.liveness_model.predict(processed_image, verbose=0)
            
            # Assuming binary classification: [fake, real]
            is_live = prediction[0][1] > 0.5
            confidence = float(prediction[0][1])
            
            return is_live, confidence
            
        except Exception as e:
            logger.error(f"Error in CNN liveness prediction: {e}")
            return False, 0.0
    
    def detect_3d_structure(self, landmarks_data: Dict) -> bool:
        """
        Detect 3D structure to identify real faces vs flat images
        
        Args:
            landmarks_data: Facial landmarks data
            
        Returns:
            True if 3D structure detected (likely real face)
        """
        if landmarks_data is None:
            return False
        
        landmarks = landmarks_data['landmarks']
        
        # Calculate depth variation using landmark z-coordinates
        # This is a simplified approach - in practice, you'd use more sophisticated 3D analysis
        try:
            # Get facial outline points
            outline_points = landmarks[0:17]  # Jawline points
            
            # Calculate variance in y-coordinates (depth approximation)
            y_variance = np.var(outline_points[:, 1])
            
            # Real faces should have more depth variation
            return y_variance > 100  # Threshold for 3D structure
            
        except Exception as e:
            logger.error(f"Error in 3D structure detection: {e}")
            return False
    
    def comprehensive_liveness_check(self, image: np.ndarray, 
                                   frame_history: List[Dict] = None) -> Dict:
        """
        Comprehensive liveness detection combining multiple methods
        
        Args:
            image: Input face image
            frame_history: Previous frame analysis results
            
        Returns:
            Dictionary with liveness analysis results
        """
        try:
            # Extract facial landmarks
            landmarks_data = self.extract_face_landmarks(image)
            
            # CNN-based liveness prediction
            cnn_live, cnn_confidence = self.predict_liveness_cnn(image)
            
            # Motion-based detection
            blink_detected = self.detect_blink(landmarks_data)
            mouth_movement = self.detect_mouth_movement(landmarks_data)
            
            # 3D structure detection
            has_3d_structure = self.detect_3d_structure(landmarks_data)
            
            # Temporal analysis (if frame history available)
            temporal_score = 0.0
            if frame_history and len(frame_history) >= self.frame_count_threshold:
                recent_blinks = sum(1 for frame in frame_history[-self.frame_count_threshold:] 
                                  if frame.get('blink_detected', False))
                temporal_score = min(recent_blinks / self.frame_count_threshold, 1.0)
            
            # Combine all scores
            scores = {
                'cnn_liveness': cnn_confidence,
                'blink_detected': 1.0 if blink_detected else 0.0,
                'mouth_movement': 1.0 if mouth_movement else 0.0,
                '3d_structure': 1.0 if has_3d_structure else 0.0,
                'temporal_activity': temporal_score
            }
            
            # Weighted combination
            weights = {
                'cnn_liveness': 0.4,
                'blink_detected': 0.2,
                'mouth_movement': 0.1,
                '3d_structure': 0.2,
                'temporal_activity': 0.1
            }
            
            final_score = sum(scores[key] * weights[key] for key in scores.keys())
            is_live = final_score > 0.6  # Threshold for live detection
            
            result = {
                'is_live': is_live,
                'confidence': final_score,
                'scores': scores,
                'timestamp': datetime.now().isoformat(),
                'blink_detected': blink_detected,
                'mouth_movement': mouth_movement,
                'has_3d_structure': has_3d_structure,
                'cnn_prediction': cnn_live,
                'cnn_confidence': cnn_confidence
            }
            
            logger.info(f"Liveness check: {is_live} (confidence: {final_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive liveness check: {e}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def continuous_liveness_verification(self, video_frames: List[np.ndarray]) -> Dict:
        """
        Perform continuous liveness verification on video frames
        
        Args:
            video_frames: List of video frames
            
        Returns:
            Overall liveness verification result
        """
        try:
            frame_results = []
            
            for frame in video_frames:
                result = self.comprehensive_liveness_check(frame, frame_results)
                frame_results.append(result)
            
            # Analyze overall results
            live_frames = sum(1 for result in frame_results if result['is_live'])
            total_frames = len(frame_results)
            live_ratio = live_frames / total_frames if total_frames > 0 else 0
            
            # Calculate average confidence
            avg_confidence = np.mean([result['confidence'] for result in frame_results])
            
            # Overall decision
            is_overall_live = live_ratio > 0.7 and avg_confidence > 0.6
            
            return {
                'is_live': is_overall_live,
                'confidence': avg_confidence,
                'live_frames_ratio': live_ratio,
                'total_frames': total_frames,
                'frame_results': frame_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in continuous liveness verification: {e}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Example usage
if __name__ == "__main__":
    # Initialize service
    liveness_service = LivenessDetectionService()
    
    # Example: Single frame liveness check
    # test_image = cv2.imread("path/to/test_face.jpg")
    # result = liveness_service.comprehensive_liveness_check(test_image)
    # print(f"Liveness result: {result}")
    
    # Example: Video stream liveness verification
    # cap = cv2.VideoCapture(0)  # Webcam
    # frames = []
    # for i in range(30):  # Collect 30 frames
    #     ret, frame = cap.read()
    #     if ret:
    #         frames.append(frame)
    # 
    # result = liveness_service.continuous_liveness_verification(frames)
    # print(f"Overall liveness: {result}")
