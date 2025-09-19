"""
AI Face Recognition Service
Nhận diện khuôn mặt sử dụng CNN models và face embedding
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import face_recognition
import pickle
import os
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionService:
    def __init__(self, model_path: str = "models/face_model.h5", 
                 face_encodings_path: str = "data/face_encodings.pkl"):
        """
        Initialize Face Recognition Service
        
        Args:
            model_path: Path to saved face recognition model
            face_encodings_path: Path to stored face encodings database
        """
        self.model_path = model_path
        self.face_encodings_path = face_encodings_path
        self.face_encodings = {}
        self.face_model = None
        self.threshold = 0.6  # Similarity threshold
        
        self._load_model()
        self._load_face_encodings()
    
    def _build_model(self) -> Model:
        """Build CNN model for face recognition using MobileNetV3"""
        base_model = MobileNetV3Large(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)  # Face embedding dimension
        
        model = Model(inputs=base_model.input, outputs=x)
        return model
    
    def _load_model(self):
        """Load or create face recognition model"""
        if os.path.exists(self.model_path):
            try:
                self.face_model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded face model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Creating new model.")
                self.face_model = self._build_model()
        else:
            self.face_model = self._build_model()
            logger.info("Created new face recognition model")
    
    def _load_face_encodings(self):
        """Load stored face encodings database"""
        if os.path.exists(self.face_encodings_path):
            try:
                with open(self.face_encodings_path, 'rb') as f:
                    self.face_encodings = pickle.load(f)
                logger.info(f"Loaded {len(self.face_encodings)} face encodings")
            except Exception as e:
                logger.warning(f"Failed to load face encodings: {e}")
                self.face_encodings = {}
        else:
            self.face_encodings = {}
    
    def _save_face_encodings(self):
        """Save face encodings to database"""
        os.makedirs(os.path.dirname(self.face_encodings_path), exist_ok=True)
        with open(self.face_encodings_path, 'wb') as f:
            pickle.dump(self.face_encodings, f)
        logger.info(f"Saved {len(self.face_encodings)} face encodings")
    
    def preprocess_face(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model input
        
        Args:
            image: Input face image
            
        Returns:
            Preprocessed image ready for model
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
    
    def extract_face_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding using CNN model
        
        Args:
            image: Preprocessed face image
            
        Returns:
            Face embedding vector
        """
        try:
            embedding = self.face_model.predict(image, verbose=0)
            return embedding.flatten()
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def add_employee_face(self, employee_id: str, image: np.ndarray) -> bool:
        """
        Add new employee face to database
        
        Args:
            employee_id: Unique employee identifier
            image: Face image
            
        Returns:
            Success status
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_face(image)
            
            # Extract embedding
            embedding = self.extract_face_embedding(processed_image)
            
            if embedding is not None:
                self.face_encodings[employee_id] = embedding
                self._save_face_encodings()
                logger.info(f"Added face for employee {employee_id}")
                return True
            else:
                logger.error(f"Failed to extract embedding for employee {employee_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding employee face: {e}")
            return False
    
    def recognize_face(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize face in image
        
        Args:
            image: Input face image
            
        Returns:
            Tuple of (employee_id, confidence_score) or (None, 0.0) if not found
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_face(image)
            
            # Extract embedding
            query_embedding = self.extract_face_embedding(processed_image)
            
            if query_embedding is None:
                return None, 0.0
            
            best_match = None
            best_score = 0.0
            
            # Compare with all stored encodings
            for employee_id, stored_embedding in self.face_encodings.items():
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > best_score and similarity > self.threshold:
                    best_score = similarity
                    best_match = employee_id
            
            if best_match:
                logger.info(f"Recognized employee {best_match} with confidence {best_score:.3f}")
                return best_match, best_score
            else:
                logger.info("No matching face found")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None, 0.0
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using face_recognition library
        
        Args:
            image: Input image
            
        Returns:
            List of face bounding boxes (top, right, bottom, left)
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_image)
            
            return face_locations
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def crop_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop face from image using face location
        
        Args:
            image: Input image
            face_location: Face bounding box (top, right, bottom, left)
            
        Returns:
            Cropped face image
        """
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        return face_image
    
    def get_face_count(self) -> int:
        """Get total number of registered faces"""
        return len(self.face_encodings)
    
    def remove_employee(self, employee_id: str) -> bool:
        """
        Remove employee face from database
        
        Args:
            employee_id: Employee ID to remove
            
        Returns:
            Success status
        """
        if employee_id in self.face_encodings:
            del self.face_encodings[employee_id]
            self._save_face_encodings()
            logger.info(f"Removed employee {employee_id} from face database")
            return True
        else:
            logger.warning(f"Employee {employee_id} not found in face database")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize service
    face_service = FaceRecognitionService()
    
    # Example: Add employee face
    # employee_image = cv2.imread("path/to/employee_face.jpg")
    # face_service.add_employee_face("EMP001", employee_image)
    
    # Example: Recognize face
    # test_image = cv2.imread("path/to/test_image.jpg")
    # faces = face_service.detect_faces(test_image)
    # for face_location in faces:
    #     face_image = face_service.crop_face(test_image, face_location)
    #     employee_id, confidence = face_service.recognize_face(face_image)
    #     print(f"Recognized: {employee_id} with confidence: {confidence}")
