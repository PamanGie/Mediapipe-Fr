import cv2
import mediapipe as mp
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionBase:
    def __init__(self, database_path="face_database.json"):
        # Inisialisasi MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        
        # Database untuk menyimpan data wajah
        self.database_path = database_path
        self.face_database = self.load_database()
        
        # Target size untuk face warping
        self.target_size = (112, 112)
        
        # Threshold untuk pengenalan wajah
        self.recognition_threshold = 0.6
    
    def load_database(self):
        """Load database wajah dari file JSON"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r') as f:
                return json.load(f)
        return {"faces": []}
    
    def save_database(self):
        """Simpan database wajah ke file JSON"""
        with open(self.database_path, 'w') as f:
            json.dump(self.face_database, f)
    
    def detect_faces(self, image):
        """Deteksi wajah menggunakan MediaPipe Face Detection"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Tambahkan margin untuk bounding box
                margin = int(0.1 * w)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(iw - x, w + 2 * margin)
                h = min(ih - y, h + 2 * margin)
                
                faces.append((x, y, w, h))
        
        return faces
    
    def extract_features(self, face_image):
        """Ekstrak fitur dari wajah"""
        # Resize ke ukuran standar
        face_resized = cv2.resize(face_image, self.target_size)
        
        # Konversi ke grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Gunakan histogram sebagai feature yang ringan
        hist = cv2.calcHist([face_gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Tambahkan beberapa fitur dari LBP (Local Binary Patterns)
        lbp = self.compute_simple_lbp(face_gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        
        # Gabungkan fitur
        features = np.concatenate((hist, lbp_hist))
        
        return features
    
    def compute_simple_lbp(self, image):
        """Hitung LBP sederhana"""
        rows, cols = image.shape
        lbp_image = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] > center) << 7
                code |= (image[i-1, j] > center) << 6
                code |= (image[i-1, j+1] > center) << 5
                code |= (image[i, j+1] > center) << 4
                code |= (image[i+1, j+1] > center) << 3
                code |= (image[i+1, j] > center) << 2
                code |= (image[i+1, j-1] > center) << 1
                code |= (image[i, j-1] > center) << 0
                lbp_image[i-1, j-1] = code
                
        return lbp_image