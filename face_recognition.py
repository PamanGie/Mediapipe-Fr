import cv2
import numpy as np
import time
from face_recognition_base import FaceRecognitionBase
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognition(FaceRecognitionBase):
    def __init__(self, database_path="face_database.json"):
        super().__init__(database_path)
        
        # Batasi frame rate untuk mengurangi beban CPU
        self.last_process_time = 0
        self.process_interval = 0.1  # Proses setiap 100ms (10 FPS)
        
        # Pengaturan tampilan
        self.show_confidence = True
        self.scale_factor = 0.75  # Faktor scaling untuk deteksi
        
        # Tambahkan variabel untuk face tracking
        self.last_faces = []
        self.face_history = []
        self.history_length = 5  # Simpan 5 frame terakhir untuk smoothing
    
    def smooth_face_boxes(self, current_faces):
        """
        Stabilkan bounding box dengan cara averaging dari beberapa frame terakhir
        """
        if not current_faces:
            self.face_history.append([])
            if len(self.face_history) > self.history_length:
                self.face_history.pop(0)
            return []
            
        # Tambahkan deteksi saat ini ke history
        self.face_history.append(current_faces)
        if len(self.face_history) > self.history_length:
            self.face_history.pop(0)
            
        # Jika belum ada cukup history, gunakan deteksi saat ini
        if len(self.face_history) < 3:
            return current_faces
            
        # Lakukan tracking sederhana berdasarkan overlap/intersection
        smoothed_faces = []
        
        # Gunakan 3 frame terakhir saja untuk smoothing
        recent_history = self.face_history[-3:]
        
        # Ambil faces dari frame terbaru
        for face in current_faces:
            x, y, w, h = face
            face_center = (x + w//2, y + h//2)
            
            # Cari faces di frame-frame sebelumnya yang mungkin sama
            matching_faces = []
            matching_faces.append(face)  # Selalu termasuk face saat ini
            
            # Periksa 2 frame terakhir
            for past_faces in recent_history[:-1]:  # Skip frame terbaru karena sudah dimasukkan
                best_match = None
                best_iou = 0.3  # Threshold IoU
                
                for past_face in past_faces:
                    px, py, pw, ph = past_face
                    past_center = (px + pw//2, py + ph//2)
                    
                    # Hitung jarak antar pusat
                    center_dist = np.sqrt((face_center[0] - past_center[0])**2 + 
                                        (face_center[1] - past_center[1])**2)
                    
                    # Jika pusat cukup dekat, hitung IoU
                    if center_dist < max(w, h) * 0.5:
                        # Intersection
                        x_left = max(x, px)
                        y_top = max(y, py)
                        x_right = min(x + w, px + pw)
                        y_bottom = min(y + h, py + ph)
                        
                        if x_right < x_left or y_bottom < y_top:
                            continue
                            
                        intersection = (x_right - x_left) * (y_bottom - y_top)
                        
                        # Union
                        area1 = w * h
                        area2 = pw * ph
                        union = area1 + area2 - intersection
                        
                        # IoU
                        iou = intersection / union
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_match = past_face
                
                if best_match:
                    matching_faces.append(best_match)
            
            # Hitung rata-rata untuk mendapatkan bounding box yang smooth
            if len(matching_faces) > 1:
                avg_x = int(sum(f[0] for f in matching_faces) / len(matching_faces))
                avg_y = int(sum(f[1] for f in matching_faces) / len(matching_faces))
                avg_w = int(sum(f[2] for f in matching_faces) / len(matching_faces))
                avg_h = int(sum(f[3] for f in matching_faces) / len(matching_faces))
                
                smoothed_faces.append((avg_x, avg_y, avg_w, avg_h))
            else:
                smoothed_faces.append(face)
                
        return smoothed_faces
    
    def recognize_face(self, image, face_box):
        """Kenali wajah berdasarkan bounding box"""
        x, y, w, h = face_box
        face_image = image[y:y+h, x:x+w]
        
        # Resize ke ukuran standar
        resized_face = cv2.resize(face_image, self.target_size)
        
        # Ekstrak fitur
        features = self.extract_features(resized_face)
        
        # Bandingkan dengan database
        best_match = None
        best_similarity = -1
        
        for face_data in self.face_database["faces"]:
            db_features = np.array(face_data["features"])
            similarity = cosine_similarity([features], [db_features])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = face_data["name"]
        
        if best_similarity > self.recognition_threshold:
            return best_match, best_similarity
        else:
            return "Unknown", best_similarity
    
    def process_frame(self, frame):
        """Proses frame untuk deteksi dan pengenalan wajah"""
        # Batasi frame rate untuk mengurangi beban CPU
        current_time = time.time()
        if current_time - self.last_process_time < self.process_interval:
            # Jika belum waktunya memproses frame baru, gunakan hasil terakhir
            if hasattr(self, 'last_processed_image') and self.last_processed_image is not None:
                return self.last_processed_image
            return frame
        
        self.last_process_time = current_time
        
        # Buat salinan frame
        image = frame.copy()
        
        # Resize image untuk mempercepat proses
        small_image = cv2.resize(image, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        
        # Deteksi wajah pada gambar yang diperkecil
        faces = self.detect_faces(small_image)
        
        # Konversi koordinat ke frame asli
        detected_faces = []
        for face_box in faces:
            x, y, w, h = [int(coord / self.scale_factor) for coord in face_box]
            detected_faces.append((x, y, w, h))
        
        # Stabilkan bounding box
        smoothed_faces = self.smooth_face_boxes(detected_faces)
        
        # Gambar semua wajah yang terdeteksi di frame asli
        for face_box in smoothed_faces:
            x, y, w, h = face_box
            
            # Kenali wajah
            name, confidence = self.recognize_face(image, (x, y, w, h))
            
            # Gambar bounding box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Tampilkan nama di atas bounding box
            label = name
            if self.show_confidence:
                label = f"{name} ({confidence:.2f})"
            
            cv2.rectangle(image, (x, y-30), (x+w, y), (0, 255, 0), -1)
            cv2.putText(image, label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tampilkan info
        info_text = "Q: Keluar | C: Toggle confidence"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Simpan hasil pemrosesan untuk digunakan jika rate dibatasi
        self.last_processed_image = image
        
        return image
    
    def run(self):
        """Jalankan pengenalan wajah dari webcam"""
        cap = cv2.VideoCapture(0)
        
        # Cek apakah database kosong
        if len(self.face_database["faces"]) == 0:
            print("Database wajah kosong! Jalankan 'face_register.py' terlebih dahulu.")
            return
        
        print(f"Database berisi {len(self.face_database['faces'])} wajah.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Gagal membaca frame dari webcam")
                break
            
            # Proses frame
            processed_frame = self.process_frame(frame)
            
            # Tampilkan
            cv2.imshow('Face Recognition', processed_frame)
            
            # Tangkap tombol keyboard
            key = cv2.waitKey(1) & 0xFF
            
            # Tekan 'q' untuk keluar
            if key == ord('q'):
                break
            
            # Tekan 'c' untuk toggle tampilan confidence
            if key == ord('c'):
                self.show_confidence = not self.show_confidence
                print(f"Confidence display: {'On' if self.show_confidence else 'Off'}")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== Sistem Pengenalan Wajah ===")
    print("Memuat database wajah...")
    
    face_recognizer = FaceRecognition()
    face_recognizer.run()