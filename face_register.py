import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

class FaceRecognitionSystem:
    def __init__(self, database_path="face_database.json"):
        # Inisialisasi MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        
        # Inisialisasi MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Database
        self.database_path = database_path
        self.face_database = self.load_database()
        
        # Pengaturan
        self.target_size = (112, 112)
        self.recognition_threshold = 0.6
        
        # Status
        self.camera_active = False
        self.registering = False
        self.recognizing = False
        self.cap = None
        self.current_frame = None
        
        # GUI components
        self.root = None
        self.panel = None
        self.status_label = None
    
    def load_database(self):
        """Load database wajah"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    return json.load(f)
            except:
                return {"faces": []}
        return {"faces": []}
    
    def save_database(self):
        """Simpan database wajah"""
        with open(self.database_path, 'w') as f:
            json.dump(self.face_database, f)
    
    def detect_faces(self, image):
        """Deteksi wajah dengan MediaPipe Face Detection"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                              int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Margin
                margin = int(0.1 * w)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(iw - x, w + 2 * margin)
                h = min(ih - y, h + 2 * margin)
                
                faces.append((x, y, w, h))
        
        return faces
    
    def extract_landmarks(self, image):
        """Ekstrak landmark wajah"""
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return None
                
            landmarks = []
            for landmark in results.multi_face_landmarks[0].landmark:
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks.append((x, y))
                
            return landmarks
    
    def warp_face(self, image, landmarks):
        """Warp wajah berdasarkan landmark"""
        if not landmarks or len(landmarks) < 468:
            return None
            
        # Pilih landmark penting
        key_points = [33, 263, 1]  # Mata kiri, mata kanan, hidung
        
        src_points = np.array([landmarks[idx] for idx in key_points], dtype=np.float32)
        
        # Titik target
        dst_points = np.array([
            [30, 45],   # Mata kiri
            [82, 45],   # Mata kanan
            [56, 70],   # Hidung
        ], dtype=np.float32)
        
        # Transformasi
        M = cv2.getAffineTransform(src_points, dst_points)
        
        # Warping
        warped_face = cv2.warpAffine(image, M, self.target_size)
        
        return warped_face
    
    def extract_features(self, face_image):
        """Ekstrak fitur dari wajah"""
        # Resize
        face_resized = cv2.resize(face_image, self.target_size)
        
        # Grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Histogram sebagai fitur
        hist = cv2.calcHist([face_gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # LBP features
        lbp = self.compute_simple_lbp(face_gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        
        # Gabungkan fitur
        features = np.concatenate((hist, lbp_hist))
        
        return features
    
    def compute_simple_lbp(self, image):
        """Hitung Local Binary Pattern sederhana"""
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
    
    def recognize_face(self, image, face_box):
        """Kenali wajah berdasarkan bounding box"""
        x, y, w, h = face_box
        face_image = image[y:y+h, x:x+w]
        
        # Resize
        resized_face = cv2.resize(face_image, self.target_size)
        
        # Ekstrak fitur
        features = self.extract_features(resized_face)
        
        # Bandingkan dengan database
        best_match = None
        best_similarity = -1
        
        for face_data in self.face_database["faces"]:
            db_features = np.array(face_data["features"])
            similarity = np.dot(features, db_features) / (np.linalg.norm(features) * np.linalg.norm(db_features))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = face_data["name"]
        
        if best_similarity > self.recognition_threshold:
            return best_match, best_similarity
        else:
            return "Unknown", best_similarity
    
    def register_face(self, name, frame):
        """Register wajah dari frame"""
        faces = self.detect_faces(frame)
        if not faces:
            return False, "Tidak ada wajah terdeteksi"
            
        x, y, w, h = faces[0]
        
        # Ekstrak landmark
        landmarks = self.extract_landmarks(frame)
        if not landmarks:
            return False, "Tidak dapat mengekstrak landmark"
            
        # Warp wajah
        warped_face = self.warp_face(frame, landmarks)
        if warped_face is None:
            return False, "Gagal melakukan face warping"
            
        # Ekstrak fitur
        features = self.extract_features(warped_face)
        
        # Simpan ke database
        self.face_database["faces"].append({
            "name": name,
            "features": features.tolist()
        })
        
        self.save_database()
        return True, f"Wajah {name} berhasil didaftarkan"
    
    def process_frame(self, frame):
        """Proses frame untuk deteksi dan pengenalan/registrasi"""
        image = frame.copy()
        faces = self.detect_faces(image)
        
        for face_box in faces:
            x, y, w, h = face_box
            
            if self.recognizing:
                # Mode pengenalan
                name, confidence = self.recognize_face(image, face_box)
                
                # Warna bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Gambar bounding box
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                
                # Tampilkan nama
                label = f"{name} ({confidence:.2f})"
                cv2.rectangle(image, (x, y-30), (x+w, y), color, -1)
                cv2.putText(image, label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            else:
                # Mode default/registrasi
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return image
    
    def start_camera(self):
        """Mulai thread kamera"""
        print("Mencoba memulai kamera...")  # Debug message
        if self.camera_active:
            print("Kamera sudah aktif")     # Debug message
            return
            
        self.camera_active = True
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Tidak dapat membuka kamera")
                print("Error: Kamera tidak dapat dibuka")  # Debug message
                self.camera_active = False
                return
                
            print("Kamera berhasil dibuka, memulai thread")  # Debug message
            threading.Thread(target=self.update_frame, daemon=True).start()
            self.status_label.config(text="Kamera aktif")
        except Exception as e:
            messagebox.showerror("Error", f"Error saat membuka kamera: {str(e)}")
            print(f"Exception saat membuka kamera: {str(e)}")  # Debug message
            self.camera_active = False
    
    def stop_camera(self):
        """Hentikan kamera"""
        print("Menghentikan kamera...")  # Debug message
        self.camera_active = False
        if self.cap is not None:
            self.cap.release()
        self.status_label.config(text="Kamera tidak aktif")
    
    def update_frame(self):
        """Update frame dari kamera (thread)"""
        print("Thread update_frame dimulai")  # Debug message
        while self.camera_active:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Tidak dapat membaca frame")
                    break
                    
                # Proses frame
                processed_frame = self.process_frame(frame)
                self.current_frame = processed_frame
                
                # Convert untuk GUI
                cv_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cv_image)
                tk_image = ImageTk.PhotoImage(image=pil_image)
                
                # Update panel secara thread-safe menggunakan after
                if self.panel is not None:
                    self.root.after(0, lambda: self.panel.configure(image=tk_image))
                    self.panel.image = tk_image
                
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                print(f"Exception di update_frame: {str(e)}")
        
        print("Thread update_frame berhenti")  # Debug message
    
    def toggle_recognition(self):
        """Toggle mode pengenalan"""
        self.recognizing = not self.recognizing
        if self.recognizing:
            self.status_label.config(text="Mode: Pengenalan Wajah")
        else:
            self.status_label.config(text="Mode: Normal")
    
    def do_register(self):
        """Register wajah dari frame saat ini"""
        if not self.camera_active or self.current_frame is None:
            messagebox.showerror("Error", "Kamera tidak aktif")
            return
            
        # Dialog untuk input nama
        name_window = tk.Toplevel(self.root)
        name_window.title("Register Wajah")
        name_window.geometry("300x150")
        name_window.resizable(False, False)
        
        tk.Label(name_window, text="Masukkan nama:").pack(pady=10)
        name_entry = tk.Entry(name_window, width=20)
        name_entry.pack(pady=10)
        name_entry.focus()
        
        def do_register_with_name():
            name = name_entry.get()
            if not name:
                messagebox.showerror("Error", "Nama tidak boleh kosong")
                return
                
            success, message = self.register_face(name, self.current_frame)
            messagebox.showinfo("Hasil Registrasi", message)
            name_window.destroy()
        
        tk.Button(name_window, text="Register", command=do_register_with_name).pack(pady=10)
    
    def register_from_file(self):
        """Register wajah dari file gambar"""
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
        )
        
        if not file_path:
            return
            
        # Dialog untuk input nama
        name_window = tk.Toplevel(self.root)
        name_window.title("Register Wajah dari File")
        name_window.geometry("300x150")
        name_window.resizable(False, False)
        
        tk.Label(name_window, text="Masukkan nama:").pack(pady=10)
        name_entry = tk.Entry(name_window, width=20)
        name_entry.pack(pady=10)
        name_entry.focus()
        
        def do_register_with_name():
            name = name_entry.get()
            if not name:
                messagebox.showerror("Error", "Nama tidak boleh kosong")
                return
                
            # Baca gambar
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Tidak dapat membaca gambar")
                return
                
            # Deteksi wajah
            faces = self.detect_faces(image)
            if not faces:
                messagebox.showerror("Error", "Tidak ada wajah terdeteksi dalam gambar")
                return
                
            x, y, w, h = faces[0]
            
            # Ekstrak landmark
            landmarks = self.extract_landmarks(image)
            if not landmarks:
                messagebox.showerror("Error", "Tidak dapat mengekstrak landmark wajah")
                return
                
            # Warp wajah
            warped_face = self.warp_face(image, landmarks)
            if warped_face is None:
                messagebox.showerror("Error", "Gagal melakukan face warping")
                return
                
            # Ekstrak fitur
            features = self.extract_features(warped_face)
            
            # Simpan ke database
            self.face_database["faces"].append({
                "name": name,
                "features": features.tolist()
            })
            
            self.save_database()
            messagebox.showinfo("Hasil Registrasi", f"Wajah {name} berhasil didaftarkan dari file")
            name_window.destroy()
        
        tk.Button(name_window, text="Register", command=do_register_with_name).pack(pady=10)
    
    def show_database(self):
        """Tampilkan database wajah"""
        if not self.face_database["faces"]:
            messagebox.showinfo("Database", "Database kosong")
            return
            
        db_window = tk.Toplevel(self.root)
        db_window.title("Database Wajah")
        db_window.geometry("300x400")
        
        # Tampilkan daftar wajah
        tk.Label(db_window, text="Wajah Terdaftar:", font=("Arial", 12, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(db_window, width=40, height=15)
        listbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        for i, face_data in enumerate(self.face_database["faces"]):
            listbox.insert(tk.END, f"{i+1}. {face_data['name']}")
        
        # Tombol hapus
        def delete_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showinfo("Info", "Pilih wajah yang akan dihapus")
                return
                
            idx = selection[0]
            name = self.face_database["faces"][idx]["name"]
            
            if messagebox.askyesno("Konfirmasi", f"Hapus wajah {name}?"):
                del self.face_database["faces"][idx]
                self.save_database()
                listbox.delete(idx)
                messagebox.showinfo("Info", f"Wajah {name} berhasil dihapus")
        
        tk.Button(db_window, text="Hapus", command=delete_selected).pack(pady=10)
    
    def on_closing(self):
        """Handle window closing"""
        print("Menutup aplikasi...")  # Debug message
        self.stop_camera()
        self.root.destroy()
    
    def create_gui(self):
        """Buat GUI"""
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        
        # Frame atas untuk kamera
        camera_frame = tk.Frame(self.root)
        camera_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel untuk kamera
        self.panel = tk.Label(camera_frame)
        self.panel.pack(fill=tk.BOTH, expand=True)
        
        # Frame bawah untuk kontrol
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Status
        status_frame = tk.Frame(control_frame)
        status_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.status_label = tk.Label(status_frame, text="Kamera tidak aktif", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Tombol-tombol
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        tk.Button(button_frame, text="Start Kamera", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Stop Kamera", command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Toggle Recognition", command=self.toggle_recognition).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Register Wajah", command=self.do_register).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Register dari File", command=self.register_from_file).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Lihat Database", command=self.show_database).pack(side=tk.LEFT, padx=5)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


if __name__ == "__main__":
    print("=== Sistem Registrasi Wajah ===")
    print("Memulai aplikasi...")
    
    try:
        app = FaceRecognitionSystem()
        app.create_gui()
        app.root.mainloop()
    except Exception as e:
        print(f"Error utama: {str(e)}")