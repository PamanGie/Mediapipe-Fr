import os
import subprocess
import sys

def clear_screen():
    """Clear the terminal screen based on OS type"""
    os.system('cls' if os.name=='nt' else 'clear')

def main():
    clear_screen()
    print("=" * 50)
    print("     SISTEM PENGENALAN WAJAH DENGAN MEDIAPIPE")
    print("=" * 50)
    print("\nMenu Utama:")
    print("1. Register Wajah Baru")
    print("2. Jalankan Pengenalan Wajah")
    print("3. Keluar")
    
    while True:
        choice = input("\nPilih menu (1-3): ")
        
        if choice == '1':
            clear_screen()
            print("Menjalankan modul registrasi wajah...\n")
            # Jalankan modul registrasi
            if os.name == 'nt':  # Windows
                subprocess.run([sys.executable, "face_register.py"], shell=True)
            else:  # Unix/Linux/Mac
                subprocess.run([sys.executable, "face_register.py"])
            input("\nTekan Enter untuk kembali ke menu utama...")
            main()  # Kembali ke menu utama
            break
            
        elif choice == '2':
            clear_screen()
            print("Menjalankan modul pengenalan wajah...\n")
            # Jalankan modul pengenalan
            if os.name == 'nt':  # Windows
                subprocess.run([sys.executable, "face_recognition.py"], shell=True)
            else:  # Unix/Linux/Mac
                subprocess.run([sys.executable, "face_recognition.py"])
            input("\nTekan Enter untuk kembali ke menu utama...")
            main()  # Kembali ke menu utama
            break
            
        elif choice == '3':
            print("\nTerima kasih telah menggunakan sistem pengenalan wajah!")
            sys.exit(0)
            
        else:
            print("Pilihan tidak valid. Silakan pilih 1-3.")

if __name__ == "__main__":
    main()