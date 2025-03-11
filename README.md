# Face Recognition System with MediaPipe

A lightweight face recognition system built using MediaPipe, OpenCV, and Python. This system provides real-time face detection, recognition, and a simple user interface for face registration.

![How Face Recognition System Works](https://i.postimg.cc/fygT3fXd/fr-mp.jpg)

## Features

- **Real-time face detection** using MediaPipe Face Detection
- **Face recognition** with feature extraction using histograms and LBP (Local Binary Patterns)
- **Stable bounding box tracking** to reduce flickering
- **Simple GUI** for face registration and management
- **JSON-based face database** for easy portability and management

## System Requirements

- Python 3.7+
- OpenCV 4.5+
- MediaPipe 0.8.9+
- NumPy
- Tkinter (for GUI)
- Pillow (PIL)
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the System

The main entry point is `run_system.py`, which provides a simple menu to register faces or run face recognition:

```bash
python run_system.py
```

### Registering Faces

To register a new face:

1. Select "Register Wajah Baru" from the main menu
2. Click "Start Kamera" to enable your webcam
3. Position your face in the camera view
4. Click "Register Wajah" and enter your name
5. The face will be registered and saved to the database

You can also register a face from an image file by clicking "Register dari File".

### Face Recognition

To start face recognition:

1. Select "Jalankan Pengenalan Wajah" from the main menu
2. The system will activate your webcam and begin detecting faces
3. Recognized faces will be displayed with a green bounding box and their name with confidence score
4. Unknown faces will be displayed with a red bounding box
5. Press 'q' to quit
6. Press 'c' to toggle confidence display

## How It Works

### System Architecture

The system consists of four main components:

1. **Face Detection**: Uses MediaPipe Face Detection to locate faces in a frame
2. **Face Landmark Detection**: Identifies key facial landmarks for face alignment
3. **Feature Extraction**: Extracts facial features using histograms and LBP
4. **Face Recognition**: Matches extracted features against stored faces using cosine similarity

### Bounding Box Stabilization

To reduce the flickering effect of bounding boxes, the system implements a smoothing algorithm that:

1. Tracks faces across multiple frames
2. Calculates IoU (Intersection over Union) between detected faces
3. Applies temporal averaging to stabilize the position and size of bounding boxes

### Database Structure

Face data is stored in a simple JSON file with the following structure:

```json
{
  "faces": [
    {
      "name": "Person Name",
      "features": [0.1, 0.2, 0.3, ...]
    },
    ...
  ]
}
```

## Project Structure

- `face_recognition_base.py`: Base class with core functionality
- `face_recognition.py`: Main recognition class
- `face_register.py`: Module for registering new faces
- `run_system.py`: Main entry point

## Future Improvements

- Add support for multiple cameras
- Implement face anti-spoofing measures
- Optimize for better performance on low-end devices
- Add support for face clustering to handle unknown faces
- Integration with YOLO for more robust object detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google MediaPipe team for the excellent framework
- OpenCV community for the computer vision tools
