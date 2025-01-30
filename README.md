# Boxing Response Trainer 🥊

A sophisticated Python application designed to measure and improve boxing reaction times using computer vision and real-time face tracking.


## 🌟 Features

- **Real-time Face Tracking**: Uses OpenCV's Haar Cascade classifier for accurate face detection
- **Camera Lag Calibration**: Advanced calibration system to account for camera delay
- **Response Time Measurement**: Precise measurement of reaction times in milliseconds
- **Visual Feedback**: Clear visual cues and on-screen instructions
- **Statistics Tracking**: Keeps track of average response times
- **Fullscreen Support**: Optimized for fullscreen operation

## 🔧 Requirements

```python
opencv-python>=4.5.0
numpy>=1.19.0
```

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/m3tamatt/BoxingReactionTime.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. Run the program:
```bash
python boxing_trainer.py
```

2. Key Controls:
- `c` - Start camera calibration
- `SPACE` - Start new reaction test
- `q` - Quit program

## 📋 Program Flow

1. **Camera Check**
   - Automatically detects available cameras
   - Initializes webcam with optimal settings

2. **Calibration Mode**
   - Measures camera lag using black/white screen transitions
   - Collects multiple samples for accuracy
   - Calculates average camera delay

3. **Training Mode**
   - Random countdown timer (3,2,1)
   - Random delay before stimulus
   - Visual stimulus (red border)
   - Movement detection and timing
   - Results display

## 🎯 How It Works

The program uses computer vision to:
1. Track the user's face position
2. Detect quick movements when responding to visual stimuli
3. Calculate precise reaction times
4. Account for camera lag through calibration

## 📊 Features in Detail

### Camera Calibration
```python
class CameraCalibrator:
    # Handles camera lag measurement
    # Uses black/white screen transitions
    # Collects multiple samples for accuracy
```

### Boxing Tracker
```python
class BoxingTracker:
    # Manages reaction time measurements
    # Handles movement detection
    # Maintains statistics
```

## 🔍 Technical Details

- Resolution: 640x480 (configurable)
- Face Detection: Haar Cascade Classifier
- Movement Threshold: Customizable sensitivity
- Timing Precision: Millisecond accuracy
- Camera Lag Compensation: Built-in calibration

## 🛠️ Customization

You can modify these parameters in the code:
- `movement_threshold`: Sensitivity of movement detection
- `required_samples`: Number of calibration samples
- `brightness_threshold`: Calibration sensitivity

## 📝 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Note

For best results:
- Use in a well-lit environment
- Maintain consistent distance from camera
- Perform calibration in a controlled lighting environment

---

Made by m3tamatt
