# Fire Detection System

This project is a real-time fire detection system using deep learning. It can detect fire in images and video streams, and includes utilities for training, testing, and real-time detection.

## Features
- Real-time fire detection
- Model training and evaluation
- Email notification demo
- Web interface for uploading and viewing results

## Project Structure
- `app.py`: Main application file
- `real_time_detection.py`: Real-time fire detection script
- `train_model.py`: Model training script
- `test_model.py`: Model testing script
- `demo_mail.py`: Email notification demo
- `utils.py`: Utility functions
- `dataset/`: Contains fire and non-fire images
- `model/`: Saved models and related files
- `templates/`, `static/`, `statics/`: Web assets
- `webpage/`: Web interface files

## Getting Started
1. Clone the repository
2. Install required Python packages (see below)
3. Run the desired script (e.g., `python real_time_detection.py`)

## Installation
Install dependencies using pip:
```bash
pip install tensorflow keras opencv-python numpy flask
```

## Usage
- **Train Model:**
  ```bash
  python train_model.py
  ```
- **Test Model:**
  ```bash
  python test_model.py
  ```
- **Real-Time Detection:**
  ```bash
  python real_time_detection.py
  ```
- **Web App:**
  ```bash
  python app.py
  ```

## License
This project is licensed under the MIT License.
