🔥 Vision-Based Fire Detection System

A real-time fire detection system powered by deep learning (MobileNetV3) and computer vision (OpenCV). The project is capable of monitoring live video feeds, detecting fire, triggering alarms, and sending email notifications with attached images when fire is detected.

📂 Project Structure
fire_detection/
│
├── app.py                      # Main web application (Flask or similar)
├── demo_mail.py                # Email notification demo script
├── detected_fire.jpg           # Sample detected fire frame
├── pubspec.yaml                # (If Flutter/Dart integration is used)
├── real_time_detection.py      # Real-time fire detection script
├── temp.py                     # Temporary script
├── tempCodeRunnerFile.py       # Code runner temp file
├── test_model.py               # Model testing script
├── train_model.py              # Model training script
├── utils.py                    # Utility functions
│
├── dataset/                    # Dataset folder
│   ├── fire/                   # Fire images
│   └── non_fire/               # Non-fire images
│
├── logs/                       # Logs folder
│   ├── train/                  # Training logs
│   └── validation/             # Validation logs
│
├── model/                      # Saved models
│   ├── fire_detection_model.h5 # Trained model
│   ├── fire_detection.h5       # Alternate model
│   └── sample_image.jpg        # Example image
│
├── static/                     # Static files for web app
│   └── uploads/                # Uploaded images
│
├── statics/                    # Duplicate static (to be cleaned)
│   └── uploads/
│
├── templates/                  # HTML templates
│   └── index.html              # Web interface
│
├── webpage/                    # Web assets & docs
│   ├── fire_back.png
│   ├── Fire_Detection_System_Report.pdf
│   ├── index.html
│   ├── script.js
│   └── style.css
│
└── README.md                   # Documentation

🚀 Features

🔥 Real-time Fire Detection using MobileNetV3 model

🎥 Live camera feed monitoring with OpenCV

🔔 Alarm system using system sound alerts

📧 Email notification with attached fire-frame image

📊 Logs for training and validation

🌐 Web interface built with Flask + HTML/CSS/JS

📊 Dataset

We used the Fire Detection Dataset (Kaggle)
 for training and evaluation.

Fire class: Images containing visible flames under different conditions

Non-fire class: Normal scenes, sunsets, and bright objects (to reduce false positives)

Data augmentation: flips, rotations, brightness/contrast adjustment, zooming

🛠️ Tech Stack

Python 3

TensorFlow / Keras

OpenCV

NumPy

Flask (Web App)

SMTP (Email Alerts)

⚙️ Installation & Setup

Clone this repository:

git clone https://github.com/your-username/fire-detection-system.git
cd fire-detection-system


Install dependencies:

pip install -r requirements.txt


(Typical requirements: tensorflow, opencv-python, numpy, flask)

Download dataset and place in dataset/:

dataset/
├── fire/
└── non_fire/


Train the model:

python train_model.py


Run real-time detection:

python real_time_detection.py

📧 Email Notifications

Configure your email credentials in real_time_detection.py or demo_mail.py.

When fire is detected:

🚨 Alarm rings

🖼️ Frame is saved as detected_fire.jpg

📧 Email alert with attached image is sent

📸 Demo

(Add screenshots or GIFs of detection output here)

🎯 Future Scope

📱 Push notifications (mobile integration)

☁️ Cloud deployment for remote monitoring

🔗 Integration with smoke & temperature sensors

🌡️ Thermal camera support

🚀 Advanced deep learning models (YOLO, EfficientNet)

👨‍💻 Authors

Atul Kumar Pal

Rahul Kumar Purty
