import tensorflow as tf
import cv2
import numpy as np
from collections import deque
import winsound
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import threading

MODEL_PATH = "model/fire_detection_model.h5"
INPUT_SIZE = (224, 224)
ALARM_COOLDOWN = 10
FIRE_PIXEL_THRESHOLD = 1000

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "fire.alert.notification.7679@gmail.com"
EMAIL_PASSWORD = "empr vcwv oupw ntnd"   # ⚠️ Use env var in real projects
RECIPIENT_EMAIL = "atulrahul704@gmail.com"


class FireDetectionSystem:
    def __init__(self):   # ✅ fixed
        self.model = tf.keras.models.load_model(MODEL_PATH)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.latest_frame = None
        self.running = True
        threading.Thread(target=self._camera_thread, daemon=True).start()

        self.alarm_active = False
        self.last_notification = 0
        self.alarm_thread = threading.Thread(target=self._alarm_handler, daemon=True)
        self.alarm_thread.start()

    def _camera_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame

    def _alarm_handler(self):
        while self.running:
            if self.alarm_active:
                winsound.Beep(2000, 1000)
            time.sleep(0.9)

    def _send_notification(self, frame):
        def send_email_async(frame_to_send):
            try:
                image_path = "detected_fire.jpg"
                cv2.imwrite(image_path, frame_to_send)

                msg = MIMEMultipart()
                msg['Subject'] = "FIRE ALERT - Emergency Notification"
                msg['From'] = EMAIL_ADDRESS
                msg['To'] = RECIPIENT_EMAIL

                text = MIMEText("🔥 Fire detected! Immediate action required!\nFrame attached.")
                msg.attach(text)

                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read(), name="fire_frame.jpg")
                    msg.attach(img)

                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                    server.starttls()
                    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())

                print("✅ Email with image sent successfully")
            except smtplib.SMTPException as e:
                print(f"❌ Email sending failed: {str(e)}")

        threading.Thread(target=send_email_async, args=(frame,), daemon=True).start()
        return True

    def _process_frame(self):
        frame = self.latest_frame
        if frame is None:
            return False, frame

        resized = cv2.resize(frame, INPUT_SIZE) / 255.0
        prediction = self.model.predict(np.expand_dims(resized, axis=0))[0][0]  # ✅ fixed

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fire_mask = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([20, 255, 255]))

        return (prediction > 0.8) and (cv2.countNonZero(fire_mask) > FIRE_PIXEL_THRESHOLD), frame

    def run(self):
        pred_history = deque(maxlen=15)

        try:
            while self.running:
                fire_detected, frame = self._process_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self.alarm_active = fire_detected

                if fire_detected and (time.time() - self.last_notification) > ALARM_COOLDOWN:
                    if self._send_notification(frame):
                        print("🔥 Fire detected - Notification with image sent")
                        self.last_notification = time.time()

                status_text = "🔥 FIRE DETECTED!" if fire_detected else "✅ Secure"
                color = (0, 0, 255) if fire_detected else (0, 255, 0)
                cv2.putText(frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow('Fire Detection', frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":   # ✅ fixed
    print("🔥 Fire Detection System Initializing...")
    detector = FireDetectionSystem()
    detector.run()
