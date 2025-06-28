import cv2
import numpy as np
import threading
import time
import pickle
import os
import serial
from deepface import DeepFace
from keras.models import load_model

try:
    ser = serial.Serial('COM9', 115200, timeout=1)
    time.sleep(2)
except Exception as e:
    print(f"Lỗi kết nối Serial: {e}")
    ser = None

model_name = "ArcFace"
detector_backend = "mtcnn"
confidence_threshold = 0.95

model_path = "face_classifier_nn.h5"
label_encoder_path = "face_label_encoder.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f"Không tìm thấy file encoder: {label_encoder_path}")

mlp_model = load_model(model_path)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_

latest_frame = None
results = []
lock = threading.Lock()
last_command_time = 0
command_interval = 5
recognition_times = {}
required_duration = 3.0

def recognize_faces():
    global latest_frame, results, last_command_time, recognition_times
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        try:
            faces = DeepFace.represent(
                img_path=frame,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )

            detected = []
            current_names = set()
            current_time = time.time()

            for face in faces:
                emb = face["embedding"]
                area = face.get("facial_area", {})

                probas = mlp_model.predict(np.array([emb]), verbose=0)[0]
                idx = np.argmax(probas)
                score = probas[idx]
                name = class_names[idx] if score > confidence_threshold else "Unknown"

                if name != "Unknown":
                    current_names.add(name)
                    if name not in recognition_times:
                        recognition_times[name] = current_time
                    elif current_time - recognition_times[name] >= required_duration:
                        if ser is not None and current_time - last_command_time >= command_interval:
                            try:
                                command = f"O:{name}"
                                ser.write(command.encode('utf-8'))
                                ser.write(b'\n')
                                last_command_time = current_time
                                print(f"Mở cửa cho {name} sau {current_time - recognition_times[name]:.2f} giây")
                                recognition_times[name] = current_time
                            except Exception as e:
                                print(f"Lỗi gửi Serial: {e}")
                else:
                    recognition_times[name] = current_time

                detected.append({
                    "identity": name,
                    "confidence": score,
                    "facial_area": area
                })

            for name in list(recognition_times.keys()):
                if name not in current_names and name != "Unknown":
                    recognition_times[name] = current_time

            with lock:
                results = detected

        except Exception as e:
            print(f"Lỗi nhận diện: {e}")

def run_realtime():
    global latest_frame, results
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return

    print("Đang nhận diện... Nhấn 'q' để thoát.")
    threading.Thread(target=recognize_faces, daemon=True).start()

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (480, 360))
        with lock:
            latest_frame = frame_resized.copy()
            current_results = results.copy()

        for res in current_results:
            name = res["identity"]
            score = res["confidence"]
            area = res["facial_area"]
            x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)

            orig_h, orig_w, _ = frame.shape
            x = int(x * orig_w / 480)
            y = int(y * orig_h / 360)
            w = int(w * orig_w / 480)
            h = int(h * orig_h / 360)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps = 1.0 / (time.time() - start_time + 1e-8)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()

if __name__ == "__main__":
    run_realtime()