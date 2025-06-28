import cv2
import os
from mtcnn import MTCNN
import numpy as np

input_dir = "dataset/Nguoi_1"
output_dir = "processed_dataset/SNguoi_1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

detector = MTCNN()

for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"Bỏ qua: {img_name} (không phải ảnh)")
        continue

    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Không thể đọc: {img_path}")
        continue

    print(f"Đang xử lý: {img_name}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        faces = detector.detect_faces(rgb_image)
    except Exception as e:
        print(f"Lỗi khi phát hiện khuôn mặt trong {img_name}: {str(e)}")
        continue

    if len(faces) > 0:
        for i, face in enumerate(faces):
            x, y, w, h = face['box']

            padding = 20
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(image.shape[1], x + w + padding)
            bottom = min(image.shape[0], y + h + padding)

            if right > left and bottom > top:
                face_image = image[top:bottom, left:right]

                if face_image.size == 0:
                    print(f"Khuôn mặt rỗng trong {img_name}")
                    continue

                try:
                    face_image_resized = cv2.resize(face_image, (224, 224))

                    output_name = f"face_{i}_{img_name}"
                    output_path = os.path.join(output_dir, output_name)
                    cv2.imwrite(output_path, face_image_resized)
                    print(f"Đã xử lý: {img_name}, khuôn mặt {i}")
                except Exception as e:
                    print(f"Lỗi khi resize hoặc lưu ảnh {img_name}: {str(e)}")
            else:
                print(f"Tọa độ không hợp lệ trong {img_name}: left={left}, top={top}, right={right}, bottom={bottom}")
    else:
        print(f"Không tìm thấy khuôn mặt trong: {img_name}")