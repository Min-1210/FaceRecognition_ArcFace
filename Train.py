from deepface import DeepFace
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import time

dataset_dir = "Anh"
database_file = "face_database.pkl"
model_name = "ArcFace"
detector_backend = "mtcnn"
max_workers = os.cpu_count()

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def process_image(img_path, model_name="ArcFace", detector_backend="mtcnn"):
    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True
        )[0]["embedding"]
        print(f"Đã xử lý: {img_path}")
        return img_path, embedding
    except Exception as e:
        print(f"Lỗi khi xử lý {img_path}: {e}")
        return img_path, None


def create_face_database():
    database = {}
    image_paths = []

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            database[person_name] = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(VALID_EXTENSIONS):
                    image_paths.append((img_path, person_name))

    if not image_paths:
        print("Không tìm thấy ảnh nào để xử lý!")
        return

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, img_path, model_name, detector_backend)
                   for img_path, _ in image_paths]

        for future in futures:
            img_path, embedding = future.result()
            if embedding is not None:
                # Tìm person_name tương ứng
                person_name = next(p_name for p, p_name in image_paths if p == img_path)
                database[person_name].append(embedding)

    with open(database_file, "wb") as f:
        pickle.dump(database, f)
    print(f"Đã lưu database vào: {os.path.abspath(database_file)}")
    print(f"Thời gian xử lý: {time.time() - start_time:.2f} giây")


if __name__ == "__main__":
    create_face_database()