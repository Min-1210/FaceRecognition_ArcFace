import numpy as np
import os
import pickle
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Thiết lập Seed để đảm bảo kết quả có thể tái lập ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Đã loại bỏ các dòng không cần thiết của torch

# --- Định nghĩa các hằng số ---
face_database_pkl = "face_database.pkl"
classifier_filename = "face_classifier_nn1"
label_encoder_filename = "face_label_encoder1.pkl"
PLOT_DIR = "src/plot"

# --- 1. Tải file face_database.pkl ---
if not os.path.exists(face_database_pkl):
    print(f"Lỗi: Không tìm thấy file database '{face_database_pkl}'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

with open(face_database_pkl, "rb") as f:
    full_database = pickle.load(f)

if not full_database:
    print("Lỗi: File database.pkl rỗng hoặc không chứa dữ liệu. Không thể huấn luyện.")
    exit()

# --- 2. Trích xuất Embeddings và Nhãn từ Keys của database.pkl ---
all_embeddings = []
all_class_names = []

print("Đang trích xuất embeddings và nhãn từ database.pkl...")
for img_path, embedding_data in full_database.items():
    norm_path = os.path.normpath(img_path)
    class_name = os.path.basename(os.path.dirname(norm_path))

    processed_embedding = None
    if isinstance(embedding_data, list):
        if len(embedding_data) > 0:
            if isinstance(embedding_data[0], np.ndarray):
                processed_embedding = embedding_data[0]
            elif all(isinstance(x, (int, float)) for x in embedding_data):
                processed_embedding = np.array(embedding_data)
    elif isinstance(embedding_data, np.ndarray):
        processed_embedding = embedding_data

    if processed_embedding is not None and processed_embedding.shape[0] > 0:
        all_embeddings.append(processed_embedding)
        all_class_names.append(class_name)
    else:
        print(f"Cảnh báo: Giá trị cho key '{img_path}' không phải là embedding hợp lệ. Bỏ qua.")


X = np.array(all_embeddings)
y_names = np.array(all_class_names)

print(f"Tổng số mẫu embeddings được trích xuất: {len(X)}")
if len(X) == 0:
    print("Không có embedding nào được trích xuất. Vui lòng kiểm tra lại cấu trúc file .pkl.")
    exit()
print(f"Kích thước của mỗi embedding: {X.shape[1]}")

print(f"Tổng số nhãn: {len(y_names)}")
print(f"Các tên lớp duy nhất tìm thấy: {np.unique(y_names)}")

# --- 3. Mã hóa nhãn (Label Encoding) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_names)

print(f"Các ID số được mã hóa: {np.unique(y_encoded)}")
print(f"Các tên người tương ứng: {label_encoder.classes_}")

# --- 4. Chia tập huấn luyện và kiểm tra ---
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Kích thước tập huấn luyện: {X_train.shape}, {y_train_encoded.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}, {y_test_encoded.shape}")

if X_train.shape[0] == 0:
    print("Lỗi: Tập huấn luyện rỗng sau khi chia. Không thể huấn luyện mô hình.")
    exit()

# --- 5. Huấn luyện Bộ phân loại bằng Neural Network ---
print("\nĐang xây dựng và huấn luyện mô hình Neural Network...")

num_classes = len(label_encoder.classes_)

model = Sequential([
    Dense(512, input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Lịch sử huấn luyện sẽ được lưu vào biến 'history'
history = model.fit(X_train, y_train_encoded,
                    epochs=100,
                    batch_size=64,
                    validation_data=(X_test, y_test_encoded),
                    callbacks=[early_stopping],
                    verbose=2) # Đặt verbose=2 để log gọn hơn

print("\n--- Huấn luyện Neural Network hoàn tất. ---")

# --- 6. Đánh giá và Vẽ biểu đồ lịch sử huấn luyện ---
print("Đang vẽ biểu đồ lịch sử huấn luyện...")

# Trích xuất dữ liệu từ đối tượng history
history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

# Số lượng epoch thực tế đã chạy (có thể dừng sớm)
epochs_range = range(1, len(train_loss) + 1)

# Tạo thư mục để lưu biểu đồ
os.makedirs(PLOT_DIR, exist_ok=True)

# Vẽ biểu đồ Loss và Accuracy
plt.figure(figsize=(14, 6))

# Biểu đồ Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, 'b-o', label='Training Loss')
plt.plot(epochs_range, val_loss, 'r-o', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Biểu đồ Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracy, 'b-o', label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, 'r-o', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_save_path = os.path.join(PLOT_DIR, "training_history.png")
plt.savefig(plot_save_path)
print(f"Đã lưu biểu đồ vào: {os.path.abspath(plot_save_path)}")
plt.show()


# --- 7. Lưu mô hình và LabelEncoder ---
model_save_path = f"{classifier_filename}.h5"
model.save(model_save_path)
print(f"Đã lưu bộ phân loại NN vào: {os.path.abspath(model_save_path)}")

with open(label_encoder_filename, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"Đã lưu LabelEncoder vào: {os.path.abspath(label_encoder_filename)}")

print("\n--- Quá trình hoàn tất. ---")
print("Bây giờ bạn có thể tải mô hình NN và LabelEncoder để nhận dạng khuôn mặt.")