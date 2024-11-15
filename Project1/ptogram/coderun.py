from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Tải mô hình
model = load_model('D:/Project1/ptogram/emotion_model.h5')

# Các nhãn cảm xúc bằng tiếng Việt
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Sử dụng webcam để dự đoán cảm xúc
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # Dự đoán cảm xúc và lấy xác suất
            prediction = model.predict(roi)[0]
            max_index = prediction.argmax()  # Chỉ số cảm xúc có xác suất cao nhất
            label = emotion_labels[max_index]
            confidence = prediction[max_index] * 100  # Tính tỉ lệ phần trăm

            # Hiển thị nhãn và tỉ lệ phần trăm
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
