import cv2
from deepface import DeepFace

# Sử dụng webcam để nhận diện cảm xúc
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Nhận diện cảm xúc
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    
    # Lấy thông tin cảm xúc và tỉ lệ phần trăm
    emotion = result[0]['dominant_emotion']
    emotion_score = result[0]['emotion'][emotion]
    
    # Hiển thị nhãn cảm xúc và tỉ lệ phần trăm
    cv2.putText(frame, f'{emotion}: {emotion_score:.2f}%', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Hiển thị khung hình
    cv2.imshow('Emotion Detector', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
