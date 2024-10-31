import cv2
import numpy as np
from gtts import gTTS
import os
import time

# Thông số tiêu cự và kích thước vật thật
focal_length = 700  # ví dụ tiêu cự (cần calibrate)
real_diameter = 10  # đường kính thật của vật (cm)

# Hàm chuyển đổi văn bản thành giọng nói
def text_to_speech(text):
    language = 'vi'  # Ngôn ngữ: tiếng Việt
    speech = gTTS(text=text, lang=language, slow=False)
    speech.save("output.mp3")
    os.system("start output.mp3")  # Phát tệp âm thanh (Windows)
    # os.system("xdg-open output.mp3")  # Linux
    # os.system("open output.mp3")  # macOS

# Hàm gọi lại để thay đổi giá trị bộ lọc GaussianBlur
def update_blur(val):
    global blur_size
    blur_size = max(3, val | 1)

# Hàm gọi lại để thay đổi thông số HoughCircles
def update_hough(param):
    pass  # Hàm này không cần thực hiện gì

# Khởi tạo kích thước bộ lọc mặc định
blur_size = 9

# Tạo cửa sổ và thanh trượt cho kích thước GaussianBlur
cv2.namedWindow("Frame")
cv2.createTrackbar("Blur Size", "Frame", blur_size, 20, update_blur)

# Tạo thanh trượt cho thông số HoughCircles
cv2.createTrackbar("dp", "Frame", 1, 10, update_hough)
cv2.createTrackbar("Min Dist", "Frame", 50, 200, update_hough)
cv2.createTrackbar("Param1", "Frame", 100, 300, update_hough)
cv2.createTrackbar("Param2", "Frame", 30, 100, update_hough)
cv2.createTrackbar("Min Radius", "Frame", 10, 100, update_hough)
cv2.createTrackbar("Max Radius", "Frame", 100, 200, update_hough)

# Mở camera
cap = cv2.VideoCapture(0)

# Biến để lưu thời điểm phát hiện gần nhất
last_detection_time = 0
reading_distance = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh sang màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Áp dụng GaussianBlur với kích thước từ thanh trượt
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 2)

    # Lấy giá trị từ thanh trượt cho HoughCircles
    dp = cv2.getTrackbarPos("dp", "Frame")
    min_dist = cv2.getTrackbarPos("Min Dist", "Frame")
    param1 = cv2.getTrackbarPos("Param1", "Frame")
    param2 = cv2.getTrackbarPos("Param2", "Frame")
    min_radius = cv2.getTrackbarPos("Min Radius", "Frame")
    max_radius = cv2.getTrackbarPos("Max Radius", "Frame")

    # Tìm các hình tròn trong ảnh
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp/10, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # Tính khoảng cách từ camera đến hình tròn
            circle_diameter = 2 * r
            distance = (focal_length * real_diameter) / circle_diameter

            # Vẽ hình tròn và hiển thị khoảng cách
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Vẽ đường tròn
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Vẽ tâm hình tròn
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Kiểm tra nếu đã đủ 2 giây từ lần phát hiện trước
            current_time = time.time()
            if current_time - last_detection_time >= 2:
                # Đọc khoảng cách bằng giọng nói
                text_to_speech(f"Khoảng cách là {distance:.2f} centimet.")
                last_detection_time = current_time  # Cập nhật thời gian phát hiện

    # Hiển thị khung hình
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
