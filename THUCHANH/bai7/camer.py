import cv2
import numpy as np

# Thông số tiêu cự và kích thước vật thật
focal_length = 700  # ví dụ tiêu cự (cần calibrate)
real_diameter = 10  # đường kính thật của vật (cm)

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
cv2.namedWindow("Frame1")
cv2.namedWindow("Frame2")
cv2.createTrackbar("Blur Size", "Frame1", blur_size, 20, update_blur)

# Tạo thanh trượt cho thông số HoughCircles
cv2.createTrackbar("dp", "Frame1", 1, 10, update_hough)
cv2.createTrackbar("Min Dist", "Frame1", 50, 200, update_hough)
cv2.createTrackbar("Param1", "Frame1", 100, 300, update_hough)
cv2.createTrackbar("Param2", "Frame1", 30, 100, update_hough)
cv2.createTrackbar("Min Radius", "Frame1", 10, 100, update_hough)
cv2.createTrackbar("Max Radius", "Frame1", 100, 200, update_hough)

# Mở 2 camera
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break

    # Chuyển ảnh sang màu xám cho cả hai khung hình
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Áp dụng GaussianBlur với kích thước từ thanh trượt
    gray1 = cv2.GaussianBlur(gray1, (blur_size, blur_size), 2)
    gray2 = cv2.GaussianBlur(gray2, (blur_size, blur_size), 2)

    # Lấy giá trị từ thanh trượt cho HoughCircles
    dp = cv2.getTrackbarPos("dp", "Frame1")
    min_dist = cv2.getTrackbarPos("Min Dist", "Frame1")
    param1 = cv2.getTrackbarPos("Param1", "Frame1")
    param2 = cv2.getTrackbarPos("Param2", "Frame1")
    min_radius = cv2.getTrackbarPos("Min Radius", "Frame1")
    max_radius = cv2.getTrackbarPos("Max Radius", "Frame1")

    # Tìm các hình tròn trong cả hai khung hình
    circles1 = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, dp=dp/10, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    circles2 = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, dp=dp/10, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    # Xử lý khung hình từ camera 1
    if circles1 is not None:
        circles1 = np.round(circles1[0, :]).astype("int")
        for (x, y, r) in circles1:
            circle_diameter = 2 * r
            distance = (focal_length * real_diameter) / circle_diameter
            cv2.circle(frame1, (x, y), r, (0, 255, 0), 4)
            cv2.circle(frame1, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(frame1, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Xử lý khung hình từ camera 2
    if circles2 is not None:
        circles2 = np.round(circles2[0, :]).astype("int")
        for (x, y, r) in circles2:
            circle_diameter = 2 * r
            distance = (focal_length * real_diameter) / circle_diameter
            cv2.circle(frame2, (x, y), r, (0, 255, 0), 4)
            cv2.circle(frame2, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(frame2, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Hiển thị cả hai khung hình
    cv2.imshow("Frame1", frame1)
    cv2.imshow("Frame2", frame2)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
