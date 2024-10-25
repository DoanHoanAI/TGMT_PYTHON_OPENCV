import cv2
import numpy as np

# Đọc ảnh từ đường dẫn
image_path = 'D:/TGMT/bai6/bai2_s/uida.jpg'  # Đường dẫn ảnh của bạn
img = cv2.imread(image_path)

# Bước 1: Hiển thị ảnh gốc
cv2.imshow("Original Image", img)

# Chuyển ảnh sang mức xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Bước 2: Hiển thị ảnh xám
cv2.imshow("Gray Image", gray)

# Làm mịn ảnh với Gaussian Blur
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Bước 3: Hiển thị ảnh sau khi làm mịn
cv2.imshow("Blurred Image", gray_blurred)

# Phát hiện hình tròn với các tham số điều chỉnh
circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,            # Độ phân giải của ảnh
    minDist=30,           # Khoảng cách tối thiểu giữa các hình tròn
    param1=50,            # Ngưỡng cho thuật toán Canny
    param2=50,            # Ngưỡng phát hiện hình tròn
    minRadius=15,         # Bán kính tối thiểu của hình tròn
    maxRadius=80          # Bán kính tối đa của hình tròn
)

# Tạo ảnh nhị phân chỉ chứa các hình tròn
binary_circles = np.zeros_like(gray)

# Vẽ các hình tròn lên ảnh nhị phân và ảnh có viền
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Vẽ hình tròn màu trắng lên ảnh nhị phân
        cv2.circle(binary_circles, (i[0], i[1]), i[2], 255, -1)  # Vùng bên trong vòng tròn là 255 (trắng)
        
        # Vẽ viền lên ảnh gốc và đánh dấu tâm
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)      # Vẽ viền màu xanh lá
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)         # Đánh dấu tâm màu đỏ

# Bước 4: Hiển thị ảnh nhị phân chứa hình tròn
cv2.imshow("Binary Circles", binary_circles)

# Bước 5: Hiển thị ảnh cuối cùng với các vòng tròn đã phát hiện
cv2.imshow("Detected Circles with Borders", img)

# Giữ tất cả các cửa sổ cho đến khi nhấn phím bất kỳ
cv2.waitKey(0)
cv2.destroyAllWindows()
