import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_energy(image):
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tính toán gradient của ảnh bằng Sobel
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Tính năng lượng bằng cách cộng độ lớn của gradient theo cả hai chiều
    energy = np.abs(gradient_x) + np.abs(gradient_y)
    return energy

def find_seam(energy):
    rows, cols = energy.shape
    seam = np.zeros((rows,), dtype=int)
    
    # Ma trận lưu tổng năng lượng đến điểm (i, j)
    M = energy.copy()
    
    # Dynamic programming để tìm tổng năng lượng nhỏ nhất cho mỗi pixel
    for i in range(1, rows):
        for j in range(cols):
            min_energy = M[i - 1, j]
            if j > 0:
                min_energy = min(min_energy, M[i - 1, j - 1])
            if j < cols - 1:
                min_energy = min(min_energy, M[i - 1, j + 1])
            M[i, j] += min_energy

    # Tìm pixel có năng lượng thấp nhất ở hàng cuối
    seam[-1] = np.argmin(M[-1])

    # Lần ngược từ hàng cuối lên để tìm đường seam
    for i in range(rows - 2, -1, -1):
        prev_x = seam[i + 1]
        start = max(prev_x - 1, 0)
        end = min(prev_x + 2, cols)
        seam[i] = start + np.argmin(M[i, start:end])
    
    return seam

def remove_seam(image, seam):
    rows, cols = image.shape[:2]
    # Tạo ảnh mới để lưu kết quả
    new_image = np.zeros((rows, cols - 1, 3), dtype=image.dtype)
    
    for i in range(rows):
        # Tạo chỉ số cho các cột
        new_image[i, :, :] = np.delete(image[i, :, :], seam[i], axis=0)
    
    return new_image

def seam_carve(image, new_width):
    while image.shape[1] > new_width:
        energy = calculate_energy(image)
        seam = find_seam(energy)
        image = remove_seam(image, seam)
    return image

# Đọc ảnh
image = cv2.imread('d:/Downloads/dgj.jpeg')

# Thay đổi kích thước ảnh
new_width = 200  # Thay đổi kích thước mong muốn
resized_image = seam_carve(image, new_width)

# Hiển thị ảnh kết quả
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title('Resized Image Using Seam Carving')
plt.axis('off')
plt.show()

# Lưu ảnh kết quả
cv2.imwrite('resized_image.jpg', resized_image)
