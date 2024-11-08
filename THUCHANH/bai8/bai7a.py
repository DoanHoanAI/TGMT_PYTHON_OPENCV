import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
anh = cv2.imread('D:/TGMT/bai8/anhbai8.jpg')  # Thay 'path_to_your_image.jpg' bằng đường dẫn ảnh của bạn
anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)  # Chuyển đổi ảnh sang RGB để hiển thị

# Lấy kích thước của ảnh
chieu_cao, do_rong = anh.shape[:2]

# Hàm hiển thị ảnh
def hien_thi_anh(anh, tieu_de="Ảnh"):
    plt.imshow(anh)
    plt.title(tieu_de)
    plt.axis('off')
    plt.show()

# 2D Dịch chuyển (Translation)
tx, ty = 200, 250
M_dich_chuyen = np.float32([[1, 0, tx], [0, 1, ty]])
anh_dich_chuyen = cv2.warpAffine(anh, M_dich_chuyen, (do_rong, chieu_cao))

# 2D Xoay (Rotation)
def xoay_anh(anh, goc):
    M_xoay = cv2.getRotationMatrix2D((do_rong/2, chieu_cao/2), goc, 1)
    return cv2.warpAffine(anh, M_xoay, (do_rong, chieu_cao))

anh_xoay_30 = xoay_anh(anh, 30)
anh_xoay_45 = xoay_anh(anh, 45)
anh_xoay_60 = xoay_anh(anh, 60)

# 2D Thu phóng (Scaling)
def thu_phong_anh(anh, sx, sy):
    return cv2.resize(anh, None, fx=sx, fy=sy)

anh_thu_phong_05_05 = thu_phong_anh(anh, 0.5, 0.5)
anh_thu_phong_2_2 = thu_phong_anh(anh, 2, 2)
anh_thu_phong_2_05 = thu_phong_anh(anh, 2, 0.5)
anh_thu_phong_05_2 = thu_phong_anh(anh, 0.5, 2)

# 2D Trượt (Shearing)
def truot_anh(anh, shx, shy):
    M_truot = np.float32([[1, shx, 0], [shy, 1, 0]])
    return cv2.warpAffine(anh, M_truot, (do_rong, chieu_cao))

anh_truot = truot_anh(anh, 0.2, 0.25)

# Hiển thị kết quả
hien_thi_anh(anh_dich_chuyen, "Ảnh sau khi dịch chuyển")
hien_thi_anh(anh_xoay_30, "Ảnh sau khi xoay 30°")
hien_thi_anh(anh_xoay_45, "Ảnh sau khi xoay 45°")
hien_thi_anh(anh_xoay_60, "Ảnh sau khi xoay 60°")
hien_thi_anh(anh_thu_phong_05_05, "Ảnh sau khi thu phóng (0.5, 0.5)")
hien_thi_anh(anh_thu_phong_2_2, "Ảnh sau khi thu phóng (2, 2)")
hien_thi_anh(anh_thu_phong_2_05, "Ảnh sau khi thu phóng (2, 0.5)")
hien_thi_anh(anh_thu_phong_05_2, "Ảnh sau khi thu phóng (0.5, 2)")
hien_thi_anh(anh_truot, "Ảnh sau khi trượt")
