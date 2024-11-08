import cv2
import numpy as np

# Đọc ảnh
anh = cv2.imread('D:/TGMT/bai8/anhbai8.jpg')
(cao, rong) = anh.shape[:2]

# 1. Dịch chuyển ảnh (Translation)
dx, dy = 200, 250
ma_tran_dich = np.float32([[1, 0, dx], [0, 1, dy]])
# Chỉ cần giữ nguyên kích thước ban đầu của ảnh
anh_da_dich = cv2.warpAffine(anh, ma_tran_dich, (rong, cao))
cv2.imshow('Ảnh đã dịch chuyển', anh_da_dich)
cv2.imwrite('D:/TGMT/bai8/anh_da_dich_chuyen.jpg', anh_da_dich)

# 2. Xoay ảnh (Rotation)
goc_xoay = [30, 45, 60]
for goc in goc_xoay:
    ma_tran_xoay = cv2.getRotationMatrix2D((rong // 2, cao // 2), goc, 1)
    # Đảm bảo kích thước ảnh không thay đổi, sử dụng cùng kích thước với ảnh ban đầu
    anh_da_xoay = cv2.warpAffine(anh, ma_tran_xoay, (rong, cao))
    cv2.imshow(f'Ảnh đã xoay {goc} độ', anh_da_xoay)
    cv2.imwrite(f'D:/TGMT/bai8/anh_da_xoay_{goc}_do.jpg', anh_da_xoay)

# 3. Thu phóng ảnh (Scaling) và thay đổi kích thước ảnh
ti_le_thu_phong = [(0.5, 0.5), (2, 2), (2, 0.5), (0.5, 2)]
for sx, sy in ti_le_thu_phong:
    # Thu phóng ảnh và thay đổi kích thước theo tỷ lệ
    new_rong = int(rong * sx)
    new_cao = int(cao * sy)
    # Sử dụng resize để thay đổi kích thước ảnh theo tỷ lệ
    anh_da_thu_phong = cv2.resize(anh, (new_rong, new_cao))
    cv2.imshow(f'Ảnh đã thu phóng {sx}-{sy}', anh_da_thu_phong)
    cv2.imwrite(f'D:/TGMT/bai8/anh_da_thu_phong_{sx}_{sy}.jpg', anh_da_thu_phong)

# 4. Trượt ảnh (Shearing)
truot_x, truot_y = 0.2, 0.25
ma_tran_truot = np.float32([[1, truot_x, 0], [truot_y, 1, 0]])
# Dùng ma trận trượt mà không thay đổi kích thước ảnh
anh_da_truot = cv2.warpAffine(anh, ma_tran_truot, (rong, cao))
cv2.imshow('Ảnh đã trượt', anh_da_truot)
cv2.imwrite('D:/TGMT/bai8/anh_da_truot.jpg', anh_da_truot)

# Hiển thị tất cả các ảnh
cv2.waitKey(0)
cv2.destroyAllWindows()
