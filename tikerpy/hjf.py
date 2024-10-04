import cv2
import numpy as np
import matplotlib.pyplot as plt

def tinh_nang_luong(anh):
    """Tinh toan ban do nang luong cua anh."""
    xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(xam, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(xam, cv2.CV_64F, 0, 1, ksize=3)
    nang_luong = np.abs(gradient_x) + np.abs(gradient_y)
    return nang_luong

def tim_duong_seam_thap_nhat(nang_luong):
    """Tim duong seam co nang luong thap nhat."""
    hang, cot = nang_luong.shape
    seam = np.zeros((hang,), dtype=int)
    M = nang_luong.copy()
    
    for i in range(1, hang):
        for j in range(cot):
            nang_luong_nho_nhat = M[i - 1, j]
            if j > 0:
                nang_luong_nho_nhat = min(nang_luong_nho_nhat, M[i - 1, j - 1])
            if j < cot - 1:
                nang_luong_nho_nhat = min(nang_luong_nho_nhat, M[i - 1, j + 1])
            M[i, j] += nang_luong_nho_nhat

    seam[-1] = np.argmin(M[-1])

    for i in range(hang - 2, -1, -1):
        prev_x = seam[i + 1]
        start = max(prev_x - 1, 0)
        end = min(prev_x + 2, cot)
        seam[i] = start + np.argmin(M[i, start:end])
    
    return seam

def tim_duong_seam_cao_nhat(nang_luong):
    """Tim duong seam co nang luong cao nhat."""
    hang, cot = nang_luong.shape
    seam = np.zeros((hang,), dtype=int)
    M = nang_luong.copy()
    
    for i in range(1, hang):
        for j in range(cot):
            nang_luong_cao_nhat = M[i - 1, j]
            if j > 0:
                nang_luong_cao_nhat = max(nang_luong_cao_nhat, M[i - 1, j - 1])
            if j < cot - 1:
                nang_luong_cao_nhat = max(nang_luong_cao_nhat, M[i - 1, j + 1])
            M[i, j] += nang_luong_cao_nhat

    seam[-1] = np.argmax(M[-1])

    for i in range(hang - 2, -1, -1):
        prev_x = seam[i + 1]
        start = max(prev_x - 1, 0)
        end = min(prev_x + 2, cot)
        seam[i] = start + np.argmax(M[i, start:end])
    
    return seam

def ve_duong_seam(anh, seam_nang_luong_thap, seam_nang_luong_cao):
    """Ve ca duong seam nang luong thap nhat va cao nhat."""
    anh_voi_seam = anh.copy()

    # Ve duong nang luong thap nhat mau xanh
    for i in range(len(seam_nang_luong_thap)):
        anh_voi_seam[i, seam_nang_luong_thap[i]] = [0, 255, 0]  # Mau xanh

    # Ve duong nang luong cao nhat mau do
    for i in range(len(seam_nang_luong_cao)):
        anh_voi_seam[i, seam_nang_luong_cao[i]] = [0, 0, 255]  # Mau do

    return anh_voi_seam

def tim_duong_seam_ngang(nang_luong):
    """Tim duong nang luong thap nhat va cao nhat theo chieu ngang."""
    hang, cot = nang_luong.shape
    seam_nang_luong_thap = np.zeros((cot,), dtype=int)
    seam_nang_luong_cao = np.zeros((cot,), dtype=int)
    
    # Tinh tong nang luong cho moi hang
    tong_nang_luong_theo_hang = np.sum(nang_luong, axis=1)
    
    # Tim hang co nang luong thap nhat va cao nhat
    seam_nang_luong_thap[:] = np.argmin(tong_nang_luong_theo_hang)
    seam_nang_luong_cao[:] = np.argmax(tong_nang_luong_theo_hang)

    return seam_nang_luong_thap, seam_nang_luong_cao

def ve_duong_seam_ngang(anh, seam_nang_luong_thap, seam_nang_luong_cao):
    """Ve duong nang luong thap nhat va cao nhat theo chieu ngang."""
    anh_voi_seam_ngang = anh.copy()

    # Ve duong nang luong thap nhat mau xanh
    anh_voi_seam_ngang[seam_nang_luong_thap, :] = [0, 255, 0]  # Mau xanh

    # Ve duong nang luong cao nhat mau do
    anh_voi_seam_ngang[seam_nang_luong_cao, :] = [0, 0, 255]  # Mau do

    return anh_voi_seam_ngang

# Doc anh
anh = cv2.imread('d:/Downloads/cvb.jpg')

# Chuyen anh sang anh xam va tinh ban do nang luong
nang_luong = tinh_nang_luong(anh)

# Tim duong seam nang luong thap nhat va cao nhat theo chieu doc
seam_nang_luong_thap = tim_duong_seam_thap_nhat(nang_luong)
seam_nang_luong_cao = tim_duong_seam_cao_nhat(nang_luong)

# Tim duong seam nang luong thap nhat va cao nhat theo chieu ngang
seam_ngang_thap, seam_ngang_cao = tim_duong_seam_ngang(nang_luong)

# Ve duong seam doc nang luong thap nhat va cao nhat
anh_voi_seam_doc = ve_duong_seam(anh, seam_nang_luong_thap, seam_nang_luong_cao)

# Ve duong seam ngang nang luong thap nhat va cao nhat
anh_voi_ca_seam = ve_duong_seam_ngang(anh_voi_seam_doc, seam_ngang_thap, seam_ngang_cao)

# Hien thi ket qua
plt.figure(figsize=(10, 10))

plt.imshow(cv2.cvtColor(anh_voi_ca_seam, cv2.COLOR_BGR2RGB))
plt.title('Anh voi duong Seam doc va ngang (Thap: Xanh, Cao: Do)')
plt.axis('off')

plt.show()

# Luu anh ket qua
cv2.imwrite('anh_voi_ca_seam.jpg', anh_voi_ca_seam)
