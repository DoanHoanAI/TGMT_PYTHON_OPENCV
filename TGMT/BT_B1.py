import cv2
import os

# Đọc ảnh màu từ đường dẫn
image = cv2.imread(r'D:\TGMT\cho_meo.jfif')

# Kiểm tra xem ảnh có đọc được hay không
if image is None:
    print("Không thể đọc ảnh từ đường dẫn đã chỉ định.")
else:
    height, width, channels = image.shape
    print(f"Chiều rộng: {width}")
    print(f"Chiều cao: {height}")
    print(f"Số kênh màu: {channels}")

    # Chuyển ảnh màu sang đen trắng (ảnh xám)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Chuyển ảnh xám sang ảnh nhị phân (binary)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Tách các kênh màu B, G, R
    B, G, R = cv2.split(image)

    # Tạo các ảnh chỉ hiển thị một kênh màu
    R_image = cv2.merge([B * 0, G * 0, R])
    G_image = cv2.merge([B * 0, G, R * 0])
    B_image = cv2.merge([B, G * 0, R * 0])

    # Tạo các ảnh với thứ tự chuyển đổi kênh màu khác nhau
    BGR_image = cv2.merge([B, G, R])
    RGB_image = cv2.merge([R, G, B])
    GRB_image = cv2.merge([G, R, B])
    BRG_image = cv2.merge([B, R, G])

    # Chuyển ảnh đen trắng và binary sang 3 kênh để ghép
    gray_image_3_channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    binary_image_3_channel = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Đảm bảo tất cả các ảnh có cùng chiều cao
    gray_image_3_channel = cv2.resize(gray_image_3_channel, (width, height))
    binary_image_3_channel = cv2.resize(binary_image_3_channel, (width, height))
    R_image = cv2.resize(R_image, (width, height))
    G_image = cv2.resize(G_image, (width, height))
    B_image = cv2.resize(B_image, (width, height))

    # Nối các ảnh theo chiều ngang
    top_row = cv2.hconcat([BGR_image, RGB_image, GRB_image, BRG_image])
    middle_row = cv2.hconcat([gray_image_3_channel, binary_image_3_channel])
    bottom_row = cv2.hconcat([R_image, G_image, B_image])

    # Đảm bảo các hàng có cùng chiều rộng bằng cách resize
    max_width = max(top_row.shape[1], middle_row.shape[1], bottom_row.shape[1])
    top_row = cv2.resize(top_row, (max_width, top_row.shape[0]))
    middle_row = cv2.resize(middle_row, (max_width, middle_row.shape[0]))
    bottom_row = cv2.resize(bottom_row, (max_width, bottom_row.shape[0]))

    # Nối các hàng để tạo thành ảnh tổng hợp
    combined_image = cv2.vconcat([top_row, middle_row, bottom_row])

    # Hiển thị ảnh kết hợp
    cv2.imshow('Combined Image', combined_image)

    # Đợi phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Tạo thư mục để lưu ảnh nếu chưa tồn tại
    output_dir = r'D:\TGMT\output_images'
    os.makedirs(output_dir, exist_ok=True)

    # Lưu ảnh tổng hợp vào thư mục
    cv2.imwrite(os.path.join(output_dir, 'combined_image.jpg'), combined_image)
