import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu từ file CSV
data = pd.read_csv('house_prices.csv')

# Chọn các đặc trưng (features) và nhãn (label)
X = data[['feature1', 'feature2', 'feature3']]  # Thay thế bằng các đặc trưng thực tế
y = data['price']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán giá nhà trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# In hệ số của mô hình
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
