import yfinance as yf
import matplotlib.pyplot as plt

# Lấy dữ liệu tỷ giá USD/VND
df = yf.download("CTG.VN", period="5y", interval="1d")

# Kiểm tra dữ liệu
print(df.head())

# Lưu lại nếu cần
df[['Low', 'High']].dropna().to_csv("VIC.csv")

# Vẽ biểu đồ
df[['Low', 'High']].plot(title="Tỷ giá USD/VND - 5 năm", figsize=(12, 5))
plt.grid(True)
plt.tight_layout()
plt.show()
