import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_interval_time_series(length=12*30,
                                   amplitude=8,
                                   offset=26,
                                   base_radius=0.5,
                                   dry_multiplier=2.0,
                                   wet_multiplier=1.0,
                                   asym_noise=1.0,            # nhiễu bất đối xứng
                                   trend_per_month=10,  # xu hướng tăng dần mỗi tháng
                                   freq='ME',
                                   seed=42):
    np.random.seed(seed)
    
    # Chuỗi thời gian
    dates = pd.date_range(start="1990-01-01", periods=length, freq=freq)
    months = np.arange(length)
    month_num = dates.month

    # Trung tâm = seasonal + trend + offset
    seasonal = amplitude * np.sin(2 * np.pi * months / 12)
    trend = 10 * (1 - np.exp(-0.01 * months))

    center = offset + seasonal + trend

    # Mùa mưa (tháng 5–10)
    is_wet = (month_num >= 5) & (month_num <= 10)

    # Bán kính cơ bản theo mùa
    base_radius_arr = np.where(is_wet,
                               base_radius * wet_multiplier,
                               base_radius * dry_multiplier)

    # Nhiễu đối xứng cho radius cơ bản
    # Hệ số giảm dần từ 1 → 0.5 về cuối
    shrink_factor = np.exp(-0.01 * (length - months))  # giảm dần

    radius = base_radius_arr * shrink_factor


    # Thêm nhiễu cho từng biên: bất đối xứng
    lower_noise = np.abs(np.random.normal(0, asym_noise, size=length))
    upper_noise = np.abs(np.random.normal(0, asym_noise, size=length))

    # Tạo khoảng
    lower = center - radius - lower_noise
    upper = center + radius + upper_noise

    # Đảm bảo không đảo ngược biên
    upper = np.maximum(upper, lower + 0.1)

    df = pd.DataFrame({
        'date': dates,
        'Low': lower,
        'High': upper
    })

    return df

# Sinh dữ liệu hẹp hơn
df = generate_interval_time_series(
    base_radius=0.5,
    dry_multiplier=2,
    wet_multiplier=1,
    asym_noise=3          # nhiễu bất đối xứng mạnh hơn
)

# Vẽ lại
plt.figure(figsize=(12, 5))
plt.fill_between(df['date'], df['Low'], df['High'], color='lightblue', alpha=0.5, label='Interval')
plt.title("Narrower Interval Time Series")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()

# Lưu
df.to_csv("./dataset/T.csv", index=False)
