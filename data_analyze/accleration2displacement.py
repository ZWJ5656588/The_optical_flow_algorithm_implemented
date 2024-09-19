import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# 加载Excel文件
file_path = r'D:\Project\振动实验数据\降采样数据\1706hz_1.xlsx'
df = pd.read_excel(file_path)

# 假设加速度数据在第一列
acceleration = df.iloc[:, 0]

# 常量定义
sampling_frequency_original = 1706  # Hz
factor = 10  # 插值加密因子

# 原始采样点数和时间步长
n_samples_original = len(acceleration)
dt_original = 1 / sampling_frequency_original
time_original = np.arange(0, n_samples_original * dt_original, dt_original)

# 插值加密后的时间步长和新采样点数
dt_enhanced = dt_original / factor
time_enhanced = np.arange(0, time_original[-1], dt_enhanced)

# 确保 time_enhanced 的最大值不超过 time_original 的最大值
if time_enhanced[-1] > time_original[-1]:
    time_enhanced = time_enhanced[time_enhanced <= time_original[-1]]

# 进行正弦插值
sinusoidal_interp = interp1d(time_original, acceleration, kind='cubic')
acceleration_enhanced = sinusoidal_interp(time_enhanced)

# 数值积分得到速度和位移
velocity = cumulative_trapezoid(acceleration_enhanced, dx=dt_enhanced, initial=0)
displacement = cumulative_trapezoid(velocity, dx=dt_enhanced, initial=0)

# 创建一个新的DataFrame用于保存插值后的位移数据
df_enhanced = pd.DataFrame({
    'Time (s)': time_enhanced,
    'Displacement (m)': displacement
})

# 将处理后的数据保存到新的Excel文件中
output_file_path = r'D:\Project\振动实验数据\降采样数据\1706hz_1_displacement_enhanced.xlsx'
df_enhanced.to_excel(output_file_path, index=False)

print("处理后的文件已保存至:", output_file_path)
