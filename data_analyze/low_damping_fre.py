import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample

# 读取Excel文件中的数据 (假设加速度数据存储在第一列)
file_path = r'D:\Project\振动实验数据\降采样数据\1706hz_1.xlsx'  # 请替换为实际文件路径
data = pd.read_excel(file_path)

# 假设加速度时程数据在第一列
acc_data = data.iloc[:, 0].values

# 原始采样频率和目标采样频率
original_fs = 1706  # 原始采样频率1706Hz
target_fs = 30  # 目标采样频率30Hz

# 计算总的样本数
num_samples = len(acc_data)

# 计算原始时刻数组
time_original = np.arange(0, num_samples) / original_fs

# 计算新的样本数（根据目标采样率）
num_samples_resampled = int(len(acc_data) * target_fs / original_fs)

# 进行降采样
acc_resampled = resample(acc_data, num_samples_resampled)

# 计算降采样后的时刻数组
time_resampled = np.arange(0, num_samples_resampled) / target_fs

# 创建降采样后的 DataFrame
resampled_df = pd.DataFrame({'Time (s)': time_resampled, 'Acceleration (m/s²)': acc_resampled})

# 将降采样后的数据保存到 Excel 文件
output_file_path = r'D:\Project\振动实验数据\降采样数据\30Hz_downsampling.xlsx'
resampled_df.to_excel(output_file_path, index=False)

# 绘制原始数据和降采样后的数据
plt.figure(figsize=(10, 6))
plt.plot(time_original, acc_data, label='Original Data (1706Hz)', alpha=0.5)
plt.plot(time_resampled, acc_resampled, label='Resampled Data (30Hz)', color='red')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration Time History (Original vs. Resampled)')
plt.legend()
plt.grid(True)
plt.show()

output_file_path