import pandas as pd
import matplotlib.pyplot as plt

# 加载Excel文件
file_path = r'D:\Project\振动实验数据\降采样数据\video_sensor_acc_comparison.xlsx'
df = pd.read_excel(file_path)

# 假设两组数据分别在第一列和第二列
acceleration1 = df.iloc[:, 1]
acceleration2 = df.iloc[:, 2]
time = df.index / 30.0  # 假设采样频率为30Hz，计算时间轴

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制第一组加速度数据，蓝色实线
plt.plot(time, acceleration1, label='Acceleration 1', color='blue', linewidth=2, linestyle='-')

# 绘制第二组加速度数据，红色虚线
plt.plot(time, acceleration2, label='Acceleration 2', color='red', linewidth=2, linestyle='--')

# 设置图形标题和标签
plt.title('Comparison of Two Acceleration Time Series', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Acceleration (m/s^2)', fontsize=14)

# 添加网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 添加图例
plt.legend(loc='upper right', fontsize=12)

# 美化图形
plt.tight_layout()

# 保存图形为图片文件
output_image_path = r'D:\备忘录\组会\comparison_acc.png'
plt.savefig(output_image_path)

# 显示图形
plt.show()
