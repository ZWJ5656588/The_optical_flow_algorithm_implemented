# 导入所需的库
import json
import os.path
import time
import matplotlib

matplotlib.use('TkAgg')  # 或者 'Qt5Agg' 设置matplotlib使用的后端
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 全局变量，用于存储关键点和帧速率相关的数据
key_points_with_fps_x = {}  # 存储每帧的关键点在x方向上的位移
key_points_with_fps_y = {}  # 存储每帧的关键点在y方向上的位移
displacement = []  # 要处理的点的位移
lists = []
key_points_indices = []  # 存储用户选择的关键点序号
initial_positions = None  # 初始位置的副本，用于计算位移
key_point_input = 0  # 用户输入的关键点序号


# 框选感兴趣区域（ROI）的回调函数
def select_roi(event, x, y, flags, param):
    global roi_pts, selecting_roi, frame_copy

    # 左键按下时记录起始点，并设置正在选择ROI的标志
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [(x, y)]
        selecting_roi = True
    # 鼠标移动时更新选择框
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi:
        frame_copy = frame.copy()
        # 绘制矩形框，颜色为绿色 (0, 255, 0)
        cv2.rectangle(frame_copy, roi_pts[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_copy)
    # 左键抬起时记录结束点，并完成ROI选择
    elif event == cv2.EVENT_LBUTTONUP:
        roi_pts.append((x, y))
        selecting_roi = False
        # 最终绘制矩形框
        cv2.rectangle(frame_copy, roi_pts[0], roi_pts[1], (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_copy)
        cv2.destroyWindow("Select ROI")  # 关闭ROI选择窗口


# 基于相位相关的光流计算函数
def phase_correlation_optical_flow(old_gray, frame_gray):
    # 使用傅里叶变换进行相位相关计算
    # 首先将图像转换为float32格式
    old_float = np.float32(old_gray)
    frame_float = np.float32(frame_gray)

    # 计算两帧之间的相位相关性
    shift = cv2.phaseCorrelate(old_float, frame_float)

    return shift


# 主要的光流处理函数，用于处理视频流中的光流跟踪
def calculate_flow(videoName):
    t = 0  # 用于记录帧数
    cap = cv2.VideoCapture(videoName)  # 打开视频文件
    color = np.random.randint(0, 255, (100, 3))  # 随机生成100种颜色用于绘制光流轨迹
    global initial_positions  # 使用全局变量存储初始位置
    ret, old = cap.read()  # 读取视频的第一帧
    old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)  # 将第一帧转换为灰度图

    # 创建一个特征点作为ROI中心点
    x_center = int((roi_pts[0][0] + roi_pts[1][0]) / 2)
    y_center = int((roi_pts[0][1] + roi_pts[1][1]) / 2)
    p0 = np.array([[[x_center, y_center]]], dtype=np.float32)  # 使用ROI中心作为初始关键点
    initial_positions = p0.copy()  # 保存初始位置

    # 在第一帧上标记检测到的特征点并显示
    for i, point in enumerate(p0):
        x, y = point.ravel()
        cv2.circle(old, (int(x), int(y)), 3, (0, 255, 0), -1)  # 使用绿色圆点标记特征点
        cv2.putText(old, str(i + 1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 在特征点处标记编号
    cv2.imshow('frame.jpg', old)  # 显示带有特征点标记的第一帧
    cv2.waitKey(2)  # 等待键盘输入（短暂停留）

    try:
        global fps
        fps = 1  # 帧速率初始值为1
        while True:
            ret, frame = cap.read()  # 读取下一帧
            if not ret:
                print("结束")  # 如果没有帧，表示视频结束
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前帧转换为灰度图

            # 计算光流位移（基于相位相关）
            shift = phase_correlation_optical_flow(old_gray, frame_gray)
            dx, dy = shift  # 获取x方向和y方向的位移量

            l1 = []  # 存储x方向的位移
            l2 = []  # 存储y方向的位移

            rows1 = frame.shape[0]  # 获取当前帧的高度
            cols1 = frame.shape[1]  # 获取当前帧的宽度
            out = np.zeros((rows1, cols1, 3), dtype='uint8')  # 创建一个空白图像，用于显示光流轨迹
            out[:rows1, :cols1] = np.dstack([frame_gray, frame_gray, frame_gray])  # 将灰度图转换为三通道图像

            # 绘制光流轨迹
            for i, point in enumerate(p0):
                a, b = point.ravel()  # 当前帧特征点的坐标
                c, d = a + dx, b + dy  # 根据位移量计算新特征点的坐标

                res1 = float(c - a)  # 计算x方向的位移
                res2 = float(d - b)  # 计算y方向的位移

                l1.append(res1)  # 将x方向位移存储到列表中
                l2.append(res2)  # 将y方向位移存储到列表中
                # 在out图像上绘制从当前帧位置到新位置的光流线
                cv2.line(out, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 3)
                cv2.circle(out, (int(c), int(d)), 3, (255, 0, 0), -1)  # 绘制新的特征点
                cv2.circle(out, (int(a), int(b)), 3, (255, 222, 173), -1)  # 绘制初始特征点
                # 显示编号
                cv2.putText(out, str(i + 1), (int(c), int(d)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('frame.jpg', out)  # 显示带有光流轨迹的图像
            k = cv2.waitKey(2) & 0xff  # 等待按键输入
            if k == 27:  # 按下ESC键退出
                break

            # 存储每一帧的x和y方向位移
            key_points_with_fps_x[f"{fps}"] = l1
            key_points_with_fps_y[f"{fps}"] = l2
            fps += 1  # 帧速率自增

    except Exception as e:  # 捕获异常
        print(e)


# 将结果保存为json格式
def save():
    with open("displacement.json", "w") as f:
        json.dump(key_points_with_fps_x, f, indent=4)  # 保存x方向位移到json文件

    global displacement
    displacement = []

    # 根据用户选择的关键点序号，生成对应的位移时程曲线
    for i in key_points_indices:
        key_displacement = [key_points_with_fps_x[str(fps)][i] for fps in range(1, len(key_points_with_fps_x) + 1)]
        displacement.append(key_displacement)

    print("位移时程曲线已生成: ", displacement)


# 绘制图像并保存
def drawing():
    fold_name = "point_displacement2"
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)  # 创建文件夹用于保存图像

    # 绘制用户选择的每个关键点的位移曲线
    for i in key_points_indices:
        if f"{i + 1}" in key_points_with_fps_x:
            plt.figure(figsize=(20, 5))  # 设置图像大小
            fps_d = []
            x_value = []
            y_value = []
            for j in range(1, fps):
                fps_d.append(key_points_with_fps_x[f"{j}"][i])
                x_value.append(j)
            y_value = fps_d

            plt.xlim(0, fps)
            plt.plot(x_value, y_value, marker="o", markersize=3)  # 绘制位移曲线
            plt.xlabel("time_fps")
            plt.ylabel("x_position")
            plt.title(f" number {i + 1} point")
            plt.tight_layout()

            file_path = os.path.join(fold_name, f"plot_{i + 1}.png")  # 保存图像路径

            plt.savefig(file_path)  # 保存图像
            plt.show()  # 显示图像
            plt.close()


# 时域信号转换到频域信号 进行傅里叶变换
def fft():
    if len(displacement) == 0:
        print("Error: No displacement data available for FFT.")
        return

    fs = 60  # 采样频率

    # 遍历每个关键点的位移时程曲线
    for idx, key_point_displacement in enumerate(displacement):
        data = np.array(key_point_displacement)

        # 获取用户输入的真实关键点序号
        key_point_number = key_points_indices[idx] + 1  # 序号从1开始显示

        # 检查数据长度是否足够
        if len(data) <= 1:
            print(f"Error: Not enough data points for FFT of key point {key_point_number}.")
            continue

        n = len(data)
        data_fft = np.fft.fft(data)  # 计算傅里叶变换

        # 计算频率轴的值
        freq = np.fft.fftfreq(n, d=1 / fs)

        # 计算幅度
        magnitude = np.abs(data_fft)

        # 找到最大幅度对应的频率
        peak_freq = freq[np.argmax(magnitude[1:n // 2])]  # 忽略直流分量
        print(f"Key Point {key_point_number} - Peak Frequency: {peak_freq} Hz")

        # 绘图展示
        plt.figure(figsize=(10, 6))
        plt.plot(freq[1:n // 2], magnitude[1:n // 2])  # 从1开始以忽略直流分量
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'FFT Analysis - Key Point {key_point_number}')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    roi_pts = []
    selecting_roi = False
    frame_copy = None
    videoName = "stabilized_output_video.avi"

    # 执行框选功能
    cap = cv2.VideoCapture(videoName)
    ret, frame = cap.read()

    frame_copy = frame.copy()
    cv2.namedWindow('Select ROI')  # 创建一个窗口用于选择ROI
    cv2.setMouseCallback('Select ROI', select_roi)  # 设置鼠标回调函数
    while True:
        cv2.imshow('Select ROI', frame_copy)  # 显示帧图像
        key = cv2.waitKey(2) & 0xFF
        if not selecting_roi and len(roi_pts) == 2:
            break

    calculate_flow(videoName)  # 进行光流计算
    global key_points_input
    # 用户输入想要记录数据和绘制图像的关键点序号
    key_points_input = input("请输入您想要记录数据和绘制图像的关键点序号，以空格分隔：")
    key_points_indices = [int(i) - 1 for i in key_points_input.split()]

    save()  # 保存位移数据
    drawing()  # 绘制位移曲线
    fft()  # 进行傅里叶变换并绘制频域图像
