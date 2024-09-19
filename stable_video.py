import cv2
import numpy as np
from tqdm import tqdm  # 用于显示进度条

# 定义全局变量，用于存储鼠标框选的ROI区域
roi = None
selecting = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

# 鼠标回调函数，用于框选ROI区域
def select_roi(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, selecting, roi

    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        selecting = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE and selecting:  # 鼠标拖动
        x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键释放
        selecting = False
        x_end, y_end = x, y
        roi = (x_start, y_start, x_end - x_start, y_end - y_start)
        print(f"选定的ROI区域：{roi}")

# 稳定视频的函数，使用光流法对静止的ROI进行补偿
def stabilize_video_with_optical_flow(input_video_path, output_video_path):
    global roi

    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)

    # 获取视频属性
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 读取第一帧
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频文件的第一帧。")
        return

    # 在第一帧上框选ROI区域
    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', select_roi)

    while True:
        display_frame = first_frame.copy()
        if selecting:  # 在拖动时显示选择区域
            cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        cv2.imshow('Select ROI', display_frame)
        key = cv2.waitKey(1)

        if key == 27:  # 按下ESC键结束选择
            print("退出选择")
            break
        elif roi is not None:  # 如果已经选择了ROI区域
            cv2.destroyWindow('Select ROI')
            break

    # 将第一帧转换为灰度图像
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # 在静止的ROI区域中检测角点（用来跟踪的特征点）
    roi_gray = prev_gray[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    if p0 is None:
        print("无法检测到足够的特征点")
        return

    # 调整特征点坐标到整个图像的坐标系
    p0[:, 0, 0] += roi[0]
    p0[:, 0, 1] += roi[1]

    # 光流参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 平移矢量的累积
    cumulative_translation = np.zeros((2,), dtype=np.float32)

    # 显示进度条
    for i in tqdm(range(1, n_frames), desc="处理中"):
        # 读取下一帧
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 将当前帧转换为灰度图像
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 计算光流，跟踪特征点在新帧中的位置
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

        # 检查是否成功计算光流
        if p1 is None or st is None:
            print(f"第 {i} 帧光流计算失败，跳过该帧")
            out.write(curr_frame)  # 跳过光流计算失败的帧
            prev_gray = curr_gray.copy()  # 更新前一帧为当前帧
            continue

        # 选择成功跟踪的点
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 处理没有成功跟踪点的情况
        if len(good_new) == 0 or len(good_old) == 0:
            print(f"第 {i} 帧没有足够的跟踪点，跳过该帧")
            out.write(curr_frame)  # 跳过光流计算失败的帧
            prev_gray = curr_gray.copy()  # 更新前一帧为当前帧
            continue

        # 计算点的平移向量
        translation = np.mean(good_new - good_old, axis=0)
        cumulative_translation += translation

        # 生成平移矩阵
        transform_matrix = np.array([[1, 0, -cumulative_translation[0]],
                                     [0, 1, -cumulative_translation[1]]], dtype=np.float32)

        # 对当前帧应用平移补偿
        stabilized_frame = cv2.warpAffine(curr_frame, transform_matrix, (width, height))

        # 写入稳定后的帧
        out.write(stabilized_frame)

        # 更新前一帧
        prev_gray = curr_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # 释放视频资源
    cap.release()
    out.release()

# 使用示例
input_video_path = 'drone_demo_1_60hz.mp4'
output_video_path = 'stabilized_output_video.avi'

# 调用视频稳定函数
stabilize_video_with_optical_flow(input_video_path, output_video_path)
