import cv2
import os
import time
import torch

"""计算输入视频的时长和帧数，并使用 CUDA 加速"""
def video_to_images_cuda(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化 CUDA 设备
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的原始帧率
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频的总帧数

    if frame_rate == 0:
        print("Error: Could not get the frame rate of the video.")
        return

    # 计算视频时长（秒）
    video_duration = total_frames / frame_rate
    print(f"Video duration: {video_duration:.2f} seconds")
    print(f"Original frame rate: {frame_rate:.2f} FPS")

    frame_count = 0
    saved_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为 CUDA 张量
        frame_tensor = torch.from_numpy(frame).to(device)

        # 保存每一帧
        file_path = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(file_path, frame)
        saved_count += 1

        frame_count += 1

    cap.release()
    end_time = time.time()
    print(f"Total frames saved: {saved_count}")
    print(f"Processing time with CUDA: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    video_path = 'unstable_light_demo_1.mp4'
    output_folder = 'video_picture'
    video_to_images_cuda(video_path, output_folder)