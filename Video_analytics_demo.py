import cv2
import cv2.legacy
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

matplotlib.style.use('bmh')


class VideoAnalytics:
    def __init__(self, video_file_name, tracker_type='MEDIANFLOW'):
        self.video_file_name = video_file_name
        self.centroids_arr = []
        self.tracker_type = tracker_type
        self.tracker = self.initialize_tracker(tracker_type)

    def initialize_tracker(self, tracker_type):
        tracker_types = {
            'BOOSTING': cv2.TrackerBoosting_create,
            'MIL': cv2.TrackerMIL_create,
            'KCF': cv2.TrackerKCF_create,
            'TLD': cv2.TrackerTLD_create,
            'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create,
            'GOTURN': cv2.TrackerGOTURN_create,
            'MOSSE': cv2.TrackerMOSSE_create,
            'CSRT': cv2.TrackerCSRT_create
        }
        return tracker_types.get(tracker_type, cv2.legacy.TrackerMedianFlow_create)()

    def track_box(self):
        video = cv2.VideoCapture(self.video_file_name)
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        bbox = cv2.selectROI(frame, False)
        ok = self.tracker.init(frame, bbox)
        timer = cv2.getTickCount()

        while True:
            centroid = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
            self.centroids_arr.append(centroid)
            ok, frame = video.read()
            if not ok:
                break
            fps = video.get(cv2.CAP_PROP_FPS)
            ok, bbox = self.tracker.update(frame)
            Time = (cv2.getTickCount() - timer) / cv2.getTickFrequency()
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(frame, self.tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "Timer: " + str(int(Time)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.imshow("Tracking", frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        video.release()
        cv2.destroyAllWindows()
        return self.centroids_arr, fps

    def extract_coordinates(self):
        centroids_arr, fps = self.track_box()
        x_coords = np.array(centroids_arr)[:, 0]
        y_coords = np.array(centroids_arr)[:, 1]
        x_axis = np.arange(len(x_coords)) / fps
        return x_coords, y_coords, x_axis, fps

    def normalize_axes(self):
        video = cv2.VideoCapture(self.video_file_name)
        ok, frame = video.read()
        bbox_norm = cv2.selectROI(frame, False)
        video.release()
        cv2.destroyAllWindows()
        x_len = bbox_norm[2]
        y_len = bbox_norm[3]
        real_x = 5
        sx = real_x / x_len
        sy = sx
        return sx, sy

    def plot_lateral_displacement(self, x_axis, x_coords, sx):
        plt.figure(figsize=(25, 6))
        plt.plot(x_axis, x_coords * sx)
        plt.title('Target Lateral Displacement')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (Inches)')
        plt.show()

    def compensate_camera_shake(self, x_coords, y_coords, sx, sy, x_axis, fps):
        fixed_centroids_arr = self.track_box()
        x_fixed = np.array(fixed_centroids_arr)[:, 0]
        y_fixed = np.array(fixed_centroids_arr)[:, 1]
        plt.figure(figsize=(25, 6))
        plt.plot(x_axis, x_fixed * sx)
        plt.title('Camera Lateral Displacement')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (in)')
        plt.show()

        x1 = np.abs(np.subtract(x_coords, x_fixed))
        y1 = np.abs(np.subtract(y_coords, y_fixed))

        plt.figure(figsize=(25, 6))
        plt.plot(x_axis, (x1 * sx))
        plt.title('Target Compensated Lateral Displacement')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (in)')
        plt.show()

        return x1, y1

    def apply_lowpass_filter(self, x1, sx, x_axis, fps):
        DURATION = len(x_axis) / fps
        SAMPLE_RATE = fps
        N = DURATION * SAMPLE_RATE

        def butter_lowpass(cutoff, nyq_freq, order=4):
            normal_cutoff = float(cutoff) / nyq_freq
            b, a = signal.butter(order, normal_cutoff, btype='lowpass')
            return b, a

        def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
            b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
            y = signal.filtfilt(b, a, data)
            return y

        cutoff_frequency = 3
        a_xf = butter_lowpass_filter(x1 * sx, cutoff_frequency, SAMPLE_RATE / 2)

        plt.figure(figsize=(20, 6))
        plt.plot(x_axis, x1 * sx, color='red', label="Original signal")
        plt.plot(x_axis, a_xf, color='blue', label="Filtered low-pass with cutoff frequency of {} Hz".format(cutoff_frequency))
        plt.title("Compensated Filtered Lateral displacement")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (inches)')
        plt.legend()
        plt.show()

        diffx = np.array(x1 * sx) - np.array(a_xf)
        plt.figure(figsize=(20, 6))
        plt.plot(diffx, color='gray', label="What has been removed" )
        plt.show()

    def run_analysis(self):
        x_coords, y_coords, x_axis, fps = self.extract_coordinates()
        sx, sy = self.normalize_axes()
        self.plot_lateral_displacement(x_axis, x_coords, sx)
        x1, y1 = self.compensate_camera_shake(x_coords, y_coords, sx, sy, x_axis, fps)
        self.apply_lowpass_filter(x1, sx, x_axis, fps)


if __name__ == "__main__":
    video_file_name = 'demo3.mp4'
    video_analytics = VideoAnalytics(video_file_name)
    video_analytics.run_analysis()