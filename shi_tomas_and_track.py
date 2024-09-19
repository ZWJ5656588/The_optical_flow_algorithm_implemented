import cv2
import cv2.legacy
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('bmh')


class ShiTomasiTracker:
    def __init__(self, video_file_name, tracker_type='MIL'):
        self.video_file_name = video_file_name
        self.tracker_type = tracker_type
        self.tracker = self.initialize_tracker(tracker_type)
        self.selected_points = []  # 存储用户框选的角点
        self.tracking_points = []  # 存储正在追踪的角点
        self.roi_pts = []
        self.selecting_roi = False

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
        return tracker_types.get(tracker_type, cv2.TrackerMIL_create)()

    def select_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_pts = [(x, y)]
            self.selecting_roi = True
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting_roi:
            frame_copy = self.frame.copy()
            cv2.rectangle(frame_copy, self.roi_pts[0], (x, y), (0, 255), 2)
            cv2.imshow("Select ROI", frame_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_pts.append((x, y))
            self.selecting_roi = False
            cv2.rectangle(self.frame, self.roi_pts[0], self.roi_pts[1], (0, 255, 0), 2)
            cv2.imshow("Select ROI", self.frame)

    def detect_and_draw_corners(self, frame, roi_pts):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        selected_roi = frame[roi_pts[0][1]:roi_pts[1][1], roi_pts[0][0]:roi_pts[1][0]]
        gray_roi = cv2.cvtColor(selected_roi, cv2.COLOR_BGR2GRAY)

        # 检测角点
        corners = cv2.goodFeaturesToTrack(gray_roi, 100, 0.01, 10)
        corners = np.int0(corners)

        for corner in corners:
            x, y = corner.ravel()
            x += roi_pts[0][0]
            y += roi_pts[0][1]
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            self.selected_points.append((x, y))

        return frame

    def track_points(self):
        video = cv2.VideoCapture(self.video_file_name)
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", self.select_roi)

        while True:
            ret, self.frame = video.read()
            if not ret:
                break
            cv2.imshow("Select ROI", self.frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        video.release()
        cv2.destroyAllWindows()

        if len(self.roi_pts) == 2:
            frame = self.detect_and_draw_corners(self.frame, self.roi_pts)
            cv2.imshow("Corners", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run(self):
        self.track_points()


if __name__ == "__main__":
    video_file_name = 'demo.mp4'
    shi_tomasi_tracker = ShiTomasiTracker(video_file_name)
    shi_tomasi_tracker.run()