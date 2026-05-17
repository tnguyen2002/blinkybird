import math
import time
from collections import deque

import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh

BLINK_DELTA = 0.3  # how far above the rolling open baseline counts as a blink

# MediaPipe Face Mesh eye landmark indices.
# Order: [outer_corner, top_a, top_b, inner_corner, bottom_b, bottom_a]
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]


def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def get_blink_ratio(eye_points, landmarks, image_w, image_h):
    """eye_points: [outer, top_a, top_b, inner, bottom_b, bottom_a]."""
    def pt(i):
        lm = landmarks[i]
        return (lm.x * image_w, lm.y * image_h)

    outer = pt(eye_points[0])
    inner = pt(eye_points[3])
    top_a = pt(eye_points[1])
    top_b = pt(eye_points[2])
    bottom_b = pt(eye_points[4])
    bottom_a = pt(eye_points[5])
    # Midpoint-of-points: measures vertical at the eye's center, where the
    # eyelid travels the most — gives the largest dynamic range on closure.
    center_top = ((top_a[0] + top_b[0]) / 2, (top_a[1] + top_b[1]) / 2)
    center_bottom = ((bottom_a[0] + bottom_b[0]) / 2,
                     (bottom_a[1] + bottom_b[1]) / 2)
    horizontal_length = euclidean_distance(outer, inner)
    vertical_length = euclidean_distance(center_top, center_bottom)
    return horizontal_length / vertical_length


class BlinkDetector:
    def __init__(self,
                 camera_index=1,
                 delta=BLINK_DELTA,
                 baseline_window=60,
                 cooldown_seconds=0.15,
                 show_window=False,
                 display_scale=0.5,
                 crop_w_ratio=0.4,
                 crop_h_ratio=0.85,
                 debug=False):
        self.delta = delta
        self.cooldown_seconds = cooldown_seconds
        self.show_window = show_window
        self.display_scale = display_scale
        self.crop_w_ratio = crop_w_ratio
        self.crop_h_ratio = crop_h_ratio
        self.debug = debug

        self.cap = cv2.VideoCapture(camera_index)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._was_blinking = False
        self._last_flap_time = 0.0
        self._ratio_window = deque(maxlen=2)
        self._baseline_window = deque(maxlen=baseline_window)

        if self.show_window:
            cv2.namedWindow('BlinkDetector')

    def poll_flap(self):
        """Grab one frame and return True if it's a fresh blink edge."""
        retval, frame = self.cap.read()
        if not retval:
            return False

        if self.crop_w_ratio < 1.0 or self.crop_h_ratio < 1.0:
            fh, fw = frame.shape[:2]
            cw, ch = int(fw * self.crop_w_ratio), int(fh * self.crop_h_ratio)
            x0, y0 = (fw - cw) // 2, (fh - ch) // 2
            frame = frame[y0:y0 + ch, x0:x0 + cw]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        is_blinking = False
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye_ratio = get_blink_ratio(LEFT_EYE_LANDMARKS, landmarks, w, h)
            right_eye_ratio = get_blink_ratio(RIGHT_EYE_LANDMARKS, landmarks, w, h)
            blink_ratio = (left_eye_ratio + right_eye_ratio) / 2
            self._ratio_window.append(blink_ratio)
            # Only feed open frames into baseline so blinks don't drag it up.
            if not self._was_blinking:
                self._baseline_window.append(blink_ratio)
            smoothed_ratio = sum(self._ratio_window) / len(self._ratio_window)
            if self._baseline_window:
                sorted_window = sorted(self._baseline_window)
                baseline = sorted_window[len(sorted_window) // 4]
            else:
                baseline = smoothed_ratio
            is_blinking = (smoothed_ratio - baseline) > self.delta
            if self.debug:
                print(f"face=YES raw={blink_ratio:.2f} smooth={smoothed_ratio:.2f} "
                      f"base={baseline:.2f} d={smoothed_ratio - baseline:+.2f} "
                      f"state={'CLOSED' if is_blinking else 'open'}")
            if self.show_window:
                # Draw eye landmarks so we can verify they track the right points.
                for idx in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS:
                    lm = landmarks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                               2, (0, 255, 0), -1)
                cv2.putText(frame, f"ratio={blink_ratio:.2f}",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA)
                if is_blinking:
                    cv2.putText(frame, "BLINKING", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2, (255, 255, 255), 2, cv2.LINE_AA)
        elif self.debug:
            print("face=NO")

        now = time.monotonic()
        flap = (
            is_blinking
            and not self._was_blinking
            and (now - self._last_flap_time) >= self.cooldown_seconds
        )
        if flap:
            self._last_flap_time = now
        self._was_blinking = is_blinking

        if self.show_window:
            if self.display_scale != 1.0:
                fh, fw = frame.shape[:2]
                frame = cv2.resize(frame,
                                   (int(fw * self.display_scale),
                                    int(fh * self.display_scale)))
            cv2.imshow('BlinkDetector', frame)
            cv2.waitKey(1)

        return flap

    def release(self):
        self.cap.release()
        self.face_mesh.close()
        if self.show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = BlinkDetector(show_window=True, debug=True)
    try:
        while True:
            if detector.poll_flap():
                print("FLAP")
            if cv2.waitKey(1) == 27:
                break
    finally:
        detector.release()
