import math
import threading
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
                 threaded=True,
                 debug=False):
        self.delta = delta
        self.cooldown_seconds = cooldown_seconds
        self.show_window = show_window
        self.threaded = threaded
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
        self._lock = threading.Lock()
        self._pending_flaps = 0
        self._latest_display_frame = None
        self._stop_event = threading.Event()
        self._thread = None

        if self.show_window:
            cv2.namedWindow('BlinkDetector')

        if self.threaded:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            self._tick()

    def _tick(self):
        """Read one frame, update edge state, push flap on open->closed edge."""
        retval, frame = self.cap.read()
        if not retval:
            return

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
            with self._lock:
                self._pending_flaps += 1
        self._was_blinking = is_blinking

        if self.show_window:
            if self.threaded:
                with self._lock:
                    self._latest_display_frame = frame
            else:
                cv2.imshow('BlinkDetector', frame)
                cv2.waitKey(1)

    def poll_flap(self):
        """Return True if a flap edge happened since the last call."""
        if not self.threaded:
            self._tick()
        with self._lock:
            if self._pending_flaps > 0:
                self._pending_flaps -= 1
                return True
            return False

    def display_window(self):
        """Call from the main thread to show the latest camera frame."""
        if not (self.show_window and self.threaded):
            return
        with self._lock:
            frame = self._latest_display_frame
        if frame is not None:
            cv2.imshow('BlinkDetector', frame)
            cv2.waitKey(1)

    def release(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
        self.face_mesh.close()
        if self.show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Synchronous mode so cv2.imshow runs on the main thread.
    detector = BlinkDetector(show_window=True, threaded=False, debug=True)
    try:
        while True:
            if detector.poll_flap():
                print("FLAP")
            if cv2.waitKey(1) == 27:
                break
    finally:
        detector.release()
