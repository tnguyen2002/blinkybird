import math
import time
from collections import deque

import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh

BLINK_DELTA = 0.3 # how far above the calibrated open baseline counts as a blink

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
                 calibration_rounds=3,
                 cooldown_seconds=0.15,
                 show_window=False,
                 display_scale=0.5,
                 crop_w_ratio=0.4,
                 crop_h_ratio=0.85,
                 debug=False):
        self.delta = delta
        self.calibration_rounds = calibration_rounds
        self.cooldown_seconds = cooldown_seconds
        self.show_window = show_window
        self.display_scale = display_scale
        self.crop_w_ratio = crop_w_ratio
        self.crop_h_ratio = crop_h_ratio
        self.debug = debug

        self._cal_open_duration = 2.0
        self._cal_countdown_step = 0.7
        self._cal_blink_duration = 0.8
        self._cal_rest_duration = 0.3

        self.camera_index = camera_index
        self.cap = self._open_camera()
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._was_blinking = False
        self._last_flap_time = 0.0
        self._ratio_window = deque(maxlen=2)
        self._baseline = None
        self._threshold = None
        self._cal_started = False
        self._reset_calibration_state()
        # Latest annotated frame in RGB, for the game to blit as a PiP overlay.
        self.latest_frame = None

    def _reset_calibration_state(self):
        self._cal_step = 'open'
        self._cal_step_start = None
        self._cal_active_round = 0
        self._cal_open_samples = []
        self._cal_blink_peaks = []
        self._cal_current_peak = 0.0

    @property
    def is_calibrating(self):
        return self._cal_started and self._threshold is None

    @property
    def needs_calibration(self):
        return not self._cal_started and self._threshold is None

    @property
    def is_ready(self):
        return self._threshold is not None

    def start_calibration(self):
        self._baseline = None
        self._threshold = None
        self._was_blinking = False
        self._ratio_window.clear()
        self._reset_calibration_state()
        self._cal_started = True

    def calibration_view(self):
        """Lines to render in the menu while calibrating. Empty when done."""
        if not self.is_calibrating:
            return {}
        step = self._cal_step
        round_label = f"Blink {self._cal_active_round + 1} of {self.calibration_rounds}"
        if step == 'open':
            return {'title': 'CALIBRATING', 'body': 'Keep eyes open'}
        if step in ('count_3', 'count_2', 'count_1'):
            return {
                'title': round_label,
                'body': 'Get ready...',
                'big': step.split('_')[1],
            }
        if step == 'blink':
            return {'title': round_label, 'big': 'BLINK!'}
        if step == 'rest':
            return {'body': 'Nice!'}
        return {}

    def reset_calibration(self):
        self._baseline = None
        self._threshold = None
        self._was_blinking = False
        self._ratio_window.clear()
        self._reset_calibration_state()
        self._cal_started = False

    def _advance_calibration(self, now):
        elapsed = now - self._cal_step_start
        step = self._cal_step
        if step == 'open' and elapsed >= self._cal_open_duration:
            self._cal_step, self._cal_step_start = 'count_3', now
        elif step == 'count_3' and elapsed >= self._cal_countdown_step:
            self._cal_step, self._cal_step_start = 'count_2', now
        elif step == 'count_2' and elapsed >= self._cal_countdown_step:
            self._cal_step, self._cal_step_start = 'count_1', now
        elif step == 'count_1' and elapsed >= self._cal_countdown_step:
            self._cal_step, self._cal_step_start = 'blink', now
            self._cal_current_peak = 0.0
        elif step == 'blink' and elapsed >= self._cal_blink_duration:
            self._cal_blink_peaks.append(self._cal_current_peak)
            self._cal_active_round += 1
            if self._cal_active_round >= self.calibration_rounds:
                self._finalize_calibration()
            else:
                self._cal_step, self._cal_step_start = 'rest', now
        elif step == 'rest' and elapsed >= self._cal_rest_duration:
            self._cal_step, self._cal_step_start = 'count_3', now

    def _finalize_calibration(self):
        if not self._cal_open_samples:
            self.reset_calibration()
            return
        sorted_open = sorted(self._cal_open_samples)
        self._baseline = sorted_open[len(sorted_open) // 4]
        peaks = [p for p in self._cal_blink_peaks if p > 0]
        gap = (sum(peaks) / len(peaks) - self._baseline) if peaks else 0.0
        # Floor at the manual delta so a missed/half-hearted calibration blink
        # doesn't leave the threshold absurdly close to the open baseline.
        self._threshold = self._baseline + max(gap * 0.5, self.delta)
        self._cal_step = 'done'
        if self.debug:
            print(f"calibration done: baseline={self._baseline:.2f} "
                  f"peaks={[f'{p:.2f}' for p in peaks]} "
                  f"threshold={self._threshold:.2f}")

    def _open_camera(self):
        """Try the configured index, then the other of {0, 1}. Returns the
        first VideoCapture that opens and yields a frame."""
        tried = []
        for idx in (self.camera_index, 1 - self.camera_index if self.camera_index in (0, 1) else 0):
            if idx in tried:
                continue
            tried.append(idx)
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    if self.debug:
                        print(f"camera: opened index {idx}")
                    self.camera_index = idx
                    return cap
            cap.release()
            if self.debug:
                print(f"camera: index {idx} failed")
        # Return a closed capture so .read() simply returns False and we'll retry.
        return cv2.VideoCapture(self.camera_index)

    def _reopen_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = self._open_camera()

    def poll_flap(self):
        """Grab one frame and return True if it's a fresh blink edge."""
        retval, frame = self.cap.read()
        if not retval:
            self._reopen_camera()
            return False

        frame = cv2.flip(frame, 1)

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
            smoothed_ratio = sum(self._ratio_window) / len(self._ratio_window)

            if self.is_calibrating:
                now = time.monotonic()
                if self._cal_step_start is None:
                    self._cal_step_start = now
                if self._cal_step == 'open':
                    self._cal_open_samples.append(blink_ratio)
                elif self._cal_step == 'blink':
                    self._cal_current_peak = max(self._cal_current_peak, blink_ratio)
                self._advance_calibration(now)
            elif self.is_ready:
                is_blinking = smoothed_ratio > self._threshold

            if self.debug:
                if self.is_calibrating:
                    print(f"face=YES raw={blink_ratio:.2f} CALIBRATING "
                          f"step={self._cal_step} round={self._cal_active_round}")
                elif self.is_ready:
                    print(f"face=YES raw={blink_ratio:.2f} smooth={smoothed_ratio:.2f} "
                          f"thr={self._threshold:.2f} d={smoothed_ratio - self._threshold:+.2f} "
                          f"state={'CLOSED' if is_blinking else 'open'}")
                else:
                    print(f"face=YES raw={blink_ratio:.2f} IDLE (awaiting calibration)")
            if self.show_window:
                # Draw eye landmarks so we can verify they track the right points.
                for idx in LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS:
                    lm = landmarks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                               2, (0, 255, 0), -1)
                cv2.putText(frame, f"ratio={blink_ratio:.2f}",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA)
                if self.is_calibrating:
                    cv2.putText(frame, "CALIBRATING", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 255), 2, cv2.LINE_AA)
                elif self.needs_calibration:
                    cv2.putText(frame, "PRESS C", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 255), 2, cv2.LINE_AA)
                elif is_blinking:
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
            self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return flap

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.face_mesh.close()


if __name__ == "__main__":
    # Standalone smoke test: still uses an OpenCV window since there's no
    # pygame surface to blit into.
    cv2.namedWindow('BlinkDetector')
    detector = BlinkDetector(show_window=True, debug=True)
    try:
        while True:
            if detector.poll_flap():
                print("FLAP")
            if detector.latest_frame is not None:
                cv2.imshow('BlinkDetector',
                           cv2.cvtColor(detector.latest_frame,
                                        cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == 27:
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()
