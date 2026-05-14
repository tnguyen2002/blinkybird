import cv2
import dlib
import math

BLINK_RATIO_THRESHOLD = 5.7
cap = cv2.VideoCapture(1)
cv2.namedWindow('BlinkDetector')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]


def midpoint (point1, point2):
    return int(point1.x + point2.x) / 2 , int(point1.y + point2.y) / 2

def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2  + (point2[1] - point1[1]) ** 2)

#eye_points arry of indexs of the yee landmarks
def get_blink_ratio(eye_points, facial_landmarks):
    corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y)
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                            facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))
    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


while True:
    retval, frame = cap.read()
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #converting to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #dlib
    faces,_,_ = detector.run(image = frame, upsample_num_times = 0,
                            adjust_threshold = 0.0)
    for face in faces:
        landmarks = predictor(frame, face)
        left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2
        if blink_ratio > BLINK_RATIO_THRESHOLD:
            #Blink detected! Do Something!
            cv2.putText(frame,"BLINKING",(10,50), cv2.FONT_HERSHEY_SIMPLEX,
                        2,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow('BlinkDetector', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
