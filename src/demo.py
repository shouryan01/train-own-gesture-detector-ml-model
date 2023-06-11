# Required Library Imports
from cv2 import cv2
import pickle
import warnings
import random
import time

warnings.filterwarnings("ignore")

# Importing Helper Classes
import detect_hands
from classes_and_constants import num_hands
from make_calculations import calculate
from classes_and_constants import classify_class

hands = detect_hands.hand_detector(max_hands=num_hands)
model = pickle.load(open("model.sav", "rb"))
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image, list = hands.find_hand_landmarks(cv2.flip(frame, 1), draw_landmarks=False)
    if list:
        height, width, _ = image.shape
        all_distance = calculate(height, width, list)
        prediction = model.predict([all_distance])[0]
        prediction_text = "You chose " + classify_class(prediction)

        cv2.putText(
            image,
            prediction_text,
            (600, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (3, 191, 8),
            6,
            cv2.LINE_AA,
        )

    cv2.imshow("Custom ML Model Demo", image)
    cv2.waitKey(1)
