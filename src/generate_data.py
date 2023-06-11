# Before running this file, please update the values in classes_and_constants.py

if __name__ == "__main__":
    # Required Library Imports
    from cv2 import cv2
    import pandas as pd
    import time

    # Importing Helper Classes
    import detect_hands
    from make_calculations import calculate
    import classes_and_constants as c
    from classes_and_constants import classify_class

    num_hands = c.num_hands
    num_class = c.num_class
    num_instance = c.num_instance
    break_time = c.break_time

    full_data = []
    data_target = 0

    hands = detect_hands.hand_detector(max_hands=num_hands)
    cap = cv2.VideoCapture(0)
    print("Now collecting data for: {}".format(classify_class(data_target)), "\n")

    def putText(text_to_put, coordinates, color):
        cv2.putText(
            image,
            text_to_put,
            coordinates,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color,
            6,
            cv2.LINE_AA,
        )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        image, list = hands.find_hand_landmarks(cv2.flip(frame, 1), draw_landmarks=True)

        if len(full_data) == 0:
            putText(
                "Collecting data for class '{}' {}/{}".format(
                    classify_class(data_target), len(full_data), num_instance
                ),  # text
                (10, 80),  # coordinates
                (3, 191, 8),  # color
            )

        if list:
            height, width, _ = image.shape
            distance_list = calculate(height, width, list)
            full_data.append(distance_list)
            print(len(full_data))

            putText(
                "Collecting data for class '{}' {}/{}".format(
                    classify_class(data_target), len(full_data), num_instance
                ),  # text
                (10, 80),  # coordinates
                (3, 191, 8),  # color
            )

        cv2.imshow("Custom ML Data Collector", image)
        cv2.waitKey(2)

        if len(full_data) >= num_instance:
            print(
                "Creating Pandas DataFrame...",
            )
            hand1_df = pd.DataFrame(full_data)
            hand1_df["y"] = data_target
            hand1_df.to_csv(
                f"class-{classify_class(data_target)}-data.csv", index=False
            )
            data_target += 1
            full_data = []
            if data_target >= num_class:
                break
            else:
                print(
                    "Get ready to train the next class .... "
                    + classify_class(data_target)
                )
                time.sleep(break_time)

    cap.release()
