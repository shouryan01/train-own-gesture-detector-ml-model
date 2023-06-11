# Train Your Own ML Hand-Detection Model!

## Why did I make this?

Google provides a [hand detection model]([url](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer#hand_landmark_model_bundle)) through [mediapipe](https://developers.google.com/mediapipe) but it only recognizes 7 gestures:

```
0 - Unrecognized gesture, label: Unknown
1 - ✊ Closed fist, label: Closed_Fist 
2 - ✋ Open palm, label: Open_Palm
3 - ☝️ Pointing up, label: Pointing_Up
4 - 👎 Thumbs down, label: Thumb_Down
5 - 👍 Thumbs up, label: Thumb_Up
6 - ✌️ Victory, label: Victory
7 - 🤟 Love, label: ILoveYou
```

They do provide documentation on [how to train custom models]([url](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/customize))  but for that you need access to a dataset of images. It is not always easy or possible to have access to the exact dataset you need.

That's why I created this repository. You can create a dataset using just your webcam, train a model and start using it right away!

## Dependencies:

There are some libraries that are required for this program to run. All the requirements are listed in **requirements.txt**. You're welcome to install them one by one, or you can simply run to install all of them at once.

```
pip install -r requirements.txt
```

## There are three easy steps to create your model:

1. In **classes_and_constants.py**, fill out the constants you want to use,

   ```
   num_hands - number of hands that you want the program to detect as a class (default is 1)
   num_class - number of different hand classes to distinguish between
   num_instance - training dataset size for each class (higher is more accurate, default is 50)
   break_time - a time delay between taking pictures of different hand classes (default is 2s)

   You must also define the names of your various classes. In this demo, I am using 'ROCK', 'PAPER', 'SCISSOR' but you can name this anything you want! Just make sure to add it to the list in the same classes_and_constants.py
   ```

2. Run generate_data.py

   - After all samples have been taken, please take your hand out of the frame of the camera so as to not contaminate other classes with bad data.

3. Run train_model.py

## For your convenience, a **demo.py** file is provided to test out your new model!

Simply run this file to see your predictions in action.
