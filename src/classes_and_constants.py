# Program Constants
num_hands = 1  # number of hands that you want the program to detect as a class (default is 1, meaning it will only be looking for one hand at a time)
num_class = 3  # number of different classes you want to distinguish between. For example, in rock-paper-scissors, there are three classes
num_instance = 50  # size of dataset. 50 is a good number, but higher is more accurate!
break_time = 3  # time delay between training of each class (this is to allow you to make the hand shape change you need in time so as to not mess up the training!)

# Simply add the class names to this list
classes = ["PAPER", "ROCK", "SCISSOR"]


def classify_class(num):
    return classes[num]
