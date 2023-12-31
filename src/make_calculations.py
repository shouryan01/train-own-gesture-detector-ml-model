import math


def list_of_ids():
    a = [i for i in range(21)]
    b = []
    for i in a:
        for j in range(len(a) - (i + 1)):
            b.append(([i, i + j + 1]))

    return b


def euclidean_distance(p1, p2):
    distance = math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    return int(distance)


def calculate(height, width, landmark_list):
    all_distance = []
    for item in list_of_ids():
        point_1 = [
            landmark_list[item[0]][0] * height,
            landmark_list[item[0]][1] * width,
        ]
        point_2 = [
            landmark_list[item[1]][0] * height,
            landmark_list[item[1]][1] * width,
        ]
        distance = euclidean_distance(point_1, point_2)
        all_distance.append(distance)

    return all_distance
