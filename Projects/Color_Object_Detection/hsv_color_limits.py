import cv2
import numpy as np


def get_color_limits(color):
    color = np.uint8([[color]])

    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    lower_limit = hsv_color[0][0][0] - 10, 100, 100
    upper_limit = hsv_color[0][0][0] + 10, 255, 255

    lower_limit = np.array(lower_limit, dtype='uint8')
    upper_limit = np.array(upper_limit, dtype='uint8')

    return lower_limit, upper_limit
'''lower_limit = (0, 0, 0)       
upper_limit = (179, 255, 100)'''
l, u = get_color_limits([0,0,0])
print(f"Lower : {l}\nUpper : {u}")