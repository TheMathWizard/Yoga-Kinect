import numpy as np
import cv2

cap = cv2.VideoCapture('depth.avi')
while(cap.isOpened()):
    print(cap.read())
    break