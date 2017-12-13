import cv2
import numpy as np
import imutils

def compare(vid1, vid2):
    cap1 = cv2.VideoCapture(vid1)
    cap2 = cv2.VideoCapture(vid2)

    cv2.namedWindow('frame')
    i = 0
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        i+=1

        frame2 = frame2*.25
        frame2 = frame2.astype(np.uint8)
        frame = cv2.add(frame1,frame2)
        #print(np.count_nonzero(frame))
        cv2.putText(frame, str(i), (10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2, cv2.LINE_AA)
        frame = imutils.resize(frame, width=750)

        if ret1 and ret2:
            cv2.imshow('frame',frame)
            cv2.waitKey(0)
        else:
            break

if __name__=='__main__':
    compare('Rahul Sir and Roshan/new aasan 1/2.avi','Rahul Sir and Roshan/new aasan 1/3_aligned_with_2.avi')
