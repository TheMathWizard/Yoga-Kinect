import numpy as np
import cv2
import time

cap = cv2.VideoCapture('10/color.avi')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))
last = time.time()

while(cap.isOpened()):
    ret, frame = cap.read()

    print(1/(time.time()-last))
    last = time.time()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        #out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
#out.release()
cv2.destroyAllWindows()