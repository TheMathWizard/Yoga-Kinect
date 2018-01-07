import cv2
import numpy as np

FRAME_DIMS = (1080,1920,3)


def drawline(frame, list, joint1, joint2):
    #if(float(list[joint1][5])>1400 or float(list[joint1][6])>1000 or float(list[joint2][5])>1400 or float(list[joint2][6])>1000 or ):
    #        return
    a = float(list[joint1][5])
    b = float(list[joint1][6])
    c = float(list[joint2][5])
    d = float(list[joint2][6])
    if(a==float('inf') or b==float('inf') or c==float('inf') or d==float('inf') or a==float('-inf') or b==float('-inf') or c==float('-inf') or d==float('-inf')):
        return
    #print(float(list[joint1][5]),float(list[joint1][6]),float(list[joint2][5]),float(list[joint2][6]))
    if(list[joint1][3]=='inferred' or list[joint2][3]=='inferred'):
        frame = cv2.line(frame, (round(a),round(b)),(round(c),round(d)), (0,255,0),2)
    else:
        frame = cv2.line(frame, (round(a),round(b)),(round(c),round(d)), (0,0,255),2)


def drawframe(list):
    frame = np.zeros(FRAME_DIMS,dtype=np.uint8)

    #lines
    if('JointType_Head' in list and 'JointType_Neck' in list):
        drawline(frame, list, 'JointType_Head', 'JointType_Neck')
    if('JointType_Neck' in list and 'JointType_SpineShoulder' in list):
        drawline(frame, list, 'JointType_SpineShoulder', 'JointType_Neck')
    if('JointType_SpineShoulder' in list and 'JointType_SpineMid' in list):
        drawline(frame, list, 'JointType_SpineShoulder', 'JointType_SpineMid')
    if('JointType_SpineMid' in list and 'JointType_SpineBase' in list):
        drawline(frame, list, 'JointType_SpineMid', 'JointType_SpineBase')
    if('JointType_SpineShoulder' in list and 'JointType_ShoulderRight' in list):
        drawline(frame, list, 'JointType_SpineShoulder', 'JointType_ShoulderRight')
    if('JointType_SpineShoulder' in list and 'JointType_ShoulderLeft' in list):
        drawline(frame, list, 'JointType_SpineShoulder', 'JointType_ShoulderLeft')
    if('JointType_SpineBase' in list and 'JointType_HipRight' in list):
        drawline(frame, list, 'JointType_SpineBase', 'JointType_HipRight')
    if('JointType_SpineBase' in list and 'JointType_HipLeft' in list):
        drawline(frame, list, 'JointType_SpineBase', 'JointType_HipLeft')

    # Right Arm
    if('JointType_ShoulderRight' in list and 'JointType_ElbowRight' in list):
        drawline(frame, list, 'JointType_ShoulderRight', 'JointType_ElbowRight')
    if('JointType_ElbowRight' in list and 'JointType_WristRight' in list):
        drawline(frame, list, 'JointType_ElbowRight', 'JointType_WristRight')
    if('JointType_WristRight' in list and 'JointType_HandRight' in list):
        drawline(frame, list, 'JointType_WristRight', 'JointType_HandRight')
    if('JointType_HandRight' in list and 'JointType_HandTipRight' in list):
        drawline(frame, list, 'JointType_HandRight', 'JointType_HandTipRight')
    if('JointType_WristRight' in list and 'JointType_ThumbRight' in list):
        drawline(frame, list, 'JointType_WristRight', 'JointType_ThumbRight')

    # Left Arm
    if('JointType_ShoulderLeft' in list and 'JointType_ElbowLeft' in list):
        drawline(frame, list, 'JointType_ShoulderLeft', 'JointType_ElbowLeft')
    if('JointType_ElbowLeft' in list and 'JointType_WristLeft' in list):
        drawline(frame, list, 'JointType_ElbowLeft', 'JointType_WristLeft')
    if('JointType_WristLeft' in list and 'JointType_HandLeft' in list):
        drawline(frame, list, 'JointType_WristLeft', 'JointType_HandLeft')
    if('JointType_HandLeft' in list and 'JointType_HandTipLeft' in list):
        drawline(frame, list, 'JointType_HandLeft', 'JointType_HandTipLeft')
    if('JointType_WristLeft' in list and 'JointType_ThumbLeft' in list):
        drawline(frame, list, 'JointType_WristLeft', 'JointType_ThumbLeft')

    # Right Leg
    if('JointType_HipRight' in list and 'JointType_KneeRight' in list):
        drawline(frame, list, 'JointType_HipRight', 'JointType_KneeRight')
    if('JointType_KneeRight' in list and 'JointType_AnkleRight' in list):
        drawline(frame, list, 'JointType_KneeRight', 'JointType_AnkleRight')
    if('JointType_AnkleRight' in list and 'JointType_FootRight' in list):
        drawline(frame, list, 'JointType_AnkleRight', 'JointType_FootRight')

    # Left Leg
    if('JointType_HipLeft' in list and 'JointType_KneeLeft' in list):
        drawline(frame, list, 'JointType_HipLeft', 'JointType_KneeLeft')
    if('JointType_KneeLeft' in list and 'JointType_AnkleLeft' in list):
        drawline(frame, list, 'JointType_KneeLeft', 'JointType_AnkleLeft')
    if('JointType_AnkleLeft' in list and 'JointType_FootLeft' in list):
        drawline(frame, list, 'JointType_AnkleLeft', 'JointType_FootLeft')

    #circles
    for _, item in list.items():
        cval = max(255, 0)
        rad = max(5, 0)
        #print(color[item[2]])
        a = float(item[5])
        b = float(item[6])
        if(a==float('inf')or b==float('inf') or a==float('-inf')or b==float('-inf')):
            return
        frame = cv2.circle(frame, (round(a),round(b)), round(rad), (cval,cval,cval), -1)

    return frame

