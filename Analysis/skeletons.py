import numpy as np
import cv2
import csv
import os
import statistics

running_mean = {}
median = {}
color = {}
last = {}
maxval = 0

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


    #cv2.imshow('frame2', frame)
    #cv2.waitKey(100)

def drawframe(frame, list):
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
        cval = max(255 - color[item[2]]*10, 0)
        rad = max(5 - color[item[2]]*.5, 0)
        #print(color[item[2]])
        a = float(item[5])
        b = float(item[6])
        if(a==float('inf')or b==float('inf') or a==float('-inf')or b==float('-inf')):
            return
        frame = cv2.circle(frame, (round(a),round(b)), round(rad), (cval,cval,cval), -1)

def smooth(row):
    global maxval
    a = list(running_mean[row[2]])
    #print('1',a)
    if a==[0,0]:
        a = [float(row[5]),float(row[6])]

    if(last[row[2]]==[0,0]):
        last[row[2]] = a

    median[row[2]][0].append(a[0])
    median[row[2]][1].append(a[1])
    if(len(median[row[2]][0])==8):
        median[row[2]][0].pop(0)
    if(len(median[row[2]][1])==8):
        median[row[2]][1].pop(0)

    if(row[3] == 'inferred'):
        med1 = statistics.median(median[row[2]][0])
        med2 = statistics.median(median[row[2]][1])
    alpha = 0.5
    running_mean[row[2]][0] = a[0]*alpha + float(row[5])*(1-alpha)
    running_mean[row[2]][1] = a[1]*alpha + float(row[6])*(1-alpha)
    #if(row[3] == 'inferred'):
        #b = [med1,med2]
    #else:
    b = running_mean[row[2]]

    t = last[row[2]]

    if(row[3]=='inferred'):
        if(row[2]=='JointType_HandTipLeft' or row[2]=='JointType_ThumbLeft' or row[2]=='JointType_HandTipRight' or row[2]=='JointType_ThumbRight' or row[2]=='JointType_WristLeft' or row[2]=='JointType_WristRight' or row[2]=='JointType_HandRight' or row[2]=='JointType_HandLeft'):
            b = running_mean[row[2]]



    color[row[2]] = abs(b[0]-t[0])+abs(b[1]-t[1])
    last[row[2]] = b[0],b[1]
    row[5] = b[0]
    row[6] = b[1]
    '''
    if(row[3]=='inferred'):
        #print('2',row[5],row[6])
        if(color[row[2]]>maxval):
            print(maxval)
            maxval = color[row[2]]
        #print(color[row[2]])'''
    #if(color[row[2]]>100):
    #    return False
    return True

def draw_skeleton(path):
    running_mean['JointType_SpineBase'] = [0,0]
    running_mean['JointType_SpineMid'] = [0,0]
    running_mean['JointType_Neck' ]= [0,0]
    running_mean['JointType_Head'] = [0,0]
    running_mean['JointType_ShoulderLeft'] = [0,0]
    running_mean['JointType_ElbowLeft'] = [0,0]
    running_mean['JointType_WristLeft'] = [0,0]
    running_mean['JointType_HandLeft'] = [0,0]
    running_mean['JointType_ShoulderRight'] = [0,0]
    running_mean['JointType_ElbowRight'] = [0,0]
    running_mean['JointType_WristRight'] = [0,0]
    running_mean['JointType_HandRight'] = [0,0]
    running_mean['JointType_HipLeft'] = [0,0]
    running_mean['JointType_KneeLeft'] = [0,0]
    running_mean['JointType_AnkleLeft'] = [0,0]
    running_mean['JointType_FootLeft'] = [0,0]
    running_mean['JointType_HipRight'] = [0,0]
    running_mean['JointType_KneeRight'] = [0,0]
    running_mean['JointType_AnkleRight'] = [0,0]
    running_mean['JointType_FootRight'] = [0,0]
    running_mean['JointType_SpineShoulder'] = [0,0]
    running_mean['JointType_HandTipLeft'] = [0,0]
    running_mean['JointType_ThumbLeft'] = [0,0]
    running_mean['JointType_HandTipRight'] = [0,0]
    running_mean['JointType_ThumbRight'] = [0,0]
    running_mean['JointType_Count'] = [0,0]

    last['JointType_SpineBase'] = [0,0]
    last['JointType_SpineMid'] = [0,0]
    last['JointType_Neck' ]= [0,0]
    last['JointType_Head'] = [0,0]
    last['JointType_ShoulderLeft'] = [0,0]
    last['JointType_ElbowLeft'] = [0,0]
    last['JointType_WristLeft'] = [0,0]
    last['JointType_HandLeft'] = [0,0]
    last['JointType_ShoulderRight'] = [0,0]
    last['JointType_ElbowRight'] = [0,0]
    last['JointType_WristRight'] = [0,0]
    last['JointType_HandRight'] = [0,0]
    last['JointType_HipLeft'] = [0,0]
    last['JointType_KneeLeft'] = [0,0]
    last['JointType_AnkleLeft'] = [0,0]
    last['JointType_FootLeft'] = [0,0]
    last['JointType_HipRight'] = [0,0]
    last['JointType_KneeRight'] = [0,0]
    last['JointType_AnkleRight'] = [0,0]
    last['JointType_FootRight'] = [0,0]
    last['JointType_SpineShoulder'] = [0,0]
    last['JointType_HandTipLeft'] = [0,0]
    last['JointType_ThumbLeft'] = [0,0]
    last['JointType_HandTipRight'] = [0,0]
    last['JointType_ThumbRight'] = [0,0]
    last['JointType_Count'] = [0,0]

    median['JointType_SpineBase'] = [[],[]]
    median['JointType_SpineMid'] = [[],[]]
    median['JointType_Neck' ]= [[],[]]
    median['JointType_Head'] = [[],[]]
    median['JointType_ShoulderLeft'] = [[],[]]
    median['JointType_ElbowLeft'] = [[],[]]
    median['JointType_WristLeft'] = [[],[]]
    median['JointType_HandLeft'] = [[],[]]
    median['JointType_ShoulderRight'] = [[],[]]
    median['JointType_ElbowRight'] = [[],[]]
    median['JointType_WristRight'] = [[],[]]
    median['JointType_HandRight'] = [[],[]]
    median['JointType_HipLeft'] = [[],[]]
    median['JointType_KneeLeft'] = [[],[]]
    median['JointType_AnkleLeft'] = [[],[]]
    median['JointType_FootLeft'] = [[],[]]
    median['JointType_HipRight'] = [[],[]]
    median['JointType_KneeRight'] = [[],[]]
    median['JointType_AnkleRight'] = [[],[]]
    median['JointType_FootRight'] = [[],[]]
    median['JointType_SpineShoulder'] = [[],[]]
    median['JointType_HandTipLeft'] = [[],[]]
    median['JointType_ThumbLeft'] = [[],[]]
    median['JointType_HandTipRight'] = [[],[]]
    median['JointType_ThumbRight'] = [[],[]]
    median['JointType_Count'] = [[],[]]

    color['JointType_SpineBase'] = 255
    color['JointType_SpineMid'] = 255
    color['JointType_Neck' ]= 255
    color['JointType_Head'] = 255
    color['JointType_ShoulderLeft'] = 255
    color['JointType_ElbowLeft'] = 255
    color['JointType_WristLeft'] = 255
    color['JointType_HandLeft'] = 255
    color['JointType_ShoulderRight'] = 255
    color['JointType_ElbowRight'] = 255
    color['JointType_WristRight'] = 255
    color['JointType_HandRight'] = 255
    color['JointType_HipLeft'] = 255
    color['JointType_KneeLeft'] = 255
    color['JointType_AnkleLeft'] = 255
    color['JointType_FootLeft'] = 255
    color['JointType_HipRight'] = 255
    color['JointType_KneeRight'] = 255
    color['JointType_AnkleRight'] = 255
    color['JointType_FootRight'] = 255
    color['JointType_SpineShoulder'] = 255
    color['JointType_HandTipLeft'] = 255
    color['JointType_ThumbLeft'] = 255
    color['JointType_HandTipRight'] = 255
    color['JointType_ThumbRight'] = 255
    color['JointType_Count'] = 255
    fps = 7
    capSize = (1400, 1000) # this is the size of my source video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path+'/smooth.avi',fourcc,fps,capSize,True)

    try:
        with open(path+'/joints.csv', 'r') as csvfile:
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #out = cv2.VideoWriter('output.avi',fourcc, 7.0, (1000,1000))
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            frame = np.zeros((1000,1400,3),dtype=np.uint8)
            list = {}

            for i, row in enumerate(reader):
                if(i%2!=0):
                    continue
                #if(row[3]=='inferred'):
                #    continue

                if(i==0):
                    l = row[4]

                if(row[4] == l):
                    ret = smooth(row)
                    if ret:
                        list[row[2]] = row
                    l = row[4]
                else:
                    drawframe(frame, list)
                    #cv2.imshow('frame', frame)
                    #cv2.waitKey(5)
                    out.write(frame)
                    frame = np.zeros((1000,1400,3),dtype=np.uint8)
                    list = {}
                    ret = smooth(row)
                    if ret:
                        list[row[2]] = row
                    l = row[4]

            drawframe(frame, list)
            #cv2.imshow('frame', frame)
            #cv2.waitKey(5)
            #cv2.destroyAllWindows()
            out.write(frame)
    except FileNotFoundError:
        print('csv file not found')
    out.release()

if __name__ == '__main__':
    '''for x in os.walk('YogaKinect'):
        print(x[0])
        draw_skeleton(x[0])'''
    draw_skeleton('YogaKinect/11')


