import os
import cv2
import csv
import numpy as np
import drawing
import copy
import matplotlib.pyplot as plt

def get_frames(path):
    frames = []
    try:
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            list = {}

            for i, row in enumerate(reader):
                if(i==0):
                    l = row[4]

                if(row[4] == l):
                    #ret = smooth(row)
                    list[row[2]] = row
                    l = row[4]
                else:
                    frames.append(list)
                    list = {}
                    list[row[2]] = row
                    l = row[4]
            frames.append(list)
    except FileNotFoundError:
        print('csv file not found')
    return frames

def exp_best(frames):
    filtered = []
    for i, frame in enumerate(frames):
        fframe = copy.deepcopy(frame)
        lx = {}
        rx = {}
        ly = {}
        ry = {}
        for k in frame:
            #print(frame[k])
            for j in range(-7,0):
                if(i+j>=0 and i+j<len(frames)):
                    if(k not in frames[i+j]):
                        continue
                    x = float(frames[i+j][k][5])
                    y = float(frames[i+j][k][6])
                    if(k not in lx):
                        lx[k] = x
                        ly[k] = y
                    else:
                        lx[k] = .2*x + .8*lx[k]
                        ly[k] = .2*y + .8*ly[k]
            for j in range(7,0,-1):
                if(i+j>=0 and i+j<len(frames)):
                    if(k not in frames[i+j]):
                        continue
                    x = float(frames[i+j][k][5])
                    y = float(frames[i+j][k][6])
                    if(k not in rx):
                        rx[k] = x
                        ry[k] = y
                    else:
                        rx[k] = .2*x + .8*rx[k]
                        ry[k] = .2*y + .8*ry[k]
            if(k in lx and k in rx):
                fframe[k][5] = str(.2*float(frame[k][5]) + .4*lx[k] + .4*rx[k])
                fframe[k][6] = str(.2*float(frame[k][6]) + .4*ly[k] + .4*ry[k])
            elif(k in lx and k not in rx):
                fframe[k][5] = str(.2*float(frame[k][5]) + .8*lx[k])
                fframe[k][6] = str(.2*float(frame[k][6]) + .8*ly[k])
            elif(k not in lx and k in rx):
                fframe[k][5] = str(.2*float(frame[k][5]) + .8*rx[k])
                fframe[k][6] = str(.2*float(frame[k][6]) + .8*ry[k])

        filtered.append(fframe)
    return filtered

def exp_center(frames):
    filtered = []
    right = {}
    for i, frame in enumerate(reversed(frames)):
        for k in frame:
            if i==0:
                right[k] = [0,0]
                right[k][0] = float(frame[k][5])
                right[k][1] = float(frame[k][6])
            else:
                right[k][0] = .2*float(frame[k][5]) + .8*right[k][0]
                right[k][1] = .2*float(frame[k][6]) + .8*right[k][1]
            #print(frame[k],right[k])
    left = {}
    for i, frame in enumerate(frames):
        fframe = copy.deepcopy(frame)
        for k in frame:
            if i==0:
                left[k] = [0,0]
                left[k][0] = float(frame[k][5])
                left[k][1] = float(frame[k][6])

            right[k][0] = (right[k][0] - (.2*float(frame[k][5])))*1.25
            right[k][1] = (right[k][1] - (.2*float(frame[k][6])))*1.25
            #print(right[k])

            fframe[k][5] = str(.4*left[k][0] + .4*right[k][0] + .2*float(frame[k][5]))
            fframe[k][6] = str(.4*left[k][1] + .4*right[k][1] + .2*float(frame[k][6]))

            left[k][0] = .8*left[k][0] + (.2*float(frame[k][5]))
            left[k][1] = .8*left[k][1] + (.2*float(frame[k][6]))
        filtered.append(fframe)
    return filtered




def exp_filter(frames):
    filtered = []
    left = {}
    #.1*frame + .45*left + .45*right
    for i, frame in enumerate(frames):
        fframe = copy.deepcopy(frame)
        for k in frame:
            if i==0:
                left[k] = [0,0]
                left[k][0] = float(frame[k][5])
                left[k][1] = float(frame[k][6])
            else:
                left[k][0] = (.8*left[k][0] + (.2*float(frame[k][5])))
                left[k][1] = (.8*left[k][1] + (.2*float(frame[k][6])))

            fframe[k][5] = str(left[k][0])
            fframe[k][6] = str(left[k][1])
            #print(fframe[k][5], fframe[k][6])
        filtered.append(fframe)
    return filtered


def median_filter(frames):
    filtered = []
    for i, frame in enumerate(frames):
        bufx = {}
        bufy = {}
        fframe = copy.deepcopy(frame)
        for j in range(-7,8):
            if(i+j>=0 and i+j<len(frames)):
                for k,v in frames[i+j].items():
                    if k not in bufx:
                        bufx[k] = []
                        bufy[k] = []
                    bufx[k].append(float(v[5]))
                    bufy[k].append(float(v[6]))

        for k in bufx:
            if k in frame:
                #print(1,fframe[k][5],fframe[k][6])
                fframe[k][5] = str(np.median(bufx[k]))
                fframe[k][6] = str(np.median(bufy[k]))
                #print(2,fframe[k][5],fframe[k][6],'\n')
        filtered.append(fframe)
    return filtered


def draw_frames(frames):
    cv2.namedWindow('skeleton')
    for i, frame in enumerate(frames):
        img = drawing.drawframe(frame)
        cv2.imshow('skeleton', img)
        cv2.waitKey(100)

def write(frames, path, name='/test.avi'):
    fps = 7
    capSize = (1600, 1000) # this is the size of my source video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path+name,fourcc,fps,capSize,True)
    for frame in frames:
        out.write(drawing.drawframe(frame))

def split(path):
    try:
        with open(path+'/joints.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')

            with open(path+'/joints1.csv', 'w') as file1:
                writer1 = csv.writer(file1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                with open(path+'/joints2.csv', 'w') as file2:
                    writer2 = csv.writer(file2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    with open(path+'/joints3.csv', 'w') as file3:
                        writer3 = csv.writer(file3, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        with open(path+'/joints4.csv', 'w') as file4:
                            writer4 = csv.writer(file4, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            with open(path+'/joints5.csv', 'w') as file5:
                                writer5 = csv.writer(file5, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                                with open(path+'/joints0.csv', 'w') as file6:
                                    writer0 = csv.writer(file6, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                                    for i, row in enumerate(reader):
                                        if(i%2!=0):
                                            continue

                                        if(row[0]=='1'):
                                            writer1.writerow(row)
                                        elif(row[0]=='2'):
                                            writer2.writerow(row)
                                        elif(row[0]=='3'):
                                            writer3.writerow(row)
                                        elif(row[0]=='4'):
                                            writer4.writerow(row)
                                        elif(row[0]=='5'):
                                            writer5.writerow(row)
                                        elif(row[0]=='0'):
                                            writer0.writerow(row)



    except FileNotFoundError:
        print('csv file not found')

def align(frames1, frames2):
    #align frames1 with frames2
    for i, frame in enumerate(frames1):
        if(i==len(frames2)):
            break
        p1 = [float(frame['JointType_SpineBase'][5]),float(frame['JointType_SpineBase'][6])]
        p2 = [float(frame['JointType_SpineShoulder'][5]),float(frame['JointType_SpineShoulder'][6])]
        p3 = [float(frame['JointType_ShoulderLeft'][5]),float(frame['JointType_ShoulderLeft'][6])]

        t1 = [float(frames2[i]['JointType_SpineBase'][5]),float(frame['JointType_SpineBase'][6])]
        t2 = [float(frames2[i]['JointType_SpineShoulder'][5]),float(frame['JointType_SpineShoulder'][6])]
        t3 = [float(frames2[i]['JointType_ShoulderLeft'][5]),float(frame['JointType_ShoulderLeft'][6])]
        #print(np.array([p1,p2,p3]), np.array([t1,t2,t3]))
        Mx = cv2.getAffineTransform(np.array([p1,p2,p3], dtype=np.float32), np.array([t1,t2,t3], dtype=np.float32))

        p1 = [float(frame['JointType_SpineBase'][6]),float(frame['JointType_SpineBase'][5])]
        p2 = [float(frame['JointType_SpineShoulder'][6]),float(frame['JointType_SpineShoulder'][5])]
        p3 = [float(frame['JointType_ShoulderLeft'][6]),float(frame['JointType_ShoulderLeft'][5])]

        t1 = [float(frames2[i]['JointType_SpineBase'][6]),float(frame['JointType_SpineBase'][5])]
        t2 = [float(frames2[i]['JointType_SpineShoulder'][6]),float(frame['JointType_SpineShoulder'][5])]
        t3 = [float(frames2[i]['JointType_ShoulderLeft'][6]),float(frame['JointType_ShoulderLeft'][5])]
        My = cv2.getAffineTransform(np.array([p1,p2,p3], dtype=np.float32), np.array([t1,t2,t3], dtype=np.float32))

        for k in frame:
            '''if(k=='JointType_ShoulderLeft'):
                print(frame[k],frames2[i][k])
                print('\n')'''
            resx = np.matmul(Mx, np.array([float(frame[k][5]), float(frame[k][6]), 1], dtype=np.float32))
            resy = np.matmul(My, np.array([float(frame[k][6]), float(frame[k][5]), 1], dtype=np.float32))
            frame[k][5] = str(resx[0])
            frame[k][6] = str(resy[0])
            '''if(k=='JointType_ShoulderLeft'):
                print(frame[k],frames2[i][k])
                print('\n')'''



def draw_movement(frames, path, param=''):
    delta = []
    for i, frame in enumerate(frames):
        if i==0:
            continue
        val = 0
        for k in frame:
            val += abs(float(frame[k][5])-float(frames[i-1][k][5])) + abs(float(frame[k][6]) - float(frames[i-1][k][6]))
        #if val>1000:
            #delta.append(0)
        else:
            delta.append(val)
    plt.plot(delta)
    plt.ylabel('delta')
    plt.show()
    plt.savefig(path+'/delta'+param+'.png')
    plt.close()

def do_all(dir):
    start = False
    for x in os.walk(dir):
        if(x[0]==dir):
            continue
        if(x[0]=='Sanjeev sir/1a'):
            start = True
        if(not start):
            continue
        print(x[0])
        split(x[0])
        for i in range(0,6):
            csvpath = x[0]+'/joints'+str(i)+'.csv'
            statinfo = os.stat(csvpath)
            if(statinfo.st_size>0):
                frames = get_frames(csvpath)
                frames = median_filter(frames)
                frames = exp_best(frames)
                draw_movement(frames, x[0], param=str(i))
                try:
                    write(frames, x[0], name='/'+str(i)+'.avi')
                except cv2.error:
                    print(str(i)+' encountered OpenCV Error')
                    os.remove(x[0]+'/'+str(i)+'.avi')
                    os.remove(x[0]+'/delta'+str(i)+'.png')

if __name__ == '__main__':
    #do_all('Sanjeev sir')
    '''frames1 = exp_best(median_filter(get_frames('Rahul Sir and Roshan/new aasan 1/joints2.csv')))
    frames2 = exp_best(median_filter(get_frames('Rahul Sir and Roshan/new aasan 1/joints3.csv')))
    align(frames2, frames1)
    write(frames2, 'Rahul Sir and Roshan/new aasan 1', name='/3_aligned_with_2.avi')'''


    path = 'Rahul Sir and Roshan/new aasan 1'
    split(path)
    for i in range(0,6):
        csvpath = path+'/joints'+str(i)+'.csv'
        statinfo = os.stat(csvpath)
        if(statinfo.st_size>0):
            frames = get_frames(csvpath)
            #frames = median_filter(frames)
            #frames = exp_best(frames)
            draw_movement(frames, path, param=str(i))
            write(frames,path,name='/'+str(i)+'.avi')