from util import *
from Joints import JointType
import numpy as np
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import csv

folders = ['Hemanti Ma\'am', 'Rahul Sir and Roshan', 'Sanjeev Sir']
f = open('timing.csv', 'w')
writer = csv.writer(f)
writer.writerow(['Person', 'Aasan', 'StartTime', 'EndTime', 'Duration'])
for folder in folders:
    subfolders = [f for f in listdir(folder) if not isfile(join(folder, f))]
    for subfolder in subfolders:
        path = folder+'/'+subfolder

        bodies = build_time_series(path+'/joints_smoothed.csv')
        references = [b.skel_list()[0] for b in build_time_series(path + '/ref.csv')]
        for body in bodies:
            #ignore_joints_name = ['FootRight', 'FootLeft']
            ignore_joints_name = []
            ignore_joints = set([JointType['JointType_'+i] for i in ignore_joints_name])
            '''
            for jtype in JointType:
                error[jtype] = 0
            '''
            min_error = 10**9
            best_cnt = -1
            best_start = '0:0:0.0'
            best_end = '0:0:0.0'
            for reference in references:
                start = None
                end = None
                total_error = 0
                cnt = 0
                error_list = []
                tot_cnt = 0
                start_of_vid = end_of_vid = None
                for (time, skel) in sorted(body.time_mp.items(), key=lambda x: x[0]): 
                    if start_of_vid is None:
                        start_of_vid = time
                    end_of_vid = time
                    tot_cnt += 1
                    error = 0
                    for jtype in JointType:
                        if jtype in ignore_joints:
                            continue
                        if jtype in skel.joints:
                            error += abs_error(skel.joints[jtype].position, 
                                    reference.joints[jtype].position)
                    error_list.append(error)
                    total_error += error
                
                THRESHOLD = np.percentile(error_list, 20)
                THRESHOLD2 = np.percentile(error_list, 50)
            
                sub = 0
                for (time, skel) in sorted(body.time_mp.items(), key=lambda x: x[0]): 
                    error = 0
                    for jtype in JointType:
                        if jtype in ignore_joints:
                            continue
                        if jtype in skel.joints:
                            error += abs_error(skel.joints[jtype].position, 
                                    reference.joints[jtype].position)
                    if error <= THRESHOLD:
                        if start is None:
                            start = time
                        end = time
                        sub = 0
                    if error <= THRESHOLD2 and start is not None:
                        cnt += 1
                        if end < time:
                            sub += 1
                cnt -= sub
                if total_error < min_error:
                    total_error = min_error
                    best_cnt = cnt
                    best_start = start
                    best_end = end 
                '''
                for (jtype, err) in sorted(error.items(), key=lambda x: x[1]):
                    if jtype in ignore_joints:
                        continue
                    print(jtype, error[jtype])
                '''
            total_time = (get_time_obj(end_of_vid) - get_time_obj(start_of_vid)).total_seconds()
            time_in_pos = timedelta(seconds=(1.0*cnt)/tot_cnt * total_time) 
            len_of_vid = (get_time_obj(getLength(path+'/color.avi')) - get_time_obj('0:0:0.0')).total_seconds()
            start_sec = (get_time_obj(best_start) - get_time_obj(start_of_vid)).total_seconds()
            end_sec = (get_time_obj(best_end) - get_time_obj(start_of_vid)).total_seconds()
            start1 = timedelta(seconds=((1.0*start_sec)/total_time)*len_of_vid)
            end1 = timedelta(seconds=((1.0*end_sec)/total_time)*len_of_vid)
            writer.writerow([folder, subfolder, str(start1), str(end1), str(time_in_pos)])
            print('Done!!!', path)
