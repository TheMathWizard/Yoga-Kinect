from util import *
from Joints import JointType
import numpy as np
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import csv
from scipy import stats
from collections import deque

import sys
sys.path.append("..")

import tracking2

def segment_time(in_csv, ref_csv, out_csv):
    bodies = build_time_series(in_csv)
    references = [b.skel_list()[0] for b in build_time_series(ref_csv)]
    with open(out_csv, 'w') as f:
        writer = csv.writer(f) 
        writer.writerow(['body_id', 'start_frame', 'end_frame'])
        for body in bodies:
            min_error = 10**9
            best_start_frame = -1
            best_end_frame = -1
            for reference in references:
                total_error = 0
                error_list = []
                start_frame = end_frame = -1
                frames_body = body.to_frames()
                frame_ref = reference.to_frame(-1)
                tracking2.align_bone_resize_2(frames_body, frame_ref)
                body_aligned = frames_to_BodyTimeSeries(frames_body)
                for (time, skel) in sorted(body_aligned.time_mp.items(), key=lambda x: x[0]): 
                    error = 0
                    for jtype in JointType:
                        if jtype in skel.joints:
                            error += abs_error(skel.joints[jtype].position, 
                                    reference.joints[jtype].position)
                    error_list.append(error)
                    total_error += error
                
                THRESHOLD = np.percentile(error_list, 20)
            
                for (time, skel) in sorted(body_aligned.time_mp.items(), key=lambda x: x[0]): 
                    error = 0
                    for jtype in JointType:
                        if jtype in skel.joints:
                            error += abs_error(skel.joints[jtype].position, 
                                    reference.joints[jtype].position)
                    if error <= THRESHOLD:
                        if start_frame == -1:
                            start_frame = skel.frame
                        end_frame = skel.frame
                if total_error < min_error:
                    total_error = min_error
                    best_start_frame = start_frame
                    best_end_frame = end_frame
            writer.writerow([body_aligned.id, best_start_frame, best_end_frame])

def calc_stats(in_csv, ref_csv, weight_file, error_data, out_csv):
    bodies = build_time_series(in_csv)
    references = [b.skel_list()[0] for b in build_time_series(ref_csv)]
    weights = get_weights(weight_file)
    with open(out_csv, 'w') as f, open(error_data, 'r') as e:
        error_list = []
        for line in e:
            error_list.append(float(line))
        writer = csv.writer(f) 
        writer.writerow(['body_id', 'Time in posture', '# of balance losses', '% perfection'])
        for body in bodies:
            min_error = 10**9
            start_of_vid = end_of_vid = None
            cnt = 0
            tot_cnt = 0
            rel_perfect = 0
            balance_loss = 0

            for reference in references:    
                THRESHOLD = np.percentile(error_list, 20)
                
                prev_err = deque()
                err_ok_cnt = 0
                in_posture = False

                for (time, skel) in sorted(body.time_mp.items(), key=lambda x: x[0]): 
                    if start_of_vid is None:
                        start_of_vid = time
                    end_of_vid = time
                    tot_cnt += 1

                    error = 0
                    for jtype in JointType:
                        if jtype in skel.joints:
                            error += weights[jtype.name]*abs_error(skel.joints[jtype].position, 
                                    reference.joints[jtype].position)
                    
                    rel_perfect += (100.0-stats.percentileofscore(error_list, error))
                    prev_err.append(error)
                    
                    if len(prev_err) > 7:
                        err = prev_err.popleft()
                        if err <= THRESHOLD:
                            err_ok_cnt -= 1

                    if error <= THRESHOLD:
                        cnt += 1
                        err_ok_cnt += 1

                    prev_in_posture = in_posture

                    if err_ok_cnt == 0:
                        in_posture = False
                    else:
                        in_posture = True

                    if prev_in_posture and (not in_posture):
                        balance_loss += 1
                # only one reference frame should be present
                break
            
            rel_perfect /= tot_cnt
            total_time = (get_time_obj(end_of_vid) - get_time_obj(start_of_vid)).total_seconds()
            time_in_pos = timedelta(seconds=(1.0*cnt)/tot_cnt * total_time)

            writer.writerow([body.id, time_in_pos, balance_loss, rel_perfect])

def process_all():
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
