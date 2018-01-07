import csv
from Skeleton import Skeleton
from BodyTimeSeries import BodyTimeSeries
from Joints import Joint, Position, ConfidenceState, JointType
from datetime import datetime
import subprocess
from enum import IntEnum

class CSVColumn(IntEnum):
    Skeleton_Id = 0
    Time = 1
    JointType = 2
    JointConfidence = 3
    Frame = 4
    X_coord = 5
    Y_coord = 6

def get_weights(file):
    weights = {}
    if file is None:
        for jtype in JointType:
            weights[jtype.name] = 1
        return weights
    with open(file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            weights[row['joint']] = float(row['weight'])
    return weights

def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
            stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    s = [x for x in result.stdout.readlines() if b"Duration" in x][0].decode('UTF-8')
    s = s.replace(',', ' ')
    return s.split()[1]

def build_time_series(joints_file):
    with open(joints_file, 'r') as f:
        reader = csv.reader(f)
        bodies = {}
        prev_time = -1
        prev_id = -1
        skel = None
        for row in reader:
            body_id = row[CSVColumn['Skeleton_Id']]
            time = row[CSVColumn['Time']]
            frame = row[CSVColumn['Frame']]
            if time != prev_time or body_id != prev_id:
                skel = Skeleton(body_id, frame)
            skel.addJoint(Joint(row[CSVColumn['JointType']], Position(
                    row[CSVColumn['X_coord']], row[CSVColumn['Y_coord']]), 
                    ConfidenceState[row[CSVColumn['JointConfidence']]]))
            if body_id not in bodies:
                bodies[body_id] = BodyTimeSeries(body_id)
            bodies[body_id].addTimeInstance(time, skel)
            prev_id = body_id
            prev_time = time
        return list(bodies.values())

def build_reference_skeleton(pos_file):
    with open(pos_file, 'r') as f:
        reader = csv.reader(f)
        skel = Skeleton()
        for row in reader:
            skel.addJoint(Joint(row[CSVColumn['JointType']], Position(
                row[CSVColumn['X_coord']], row[CSVColumn['Y_coord']]), 
                ConfidenceState[row[CSVColumn['JointConfidence']]]))
        return skel

def frames_to_BodyTimeSeries(frames):
    _id = [val[CSVColumn['Skeleton_Id']] for val in frames[0].values()][0]
    body_series = BodyTimeSeries(_id)
    for frame in frames:
        time = [val[CSVColumn['Time']] for val in frame.values()][0]
        body_series.addTimeInstance(time, frame_to_skel(frame, _id))
    return body_series

def frame_to_skel(frame, _id):
    frame_num = [val[4] for val in frame.values()][0]
    skel = Skeleton(_id, frame_num)
    for key, val in frame.items():
        skel.addJoint(Joint(key, Position(
            val[CSVColumn['X_coord']], val[CSVColumn['Y_coord']]), 
            ConfidenceState[val[CSVColumn['JointConfidence']]]))
    return skel

def abs_error(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)

def get_time_obj(s):
    time = s.split('.')[0]
    return datetime.strptime(time, '%H:%M:%S')
