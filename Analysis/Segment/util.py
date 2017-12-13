import csv
from Skeleton import Skeleton
from BodyTimeSeries import BodyTimeSeries
from Joints import Joint, Position, ConfidenceState
from datetime import datetime
import subprocess

def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
            stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    s = [x for x in result.stdout.readlines() if b"Duration" in x][0].decode('UTF-8')
    s = s.replace(',', ' ')
    return s.split()[1]

def build_time_series(joints_file):
    with open(joints_file, 'r') as f:
        reader = csv.DictReader(f)
        bodies = {}
        prev_time = -1
        prev_id = -1
        skel = None
        for row in reader:
            body_id = row['Skeleton_Id']
            time = row['Time']
            if time != prev_time or body_id != prev_id:
                skel = Skeleton(body_id)
            skel.addJoint(Joint(row['JointType'], Position(
                    row['X_coord'], row['Y_coord']), 
                    ConfidenceState[row['JointConfidence']]))
            if body_id not in bodies:
                bodies[body_id] = BodyTimeSeries(body_id)
            bodies[body_id].addTimeInstance(time, skel)
            prev_id = body_id
            prev_time = time
        return list(bodies.values())

def build_reference_skeleton(pos_file):
    with open(pos_file, 'r') as f:
        reader = csv.DictReader(f)
        skel = Skeleton()
        for row in reader:
            skel.addJoint(Joint(row['JointType'], Position(
                row['X_coord'], row['Y_coord']), 
                ConfidenceState[row['JointConfidence']]))
        return skel
     
def abs_error(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)


def get_time_obj(s):
    time = s.split('.')[0]
    return datetime.strptime(time, '%H:%M:%S')
