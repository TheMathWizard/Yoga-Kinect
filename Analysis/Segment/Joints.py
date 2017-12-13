from enum import Enum

class JointType(Enum):
    JointType_Head = 1
    JointType_Neck = 2
    JointType_SpineShoulder = 3
    JointType_SpineMid = 4
    JointType_SpineBase = 5
    JointType_FootLeft = 6
    JointType_ShoulderRight = 7
    JointType_ShoulderLeft = 8
    JointType_HipRight = 9
    JointType_HipLeft = 10
    JointType_ElbowRight = 11
    JointType_WristRight = 12
    JointType_HandRight = 13
    JointType_HandTipRight = 14
    JointType_ThumbRight = 15
    JointType_ElbowLeft = 16
    JointType_WristLeft = 17
    JointType_HandLeft = 18
    JointType_HandTipLeft = 19
    JointType_KneeRight = 20
    JointType_AnkleRight = 21
    JointType_FootRight = 22
    JointType_KneeLeft = 23
    JointType_AnkleLeft = 24
    JointType_ThumbLeft = 25

class ConfidenceState(Enum):
    tracked = 1
    inferred = 0

class Position:
    def __init__(self, x = -1, y = -1, z = -1):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class Joint:
    def __init__(self, _type = None, pos = None, state = ConfidenceState.inferred):
        self.type = JointType[_type]
        self.position = pos
        self.state = state

