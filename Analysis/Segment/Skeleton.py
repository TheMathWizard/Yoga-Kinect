class Skeleton:
    def __init__(self, _id = -1):
        self.id = _id
        self.joints = {}

    def addJoint(self, joint):
        self.joints[joint.type] = joint
