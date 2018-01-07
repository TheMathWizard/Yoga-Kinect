from Joints import JointType

class Skeleton:
    def __init__(self, _id = -1, _frame = -1):
        self.id = _id
        self.frame = _frame
        self.joints = {}

    def addJoint(self, joint):
        self.joints[joint.type] = joint

    def to_frame(self, time):
    	mp = {}
    	for jtype in JointType:
    		if jtype in self.joints:
    			joint = self.joints[jtype]
    			mp[jtype.name] = [id, time, jtype.name, joint.state.name, self.frame, 
    				joint.position.x, joint.position.y]
    	return mp
