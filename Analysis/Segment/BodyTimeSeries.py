class BodyTimeSeries:
    def __init__(self, _id):
        self.time_mp = {}
        self.id = _id

    def addTimeInstance(self, time, skeleton):
        self.time_mp[time] = skeleton

    def skel_list(self):
        return list(self.time_mp.values())

    def to_frames(self):
    	frames = []
    	for (time, skel) in sorted(self.time_mp.items(), key=lambda x: x[0]):
    		frames.append(skel.to_frame(time))
    	return frames
