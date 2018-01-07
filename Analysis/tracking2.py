import os
import cv2
import csv
import numpy as np
import drawing
import copy
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pickle
import shutil
import math

INF = 10**9
FPS = 7

weight = {}
weight['JointType_SpineBase'] = 1
weight['JointType_SpineMid'] = 1
weight['JointType_Neck' ] = 1
weight['JointType_Head'] = 1
weight['JointType_ShoulderLeft'] = 1
weight['JointType_ElbowLeft'] = 1
weight['JointType_WristLeft'] = 1
weight['JointType_HandLeft'] = 1
weight['JointType_ShoulderRight'] = 1
weight['JointType_ElbowRight'] = 1
weight['JointType_WristRight'] = 1
weight['JointType_HandRight'] = 1
weight['JointType_HipLeft'] = 1
weight['JointType_KneeLeft'] = 1
weight['JointType_AnkleLeft'] = 1
weight['JointType_FootLeft'] = 1
weight['JointType_HipRight'] = 1
weight['JointType_KneeRight'] = 1
weight['JointType_AnkleRight'] = 1
weight['JointType_FootRight'] = 1
weight['JointType_SpineShoulder'] = 1
weight['JointType_HandTipLeft'] = 1
weight['JointType_ThumbLeft'] = 1
weight['JointType_HandTipRight'] = 1
weight['JointType_ThumbRight'] = 1
weight['inferred'] = 1
weight['tracked'] = 1

total_weight = 25   #Summation of all weights

skeleton_graph = {
	'JointType_SpineBase' : ['JointType_SpineMid', 'JointType_HipRight', 'JointType_HipLeft'],
	'JointType_SpineMid' : ['JointType_SpineBase', 'JointType_SpineShoulder'],
	'JointType_Neck' : ['JointType_Head', 'JointType_SpineShoulder'],
	'JointType_Head' : ['JointType_Neck'],
	'JointType_ShoulderLeft' : ['JointType_SpineShoulder', 'JointType_ElbowLeft'],
	'JointType_ElbowLeft' : ['JointType_ShoulderLeft', 'JointType_WristLeft'],
	'JointType_WristLeft' : ['JointType_ElbowLeft', 'JointType_HandLeft', 'JointType_ThumbLeft'],
	'JointType_HandLeft' : ['JointType_WristLeft', 'JointType_HandTipLeft'],
	'JointType_ShoulderRight' : ['JointType_SpineShoulder', 'JointType_ElbowRight'],
	'JointType_ElbowRight' : ['JointType_ShoulderRight', 'JointType_WristRight'],
	'JointType_WristRight' : ['JointType_ElbowRight', 'JointType_HandRight', 'JointType_ThumbRight'],
	'JointType_HandRight' : ['JointType_WristRight', 'JointType_HandTipRight'],
	'JointType_HipLeft' : ['JointType_SpineBase', 'JointType_KneeLeft'],
	'JointType_KneeLeft' : ['JointType_HipLeft', 'JointType_AnkleLeft'],
	'JointType_AnkleLeft' : ['JointType_KneeLeft', 'JointType_FootLeft'],
	'JointType_FootLeft' : ['JointType_AnkleLeft'],
	'JointType_HipRight' : ['JointType_SpineBase', 'JointType_KneeRight'],
	'JointType_KneeRight' : ['JointType_HipRight', 'JointType_AnkleRight'],
	'JointType_AnkleRight' : ['JointType_KneeRight', 'JointType_FootRight'],
	'JointType_FootRight' : ['JointType_AnkleRight'],
	'JointType_SpineShoulder' : ['JointType_Neck', 'JointType_SpineMid', 'JointType_ShoulderRight', 'JointType_ShoulderLeft'],
	'JointType_HandTipLeft' : ['JointType_HandLeft'],
	'JointType_ThumbLeft' : ['JointType_WristLeft'],
	'JointType_HandTipRight' : ['JointType_HandRight'],
	'JointType_ThumbRight' : ['JointType_WristRight']
}
skeleton_edges = [['JointType_Head','JointType_Neck'], ['JointType_Neck','JointType_SpineShoulder'], ['JointType_SpineShoulder','JointType_SpineMid'], ['JointType_SpineMid','JointType_SpineBase'], ['JointType_SpineShoulder','JointType_ShoulderRight'], ['JointType_SpineShoulder','JointType_ShoulderLeft'], ['JointType_SpineBase','JointType_HipRight'], ['JointType_SpineBase','JointType_HipLeft'], ['JointType_ShoulderRight','JointType_ElbowRight'], ['JointType_ElbowRight','JointType_WristRight'], ['JointType_WristRight','JointType_HandRight'], ['JointType_HandRight','JointType_HandTipRight'], ['JointType_WristRight','JointType_ThumbRight'], ['JointType_ShoulderLeft','JointType_ElbowLeft'], ['JointType_ElbowLeft','JointType_WristLeft'], ['JointType_WristLeft','JointType_HandLeft'], ['JointType_HandLeft','JointType_HandTipLeft'], ['JointType_WristLeft','JointType_ThumbLeft'], ['JointType_HipRight','JointType_KneeRight'], ['JointType_KneeRight','JointType_AnkleRight'], ['JointType_AnkleRight','JointType_FootRight'], ['JointType_HipLeft','JointType_KneeLeft'], ['JointType_KneeLeft','JointType_AnkleLeft'], ['JointType_AnkleLeft','JointType_FootLeft']]

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

def exp_best(frames, half_life=0.5):
	window_size = int(math.ceil(2*FPS*half_life))
	alpha = 1 - 0.5**(1.0/(window_size//2))
	filtered = []
	for i, frame in enumerate(frames):
		fframe = copy.deepcopy(frame)
		lx = {}
		rx = {}
		ly = {}
		ry = {}
		for k in frame:
			#print(frame[k])
			for j in range(-window_size,0):
				if(i+j>=0 and i+j<len(frames)):
					if(k not in frames[i+j]):
						continue
					x = float(frames[i+j][k][5])
					y = float(frames[i+j][k][6])
					if(k not in lx):
						lx[k] = x
						ly[k] = y
					else:
						lx[k] = alpha*x + (1-alpha)*lx[k]
						ly[k] = alpha*y + (1-alpha)*ly[k]
			for j in range(window_size,0,-1):
				if(i+j>=0 and i+j<len(frames)):
					if(k not in frames[i+j]):
						continue
					x = float(frames[i+j][k][5])
					y = float(frames[i+j][k][6])
					if(k not in rx):
						rx[k] = x
						ry[k] = y
					else:
						rx[k] = alpha*x + (1-alpha)*rx[k]
						ry[k] = alpha*y + (1-alpha)*ry[k]
			if(k in lx and k in rx):
				fframe[k][5] = str(alpha*float(frame[k][5]) + (1-alpha)/2*lx[k] + (1-alpha)/2*rx[k])
				fframe[k][6] = str(alpha*float(frame[k][6]) + (1-alpha)/2*ly[k] + (1-alpha)/2*ry[k])
			elif(k in lx and k not in rx):
				fframe[k][5] = str(alpha*float(frame[k][5]) + (1-alpha)*lx[k])
				fframe[k][6] = str(alpha*float(frame[k][6]) + (1-alpha)*ly[k])
			elif(k not in lx and k in rx):
				fframe[k][5] = str(alpha*float(frame[k][5]) + (1-alpha)*rx[k])
				fframe[k][6] = str(alpha*float(frame[k][6]) + (1-alpha)*ry[k])

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


def median_filter(frames, window_size=2):
	filtered = []
	window_frame = int(math.ceil(window_size*FPS/2.0))
	for i, frame in enumerate(frames):
		bufx = {}
		bufy = {}
		fframe = copy.deepcopy(frame)
		for j in range(-window_frame,window_frame+1):
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

def write(frames, path, name='/test.avi', overColor=True):
	global FPS
	capSize = (1920, 1080) # this is the size of my source video
	codec_x264 = cv2.VideoWriter_fourcc('X', '2', '6', '4')
	out = cv2.VideoWriter(path+name,codec_x264,FPS,capSize,True)
	if(overColor):
		cap = cv2.VideoCapture(path+'/color.avi')
		counter = 0
		while(cap.isOpened()):
			ret, color_frame = cap.read()
			if not ret:
				print('Cannot open video file!')
				break
			else:
				#print(counter)
				if(counter==len(frames)):
					break
				frame = frames[counter]
				resized_frame = cv2.resize(color_frame, (1920, 1080))
				out.write(cv2.add(drawing.drawframe(frame),resized_frame))
				counter += 1
		cap.release()
	else:
		for frame in frames:
			out.write(drawing.drawframe(frame))
	out.release()


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

def traverse(frame, parent, current, delta):
	frame[current][5] = str(float(frame[current][5]) + delta[0])
	frame[current][6] = str(float(frame[current][6]) + delta[1])
	for edge in skeleton_graph[current]:
		if(edge!=parent):
			traverse(frame, current, edge, delta)

def scale_bone(frame1, frame2, p1, p2):
	s1 = np.array([float(frame1[p1][5]), float(frame1[p1][6])])
	s2 = np.array([float(frame1[p2][5]), float(frame1[p2][6])])
	t1 = np.array([float(frame2[p1][5]), float(frame2[p1][6])])
	t2 = np.array([float(frame2[p2][5]), float(frame2[p2][6])])
	new_s2 = s1 + (LA.norm(t2-t1)/LA.norm(s2-s1))*(s2-s1)
	del_s2 = new_s2-s2
	traverse(frame1, p1, p2, del_s2)

def scale_bones(frame1, frame2):
	for pair in skeleton_edges:
		scale_bone(frame1, frame2, pair[0], pair[1])

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

def align_trans_mframe(frames, m_frame):
	sum2 = [0,0]
	num = 0
	for k in m_frame:
		sum2[0] += float(m_frame[k][5])
		sum2[1] += float(m_frame[k][6])
		num += 1
	cen2 = [sum2[0]/num, sum2[1]/num]
	for i, frame in enumerate(frames):
		sum1 = [0,0]
		num = 0
		for k in frame:
			sum1[0] += float(frame[k][5])
			sum1[1] += float(frame[k][6])
			num += 1
		cen1 = [sum1[0]/num, sum1[1]/num]

		for k in frame:
			frame[k][5] = str(float(frame[k][5]) + cen2[0] - cen1[0])
			frame[k][6] = str(float(frame[k][6]) + cen2[1] - cen1[1])

def align_trans(frames1, frames2):
	#Align the centroid of frames
	for i, frame in enumerate(frames1):
		if(i==len(frames2)):
			break
		sum1 = [0,0]
		num = 0
		for k in frame:
			sum1[0] += float(frame[k][5])
			sum1[1] += float(frame[k][6])
			num += 1
		cen1 = [sum1[0]/num, sum1[1]/num]
		sum2 = [0,0]
		num = 0
		for k in frames2[i]:
			sum2[0] += float(frames2[i][k][5])
			sum2[1] += float(frames2[i][k][6])
			num += 1
		cen2 = [sum2[0]/num, sum2[1]/num]

		for k in frame:
			frame[k][5] = str(float(frame[k][5]) + cen2[0] - cen1[0])
			frame[k][6] = str(float(frame[k][6]) + cen2[1] - cen1[1])

def align_trans_2(frames1, frame2):
	#Align the centroid of frames
	for frame in frames1:
		sum1 = [0,0]
		num = 0
		for k in frame:
			sum1[0] += float(frame[k][5])
			sum1[1] += float(frame[k][6])
			num += 1
		cen1 = [sum1[0]/num, sum1[1]/num]
		sum2 = [0,0]
		num = 0
		for k in frame2:
			sum2[0] += float(frame2[k][5])
			sum2[1] += float(frame2[k][6])
			num += 1
		cen2 = [sum2[0]/num, sum2[1]/num]

		for k in frame:
			frame[k][5] = str(float(frame[k][5]) + cen2[0] - cen1[0])
			frame[k][6] = str(float(frame[k][6]) + cen2[1] - cen1[1])

def align_trans_weighted(frames1, frames2):
	#Align the centroid of frames
	for i, frame in enumerate(frames1):
		if(i==len(frames2)):
			break
		sum1 = [0,0]
		num = 0
		for k in frame:
			sum1[0] += weight[frame[k][3]]*weight[k]*float(frame[k][5])
			sum1[1] += weight[frame[k][3]]*weight[k]*float(frame[k][6])
			num += weight[k]*weight[frame[k][3]]
		cen1 = [sum1[0]/num, sum1[1]/num]
		sum2 = [0,0]
		num = 0
		for k in frames2[i]:
			sum2[0] += weight[frames2[i][k][3]]*weight[k]*float(frames2[i][k][5])
			sum2[1] += weight[frames2[i][k][3]]*weight[k]*float(frames2[i][k][6])
			num += weight[k]*weight[frames2[i][k][3]]
		cen2 = [sum2[0]/num, sum2[1]/num]

		for k in frame:
			frame[k][5] = str(float(frame[k][5]) + cen2[0] - cen1[0])
			frame[k][6] = str(float(frame[k][6]) + cen2[1] - cen1[1])

# align frame1 w.r.t. frame2
def align_bone_resize(frames1, frames2):
	#Align the centroid of frames
	for i, frame in enumerate(frames1):
		if(i==len(frames2)):
			break
		scale_bones(frame, frames2[i])
	align_trans(frames1,frames2)

def align_bone_resize_2(frames1, frame2):
	#Align the centroid of frames
	
	for frame in frames1:
		scale_bones(frame, frame2)
	align_trans_2(frames1,frame2)

def align_bone_resize_weighted(frames1, frames2):
	#Align the centroid of frames
	
	for i, frame in enumerate(frames1):
		if(i==len(frames2)):
			break
		scale_bones(frame, frames2[i])
	align_trans_weighted(frames1,frames2)

def error_vs_mframe(frames, m_frame):
	err_frames = []
	for frame in frames:
		sum = 0
		num_infer = 0
		entry = {}
		for k in frame:
			if(k in m_frame):
				val = weight[k]*LA.norm(np.array([float(frame[k][5]),float(frame[k][6])]) - np.array([float(m_frame[k][5]),float(m_frame[k][6])]))
				if(frame[k][3]=='inferred' or m_frame[k][3]=='inferred'):
					val = weight['inferred']*val
					num_infer += weight[k]*(1-weight['inferred'])
				entry[k] = val
				sum += val
		if((total_weight-num_infer)==0):
			print('No skeleton detected!')
			entry['Average'] = 'No skeleton detected'
			err_frames.append(entry)
		else:
			entry['Average'] = sum/(total_weight-num_infer)
			err_frames.append(entry)
	return err_frames



def error(frames1, frames2):
	err_frames = []
	for i, frame in enumerate(frames1):
		if(i==len(frames2)):
			break
		sum = 0
		for k in frame:
			if(k in frames2[i]):
				val = weight[k]*LA.norm(np.array([float(frame[k][5]),float(frame[k][6])]) - np.array([float(frames2[i][k][5]),float(frames2[i][k][6])]))
				if(frame[k][3]=='inferred' or frames2[i][k][3]=='inferred'):
					val = weight['inferred']*val
				sum += val
		err_frames.append(sum/total_weight)
	return err_frames

def delta(frames):
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
	return np.array(delta)

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

def do_all2(dir):
	for x in os.walk(dir):
		if(x[0]==dir):
			continue
		print(x[0])
		for i in range(0,6):
			csvpath = x[0]+'/joints'+str(i)+'.csv'
			statinfo = os.stat(csvpath)
			if(statinfo.st_size>0):
				'''frames = get_frames(csvpath)
				frames = median_filter(frames)
				frames = exp_best(frames)
				pickle.dump( frames, open( x[0]+"/best_skeleton"+str(i)+".p", "wb" ) )'''
				frames = pickle.load( open( x[0]+"/best_skeleton"+str(i)+".p", "rb" ) )
				try:
					write(frames, x[0], name='/'+str(i)+'.avi')
				except cv2.error:
					print(str(i)+' encountered OpenCV Error')
					write(frames, x[0], name='/'+str(i)+'.avi', overColor=False)
				'''
				min = np.argmin(delta(frames))
				print(min)
				align_trans_mframe(frames,frames[min])
				errors = error_vs_mframe(frames,frames[min])
				f = open('out/'+x[0]+'/min.txt', 'w')
				f.write('Minimum occurred at '+str(min))
				f.close()
				with open('out/'+x[0]+'/errors3.csv', 'w') as file:
					header = []
					for k in weight:
						if(k!='inferred'):
							header.append(k)
					header.append('Average')
					writer = csv.DictWriter(file, fieldnames = header)
					head = {}
					for h in header:
						head[h] = h
					writer.writerow(head)
					for row in errors:
						writer.writerow(row)'''

def copy_videos(dir):
	for x in os.walk(dir):
		if(x[0]==dir):
			continue
		print(x[0])
		for i in range(0,6):
			if(os.path.exists(x[0]+'/'+str(i)+'.avi')):
				shutil.copy(x[0]+'/'+str(i)+'.avi', 'out/'+x[0])

def write_csv(frames, file_name):
	with open(file_name, 'w') as csvfile:
		csv_writer = csv.writer(csvfile)
		for frame in frames:
			csv_writer.writerows(list(frame.values()))

def write_video(frames, out_vid, background_vid, start_frame, end_frame):
	fps = 7
	capSize = (1920, 1080) # this is the size of my source video
	codec_x264 = cv2.VideoWriter_fourcc('X', '2', '6', '4')
	out = cv2.VideoWriter(out_vid,codec_x264,fps,capSize,True)
	if background_vid is not None:
		cap = cv2.VideoCapture(background_vid)
		counter = 1
		while(cap.isOpened()):
			ret, color_frame = cap.read()
			if not ret:
				print('Cannot open video file!')
				break
			else:
				#print(counter)
				if(counter-start_frame==len(frames)):
					break
				if counter < start_frame:
					counter += 1
					continue
				frame = frames[counter-start_frame]
				resized_frame = cv2.resize(color_frame, (1920, 1080))
				out.write(cv2.add(drawing.drawframe(frame),resized_frame))
				counter += 1
		cap.release()
	else:
		for frame in frames:
			out.write(drawing.drawframe(frame))
	out.release()

def strip_frames(frames, start_frame, end_frame):
	result = []
	for frame in frames:
		for joint in frame.values():
			frame_num = int(joint[4])
			if frame_num >= start_frame and frame_num <= end_frame:
				result.append(frame)
			break
	return result

def process_video(in_csv, rgb_vid, depth_vid, background, out_vid, out_csv, 
	start_frame, end_frame, median_window_size, exp_half_life):
	global INF
	if end_frame == -1:
		end_frame = INF
	frames = get_frames(in_csv)
	frames = strip_frames(frames, start_frame, end_frame)
	frames = median_filter(frames, median_window_size)
	frames = exp_best(frames, exp_half_life)
	write_csv(frames, out_csv)
	background_vid = None
	if background == 'rgb':
		background_vid = rgb_vid
	elif background == 'depth':
		background_vid = depth_vid
	write_video(frames, out_vid, background_vid, start_frame, end_frame)
	#print(len(frames))
	#print(frames[0])
	#draw_movement(frames, path, param=str(i))
	'''
	try:
		write(frames, path, name='/skel_depth.avi')
	except cv2.error:
		print('encountered OpenCV Error')
		#os.remove(x[0]+'/'+str(i)+'.avi')
		#os.remove(x[0]+'/delta'+str(i)+'.png')
	'''

def assign_weights(w_file):
	with open(w_file, 'r') as f:
		csv_reader = csv.DictReader(f)
		for row in reader:
			weight[row['joint']] = row['weight']

def align_frames(in_csv, ref_csv, out_csv, weights):
	if weights is not None:
		assign_weights(weights)
	frames = get_frames(in_csv)
	ref = get_frames(ref_csv)
	align_bone_resize_weighted(frames, ref)
	write_csv(frames, out_csv)

if __name__ == '__main__':
	frames = get_frames('../../data/test_data_2/joints3.csv')
	print(frames[0])
	#do_all('Sanjeev sir')
	'''
	frames1 = exp_best(median_filter(get_frames('Rahul Sir and Roshan/new aasan 1/joints2.csv')))
	frames2 = exp_best(median_filter(get_frames('Rahul Sir and Roshan/new aasan 1/joints3.csv')))
	align_bone_resize(frames2, frames1)
	print('alignment complete!')
	write(frames2, 'Rahul Sir and Roshan/new aasan 1', name='/3_aligned_with_2.avi')'''

	#plt.plot(errors)
	#plt.ylabel('error')
	#plt.show()
	#plt.close()

	'''
	path = 'Rahul Sir and Roshan/new aasan 1'
	#split(path)
	for i in range(0,6):
		csvpath = path+'/joints'+str(i)+'.csv'
		statinfo = os.stat(csvpath)
		if(statinfo.st_size>0):
			frames = get_frames(csvpath)
			frames = median_filter(frames)
			frames = exp_best(frames)
			#draw_movement(frames, path, param=str(i))
			write(frames,path,name='/'+str(i)+'.avi')'''
