from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from threading import Thread
from queue import Queue
import numpy as np

import time
import cv2
import sys, os
from datetime import datetime
import time
import csv
import ctypes
import _ctypes
import pygame
import sys
import colour

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                   pygame.color.THECOLORS["blue"],
                   pygame.color.THECOLORS["green"],
                   pygame.color.THECOLORS["orange"],
                   pygame.color.THECOLORS["purple"],
                   pygame.color.THECOLORS["yellow"],
                   pygame.color.THECOLORS["violet"]]

joint_types = ['JointType_SpineBase',
               'JointType_SpineMid',
               'JointType_Neck',
               'JointType_Head',
               'JointType_ShoulderLeft',
               'JointType_ElbowLeft',
               'JointType_WristLeft',
               'JointType_HandLeft',
               'JointType_ShoulderRight',
               'JointType_ElbowRight',
               'JointType_WristRight',
               'JointType_HandRight',
               'JointType_HipLeft',
               'JointType_KneeLeft',
               'JointType_AnkleLeft',
               'JointType_FootLeft',
               'JointType_HipRight',
               'JointType_KneeRight',
               'JointType_AnkleRight',
               'JointType_FootRight',
               'JointType_SpineShoulder',
               'JointType_HandTipLeft',
               'JointType_ThumbLeft',
               'JointType_HandTipRight',
               'JointType_ThumbRight'
               ]


class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        self.now = 0

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body |
            PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface(
            (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data
        self._bodies = None

        self._joints_with_time = None
        self.path = None
        # self._cnt = 0
        self._frameno = 0
        self._video_frameno = None
        self._timestamps = None
        self._video_color = None
        self._video_depth = None
        self.MOUSE_BUTTON_DOWN = 1
        self.MOUSE_BUTTON_UP = 0

        self.clicked = False
        self.prev_mouse_state = self.MOUSE_BUTTON_DOWN

        self.isRecording = False
        #multithreading queue
        self.Q = Queue()
        self.Q2 = Queue()
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t2 = Thread(target=self.update2, args=())
        self.t2.daemon = True
        self.last = time.time()

    def update(self):
        print('thread started')
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if not self.isRecording:
                return
            if(self.Q.qsize()>0):
                #print('eating color frame')
                # read the next frame from the file
                frame = self.Q.get()
                self._video_color.write(frame)

    def update2(self):
        print('thread started')
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if not self.isRecording:
                return

            if (self.Q2.qsize() > 0):
                #print('eating depth frame')
                # read the next frame from the file
                frame = self.Q2.get()
                self._video_depth.write(frame)


    def csv_writer(self, filename, data):
        with open(filename, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in data:
                writer.writerow(line)

    def store_joints(self, joints, jointPoints, person):
        for i in range(0, 25):
            joint = joints[i]
            joint_coord = jointPoints[i]

            if (joint.TrackingState == 2):
                # print(i," is tracked.")
                self._joints_with_time.append(
                    [person, self.now, joint_types[i], 'tracked', self._frameno, jointPoints[i].x, jointPoints[i].y])
            elif (joint.TrackingState == 1):
                # print(i," is inferred.")
                self._joints_with_time.append(
                    [person, self.now, joint_types[i], 'inferred', self._frameno, jointPoints[i].x, jointPoints[i].y])

            if (len(self._joints_with_time) == 100):
                self.csv_writer(self.path + '/joints.csv', self._joints_with_time)
                self._joints_with_time = []

    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked):
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        # self.csv_writer([[joint0,jointPoints[joint0].x,jointPoints[joint0].y],[joint1,jointPoints[joint1].x,jointPoints[joint1].y]])

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except:  # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder,
                            PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);

        # Right Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight,
                            PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight,
                            PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight,
                            PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight,
                            PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight,
                            PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft,
                            PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft,
                            PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight,
                            PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight,
                            PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def startRecording(self):
        self._joints_with_time = []
        self.path = str(datetime.now().date()) + ' at ' + str(datetime.now().hour) + '.' + str(
            datetime.now().minute) + '.' + str(datetime.now().second)
        os.mkdir(self.path)
        # self._cnt = 0
        self._frameno = 0
        self._video_frameno = [0, 0]
        self._timestamps = [[], []]
        codec_x264 = cv2.VideoWriter_fourcc('X', '2', '6', '4')
        self._video_color = cv2.VideoWriter(self.path + '/color.avi', codec_x264, 10,
                                            (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height))
        codec_none = cv2.VideoWriter_fourcc('D', 'I', 'B', ' ')
        self._video_depth = cv2.VideoWriter(self.path + '/depth.avi', codec_none, 10,
                                            (self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height))
        self.t.start()
        self.t2.start()
        #time.sleep(1)

    def stopRecording(self):
        self._video_color.release()
        self._video_depth.release()

    def recordingButton(self, start_msg, stop_msg, x, y, w, h, start_col, stop_col):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        # print(click)
        if x + w > mouse[0] > x and y + h > mouse[1] > y and click[0] and self.prev_mouse_state == self.MOUSE_BUTTON_UP:
            if self.isRecording:
                self.stopRecording()
            else:
                self.startRecording()

            self.isRecording = not self.isRecording
            self.clicked = not self.clicked
            # pygame.draw.rect(screen, ac,(x,y,w,h))
        if self.clicked:
            msg = stop_msg
            col = stop_col
        else:
            msg = start_msg
            col = start_col
        pygame.draw.rect(self._screen, col, (x, y, w, h))

        self.prev_mouse_state = click[0]
        smallText = pygame.font.SysFont("comicsansms", 20)
        textSurf, textRect = self.text_objects(msg, smallText)
        textRect.center = ((x + (w / 2)), (y + (h / 2)))
        self._screen.blit(textSurf, textRect)

    def text_objects(self, text, font):
        textSurface = font.render(text, True, colour.BLACK)
        return textSurface, textSurface.get_rect()

    def handle(self, frame, type):
        #if(type=='color'):
        #    if((not self.Q.full()) and self._video_frameno[0]%3==0):
                #print('frame rate:', self._clock.get_fps())
                #print('Q1', self.Q.qsize())
                #rate = 1/(time.time()-self.last)
                #self.last = time.time()
                #print(rate)
                #self.Q.put(frame)

        if(type=='depth'):
            if ((not self.Q2.full()) and self._video_frameno[1] % 3 == 0):
                # print('frame rate:', self._clock.get_fps())
                #print('Q2',self.Q2.qsize())
                rate = 1 / (time.time() - self.last)
                self.last = time.time()
                print(rate)
                self.Q2.put(frame)

    def run(self):
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    self._done = True  # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE:  # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                                           pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

            # --- Game logic should go here

            # --- Getting frames and drawing
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data
            if self.isRecording:
                if not (self._kinect.has_new_color_frame() and self._kinect.has_new_depth_frame() and (self._bodies is not None)):
                    continue

            self.now = str(datetime.now().time())
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)

                if self.isRecording:
                    #start = time.time()
                    frame1 = frame.reshape(
                        (self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4))
                    #end = time.time()
                    #print("Frame reshaping: ", end - start)
                    #start = time.time()
                    frame2 = cv2.cvtColor(frame1, cv2.COLOR_RGBA2RGB)
                    #end = time.time()
                    #print("Frame conversion: ", end-start)
                    self._video_frameno[0] += 1
                    self._timestamps[0].append([self._video_frameno[0], self.now])
                    if (len(self._timestamps[0]) == 10):
                        self.csv_writer(self.path + '/color_timestamps.csv', self._timestamps[0])
                        self._timestamps[0] = []
                    #start = time.time()
                    self.handle(frame2,'color')
                    #self._video_color.write(frame2)
                    #end = time.time()
                    #print("Frame writing: ", end - start)

            if self._kinect.has_new_depth_frame() and self.isRecording:
                frame = self._kinect.get_last_depth_frame()
                frame1 = frame.reshape((self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
                # cv2.imshow('frame1', frame1)
                # cv2.waitKey(5)
                self._video_frameno[1] += 1
                self._timestamps[1].append([self._video_frameno[1], self.now])
                if (len(self._timestamps[1]) == 10):
                    self.csv_writer(self.path + '/depth_timestamps.csv', self._timestamps[1])
                    self._timestamps[1] = []
                self.handle(frame1,'depth')
                #self._video_depth.write(frame1)
            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame():
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface
            if self._bodies is not None:
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked:
                        continue

                    joints = body.joints
                    # convert joint coordinates to color space
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])
                    if self.isRecording:
                        self.store_joints(joints, joint_points, i)

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size)
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0, 0))
            surface_to_draw = None
            self.recordingButton("Start", "Stop", 30, 30, 85, 45, colour.GREEN, colour.RED)
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            #self._clock.tick(60)
            self._frameno = self._frameno + 1



        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()

        pygame.quit()


__main__ = "Kinect v2 Body Game"
'''
try:
    os.makedirs("Snaps")
except OSError:
    pass
'''

game = BodyGameRuntime();
game.run();