from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np

import time
import cv2
import sys, os
from datetime import datetime
import csv
import ctypes
import _ctypes
import pygame
import sys

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

joint_types =  ['JointType_SpineBase',
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
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body |
                                                       PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data
        self._bodies = None

        self._joints_with_time = []

        self._frameno = 0
        codec = cv2.VideoWriter_fourcc('X', '2', '6', '4')
        self._video_color = cv2.VideoWriter('color.avi', codec, 10, (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height))
        self._video_depth = cv2.VideoWriter('depth.avi', -1, 10,
                                      (self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height))

    def csv_writer(self, data):
        with open("output.csv", "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in data:
                writer.writerow(line)

    def store_joints(self, joints, jointPoints, person):
        for i in range(0,25):
            joint = joints[i]
            joint_coord = jointPoints[i]
            if(joint.TrackingState==2):
                #print(i," is tracked.")
                self._joints_with_time.append([person, str(datetime.now().time()), joint_types[i],'tracked', self._frameno])
            elif (joint.TrackingState==1):
                #print(i," is inferred.")
                self._joints_with_time.append([person, str(datetime.now().time()), joint_types[i],'inferred', self._frameno])

            if(len(self._joints_with_time)==100):
                self.csv_writer(self._joints_with_time)
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

        #self.csv_writer([[joint0,jointPoints[joint0].x,jointPoints[joint0].y],[joint1,jointPoints[joint1].x,jointPoints[joint1].y]])

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);

        # Right Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

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

    def surface_to_string(self, surface):
        return pygame.image.tostring(surface, 'RGB')

    def run(self):
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            # --- Game logic should go here

            # --- Getting frames and drawing
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame1 = frame.reshape((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4))
                frame2 = cv2.cvtColor(frame1,cv2.COLOR_RGBA2RGB)
                self._video_color.write(frame2)

            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                frame1 = frame.reshape((self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))
                print(frame1.dtype)
                self._video_depth.write(frame1)
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
                    self.store_joints(joints, joint_points, i)

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size)
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)
            self._frameno = self._frameno + 1

            print(self._clock.get_fps())

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        self._video_color.release()
        self._video_depth.release()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
'''
try:
    os.makedirs("Snaps")
except OSError:
    pass'''

game = BodyGameRuntime();
game.run();