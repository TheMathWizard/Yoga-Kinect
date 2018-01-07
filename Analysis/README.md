# Skeletal Joints Analysis from the Microsoft Kinect

* skeleton.py - program for generating videos and joint smoothing
* align.py - align one yoga exercise to another master/reference frame
* Segment/segment.py - segment the start and end of posture
* Segment/posture_data.py - print data about the posture

Further help can be seen using the "--help" flag of each application. 

## Prerequisites

The code runs with Python 3, so make sure you have latest version installed before testing. This program has several python library dependencies, most important of which are:

* OpenCV 3 with ffmpeg
* Matplotlib 
* NumPy
* Scipy

## Conversion to an Application

If you want to deploy the program as an executable application that can be directly downloaded by anyone and used, follow these instuction:

* Successfully run the program in an operating system for which you want to create the executable application.
* Install PyInstaller (pip install pyinstaller)
* Run 'pyinstaller version2.py' to generate the executable. Refer to http://www.pyinstaller.org/
