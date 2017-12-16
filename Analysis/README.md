# Skeletal Joints Analysis from the Microsoft Kinect

tracking2.py - The main program for smoothing, alignment & error calculation
drawing.py - To draw skeletons
compare.py - To compare 2 videos by superimposing one on the other
skeletons.py - Old analysis code

## Prerequisites

The code runs with Python 3, so make sure you have latest version installed before testing. This program has several python library dependencies, most important of which are:

* OpenCV 3 with ffmpeg
* Matplotlib 
* NumPy

## Conversion to an Application

If you want to deploy the program as an executable application that can be directly downloaded by anyone and used, follow these instuction:

* Successfully run the program in an operating system for which you want to create the executable application
* Install PyInstaller (pip install pyinstaller)
* Run 'pyinstaller version2.py' to create the application
