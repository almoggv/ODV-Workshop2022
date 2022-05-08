######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# We added a method of drawing boxes and labels using OpenCV.

#working on distance_testing branch

# Import packages
import os
import argparse
import sys
import time
import importlib.util
from ODV import *

def main():
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--log', help='Turn logger on',
                        default=0)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    LOG_LVL = args.log

    # Construct ODV Class   
    odv = ODV(MODEL_NAME, GRAPH_NAME, LABELMAP_NAME, min_conf_threshold, resW, resH, LOG_LVL)

    odv.run_detection()

if __name__ == '__main__':
   main()


#method for odv class

    

