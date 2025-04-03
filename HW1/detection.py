import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped

def draw_lines(img,x1,y1,x2,y2):
    start_point = (x1, y1)
    end_point = (x2, y2)
    color = (0, 255, 0) # green
    thickness = 1 # 寬度
    cv2.line(img, start_point, end_point, color, thickness) 

    return img

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    gif=cv2.VideoCapture('data/detect/video.gif')
    f1=open('Adaboost_pred.txt','w')
    while True:
      #ret,frame = gif.read()
      ret,frame = gif.read()
      if ret==False:
        break
      f=open(dataPath,'r')
      for line in f.readlines():
        s=line.split(' ')
        if int(s[0])==76:
          continue
        parking_space_image=crop(int(s[0]),int(s[1]),int(s[2]),int(s[3]),int(s[4]),int(s[5]),int(s[6]),int(s[7]),frame)
        reframe=cv2.resize(parking_space_image,(36,16),interpolation=cv2.INTER_AREA)
        reframe = cv2.cvtColor(reframe, cv2.COLOR_BGR2GRAY)
        if clf.classify(reframe)==1:
          frame=draw_lines(frame,int(s[0]),int(s[1]),int(s[2]),int(s[3]))
          frame=draw_lines(frame,int(s[0]),int(s[1]),int(s[4]),int(s[5]))
          frame=draw_lines(frame,int(s[4]),int(s[5]),int(s[6]),int(s[7]))
          frame=draw_lines(frame,int(s[2]),int(s[3]),int(s[6]),int(s[7]))
          f1.write('1 ')
        else:
          f1.write('0 ')
      f1.write('\n')
      cv2.namedWindow('img', cv2.WINDOW_NORMAL)
      cv2.resizeWindow("img",1280 , 800)
      cv2.imshow("img",frame)
      cv2.waitKey(1)
    cv2.destroyALLWindows()
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
