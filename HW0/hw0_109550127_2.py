import cv2
import numpy as np
cap = cv2.VideoCapture('video.mp4')
lastframe = None
while True:
    ret,frame = cap.read()
    frame_preprocess=frame
    if(lastframe is None):lastframe=frame_preprocess
    fgmask_diff=cv2.absdiff(frame_preprocess,lastframe)
    b = np.zeros(fgmask_diff.shape[:2], dtype = np.uint8)
    r = np.zeros(fgmask_diff.shape[:2], dtype = np.uint8)
    fgmask_diff = cv2.merge([b, fgmask_diff[:, :, 0], r])
    lastframe=frame_preprocess
    conbine = np.hstack((frame,fgmask_diff))
    #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow("Conbined",0)
    cv2.resizeWindow("Conbined",1920,540)
    cv2.imshow("Conbined",conbine)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyALLWindows()
