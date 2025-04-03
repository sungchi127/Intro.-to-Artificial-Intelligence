import cv2
img = cv2.imread('image.png')
f=open('bounding_box.txt','r')
for line in f.readlines():
    s=line.split(' ')
    cv2.rectangle(img,(int(s[0]),int(s[1])),(int(s[2]),int(s[3])),(0,0,255),2)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyALLWindow()