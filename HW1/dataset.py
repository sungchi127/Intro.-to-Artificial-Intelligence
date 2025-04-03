import os
import cv2

def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset=[]
    pathc=dataPath+'/car'
    pathnoc=dataPath+'/non-car'
    for filename in os.listdir(pathc):
        img = cv2.imread(os.path.join(pathc,filename))
        img = cv2.resize(img, (36, 16), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            dataset.append((img,1))
    for filename in os.listdir(pathnoc):
        img = cv2.imread(os.path.join(pathnoc,filename))
        img = cv2.resize(img, (36, 16), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            dataset.append((img,0))
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset
