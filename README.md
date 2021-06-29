# Ai-Camera
#Import ComputerVision library
import cv2
import numpy as np

#Set the camera to detect the object every 0.5 second
thres = 0.5 # Threshold to detect object
#nms_threshold = 0.2

#State the no. of ID camera used for object detection
cap = cv2.VideoCapture(0)
#Setting camera parameter
cap.set(3,640)
cap.set(4,480)

#Property of the Element Interface
classNames= []
#We call the files that contain the list of objects that can be detected
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#Default setting for ComputerVision
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#Loop when the camera detect the object listed in file
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) !=0:
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0, 255, 0), thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img, str(round(confidence*100,2)), (box[0] + 150, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    #To provide a visual output when the camera detecting an object
    cv2.imshow("Output",img)
    cv2.waitKey(1)
