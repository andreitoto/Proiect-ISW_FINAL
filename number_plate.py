import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures

greyToggle = True #set this to True to enable grey debugging mode 


harcascade = "model\haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

cap.set(3, 440) 
cap.set(4, 780) 

min_area = 500
count = 0


while True:
      img_resp=urllib.request.urlopen(url) 
      imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8) 
      img = cv2.imdecode(imgnp,-1) 

      plate_cascade = cv2.CascadeClassifier(harcascade) 
     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4) 

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
           cv2.rectangle(img, (x,y), (x+w, y+h), (0,355,0), 2) 
           cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (355, 0, 355), 2) 

           img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi) 




    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)  
        cv2.rectangle(img, (0,200), (640,300), (0,355,0), cv2.FILLED) 
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 355), 2) 
        cv2.imshow("Results",img) 
        cv2.waitKey(500)
     

