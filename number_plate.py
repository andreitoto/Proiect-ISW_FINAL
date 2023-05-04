import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures

greyToggle = False #set this to True to enable grey debugging mode 


harcascade = "model\haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

cap.set(3, 640) #width 
cap.set(4, 480)  #height 

min_area = 500
count = 0
url=’http://192.168.0.198/cam-hi.jpg’ 


while True:
      img_resp=urllib.request.urlopen(url)  #fetch image from ur
      imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8) #convert image to bytearray 
      img = cv2.imdecode(imgnp,-1)  #convert image to bytearray 

      plate_cascade = cv2.CascadeClassifier(harcascade) #initialize cascade classifier
     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #process image in grayscale 

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)  #detect plates

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
           cv2.rectangle(img, (x,y), (x+w, y+h), (0,355,0), 2) #draw box 
           cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (355, 0, 355), 2) #draw box label 

           img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi) #display image inside box
            
            if greyToggle is True: 
cv2.imshow("Result", img_gray)
else: 
cv2.imshow("Result", img) 




    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)  #write screenshot to disk 
        cv2.rectangle(img, (0,200), (640,300), (0,355,0), cv2.FILLED) #draw box
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 355), 2) 
        cv2.imshow("Results",img) 
        cv2.waitKey(500)
     

