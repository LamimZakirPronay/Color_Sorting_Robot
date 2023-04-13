import cv2
import numpy as np
import cv2
import numpy as np
cap=cv2.VideoCapture(0)
lower_range=np.array([0,71,154])
upper_range=np.array([4,253,253])
lower_green=np.array([40,50,50])
upper_green=np.array([80,255,255])
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,480))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_range,upper_range)
    mask_green=cv2.inRange(hsv,lower_green,upper_green)

    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    _,mask2=cv2.threshold(mask_green,254,255,cv2.THRESH_BINARY)
    
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnts2,_=cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   
   
   
    
    flag_red=False
    for c in cnts:
        x=60
        if cv2.contourArea(c)>x:
            flag_red=True
            x,y,w,h=cv2.boundingRect(c)
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,("DETECT"),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            
    if flag_red:
    
        cv2.putText(frame, 'Red Detected', (10,100), font, 1, (255,0,0), 2, cv2.LINE_AA)
        
    flag_green=False
    for c in cnts2:
        x=60
        if cv2.contourArea(c)>x:
            flag_green=True
            x,y,w,h=cv2.boundingRect(c)
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,("DETECT"),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            
    if flag_green:
    
        cv2.putText(frame, 'Green Detected', (10,100), font, 1, (255,0,0), 2, cv2.LINE_AA)
        
            
    cv2.imshow("FRAME",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("mask2",mask2)
    frame[:,:,:]=(0,0,0)
    
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()