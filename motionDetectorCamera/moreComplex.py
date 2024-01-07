import cv2
import numpy as numpy
import time
cap = cv2.VideoCapture('./cam.mp4')
pTime=0

success, img1 = cap.read()
success, img2 = cap.read()

while True: 
    if not cap.isOpened():
        print("Error: Could not open the video capture.")
        break
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 30, 2, cv2.THRESH_BINARY)
    loose = cv2.dilate(thresh, None)
    # Find contours of moving objects
    cnts, _ = cv2.findContours(loose, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img1, cnts, -1, (255,255,0), 2)
    #rectangle
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < 200:
            continue
        cv2.rectangle(img1, (x,y), (x+w, y+h), (255,255,0), 2)
        cv2.putText(img1, "Movement Detected", (10,200), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img1, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    cv2.imshow("Image", img1)
    img1=img2
    success, img2 = cap.read()

    key = cv2.waitKey(10)
    if key == ord('q'):  
        break
cap.release()  
cv2.destroyAllWindows()