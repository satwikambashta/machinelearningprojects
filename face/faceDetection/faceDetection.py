import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime=0

mpFaceDet = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDet = mpFaceDet.FaceDetection(0.8)


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDet.process(imgRGB)
    # print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            #OR
            print(detection.location_data.relative_bounding_box)
            boundingboxC=detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            boundingbox = int(boundingboxC.xmin*iw), int(boundingboxC.ymin*ih),int(boundingboxC.width*iw), int(boundingboxC.height *ih)
            cv2.rectangle(img, boundingbox, (255,0,255),3)
            cv2.putText(img,f'confidence: {int(detection.score[0]*100)}%', (boundingbox[0], boundingbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

            


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS: {int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        
    cv2.imshow("Image", img)
    key = cv2.waitKey(10)
    if key == ord('q'):           
        break