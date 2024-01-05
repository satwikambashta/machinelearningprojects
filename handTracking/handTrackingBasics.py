import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

rtHands = mp.solutions.hands
hands = rtHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime=cTime=0

# Hand Landmarks
# 0. WRIST
# 1. THUMB_CMC
# 2. THUMB_MCP
# 3. THUMB_IP
# 4. THUMB_TIP
# 5. INDEX_FINGER_MCP
# 6. INDEX_FINGER_PIP
# 7. INDEX_FINGER_DIP
# 8. INDEX_FINGER_TIP
# 9. MIDDLE_FINGER_MCP
# 10. MIDDLE_FINGER_PIP
# 11. MIDDLE_FINGER_DIP
# 12. MIDDLE_FINGER_TIP
# 13. RING_FINGER_MCP
# 14. RING_FINGER_PIP
# 15. RING_FINGER_DIP
# 16. RING_FINGER_TIP
# 17. PINKY_MCP
# 18. PINKY_PIP
# 19. PINKY_DIP
# 20. PINKY_TIP

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #converted to rgb since the function can take only rgb values
    results = hands.process(imgRGB)
#     print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handy in results.multi_hand_landmarks:
            #get id number and landmark location
            #check index number
            for id, landmarks in enumerate(handy.landmark):
                h,w,c=img.shape #gives image height and width
                #convert ratio to pixels
                cx, cy=int(landmarks.x*w), int(landmarks.y*h)
                #print location of each landmark (total 21)
                print(id, cx, cy)
                if id==0:
                    cv2.circle(img, (cx,cy), 25, (255,255,255), cv2.FILLED)



            mpDraw.draw_landmarks(img, handy, rtHands.HAND_CONNECTIONS)
            
    #FPS
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    #display on screen
    #frames, position, font, scale, color, thickness
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):           
        break