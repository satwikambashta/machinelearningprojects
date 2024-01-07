import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap=cv2.VideoCapture(0)
# cap = cv2.VideoCapture('./conor.mp4')

pTime=0
while True:
    if not cap.isOpened():
        print("Error: Could not open the video capture.")
        break
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            cv2.putText(img, "Movement detected!!", (20,40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

    
    #Display FPS
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):           
        break