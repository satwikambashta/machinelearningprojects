import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# 0. nose
# 1. left_eye_inner
# 2. lefteye
# 3. left_eye_outer
# 4. right_eye_inner
# 5. right eye
# 6. right eye_outer
# 7. left ear
# 8. right_ear
# 9. mouth_left
# 10. mouth_right
# 11. left shoulder
# 12. right shoulder
# 13. left elbow
# 14. right_elbow
# 15. left wrist
# 16. right wrist
# 17. left_pinky
# 18. right pinky
# 19. left_index
# 20. right_index
# 21. left_thumb
# 22. right thumb
# 23. left hip
# 24. right hip
# 25. left knee
# 26. right knee
# 27. left_ankle
# 28. right ankle
# 29. left heel
# 30. right heel
# 31. left_foot_index
# 32. right foot_index

# cap=cv2.VideoCapture(0)
cap = cv2.VideoCapture('./conor.mp4')
pTime=0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape #ratio to pixel
            cx, cy =int(landmark.x*w), int(landmark.y*h)
            cv2.circle(img, (cx,cy), 4, (255,255,255), cv2.FILLED)
            print(id, 'x:', cx,'y:', cy, 'visibility:',landmark.visibility)

    

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(10)
    if key == ord('q'):           
        break