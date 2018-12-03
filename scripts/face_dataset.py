import cv2
import os
cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, +1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is None or gray is None:
        continue

    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Save the captured image to the datasets folder
        cv2.imwrite("data/dataset/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' to exit data collection
    if k == 27:
        break
    elif count >= 500: # Take 500 face samples and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
