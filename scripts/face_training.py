import cv2
import numpy as np
from PIL import Image
import os

data_path = 'data/dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml");

def getImagesAndLabels(data_path):
    imagePaths = [os.data_path.join(data_path,f) for f in os.listdir(data_path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        if os.data_path.split(imagePath)[-1][0] == '.':
            continue
        PIL_img = Image.open(imagePath).convert('L') # convert to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.data_path.split(imagePath)[-1].split(".")[0])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(data_path)
recognizer.train(faces, np.array(ids))
# Save the model to trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() works on Mac, but not on Pi
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
