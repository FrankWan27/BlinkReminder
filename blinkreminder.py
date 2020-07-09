import numpy as np
import cv2
import time
from imutils.video import VideoStream
from PIL import Image
from keras.models import model_from_json

faces = []
lastBlink = time.perf_counter()
lastPing = 0

def loadModel():
    #Model structure and weights from https://github.com/Guarouba/face_rec
    modelJSON = open('data/model.json', 'r')
    modelStructure = modelJSON.read()
    modelJSON.close()
    model = model_from_json(modelStructure)
    model.load_weights('data/model.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def loadClassifiers():
    faceClassifier = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    glassesClassifier = cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')
    leftEyeClassifier = cv2.CascadeClassifier('data/haarcascade_lefteye_2splits.xml')
    rightEyeClassifier = cv2.CascadeClassifier('data/haarcascade_righteye_2splits.xml')
    return (faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier)

def predict(img, model):
    img = Image.fromarray(img, 'RGB').convert('L')
    img = np.array(img.resize((24, 24))).astype('float32')
    img /= 255
    img = img.reshape(1, 24, 24, 1)
    prediction = model.predict(img)
    return prediction

def detectEyes(webcam, scale, threshold, model, faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier):
    global faces
    global lastBlink
    global lastPing

    frame = webcam.read()
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceSize = ((int)(100 * scale), (int)(100 * scale))
    eyeSize = ((int)(15 * scale), (int)(15 * scale))

    # Detect faces
    facesTemp = faceClassifier.detectMultiScale(gray, 1.1, 2, minSize=faceSize)

    # if we lose face tracking, we keep a persistant frame
    if len(facesTemp) > 0:
        faces = facesTemp

    for (x,y,w,h) in faces:
        #draw face frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #used for drawing
        face = frame[y:y+h,x:x+w]
        #used for classification
        faceGray = gray[y:y+h,x:x+w]

        #check if detected eye in glasses
        glasses = glassesClassifier.detectMultiScale(faceGray, 1.1, 3, minSize=eyeSize)

        if len(glasses) > 0:
            #draw eye frame
            for (ex,ey,ew,eh) in glasses:
                cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (255, 255, 255), 2)
        else:
            leftFace = face[:, int(w/2):]
            leftGray = gray[:, int(w/2):]

            rightFace = face[:, :int(w/2)]
            rightGray = gray[:, :int(w/2)]

            #check if detected left or right eye
            left = leftEyeClassifier.detectMultiScale(leftFace, 1.1, 3, minSize=eyeSize)
            right = rightEyeClassifier.detectMultiScale(rightFace, 1.1, 3, minSize = eyeSize) 

            bothOpen = True
            blinkThresh = 0.2


            #we will assume you blinked if either eye is closed
            for (ex,ey,ew,eh) in left:
                color = (0, 255, 0)

                if predict(leftFace[ey:ey+eh,ex:ex+ew],model) < blinkThresh:
                    bothOpen = False
                    color = (0, 0, 255)
                cv2.rectangle(leftFace,(ex,ey),(ex+ew,ey+eh),color,2)

            for (ex,ey,ew,eh) in right:
                color = (0, 255, 0)

                if predict(rightFace[ey:ey+eh,ex:ex+ew],model) < blinkThresh:
                    bothOpen = False
                    color = (0, 0, 255)
                cv2.rectangle(rightFace,(ex,ey),(ex+ew,ey+eh),color,2)

            #if one eye is predicted to be closed, we assume we blinked
            if not bothOpen:
                lastPing = 0
                lastBlink = time.perf_counter()


    timeSinceBlink = time.perf_counter() - lastBlink

    cv2.putText(frame, "Ping if you haven't blinked for " + str(threshold) + " seconds. (Use W and S to change)", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, str(round(timeSinceBlink, 1)) + "s since last blink", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, (int)(255 - timeSinceBlink * 10), (int)(timeSinceBlink * 30)), 1)
    alert(timeSinceBlink, threshold)
    return frame

    
def alert(timeSinceBlink, threshold):
    global lastPing
    if timeSinceBlink >= threshold:
        timer = (int)(timeSinceBlink / threshold)
        if timer != lastPing:
            lastPing = timer
            print('\a', end ="", flush=True)

if __name__ == "__main__":

    #loads ML models and starts webcam
    (faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier) = loadClassifiers()
    model = loadModel()
    webcam = VideoStream(src=0).start()

    #User can adjust these parameters (video size and ping timer threshold)
    scale = 1
    threshold = 8.0

    while True:
        frame = detectEyes(webcam, scale, threshold, model, faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier)
        cv2.imshow("Blink Reminder", frame)

        
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q') or keypress == 27: #Quit on keypress Q or ESC
            break
        elif keypress == ord('=') and (scale < 2): #Video scale with + and - keys
            scale += 0.1
        elif keypress == ord('-') and (scale > 0.3):
            scale -= 0.1
        elif keypress == ord('w') and threshold < 10: #Increase/Decrease threshold with arrow keys
            threshold += 0.5
        elif keypress == ord('s') and threshold > 3:
            threshold -= 0.5


    cv2.destroyAllWindows()
    webcam.stop()
