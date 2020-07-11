import cv2
import time
import sys
import os
from imutils.video import VideoStream

# PyInstaller adds this attribute
if getattr(sys, 'frozen', False):
    # Running in a bundle
    currentPath = sys._MEIPASS
else:
    # Running in normal Python environment
    currentPath = os.path.dirname(__file__)

#globals
faces = []
lastBlink = time.perf_counter()
lastPing = 0

def loadClassifiers():
    faceClassifier = cv2.CascadeClassifier(os.path.join(currentPath, 'data/haarcascade_frontalface_alt.xml'))
    glassesClassifier = cv2.CascadeClassifier(os.path.join(currentPath, 'data/haarcascade_eye_tree_eyeglasses.xml'))
    leftEyeClassifier = cv2.CascadeClassifier(os.path.join(currentPath, 'data/haarcascade_lefteye_2splits.xml'))
    rightEyeClassifier = cv2.CascadeClassifier(os.path.join(currentPath, 'data/haarcascade_righteye_2splits.xml'))
    return (faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier)

def detectEyes(webcam, scale, threshold, faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier):
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
        glasses = glassesClassifier.detectMultiScale(faceGray, 1.1, 1, minSize=eyeSize)

        if len(glasses) > 0:
            #draw eye frame
            for (ex,ey,ew,eh) in glasses:
                cv2.rectangle(face, (ex,ey), (ex+ew,ey+eh), (255, 255, 255), 2)
        else:
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
    webcam = VideoStream(src=0).start()

    #User can adjust these parameters (video size and ping timer threshold)
    scale = 1
    threshold = 8.0

    while True:
        frame = detectEyes(webcam, scale, threshold, faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier)
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
