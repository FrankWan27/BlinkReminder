from imutils.video import VideoStream
import cv2
from keras.models import model_from_json

faces = []

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

def detectEyes(webcam, scale, model, faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier):
    global faces

    frame = webcam.read()
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceSize = ((int)(100 * scale), (int)(100 * scale))

    # Detect faces
    facesTemp = faceClassifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=faceSize,
    )

    # if we lose face tracking, we keep a persistant frame
    if len(facesTemp) > 0:
        faces = facesTemp

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)



    return frame


if __name__ == "__main__":

    #loads ML models and starts webcam
    (faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier) = loadClassifiers()
    model = loadModel()
    webcam = VideoStream(src=0).start()
    scale = 1

    while True:
        frame = detectEyes(webcam, scale, model, faceClassifier, glassesClassifier, leftEyeClassifier, rightEyeClassifier)
        cv2.imshow("Blink Reminder", frame)

        #Quit on keypress Q or ESC
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q') or keypress == 27: 
            break
        elif keypress == ord('=') and (scale < 2):
            scale += 0.1
        elif keypress == ord('-') and (scale > 0.3):
            scale -= 0.1

    cv2.destroyAllWindows()
    webcam.stop()
