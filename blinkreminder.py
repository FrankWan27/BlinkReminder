from imutils.video import VideoStream
import cv2

def init():

    return VideoStream(src=0).start()

if __name__ == "__main__":

    #starts webcam and loads ML models
    webcam = init()

    while True:
        frame = webcam.read()
        cv2.imshow("Blink Reminder", frame)

        #Quit on keypress Q or ESC
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27: 
            break
    cv2.destroyAllWindows()
    webcam.stop()
