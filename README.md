# Blink Reminder
Eye strain is a big problem for people who look at a computer screen all day. One of the biggest causes of eyestrain is being so focused on the screen that you forget to blink. A healthy person blinks 10-15 times a minute, but computer users can be observed to have up to a 80% reduction in blink rate. 

This program attempts to notify the user through a sound alert if it detects the user hasn't blinked in an extended period of time (8 seconds by default).

Face, glasses, left eye, and right eye detection uses [OpenCV with Haar Cascade Classifiers](https://github.com/opencv/opencv/tree/master/data/haarcascades), and the program uses [Jordan Van Eetveldt's pre-trained model](https://github.com/Guarouba/face_rec) to classify whether an eye appears to be opened or closed. 

## Installation

### Clone or Download

- Clone this repo to your local machine using 
```git clone https://github.com/FrankWan27/BlinkReminder.git```
- Or just download and extract the zip file instead

### Dependencies

- [Python 3 ](https://www.python.org/downloads/)
- [Numpy](https://numpy.org/) - used for matrix manipulation in numpy.ndarray

  ```pip install numpy```
- [OpenCV](https://opencv.org/) - open source computer vision library
 
  ```pip install opencv-python```
- [imutils](https://pypi.org/project/imutils/) - used to enable videostream (webcam)
 
  ```pip install imutils```
- [Pillow](https://python-pillow.org/) - Python imaging library
 
  ```pip install Pillow```
 
### Usage
To start running the application, run in the parent directory
 
 ```python blinkreminder.py```
- Use W and S to increase or decrease the blink reminder timer. 
- Use + and - to increase the videostream size. (Impacts detection accuracy at the cost of performance)

## Authors

* **Frank Wan** - [Github](https://github.com/FrankWan27)
