# Blink Reminder

Eye strain is a big problem for people who look at a computer screen all day. One of the biggest causes of eyestrain is being so focused on the screen that you forget to blink. A healthy person blinks 10-15 times a minute, but computer users can be observed to have up to a 80% reduction in blink rate. 

This program uses facial recognition to notify the user through a sound alert if it detects the user hasn't blinked in an extended period of time (8 seconds by default).

Face, glasses, left eye, and right eye detection uses [OpenCV with Haar Cascade Classifiers](https://github.com/opencv/opencv/tree/master/data/haarcascades), and the program uses [Jordan Van Eetveldt's pre-trained model](https://github.com/Guarouba/face_rec) to classify whether an eye appears to be opened or closed. 
## Example
![Example](https://github.com/FrankWan27/FrankWan27.github.io/blob/master/images/blink.png)
## Installation

If you're on windows, you can easily download and run the latest release here: https://github.com/FrankWan27/BlinkReminder/releases/tag/v1.0

### Clone or Download

- If you're on mac or linux, you can clone this repo to your local machine using 
```git clone https://github.com/FrankWan27/BlinkReminder.git```

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
- Press Q to quit

## Authors

* **Frank Wan** - [Github](https://github.com/FrankWan27)
