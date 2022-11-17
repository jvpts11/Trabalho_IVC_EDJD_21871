import cv2
import numpy as np

color_lower_limit = np.array([110, 100, 20])
color_upper_limit = np.array([130, 245, 255])

def cv_setup(game):
    cv_init(game)
    cv_update(game)

def cv_init(game):
    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)


def cv_update(game):
    cap = game.cap
    if not cap.isOpened():
        cap.open(-1)
    ret, image = cap.read()
    image = image[:, ::-1, :]
    cv_process(image)
    cv_output(image)
    # game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,color_lower_limit,color_upper_limit )
    image.shape[1]/2
    cv2.imshow("Processed", mask)
    pass


def cv_output(image):
    cv2.imshow("Original", image)
    # rest of output rendering
    cv2.waitKey(1)

