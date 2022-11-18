import cv2
import numpy as np

color_lower_limit = np.array([36, 25, 25])
color_upper_limit = np.array([86, 255,255])

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
    cv_process(image, game)
    #cv_output(image)
    # game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(image, game):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,color_lower_limit,color_upper_limit )
    half_size = get_image_half_size(image)
    converted_half_size = int(half_size)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image_copy, (x,y), (x + w, y + h),(0, 255, 0), 2)
                if converted_half_size < x:
                    game.paddle.move(-10)
                if converted_half_size > x:
                    game.paddle.move(10)
                print(converted_half_size)
                print(x)
    cv2.imshow("Original", image_copy)
    cv2.imshow("Processed", mask)
    pass


def get_image_half_size(image):
    half_size = image.shape[1] / 2
    return half_size

