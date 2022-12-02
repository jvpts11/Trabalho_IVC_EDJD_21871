import cv2
import numpy as np


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
    ret2, image2 = cap.read()
    image = image[:, ::-1, :]
    image2 = image2[:,::-1,:]
    cv_process(image,image2, game)
    #cv_output(image)
    # game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(image,image2, game):
    image_copy = image.copy() #cópia da imagem original para não haver erros
    half_size_image = get_image_half_size(image)
    diff = cv2.absdiff(image,image2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur,20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 9000:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(image_copy,(x,y),(x + w, y + h),(0, 255, 0), 2 )
                check = check_is_growing(x,half_size_image)
                if check is True:
                    game.paddle.move(-10)
                if check is False:
                    game.paddle.move(10)

    cv2.imshow("Original", image_copy)
    cv2.imshow("Difference",thresh)
    pass


def check_is_growing(posX, otherX):
    if posX > otherX:
        return True
    else:
        return False

def get_image_half_size(image):
    half_size = image.shape[1] / 2
    return half_size

