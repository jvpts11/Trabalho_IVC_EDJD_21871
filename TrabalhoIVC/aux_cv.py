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
    diff = cv2.absdiff(image,image2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(blur,20, 255, cv2.THRESH_BINARY)


    cv2.imshow("Original", image_copy)
    cv2.imshow("Difference",thresh)
    pass


def get_image_half_size(image):
    half_size = image.shape[1] / 2
    return half_size

