import cv2
import yolov5 as yolo
import numpy as np

def yolo_init():
    model = yolo.load('../yolov5n.pt')
    print(model.names)
    model.conf = 0.30
    return model

model = yolo_init()

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
    cv_process(image,game)
    #cv_output(image)
    # game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(image,game):
    image_copy = image.copy() #cópia da imagem original para não haver erros
    half_size_image = get_image_half_size(image)


    cv2.imshow("Original", image_copy)
    pass

def convert_to_rgb(image_name):
    rgbImage = cv2.cvtColor(image_name, cv2.COLOR_BGR2RGB)
    return rgbImage

def check_is_growing(posX, otherX):
    if posX > otherX:
        return True
    else:
        return False

def get_image_half_size(image):
    half_size = image.shape[1] / 2
    return half_size

