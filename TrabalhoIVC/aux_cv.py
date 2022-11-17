import cv2


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

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv_process(hsv)
    cv_output(hsv)
    # game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(image):
    # main image processing code
    pass


def cv_output(image):
    cv2.imshow("Image", image)
    # rest of output rendering
    cv2.waitKey(1)

