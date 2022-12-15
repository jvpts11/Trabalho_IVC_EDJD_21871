import cv2
import yolov5 as yolo
import numpy as np


x_variance_range = 20

def yolo_init():
    model = yolo.load('../yolov5n.pt')
    print(model.names)
    model.conf = 0.23
    return model


def haar_cascades_init():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return cascade


cascade = haar_cascades_init()
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
    result_image, x_pos = yolo_approach(image_copy)
    #result_image, x_pos = viola_jones(image_copy)
    check = check_is_growing(x_pos,half_size_image)
    if x_pos is not 0:
        if check is True:
            game.paddle.move(-10)
        if check is False:
            game.paddle.move(10)
    else:
        pass

    cv2.imshow("Original", result_image)
    pass


def yolo_approach(image):
    image_copy = image.copy()
    image_in_rgb = convert_to_rgb(image)
    result = model(image_in_rgb)
    x_pos = 0
    try:
        for pred in enumerate(result.pred):
            im = pred[0]
            im_boxes = pred[1]
            for *box, conf, cls in im_boxes:
                box_class = int(cls)
                conf = float(conf)
                x_pos = float(box[0])
                y_pos = float(box[1])
                w = float(box[2]) - x_pos
                h = float(box[3]) - y_pos
                pt1 = np.array(np.round((float(box[0]), float(box[1]))), dtype=int)
                pt2 = np.array(np.round((float(box[2]), float(box[3]))), dtype=int)
                box_color = (255, 0, 0)
                cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=box_color, thickness=1)
                text_format = "{}:{:.2f}".format(result.names[box_class], conf)
                cv2.putText(img=image, text=text_format,
                            org=np.array(np.round((float(box[0]), float(box[1] - 1))), dtype=int),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=box_color, thickness=1)
                return image, x_pos
    except:
        return image_copy, x_pos


def viola_jones(image):
    gray_image = convert_to_gray(image)
    eyes = cascade.detectMultiScale(gray_image,1.3,5)
    x = 0
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return image, x


def convert_to_gray(image_name):
    grayImage = cv2.cvtColor(image_name,cv2.COLOR_BGR2GRAY)
    return grayImage


def convert_to_rgb(image_name):
    rgbImage = cv2.cvtColor(image_name, cv2.COLOR_BGR2RGB)
    return rgbImage


def check_is_growing(posX, otherX):
    if posX > otherX:
        return True
    else:
        return False


def check_x_variance(x_value):
    if x_value > x_variance_range:
        return True


def get_image_half_size(image):
    half_size = image.shape[1] / 2
    return half_size

