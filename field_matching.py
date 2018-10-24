# coding: utf-8
import sys
import cv2
import os.path
import matplotlib.pyplot as plt

PUYO_W = 72
PUYO_H = 67
PUYO_N_ROWS = 12
PUYO_N_COLS = 6

def load_image(filename):
    # imread はファイルが存在しなくてもエラーにならない！！
    if not os.path.exists(filename):
        raise RuntimeError("not found: {}".format(filename))

    img = cv2.imread(filename)

    # matplotlibでそのまま出力できるように色を変換しておく
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_pattern_images():
    color_names = ["red", "blue", "purple", "yellow"]
    result = {}
    for name in color_names:
        sys.stderr.write("load: {}\n".format(name))
        result[name] = load_image("patterns/" + name + "-0.png")
    return result


def load_screen_image():
    return load_image("frame-27s.png")


def crop_to_field_of_1p(screen_image):
    pixel_of_puyo = (PUYO_W, PUYO_H)
    field_size_by_puyo = (6, 12)

    (w, h) = map(lambda lhs, rhs: lhs * rhs, pixel_of_puyo, field_size_by_puyo)
    (x, y) = (296, 116)

    return screen_image[y:int(y+h), x:int(x+w)]


def extract_cell_at(field_image, row, col):
    if not (0 <= row < PUYO_N_ROWS) or not (0 <= col < PUYO_N_COLS):
        raise RuntimeError("Out of puyo field: row={}, col={}".format(row, col))

    y = (PUYO_N_ROWS - row - 1) * PUYO_H
    x = col * PUYO_W
    return field_image[int(y):int(y+PUYO_H), int(x):int(x+PUYO_W)]


def main():
    img = cv2.imread("sample.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


if __name__ == "__main__":
    main()
