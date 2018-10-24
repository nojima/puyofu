# coding: utf-8
import sys
import cv2
import numpy as np
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
    color_names = ["red", "blue", "purple", "yellow", "empty"]
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


def diff_image(image_a, image_b):
    return np.abs(image_a.astype(float) / 256.0 - image_b.astype(float) / 256.0)


def detect_puyo(puyo_image, patterns):
    best_match = None
    best_match_diff = 100000000
    for name, pattern in patterns.iteritems():
        diff = diff_image(puyo_image, pattern).sum()
        if diff < best_match_diff:
            best_match = name
            best_match_diff = diff
    return best_match


def detect_all_puyo(field_image, patterns):
    result = []
    for row in xrange(0, PUYO_N_ROWS):
        row_result = []
        for col in xrange(0, PUYO_N_COLS):
            cell_image = extract_cell_at(field_image, row, col)
            row_result.append(detect_puyo(cell_image, patterns))
        result.append(row_result)
    return result


def construct_field_image(field_data, patterns):
    result = np.empty(shape=(PUYO_N_ROWS * PUYO_H, PUYO_N_COLS * PUYO_W, 3), dtype=np.uint8)
    for row, row_data in enumerate(field_data):
        for col, data in enumerate(row_data):
            y = (PUYO_N_ROWS - row - 1) * PUYO_H
            x = col * PUYO_W
            result[y:y+PUYO_H, x:x+PUYO_W, :] = patterns[data]
    return result


def main():
    img = cv2.imread("sample.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


if __name__ == "__main__":
    main()
