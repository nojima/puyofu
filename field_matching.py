# coding: utf-8
import os.path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

PUYO_W = 72
PUYO_H = 67
PUYO_N_ROWS = 12
PUYO_N_COLS = 6

CROSS_MARK_DETECT_THRESHOLD = 1300
TSUMO_DETECT_THRESHOLD = 2500

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


def crop_to_next_puyo_of_1p(screen_image):
    x = 799
    y = 144
    return screen_image[y:y+2*PUYO_H, x:x+PUYO_W]


def crop_to_double_next_puyo_of_1p(screen_image):
    x = 864
    y = 279
    w = 51
    h = 51
    img = screen_image[y:y+2*h, x:x+w]

    # サイズが普通のぷよより小さいので、無理やり拡大して揃える
    return cv2.resize(img, (PUYO_W, PUYO_H * 2))


def crop_to_score_cross_mark_of_1p(screen_image):
    x = int(344 / 1280.0 * 1920.0)
    y = int(629 / 720.0 * 1080.0)
    w = int(36 / 1280.0 * 1920.0)
    h = int(43 / 720.0 * 1080.0)
    return screen_image[y:y+h, x:x+w]


def extract_cell_at(field_image, row, col):
    if not (0 <= row < PUYO_N_ROWS) or not (0 <= col < PUYO_N_COLS):
        raise RuntimeError("Out of puyo field: row={}, col={}".format(row, col))

    y = (PUYO_N_ROWS - row - 1) * PUYO_H
    x = col * PUYO_W
    return field_image[int(y):int(y+PUYO_H), int(x):int(x+PUYO_W)]


def diff_image(image_a, image_b, mask=None):
    a = image_a.astype(float) / 255.0
    b = image_b.astype(float) / 255.0
    if mask is None:
        return np.abs(a - b)
    else:
        return np.abs(a * mask - b * mask)


def detect_puyo(puyo_image, mask_image, patterns):
    best_match = None
    best_match_diff = 100000000
    mask = mask_image.astype(float) / 255.0
    for name, pattern in patterns.iteritems():
        diff = diff_image(puyo_image, pattern, mask).sum()
        if diff < best_match_diff:
            best_match = name
            best_match_diff = diff
    return best_match


def detect_all_puyo(field_image, mask_image, patterns):
    mask = mask_image.astype(float) / 255.0
    result = []
    for row in xrange(0, PUYO_N_ROWS):
        row_result = []
        for col in xrange(0, PUYO_N_COLS):
            cell_image = extract_cell_at(field_image, row, col)
            row_result.append(detect_puyo(cell_image, mask, patterns))
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


def detect_next_puyo_of_1p(screen_image, mask_image, patterns):
    next_image = crop_to_next_puyo_of_1p(screen_image)
    p1 = detect_puyo(next_image[:PUYO_H, :], mask_image, patterns)
    p2 = detect_puyo(next_image[PUYO_H:, :], mask_image, patterns)
    return [p1, p2]


def detect_double_next_puyo_of_1p(screen_image, mask_image, patterns):
    next_image = crop_to_double_next_puyo_of_1p(screen_image)
    p1 = detect_puyo(next_image[:PUYO_H, :], mask_image, patterns)
    p2 = detect_puyo(next_image[PUYO_H:, :], mask_image, patterns)
    return [p1, p2]


def is_cross_mark_exists_on_1p(screen_image, cross_mark_pattern):
    actual = crop_to_score_cross_mark_of_1p(screen_image)
    diff = diff_image(actual, cross_mark_pattern)
    return diff.sum() < CROSS_MARK_DETECT_THRESHOLD


# ツモる瞬間のフレームを検出する。フレームのインデクスの配列を返す。
def detect_tsumo_frames(frames):
    slide_size = 3

    frame_diffs1 = []
    frame_diffs2 = []
    for i in xrange(slide_size, len(frames)):
        curr = frames[i]
        curr_next = crop_to_next_puyo_of_1p(curr)
        curr_dn = crop_to_double_next_puyo_of_1p(curr)

        prev = frames[i-slide_size]
        prev_next = crop_to_next_puyo_of_1p(prev)
        prev_dn = crop_to_double_next_puyo_of_1p(prev)

        frame_diffs1.append(diff_image(curr_next, prev_next).sum())
        frame_diffs2.append(diff_image(curr_dn, prev_dn).sum())

    return np.where(
        np.logical_and(
            np.diff(frame_diffs1) > TSUMO_DETECT_THRESHOLD,
            np.diff(frame_diffs2) > TSUMO_DETECT_THRESHOLD
        )
    )[0] + np.array(slide_size)


def make_field_data_list(frames, tsumo_frame_indices, mask_image, patterns):
    field_data_list = []
    for i in tsumo_frame_indices:
        frame = frames[i]
        data = detect_all_puyo(crop_to_field_of_1p(frame), mask_image, patterns)
        field_data_list.append(data)
    return field_data_list


def detect_move(data_prev, data_curr):
    diffs = []
    for row in xrange(len(data_curr)):
        for col in xrange(len(data_curr[row])):
            if data_curr[row][col] != data_prev[row][col]:
                diffs.append((row, col, data_curr[row][col]))
    if len(diffs) > 2:
        raise RuntimeError("Failed to detect move: {}".format(diffs))
    return diffs


def print_moves(field_data_list):
    for i in xrange(1, len(field_data_list)):
        try:
            print i,
            print detect_move(field_data_list[i-1], field_data_list[i])
        except RuntimeError as e:
            print e


def main():
    img = cv2.imread("sample.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


if __name__ == "__main__":
    main()
