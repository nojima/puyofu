# coding: utf-8
import argparse
import collections
import os.path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PUYO_W = 72
PUYO_H = 67
PUYO_N_ROWS = 12
PUYO_N_COLS = 6

CROSS_MARK_DETECT_THRESHOLD = 1300
GAMEOVER_BACKGROUND_THRESHOLD = 10000
TSUMO_DETECT_THRESHOLD = 2500

def load_image(filename):
    logger.info("Load image: {}".format(filename))

    # imread はファイルが存在しなくてもエラーにならない！！
    if not os.path.exists(filename):
        raise RuntimeError("not found: {}".format(filename))

    img = cv2.imread(filename)

    # matplotlibでそのまま出力できるように色を変換しておく
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_pattern_images():
    color_names = ["red", "blue", "purple", "yellow", "ojama", "empty"]
    result = {}
    for name in color_names:
        result[name] = load_image("{}/patterns/{}-0.png".format(BASE_DIR, name))
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


def crop_to_top_area_of_1p(screen_image):
    x = 279
    y = 224
    w = 443
    h = 75
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
    above_top = [False] * PUYO_N_COLS
    for row in xrange(0, PUYO_N_ROWS):
        row_result = []
        for col in xrange(0, PUYO_N_COLS):
            if above_top[col]:
                # empty なマスよりも上にはぷよは存在しないはず
                row_result.append("empty")
            else:
                cell_image = extract_cell_at(field_image, row, col)
                puyo = detect_puyo(cell_image, mask, patterns)
                if puyo == "empty":
                    above_top[col] = True
                row_result.append(puyo)
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


Event = collections.namedtuple("Event", ["time", "kind", "field", "move", "chain"])


def make_event_list(
        frames,
        tsumo_frame_indices,
        chain_start_frame_indices,
        gameover_frame_index,
        mask_image,
        patterns):
    event_points = []
    for i in tsumo_frame_indices:
        event_points.append((i, "tsumo"))
    for i in chain_start_frame_indices:
        event_points.append((i, "chain"))
    event_points.append((gameover_frame_index, "gameover"))
    event_points = sorted(event_points)

    logger.debug("Event points: {}".format(event_points))

    events = []
    state = "NORMAL"
    prev_field = None
    chain = 1

    for (i, which) in event_points:
        frame = frames[i]
        field = detect_all_puyo(crop_to_field_of_1p(frame), mask_image, patterns)

        if state == "NORMAL":
            if which == "chain":
                assert prev_field is not None
                chain = 1
                move = detect_move(prev_field, field)
                event = Event(time=i, kind="ChainStart", field=field, move=move, chain=chain)
                state = "IN_CHAIN"
            elif which == "tsumo":
                if prev_field is None:
                    # 最初のツモを引いた瞬間なので、イベントとしては記録しない
                    event = None
                else:
                    move = detect_move(prev_field, field)
                    event = Event(time=i, kind="Stack", field=field, move=move, chain=0)
            elif which == "gameover":
                event = Event(time=i, kind="GameOver", field=None, move=None, chain=0)
                state = "GAMEOVER"
            else:
                raise RuntimeError("BUG: {}".format(which))
        elif state == "IN_CHAIN":
            if which == "chain":
                chain += 1
                event = Event(time=i, kind="ChainProgress", field=field, move=None, chain=chain)
            elif which == "tsumo":
                event = Event(time=i, kind="ChainEnd", field=field, move=None, chain=0)
                state = "NORMAL"
            elif which == "gameover":
                event = Event(time=i, kind="GameOver", field=None, move=None, chain=0)
                state = "GAMEOVER"
            else:
                raise RuntimeError("BUG: {}".format(which))
        elif state == "GAMEOVER":
            event = None
        else:
            raise RuntimeError("BUG: {}".format(state))

        if event is not None:
            logger.debug("Event: time={}, kind={}, move={}".format(event.time, event.kind, event.move))
            events.append(event)
        prev_field = field

    return events


def detect_move(data_prev, data_curr):
    diffs = []
    for row in xrange(len(data_curr)):
        for col in xrange(len(data_curr[row])):
            if data_curr[row][col] != data_prev[row][col]:
                # おじゃまぷよは move に含めない
                if data_curr[row][col] == "ojama":
                    continue
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


def is_vanishing_start_frame_of_1p(prev_frame, curr_frame, cross_mark_pattern):
    return (
        not is_cross_mark_exists_on_1p(prev_frame, cross_mark_pattern) and
        is_cross_mark_exists_on_1p(curr_frame, cross_mark_pattern)
    )


def detect_chain_start_frames(frames, cross_mark_pattern, tsumo_frame_indices):
    tsumo_frame_indices = set(tsumo_frame_indices)

    chain_start_frame_indices = []
    for i in xrange(1, len(frames)):
        if is_vanishing_start_frame_of_1p(frames[i-1], frames[i], cross_mark_pattern):
            chain_start_frame_indices.append(i)

    return chain_start_frame_indices


def detect_gameover_frame(frames, background_pattern):
    for i in xrange(1, len(frames)):
        img = crop_to_top_area_of_1p(frames[i])
        if diff_image(img, background_pattern).sum() < GAMEOVER_BACKGROUND_THRESHOLD:
            return i
    return None


def pretty_print_events(events, patterns):
    for event in events:
        if event.field:
            img = construct_field_image(event.field, patterns) / 2
            if event.move:
                for row, col, puyo in event.move:
                    y = (PUYO_N_ROWS - row - 1) * PUYO_H
                    x = col * PUYO_W
                    img[y:y+PUYO_H, x:x+PUYO_W, :] = patterns[puyo]
            fig, ax = plt.subplots()
            ax.set_title("time={}, kind={}".format(event.time, event.kind))
            ax.imshow(img)


def generate_puyofu(events):
    def fugou_one(move):
        row, col, puyo = move
        kansuuji = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
        kanji_puyo = {
            "red": "赤",
            "blue": "青",
            "purple": "紫",
            "yellow": "黄",
        }
        return "▲{}{}{}".format(col + 1, kansuuji[row], kanji_puyo[puyo])

    def fugou(move):
        return "  ".join(fugou_one(m) for m in move)

    for event in events:
        if event.kind == "Stack":
            print "{: >5}:  {}".format(event.time, fugou(event.move))
        elif event.kind == "ChainStart":
            print "{: >5}:  {} 発火".format(event.time, fugou(event.move))
        elif event.kind == "ChainProgress":
            print "{: >5}:  ★{}連鎖".format(event.time, event.chain)
        elif event.kind == "ChainEnd":
            print "{: >5}:  ★連鎖終了".format(event.time)
        elif event.kind == "GameOver":
            print "{: >5}:  ▲投了".format(event.time)


# 動画をメモリ上に展開する (長い動画だとやばい)
def load_movie(movie_path):
    cap = cv2.VideoCapture(movie_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open movie file: movie_path={}".format(movie_path))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        logger.debug("Loading movie frame #{}...".format(len(frames) + 1))

        # matplotlib でそのまま画像を表示できるように色を変換しておく
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(img)

    logger.info("Movie loaded")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("movie_path", metavar="MOVIE")
    args = parser.parse_args()

    # 解析に使う画像をロードする
    patterns = load_pattern_images()
    cross_mark_image = load_image(os.path.join(BASE_DIR, "cross_mark.png"))
    cross_mark_image = cv2.resize(cross_mark_image, (54, 64))  # サイズをミスってるので修正
    background_image = load_image(os.path.join(BASE_DIR, "background_pattern.png"))
    mask_image = load_image(os.path.join(BASE_DIR, "puyo_mask.png"))

    # 動画をロードする
    frames = load_movie(args.movie_path)

    # 重要なフレームを検出する
    logger.info("Detect important frames...")
    tsumo_frame_indices = detect_tsumo_frames(frames)
    chain_start_frame_indices = detect_chain_start_frames(frames, cross_mark_image, tsumo_frame_indices)
    gameover_frame_index = detect_gameover_frame(frames, background_image)

    # イベントリストを作る
    logger.info("Building event list...")
    events = make_event_list(
        frames,
        tsumo_frame_indices,
        chain_start_frame_indices,
        gameover_frame_index,
        mask_image,
        patterns)

    # 棋譜を生成して出力する
    logger.info("Generating puyofu...")
    generate_puyofu(events)


if __name__ == "__main__":
    main()
