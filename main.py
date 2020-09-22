import cv2 as cv2
import time
import numpy as np
from board import Board
import board_reader as board_reader


def solve_boards_from_disk():
    BOARD_COUNT = 4
    for i in range(BOARD_COUNT):
        start = time.time()
        board = board_reader.read_board_from_disk(f'boards/{i + 1}.jpg')
        # board.save(f'boards/{i + 1}.npy')
        # board = Board.load(f'boards/{i + 1}.npy')
        end = time.time()

        print(f'Board {i + 1} Read in {end - start:.3f} s')

        start = time.time()
        board.solve()
        end = time.time()

        print(f'Board {i + 1} Solved in {end - start:.3f} s')
        board.print()


def solve_board_from_web_cam():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("raw", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("board", cv2.WINDOW_AUTOSIZE)
    while True:
        success, img = cam.read()
        if success:
            key = cv2.waitKey(10)
            if key == 27:
                break
            try:
                board = board_reader.read_board_from_image(img)
                board.solve()
                if board.is_solved():
                    board.print()
                cv2.imshow("raw", img)
            except Exception as e:
                print(e)
                continue

    # cv2.destroyWindow("raw")
    cv2.destroyWindow("board")


# solve_boards_from_disk()
solve_board_from_web_cam()
