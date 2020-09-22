import time
import numpy as np
from board import Board
import board_reader as board_reader

BOARD_COUNT = 4


def read_boards():
    for i in range(BOARD_COUNT):
        start = time.time()
        board = board_reader.read_board(f'boards/{i + 1}.jpg')
        end = time.time()

        print(f'Board {i + 1} Read in {end - start:.3f} s')
        board.save(f'boards/{i + 1}.npy')


def solve_boards():
    for i in range(BOARD_COUNT):
        board = Board.load(f'boards/{i + 1}.npy')

        start = time.time()
        board.solve()
        end = time.time()

        print(f'Board {i + 1} Solved in {end - start:.3f} s')
        board.print()


read_boards()
solve_boards()
