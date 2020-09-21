import time
import numpy as np
from board import Board
import board_reader as board_reader

BOARD_COUNT = 4

def read_boards():
    for i in range(BOARD_COUNT):
        start = time.time()
        values = board_reader.read_values(f'boards/{i + 1}.jpg')
        end = time.time()
        print(f'Board {i + 1} Read in {end - start:.3f} s')
        np.save(f'boards/{i + 1}.npy', values)

def solve_boards():
    for i in range(BOARD_COUNT):
        values = np.load(f'boards/{i + 1}.npy')
        board = Board(values)

        start = time.time()
        board.solve()
        end = time.time()
        print(f'Board Solved in {end - start:.3f} s')
        print(board)

read_boards()
solve_boards()
