import time
import board_reader as board_reader

start = time.time()
board = board_reader.read_board('boards/4.jpg')
end = time.time()

print(f'Board Read in {end - start:.3f} s')

start = time.time()
board.solve()
end = time.time()

print(f'Board Solved in {end - start:.3f} s')
print(board)
