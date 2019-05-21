import board_reader as board_reader

board = board_reader.read_board('boards/1.jpg')
print(board.transpose())

