from cell import Cell
import colorama
from colorama import Fore, Style
colorama.init()

class Board:
    def __init__(self, values):
        self.cells = self.create_cells(values)
    
    def __str__(self):
        ret = ''
        for y in range(9):
            for x in range(9):
                cell = self.cells[x][y]
                if cell.value == 0:
                    ret += ' '
                else:
                    color_prefix = ''
                    if cell.is_starting_cell():
                        color_prefix = Fore.BLUE
                    elif cell.is_solved():
                        color_prefix = Fore.GREEN
                    ret += color_prefix + str(self.cells[x][y].value) + Style.RESET_ALL                
                ret += ' '
            ret += '\n'
        return ret

    def create_cells(self, values):
        cells = [[Cell(x, y, values[x][y]) for x in range(9)] for y in range(9)]
        for y in range(9):
            row = []
            for x in range(9):
                value = values[x][y]
                cell = Cell(x, y, value)
                cells[x][y] = cell
                cell.set_row(row)
                row.append(cell)

        for x in range(9):
            col = []
            for y in range(9):
                cell = cells[x][y]
                cell.set_col(col)
                col.append(cell)

        for y_offset in range(0, 9, 3):
            for x_offset in range(0, 9, 3):
                section = []
                for x in range(3):
                    for y in range(3):
                        cell = cells[x + x_offset][y + y_offset]
                        cell.set_section(section)
                        section.append(cell)
        return cells

    def interate_cells(self):
        for y in range(9):
            for x in range(9):
                yield self.cells[x][y]

    def is_solved(self):
        for cell in self.interate_cells():
            if not cell.is_solved():
                return False
        return True

    def solve(self):
        for cell in self.interate_cells():
            cell.remove_from_related()
                  
        count = 0
        for i in range(50):            
            count = i
            for cell in self.interate_cells():
                cell.check_possible_values()
                cell.check_unique_possible_values()
            if self.is_solved():
                break
        print(('Solved' if self.is_solved() else 'Unsolved') + f' after {count} Iterations')
