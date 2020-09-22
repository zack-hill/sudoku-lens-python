from cell import Cell
import itertools as it
import numpy as np
import time

class Board:
    def __init__(self, values):
        self.cells = []
        self.columns = []
        self.rows = []
        self.sections = []
        # Create cells
        for y in range(9):
            self.cells.append([])
            for x in range(9):
                self.cells[y].append(Cell(values[x][y]))
        # Set up rows
        for y in range(9):
            row = [cell for cell in self.cells[y]]
            for cell in row:
                cell.row = row
            self.rows.append(row)
        # Set up columns
        for x in range(9):
            col = []
            for y in range(9):
                cell = self.cells[y][x]
                col.append(cell)
                cell.column = col
            self.columns.append(col)
        # Set up sections
        for y_offset in range(0, 9, 3):
            for x_offset in range(0, 9, 3):
                section = []
                for y in range(3):
                    for x in range(3):
                        cell = self.cells[y+y_offset][x+x_offset]
                        section.append(cell)
                        cell.section = section
                self.sections.append(section)
        
        for y in range(9):
            for x in range(9):
                cell = self.cells[y][x]
                if cell.value != 0:
                    cell.clear_value_from_related()


    def print(self):
        for y in range(9):
            if y in [3, 6]:
                print('------+-------+------')
            row = ""
            for x in range(9):
                if x in [3, 6]:
                    row += "| "
                row += str(self.cells[y][x]) + " "
            print(row.rstrip())

    def is_solved(self):
        for x in range(9):
            for y in range(9):
                if self.cells[y][x].value == 0:
                    return False
        return True
    
    def solve(self):
        # Try to solve the board heuristically using the process of elimination
        self.__solve_using_elimination()
        if self.is_solved():
            return
        # Fallback to brute force using backtracking algorithm
        self.__solve_using_backtracking()

    def save(self, path):
        values = np.zeros(shape=(9, 9), dtype=np.int)        
        for y in range(9):
            for x in range(9):
                values[x][y] = self.cells[y][x].value
        np.save(path, values)

    def load(path):
        values = np.load(path)
        return Board(values)

    def __solve_using_elimination(self):
        # Loop until we have made a pass over the board where we did
        # not find a value of a cell. This means we have either
        # finished or hit a dead end.
        found_value = True
        while found_value:
            found_value = False
            for y in range(9):
                for x in range(9):
                    cell = self.cells[y][x]
                    # Skip cells that already have a value
                    if cell.value != 0:
                        continue
                    # Compare the cell's possible values to other cells in its row,
                    # column, and section looking for one that is unique to the cell
                    unique_cell_possible_value = self.__get_unique_possible_value(cell)
                    if unique_cell_possible_value is not None:
                        cell.set_value(unique_cell_possible_value)
                        found_value = True

    @staticmethod
    def __get_cell_group_values(cell):
        cells = it.chain(cell.row, cell.column, cell.section)
        other_cells = filter(lambda x: x != cell, cells)
        other_values = set(map(lambda x: x.value, other_cells))
        return other_values

    @staticmethod
    def __get_unique_possible_value(cell):
        for group in (cell.row, cell.column, cell.section):
            # Build a unique set of possible values for all cells in the group
            # except for the current cell.
            other_possible = set()
            for other_cell in group:
                if other_cell != cell:
                    other_possible.update(other_cell.possible_values)
            # Get the set of possible values the the cell has that the other
            # cells don't. This should only ever contain one or zero items.
            unique_to_cell = set(cell.possible_values).difference(other_possible)
            # Return the unique possible value if one is found.
            if len(unique_to_cell) == 1:
                return unique_to_cell.pop()
        return None

    def __solve_using_backtracking(self):
        unsolved_cells = []
        # Create a stack of all unsolved cells
        for y in range(9):
            for x in range(9):
                cell = self.cells[y][x]
                if cell.value == 0:
                    # Capture the cell's possible values
                    unsolved_cells.append([cell, set(cell.possible_values)])
        solved_cells = []
        is_backtracking = False
        while len(unsolved_cells) > 0:
            # Pop a cell off of the appropriate stack
            cell, possible_values = solved_cells.pop() if is_backtracking else unsolved_cells.pop()
            # Remove any values from other cells in the current cell's groups from
            # this cell's set of possible values
            possible_values = possible_values.difference(self.__get_cell_group_values(cell))
            if len(possible_values) > 0:
                # Set the cell's value to the first of the possible values
                cell.value = possible_values.pop()
                # Move the cell onto the solved cell stack! (Hint: it probably won't last for long)
                solved_cells.append([cell, possible_values])
                is_backtracking = False
                continue
            # If there are no valid values, put the cell back on the unsolved stack and reset the possible values.
            unsolved_cells.append([cell, set(cell.possible_values)])
            cell.value = 0
            is_backtracking = True
