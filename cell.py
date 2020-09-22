import itertools as it
import colorama
from colorama import Fore, Style
colorama.init()

class Cell:
    def __init__(self, initial_value):
        self.row = []
        self.column = []
        self.section = []
        self.is_starting_cell = initial_value != 0
        if self.is_starting_cell:
            self.value = initial_value
            self.possible_values = []
        else:
            self.reset()

    def __str__(self):
        color = ''
        if self.is_starting_cell:
            color = Fore.BLUE
        elif self.value != 0:
            color = Fore.GREEN
        return color + str(self.value if self.value != 0 else "□") + Style.RESET_ALL

    def reset(self):
        self.value = 0
        self.possible_values = list(range(1, 10))

    def remove_possible_value(self, value):
        if value in self.possible_values:
            self.possible_values.remove(value)
            self.check_possible_values()

    def check_possible_values(self):
        if len(self.possible_values) == 1:
            self.set_value(self.possible_values[0])

    def set_value(self, value):
        self.value = value
        self.possible_values.clear()
        self.clear_value_from_related()

    def clear_value_from_related(self):
        for cell in it.chain(self.row, self.column, self.section):
            cell.remove_possible_value(self.value)
