class Cell:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.initial_value = value
        self.value = value
        self.possible_values = set([x + 1 for x in range(9)] if value == 0 else [])
        self.row = []
        self.col = []
        self.section = []

    def __repr__(self):
        return str(self.value)

    def set_row(self, row):
        self.row = row

    def set_col(self, col):
        self.col = col

    def set_section(self, section):
        self.section = section

    def remove_from_related(self):
        if self.value == 0:
            return
        for cell in self.row + self.col + self.section:
            cell.remove_possible_value(self.value)

    def set_value(self, value):
        self.value = value
        self.possible_values.clear()
        self.remove_from_related()

    def remove_possible_value(self, value):
        if value in self.possible_values:
            self.possible_values.remove(value)

    def check_possible_values(self):
        if len(self.possible_values) == 1:
            self.set_value(self.possible_values.pop())
    
    def check_unique_possible_values(self):
        if self.value != 0:
            return
        self.check_unique_possible_value(self.row)
        self.check_unique_possible_value(self.col)
        self.check_unique_possible_value(self.section)

    def check_unique_possible_value(self, cells):
        if self.value != 0:
            return
        other_possible = set()
        for cell in cells:
            if cell == self:
                continue
            other_possible = other_possible.union(cell.possible_values)
        difference = self.possible_values - other_possible
        if len(difference) == 1:
            self.set_value(difference.pop())

    def is_solved(self):
        return self.value != 0

    def is_starting_cell(self):
        return self.initial_value != 0
