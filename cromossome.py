class Cromossome:
    def __init__(self):
        self.string = None
        self.fitness = None
        self.p_optimal = True
        self.rank = None

    def partially_less_than(self, b):
        has_less = False

        for x, y in zip(self.fitness, b.fitness):
            if(x > y):
                return False
            elif(x < y):
                has_less = True

        return has_less
