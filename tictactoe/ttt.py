class TicTacToe:
    """ Game of TicTacToe. Alternates between XO plays.
    Determines win or lose.
    """
    def __init__(self, size=3):
        # Defines the grid
        self.grid = []
        self.size = size
        self.turn = 1
        for i in range(size):
            for j in range(size):
                self.grid.append(0)

    def place(self, i, j):
        # plays if spot is empty
        if self.grid[i*self.size + j] == 0:
            self.grid[i*self.size + j] = self.turn
            self.turn = self.turn*-1 # switch turn
            return True
        return False

    def check_win(self):
        """ Checks if a win occurred.

        Output:
            returns 1 if player 1 got <size> in a row
            returns -1 if player -1 got <size> in a row
            returns 0 if no win
            returns 2 if draw - no more spaces available to play
        """
        # must fit within <size>, so start at borders, walk along edge 
        # just need left and down direction
        # starting with first element - if none, skip
        # corners can go diagonal
        size = self.size
        for i in range(size): # go along left edge
            # check row right
            val = self.grid[i*self.size + 0]
            if val == 0:
                continue
            j = 0
            while j < size and self.grid[i*size + j] == val:
                j = j + 1

            if j == size:
                return val # winner

        # check diagonal
        j = 0
        val = self.grid[0]
        if val != 0:
            while j < size and self.grid[j*size + j] == val:
                j = j + 1

            if j == size:
                return val

        # check top row now
        for i in range(size): # go along left edge
            # check row right
            val = self.grid[i]
            if val == 0:
                continue
            j = 0
            while j < size and self.grid[j*size + i] == val:
                j = j + 1

            if j == size:
                return val # winner

        # check if no zeros exist
        for i in range(size):
            for j in range(size):
                if self.grid[i*size + j] == 0:
                    return 0 # moves available

        return 2 # draw

    def show_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i*self.size + j] == 1:
                    print('X ', end='')
                elif self.grid[i*self.size + j] == -1:
                    print('O ', end='')
                else:
                    print('- ', end='')

            print('')

    def flatten_grid(self):
        return self.grid

while True:
    t = TicTacToe(3)
    while t.check_win() == 0:
        t.show_grid()
        while True:
            x = int(input('y position: '))
            y = int(input('x position: '))
            if 0 > x or x > t.size or 0 > y or y > t.size:
                continue
            if t.place(x,y):
                break

            
