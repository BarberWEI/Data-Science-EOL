class Bacterium:
    def __init__(self, loc_x, loc_y, r):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.r = r
    
    def get_loc(self):
        return [self.loc_x, self.loc_y]