import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        """initialize a checkerboard pattern
        
        Args:
            resolution (int): image resolution
            tile_size (int): size of each tile in pixels
        """
        # resolution should be evenly divisible by 2*tile_size
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution should be evenly divisible by 2 x tile_size")
            
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        """create the checkerboard pattern
        
        Returns:
            np.ndarray: copy of the output pattern
        """
        # black and white tiles
        black_tile = np.zeros((self.tile_size, self.tile_size))
        white_tile = np.ones((self.tile_size, self.tile_size))
        
        # a row of black and white tiles
        first_row_base = np.concatenate([black_tile, white_tile], axis=1)
        first_row_full = np.tile(first_row_base, (1, self.resolution//(2*self.tile_size)))
        
        # a row of white and black tiles
        second_row_base = np.concatenate([white_tile, black_tile], axis=1)
        second_row_full = np.tile(second_row_base, (1, self.resolution//(2*self.tile_size)))
        
        # stack rows vertically
        full_row = np.vstack([first_row_full, second_row_full])
        # self.output = full_row
        self.output = np.tile(full_row, (self.resolution//(2*self.tile_size), 1))

        return self.output.copy()

    def show(self):
        """display the pattern"""
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        """initialize a circle pattern
        
        Args:
            resolution (int): image resolution
            radius (int): radius of the circle
            position (tuple): (x,y) coordinates of circle center
        """
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        """create the circle pattern
        
        Returns:
            np.ndarray: copy of the output pattern
        """
        # coordinate grids
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # calculating distances from center using euclidean distance
        distances = np.sqrt((X - self.position[0])**2 + (Y - self.position[1])**2)
        
        # binary circle mask
        self.output = distances <= self.radius
        
        return self.output.copy()

    def show(self):
        """display the pattern"""
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        """initialize an RGB spectrum pattern
        
        Args:
            resolution (int): image resolution
        """
        self.resolution = resolution
        self.output = None

    def draw(self):
        """create the RGB spectrum pattern
        
        Returns:
            np.ndarray: copy of the output pattern
        """
        # coordinate grids between 0 and 1
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # RGB channels
       
        red = X
       
        green = Y
        
        blue = 1.0 - X
        
        # stack channels to have RGB image
        self.output = np.dstack((red, green, blue))
        
        return self.output.copy()

    def show(self):
        """display the pattern"""
        if self.output is None:
            self.draw()
        plt.imshow(self.output)
        plt.show()
