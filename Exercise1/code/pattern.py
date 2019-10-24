import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution, resolution));

    def draw(self):
        size = (self.tile_size, self.tile_size)
        tile = np.ones(size, dtype=bool)
        tile_bw = np.concatenate((tile * 0, tile), axis=0)
        tile_wb = np.concatenate((tile, tile * 0), axis=0)
        tile22 = np.concatenate((tile_bw, tile_wb), axis=1)

        repetition = int(self.resolution / (self.tile_size * 2)) + 1
        self.output = np.tile(tile22, (repetition, repetition))
        self.output = self.output[:self.resolution, :self.resolution]
        copy = np.copy(self.output)
        return copy

    def show(self):
        plt.title('Checkerboard')
        plt.imshow(self.output, cmap='bone')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((resolution, resolution, 3),)

    def draw(self):
        # upper_left  blue:   [0, 0, 225]
        # upper_right red :   [255, 0, 0]
        # lower_left  Cyan:   [0, 255, 255]
        # lower_right yellow: [255, 255, 0]

        r = np.linspace(0, 1, num=self.resolution)
        r = np.tile(r, (self.resolution, 1))
        g = r.T
        b = np.linspace(1, 0, num=self.resolution)
        b = np.tile(b, (self.resolution, 1))

        self.output[:, :, 0] = r
        self.output[:, :, 1] = g
        self.output[:, :, 2] = b

        copy = np.copy(self.output)
        return copy

    def show(self):
        plt.title('Spectrum')
        plt.imshow(self.output)
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((resolution, resolution), dtype=bool)

    def draw(self):
        yy, xx = np.ogrid[:self.resolution, :self.resolution]
        circle = (xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2 <= self.radius ** 2
        print(circle)
        self.output[circle] = 1
        copy = np.copy(self.output)
        return copy

    def show(self):
        plt.title('Circle')
        plt.imshow(self.output, cmap='bone')
        plt.show()

test = Circle(100, 2, (2, 2))
test.draw()