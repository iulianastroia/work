import math


class Point:
    def __init__(self, initx, inity):
        self.x = initx
        self.y = inity

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distance_to_line(self, p1, p2):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff * self.x - x_diff * self.y + p2.x * p1.y - p2.y * p1.x)
        den = math.sqrt(y_diff ** 2 + x_diff ** 2)
        return num / den
