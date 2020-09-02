

class RunningMeanMeter:

    def __init__(self, window_size=None):
        self.window_size = window_size
        self.window = []

        self.sum = 0.0

    def add(self, num):

        self.window.append(num)
        self.sum += num

        if self.window is not None and len(self.window) >= self.window_size:
            self.sum -= self.window[0]
            self.window = self.window[1:]

    def mean(self):
        return self.sum / len(self.window)


class MeanMeter:

    def __init__(self):
        self.sum = 0.0
        self.num_items = 0

    def add(self, num):
        self.sum += num
        self.num_items += 1

    def mean(self):
        return self.sum / self.num_items
