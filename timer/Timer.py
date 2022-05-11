import time


class Timer:
    def __init__(self):
        self.started = None
        self.finished = None

    def startTime(self):
        self.started = time.perf_counter()

    def finishTime(self):
        self.finished = time.perf_counter()
        distance = (self.finished - self.started) * 1000
        print(f"{distance} ms'de tamamlandı.")
