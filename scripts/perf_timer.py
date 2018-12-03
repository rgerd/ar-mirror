import time

class PerfTimer:
    def __init__(self):
        self.times = [('start', time.time())]

    def measure(self, name):
        self.times.append((name, time.time()))

    def log(self):
        durations = []
        total_time = self.times[-1] - self.times[0]
        for i in range(len(self.times)):
            name = self.times[i + 1]
            duration = round(self.times[i + 1] - self.times[i], 3)
            percent = str(round((duration / total_time) * 100, 1)) + "%"
            durations.append((name, duration, percent))

        print(durations)