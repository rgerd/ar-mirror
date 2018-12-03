import time

class PerfTimer:
    def __init__(self):
        self.times = [('start', time.time())]

    def measure(self, name):
        self.times.append((name, time.time()))

    def log(self):
        durations = []
        total_time = self.times[-1][1] - self.times[0][1]

        for i in range(len(self.times) - 1):
            name = self.times[i + 1][0]
            duration = self.times[i + 1][1] - self.times[i][1]
            pretty_duration = round(duration * 1000, 2)
            percent = str(round((duration / total_time) * 100, 1)) + "%"
            durations.append((name, pretty_duration, percent))

        print(round(total_time * 1000, 2), durations)