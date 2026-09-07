class Tracker:

    def __init__(self):
        self.previous = None

    def update(self, detection):

        if detection is None:
            return None

        x = detection["x"]
        y = detection["y"]

        if self.previous is None:
            vx = 0
            vy = 0
        else:
            vx = x - self.previous[0]
            vy = y - self.previous[1]

        self.previous = (x, y)

        return {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy
        }
