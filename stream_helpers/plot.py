import astropy.units as u

__all__ = ['StreamRidgeBuilder']


class StreamRidgeBuilder:

    def __init__(self, ax, proj):
        self.ax = ax
        self.reset()
        fig = self.ax.figure

        self.proj = proj
        self.cids = {}
        self.cids['mouse'] = fig.canvas.mpl_connect('button_press_event', self)
        self.cids['key'] = fig.canvas.mpl_connect('key_press_event', self)

    def reset(self):
        self.line = None
        self.xs = list()
        self.ys = list()

    def __call__(self, event):
        if event.inaxes != self.ax:
            return

        self.xs.append(event.xdata)
        self.ys.append(event.ydata)

        if self.line is None:
            self.line, = self.ax.plot(self.xs, self.ys, color='tab:red')
        else:
            self.line.set_data(self.xs, self.ys)

        self.line.figure.canvas.draw()

        if event.key == 'escape':
            if len(self.xs) == 0:
                self.reset()
                return

            endpoints = [self.proj.xy2ang(x, y, lonlat=True)
                         for x, y in zip(self.xs, self.ys)] * u.deg
            print("Track in ICRS coordinates (degrees):")
            print(endpoints.to_value(u.deg).tolist())

            self.line.set_data([], [])
            self.reset()
