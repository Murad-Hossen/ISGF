class addSubtract:
    def __init__(self):
        self.a = 0
        self.b = 0

    def add(self):
        return self.a + self.b

    def subtract(self):
        return self.a - self.b
    
    def graph_to_binary_image(self, graph, image_size=700, radius_scale=2.0):
        all_x = []
        all_y = []
        for s, e in graph.edges():
            pts = graph[s][e]['pts']
            if pts.ndim != 2 or pts.shape[1] < 2:
                continue
            all_x.extend(pts[:,1])
            all_y.extend(pts[:,0])
        if not all_x:
            return np.zeros((image_size, image_size), dtype=bool)
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        extent = max(x_max - x_min, y_max - y_min) * 1.1
        if extent == 0:
            return np.zeros((image_size, image_size), dtype=bool)
        pixel_size = extent / (image_size - 1)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        im = np.zeros((image_size, image_size), dtype=bool)
        def scale_coord(coord, center, extent):
            return int(((coord - center) / extent + 0.5) * (image_size - 1))
        for s, e in graph.edges():
            pts = graph[s][e]['pts']
            if len(pts) < 2:
                continue
            xs = pts[:,1]
            ys = pts[:,0]
            radius = max(1, int(radius_scale))
            for i in range(len(pts) - 1):
                x1 = scale_coord(xs[i], x_center, extent)
                y1 = scale_coord(ys[i], y_center, extent)
                x2 = scale_coord(xs[i+1], x_center, extent)
                y2 = scale_coord(ys[i+1], y_center, extent)
                rr, cc = draw.line(y1, x1, y2, x2)
                valid = (0 <= rr) & (rr < image_size) & (0 <= cc) & (cc < image_size)
                rr, cc = rr[valid], cc[valid]
                for j in range(len(rr)):
                    r, c = rr[j], cc[j]
                    rr_disk, cc_disk = draw.disk((r, c), radius, shape=im.shape)
                    im[rr_disk, cc_disk] = True
        return im