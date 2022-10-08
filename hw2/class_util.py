class NumberBin():
    def __init__(self, bins_count: int, pixels: int, pseudo_count = 1):    
        # number of bins for each pixel
        self.bins_count = bins_count

        # size of the image is size * size
        self.pixels = pixels + 1

        # 每一個pixel有bins_count個bin
        self.pixel_bins = [[0 for _ in range(self.bins_count)] for _ in range(self.pixels)]

    def add(self, feature):
        for i in range(self.pixels):
            print(feature[i])
            self.pixel_bins[i].add(feature[i])

    def __str__(self):
        res = '['
        for i in range(self.pixels):
            res += '['
            for j in range(self.bins_count):
                res += str(self.pixel_bins[i][j])
                res += ' '
            res += '], '

        res += ']'

        return res