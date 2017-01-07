import numpy as np

def jointHistogram(img, boundary, bin_size):

    interval = (boundary[1] - boundary[0] + 1) / bin_size
    
    if len(img.shape) > 2:
        hist_size = bin_size ** img.shape[2]
        img_bin = np.zeros([img.shape[0], img.shape[1]], np.int32)
        for i in range(img.shape[2]):
            img_bin += ((img[:, :, i] - boundary[0]) / interval) * (bin_size ** i)
    else:
        hist_size = bin_size
        img_bin = (img - boundary[0]) / interval

    unique, count = np.unique(img_bin, return_counts=True)


    histogram = np.zeros([hist_size])
    for u, c in zip(unique, count):
        histogram[u] = c

    return histogram
