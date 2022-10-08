from cgi import test
import sys
from tokenize import Number
from unicodedata import category
from PIL import Image 
from class_util import NumberBin
import math
from tqdm import tqdm
# image starts from the 17th byte of image file
TRAIN_DATA_OFFSET = 16 

# label starts from the 9th byte of label file
LABEL_DATA_OFFSET = 8

# the metadata of an image
PIXELS = 28 * 28
ROWS = 28
COLS = 28
MAX_PIXEL_VALUE = 255
DIGITS = 10

# number of items
TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000

# file path
TRAIN_IMAGES_PATH = './train-images-idx3-ubyte/train-images.idx3-ubyte'
t10k_IMAGES_PATH = './t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
TRAIN_LABELS_PATH = './train-labels-idx1-ubyte/train-labels.idx1-ubyte'
t10k_LABELS_PATH = './t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'

# Bins 
BINS_COUNT = 32
BINS_SIZE = (MAX_PIXEL_VALUE + 1) // BINS_COUNT


PSEUDO_COUNT = 1

def trace(func):
    def wrapper(*args):
        print(f'Processing: {func.__name__}')
        res = func(*args)
        print(f'Done: {func.__name__}\n')
        return res
    return wrapper

def main():
    train_data = read_data(TRAIN_IMAGES_PATH, 'images', TRAIN_DATA_SIZE)
    train_labels = read_data(TRAIN_LABELS_PATH, 'labels', TRAIN_DATA_SIZE)

    # bins[i] is the total count of grayscale value that belongs to [i * 8, i * 8 + 7]
    bins = [[0 for _ in range(BINS_COUNT)] for _ in range(PIXELS)]

    # numbers[i] is the bins of the ith number
    numbers = [[0 for _ in range(BINS_COUNT)] for _ in range(DIGITS)]

    # freq[i] is the frequency of the ith number
    label_freq = [0 for _ in range(DIGITS)]


    number_bins = [NumberBin(BINS_COUNT, PIXELS) for _ in range(DIGITS)]

    # counts and puts the result in bins and numbers
    # extract_feature(train_data, train_labels, bins, numbers, freq)
    extract_feature_by_position(train_data, train_labels, number_bins, bins, label_freq)

    train_error_rate = get_error_rate(train_data, train_labels, number_bins, label_freq, TRAIN_DATA_SIZE)

    release_list(train_data)
    release_list(train_labels)

    test_data = read_data(t10k_IMAGES_PATH, 'images', TEST_DATA_SIZE)
    test_labels = read_data(t10k_LABELS_PATH, 'labels', TEST_DATA_SIZE)

    test_error_rate = get_error_rate(test_data, test_labels, number_bins, label_freq,TEST_DATA_SIZE)

    print(f'Train Error Rate: {train_error_rate}, Test Error Rate: {test_error_rate}')

@trace 
def get_error_rate(data: list, labels: list, number_bins: list, label_freq: list, dataset_size: int):
    error_count = 0
    for i in tqdm(range(dataset_size)):
        d = data[i]
        l = labels[i]
        data_bin = [x // BINS_SIZE for x in d]

        pred = 0
        cur_max = float('-inf')
        for p in range(DIGITS):
            prob_in_ln = get_ln_prob(data_bin, p, number_bins[p], label_freq)
            if prob_in_ln > cur_max:
                cur_max = prob_in_ln
                pred = p

        if l != pred:
            error_count += 1
    return error_count / dataset_size


def get_ln_prob(data_bin: list, p: int, number_bin: list, label_freq: list):
    ln_prior = math.log(label_freq[p]) - math.log(sum(label_freq))
    ln_likelihood = 0


    for i in range(PIXELS):
        # bins at the ith pixel
        bins = number_bin.pixel_bins[i]
        discrete_value = data_bin[i]

        ln_likelihood += math.log(bins[discrete_value] if bins[discrete_value] != 0 else PSEUDO_COUNT) - math.log(sum(bins))

    return ln_prior + ln_likelihood


# ref: https://stackoverflow.com/questions/12417498/how-to-release-used-memory-immediately-in-python-list
def release_list(ls):
    del ls[:]
    del ls


@trace 
def extract_feature_by_position(data: list, labels: list, NumberBins: list, bins: list, label_freq: list):
    for i in tqdm(range(TRAIN_DATA_SIZE)):
        d = data[i]
        l = labels[i]
        label_freq[l] += 1

        for j in range(PIXELS):
            pixel = d[j]
            category = get_catogory(pixel)
            
            NumberBins[l].pixel_bins[j][category] += 1
            bins[j][category] += 1


def get_catogory(pixel):
    return pixel // BINS_SIZE


@trace
def read_data(file_path, type, items):
    offset = TRAIN_DATA_OFFSET if type == 'images' else LABEL_DATA_OFFSET

    # 28 * 28 for image, 1 for label

    data = []

    with open(file_path, 'rb') as f:
        f.seek(offset)
        for _ in tqdm(range(items)):
            if type == 'images':
                # Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
                # 注意: 這邊跟grayscale使用的數值相反，但不影響
                image = [int.from_bytes(bytes=f.read(1), byteorder='big', signed=False) for _ in range(PIXELS)]
                data.append(image)
            else:
                data.append(int.from_bytes(bytes=f.read(1), byteorder='big', signed=False))

    return data




if __name__ == '__main__':
    main()
