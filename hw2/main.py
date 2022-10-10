from class_util import NumberBin
import math
from tqdm import tqdm
import argparse

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

def main(mode):

    # discrete mode
    if mode == 0:
        discrete_classifier()
    elif mode == 1:
        cont_classifier()


def get_train_dataset():
    train_data = read_data(TRAIN_IMAGES_PATH, 'images', TRAIN_DATA_SIZE)
    train_labels = read_data(TRAIN_LABELS_PATH, 'labels', TRAIN_DATA_SIZE)
    return train_data, train_labels

def get_test_dataset():
    test_data = read_data(t10k_IMAGES_PATH, 'images', TEST_DATA_SIZE)
    test_labels = read_data(t10k_LABELS_PATH, 'labels', TEST_DATA_SIZE)
    return test_data, test_labels

def cont_classifier():
    train_data, train_labels = get_train_dataset()

    # the only difference of cont and discrete version is the likelihood part

    label_freq, train_pixel_distribution, digit_pixel_distribution = extract_cont_feature_by_position(train_data, train_labels)


    # print(digit_pixel_distribution[0])
    data_MLE_params, digit_MLE_params = get_MLE_params(train_pixel_distribution, digit_pixel_distribution)
    # print(digit_MLE_params[0])


    # train_error_rate, total_scores, predictions = get_cont_error_rate(train_data, train_labels, digit_MLE_params, label_freq, TRAIN_DATA_SIZE)


    test_data, test_labels = get_test_dataset()

    test_error_rate, total_scores, predictions = get_cont_error_rate(test_data, test_labels, digit_MLE_params, label_freq, TEST_DATA_SIZE)

    images = cont_plot_imagination(digit_MLE_params, label_freq)

    write_result(images, test_error_rate, total_scores, predictions, 1, TEST_DATA_SIZE)


@trace 
def get_cont_error_rate(data: list, labels: list, digit_MLE_params: list, label_freq: list, dataset_size: int):
    error_count = 0

    # (dataset_size, digits)
    total_scores = []

    # (dataset_size, 2(prediction, label))
    predictions = []

    for i in tqdm(range(dataset_size), "Calculating Error Rate"):
        d = data[i]
        l = labels[i]

        pred = 0
        cur_max = float('-inf')
        
        scores = []

        for p in range(DIGITS):
            prob_in_ln = get_cont_ln_prob(d, digit_MLE_params, label_freq, p)

            if prob_in_ln > cur_max:
                cur_max = prob_in_ln
                pred = p

            scores.append(-prob_in_ln)

        if pred != l:
            error_count += 1

        s = sum(scores)
        for i in range(DIGITS):
            scores[i] /= s
        total_scores.append(scores)

        predictions.append([pred, l])

    return error_count / dataset_size, total_scores, predictions


def get_cont_ln_prob(data: list, digit_MLE_params: list, label_freq: list, p: int):
    prior_ln = math.log(label_freq[p]) - math.log(sum(label_freq))
    likelihood_ln = 0
    for i in range(PIXELS):
        pixel = data[i]
        mean, std = digit_MLE_params[p][i]
        likelihood_ln += get_Gaussian_ln_prob(mean, std, pixel)
    return prior_ln + likelihood_ln



def get_Gaussian_ln_prob(mean: float, std: float, x: int):
    # precedence of power is higher than unary operator 
    return -math.log(std) - math.log(2 * math.pi) / 2 - ((x - mean) / std) ** 2 / 2 


@trace
# get the mean and std of original data and digit 
def get_MLE_params(train_pixel_distribution, digit_pixel_distribution):

    # (pixels, 2)
    data_MLE_params = []

    # (digits, pixels, 2)
    digit_MLE_params = []


    for d in tqdm(train_pixel_distribution, desc="Calculating mean and std of dataset"):
        mean, std = get_params(d)
        data_MLE_params.append([mean, std])

    for i in tqdm(range(DIGITS), desc="Calculating mean and std of digits"):
        MLE_params = []
        for d in digit_pixel_distribution[i]:
            mean, std = get_params(d)
            MLE_params.append([mean, std])
        digit_MLE_params.append(MLE_params)


    
    return data_MLE_params, digit_MLE_params


# return the mean and standard deviation of the MLE
def get_params(distribution):
    # should use the pixel value as x, instead of count
    N = len(distribution)
    mean = sum(distribution) / N
    std = math.sqrt(sum([(x - mean) ** 2 for x in distribution]) / N)

    return mean, std

@trace
def extract_cont_feature_by_position(train_data: list, train_labels: list, episilon=0.85):
    label_freq = [0 for _ in range(DIGITS)]


    # (digits, pixels, MAX_PIXEL_VALUE)
    digit_pseudo_checker = [[[False for _ in range(MAX_PIXEL_VALUE + 1)] for _ in range(PIXELS)] for _ in range(DIGITS)]

    # (pixels, [list of pixel values of that pixel])
    train_pixel_distribution = [[] for _ in range(PIXELS)]

    # (digits, pixels, [list of pixel values of that pixel])
    digit_pixel_distribution = [[[] for _ in range(PIXELS)] for _ in range(DIGITS)]

    for i in tqdm(range(TRAIN_DATA_SIZE), desc="Extracting features"):
        d = train_data[i]
        l = train_labels[i]
        label_freq[l] += 1
        for j in range(PIXELS):
            pixel_value = d[j]

            digit_pseudo_checker[l][j][pixel_value] = True

            train_pixel_distribution[j].append(pixel_value)
            digit_pixel_distribution[l][j].append(pixel_value)

    # randomly adding noise to avoid 0 standard deviation 
    for i in tqdm(range(PIXELS), desc="Adding pseudocount"):
        for pixel_value in range(MAX_PIXEL_VALUE + 1):
            for digit in range(DIGITS):
                if not digit_pseudo_checker[digit][i][pixel_value]:
                    digit_pixel_distribution[digit][i].append(pixel_value)


    
    return label_freq, train_pixel_distribution, digit_pixel_distribution


@trace
def discrete_classifier():
    
    train_data, train_labels = get_train_dataset()
    # bins[i] is the total count of grayscale value that belongs to [i * 8, i * 8 + 7]
    bins = [[0 for _ in range(BINS_COUNT)] for _ in range(PIXELS)]


    # label_freq[i] is the frequency of the ith number
    label_freq = [0 for _ in range(DIGITS)]


    number_bins = [NumberBin(BINS_COUNT, PIXELS) for _ in range(DIGITS)]

    extract_feature_by_position(train_data, train_labels, number_bins, bins, label_freq)

    # train_error_rate = get_error_rate(train_data, train_labels, number_bins, label_freq, TRAIN_DATA_SIZE)


    test_data, test_labels = get_test_dataset()
    
    test_error_rate, total_scores, predictions = get_error_rate(test_data, test_labels, number_bins, label_freq,TEST_DATA_SIZE)

    images = plot_imagination(number_bins, label_freq)

    write_result(images, test_error_rate, total_scores, predictions, 0, TEST_DATA_SIZE)
    # print(test_error_rate)

@trace 
def get_error_rate(data: list, labels: list, number_bins: list, label_freq: list, dataset_size: int):
    error_count = 0
    total_scores = []
    predictions = []
    


    for i in tqdm(range(dataset_size), "Calculating error rate"):
        d = data[i]
        l = labels[i]
        data_bin = [x // BINS_SIZE for x in d]

        scores = []

        pred = 0
        cur_max = float('-inf')
        for p in range(DIGITS):
            prob_in_ln = get_ln_prob(data_bin, p, number_bins[p], label_freq)
            if prob_in_ln > cur_max:
                cur_max = prob_in_ln
                pred = p
            scores.append(-prob_in_ln)

        if l != pred:
            error_count += 1

        s = sum(scores)
        for i in range(DIGITS):
            scores[i] /= s

        predictions.append([pred, l])
        total_scores.append(scores)

    return error_count / dataset_size, total_scores, predictions


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
# delete unused list to release memory 



@trace 
def extract_feature_by_position(data: list, labels: list, NumberBins: list, bins: list, label_freq: list):
    for i in tqdm(range(TRAIN_DATA_SIZE), desc="Extracting features"):
        d = data[i]
        l = labels[i]
        label_freq[l] += 1

        for j in range(PIXELS):
            pixel_value = d[j]
            category = get_catogory(pixel_value)
            
            NumberBins[l].pixel_bins[j][category] += 1
            bins[j][category] += 1


def get_catogory(pixel: int):
    return pixel // BINS_SIZE


@trace
def read_data(file_path: str, type: str, items: int):
    offset = TRAIN_DATA_OFFSET if type == 'images' else LABEL_DATA_OFFSET

    # 28 * 28 for image, 1 for label

    data = []

    with open(file_path, 'rb') as f:
        f.seek(offset)
        for _ in tqdm(range(items), desc="Reading data"):
            if type == 'images':
                # Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
                # 注意: 這邊跟grayscale使用的數值相反，但不影響
                image = [int.from_bytes(bytes=f.read(1), byteorder='big', signed=False) for _ in range(PIXELS)]
                data.append(image)
            else:
                data.append(int.from_bytes(bytes=f.read(1), byteorder='big', signed=False))

    return data


@trace 
def plot_imagination(number_bins: list, label_freq: list, threshold_category=16):
    # pixel 129 starts from the 16th bin

    images = [[[0 for _ in range(COLS)] for _ in range(ROWS)] for _ in range(DIGITS)]

    for l in tqdm(range(DIGITS), desc="Plotting imagination number"):
        for pixel in range(PIXELS):
            score = 0
            for category in range(threshold_category, BINS_COUNT):
                score += number_bins[l].pixel_bins[pixel][category]
            if score * 2.5 >= label_freq[l]:
                r = pixel // ROWS
                c = pixel % COLS
                images[l][r][c] = 1

    return images
    

@trace 
def cont_plot_imagination(digit_MLE_params: list, label_freq: list, threshold=128):
    # plot 1 if mean >= threshold else 0
    images = [[[0 for _ in range(COLS)] for _ in range(ROWS)] for _ in range(DIGITS)]

    for l in tqdm(range(DIGITS), desc="Plotting imagination number"):
        for pixel in range(PIXELS):
            mean = digit_MLE_params[l][pixel][0]
            if mean >= threshold * 0.85:
                r = pixel // ROWS
                c = pixel % COLS
                images[l][r][c] = 1

    return images


@trace
def write_result(images, error_rate, total_scores, predictions, mode, dataset_size):
    file_path = "./cont_result.txt" if mode == 1 else "./result.txt"

    with open(file_path, 'w') as f:
        for i in tqdm(range(dataset_size), desc='Writing ln(prob)'):
            f.write('Posterior (in log scale):\n')
            for d in range(DIGITS):
                f.write(f'{d}: {total_scores[i][d]}\n')
            f.write(f'Prediction: {predictions[i][0]}, Ans: {predictions[i][1]}\n\n')
            
        f.write('Imagination of numbers in Bayesian classifier:\n\n')

        for i in tqdm(range(DIGITS), desc='Plotting image'):
            f.write(f'{i}: \n')
            for r in range(ROWS):
                for c in range(COLS):
                    f.write(f'{images[i][r][c]} ')
                f.write('\n')

            f.write('\n')


        f.write(f'Error rate: {error_rate}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", help="toggle mode: 0 for discrete, 1 for continuous", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    
    # 0 for discrete mode, 1 for continuous mode
    mode = args.mode

    main(mode)
