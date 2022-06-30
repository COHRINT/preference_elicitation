import pickle as pk
import sys
import random
from matplotlib import image
import numpy as np


############################################################
### IO
############################################################
def disp_to_term(msg):
    sys.stdout.write(msg + '\r')
    sys.stdout.flush()


def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print("Pickle file cannot be opened.")
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print('load_pickle failed once, trying again')
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious


def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()


def sample_image(filename, save_path, set_name, n_samples, sample_size):
    """Randomly samples and saves images to a given directory."""
    data = image.imread(filename)
    x_avail = data.shape[0] - sample_size - 1
    y_avail = data.shape[1] - sample_size - 1
    x_set = range(0, x_avail)
    y_set = range(0, y_avail)

    for s in range(0, n_samples+1):
        x_sample = random.choice(x_set)
        y_sample = random.choice(y_set)
        new_img = data[x_sample:x_sample + sample_size, y_sample:y_sample + sample_size]
        path = str(save_path + set_name + str(s) + ".jpeg")
        # print(path)
        image.imsave(path, new_img)

