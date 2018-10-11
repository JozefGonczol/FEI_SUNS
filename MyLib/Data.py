import os
import os.path
import pickle
import urllib.request
import urllib.error
import cv2
import random
from random import randint
import numpy
import matplotlib.pyplot as plt


def mk_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")


def download_data(url):
    name = url.split('/')[-1]
    mk_data_dir()
    if not os.path.exists("data/" + name):
        print("Downloading data")
        try:
            urllib.request.urlretrieve(url, "data/" + name)
            print("Data downloaded")
        except urllib.error:
            print("¯\_(ツ)_/¯ Downloading from " + url + " failed")

    else:
        print("File " + name + " is already downloaded")
    return name


def cache_images(folder):
    all_fruits = {}
    print("Caching {} data".format(folder))
    for dir in os.listdir("data/{}".format(folder)):
        if os.path.isfile("data/{}/{}".format(folder, dir)):
            continue
        jedno_ovocie = list()
        fruit_name = dir.split(" ")[0]
        for image_name in os.listdir("data/{}/{}".format(folder, dir)):
            img = cv2.imread("data/{}/{}/{}".format(folder, dir, image_name))
            imgBGV = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            jedno_ovocie.append(img)
        if fruit_name in all_fruits:
            all_fruits[fruit_name].extend(jedno_ovocie)
        else:
            all_fruits[fruit_name] = jedno_ovocie

    print("From {} was cached:".format(folder))
    for i, key in enumerate(all_fruits):
        print("{}. name:{} size:{}".format(i, key, len(all_fruits[key])))
    return all_fruits


def show_cached(all_fruits):
    plt.axes(frameon=False)
    num_fruits = len(all_fruits.keys())
    k = 1
    for fruit in all_fruits:
        samples = random.sample(all_fruits[fruit], 3)
        for img in samples:
            plt.subplot(num_fruits, 3, k)
            plt.axis('off')
            k = k + 1
            plt.imshow(img)
    plt.show()


def show_cached_2(all_fruits):
    num_fruits = len(all_fruits.keys())
    k = 1
    for fruit in all_fruits:
        samples = random.sample(all_fruits[fruit], 3)
        for img in samples:
            cv2.imshow('Image', img)
            cv2.waitKey(5)


def normalize_and_pickle(all_fruits, folder, depth):
    for fruit in all_fruits:
        print("Pickling {}/{}".format(folder, fruit))
        data = []
        if os.path.isfile("data/{}/{}.pickle".format(folder, fruit)):
            print("Skipping...")
            continue
        for image in all_fruits[fruit]:
            data.append(image / depth - 0.5)
        random.shuffle(data)
        pickle.dump(data, open("data/{}/{}.pickle".format(folder, fruit), 'wb'))


def cache_pickled(folder):
    all_fruits = {}
    print("Caching pickled {} data".format(folder))
    for dir in os.listdir("data/{}".format(folder)):
        fruit_name = dir.split('.')[0]
        if os.path.isdir("data/{}/{}".format(folder, dir)):
            continue
        fruits = pickle.load(open("data/{}/{}".format(folder, dir), 'rb'))
        all_fruits[fruit_name] = fruits
    return all_fruits


def equalize_classes(all_fruits):
    min = 999999
    for fruit in all_fruits:
        cnt = len(all_fruits[fruit])
        if cnt < min:
            min = cnt

    for fruit in all_fruits:
        all_fruits[fruit] = random.sample(all_fruits[fruit], min)

    return all_fruits


def split_data(test_data, train_data):
    train_labels = []
    train_data_list = []
    test_labels = []
    test_data_list = []
    valid_labels = []
    valid_data_list = []

    for key in train_data:
        for data in train_data[key]:
            train_labels.append(key)
            train_data_list.append(data)

    for key in test_data:
        for data in test_data[key]:
            test_labels.append(key)
            test_data_list.append(data)

    half = len(test_labels) // 2

    test_labels = numpy.array(test_labels)
    test_data_list = numpy.array(test_data_list)
    permutation = numpy.random.permutation(len(test_labels))
    test_labels = test_labels[permutation]
    test_data_list = test_data_list[permutation]

    valid_labels = test_labels[half:]
    valid_data_list = test_data_list[half:]

    test_labels = test_labels[:half]
    test_data_list = test_data_list[:half]

    # shufle train data
    train_labels = numpy.array(train_labels)
    train_data_list = numpy.array(train_data_list)
    permutation = numpy.random.permutation(len(test_labels))
    train_labels = train_labels[permutation]
    train_data_list = train_data_list[permutation]

    all_pickle = {
        'train_data': train_data_list,
        'train_labels': train_labels,
        'test_data': test_data_list,
        'test_labels': test_labels,
        'valid_data': valid_data_list,
        'valid_labels': valid_labels,
    }

    pickle.dump(all_pickle, open("data/all_data.pickle", 'wb'))

    print("Pickling")


def check_all_pickle():
    data = pickle.load(open("data/all_data.pickle", 'rb'))

    test = randint(0, len(data['test_labels']))
    train = randint(0, len(data['train_labels']))
    valid = randint(0, len(data['valid_labels']))

    cv2.imshow(data['test_labels'][test], data['test_data'][test])
    cv2.waitKey()

    cv2.imshow(data['train_labels'][train], data['train_data'][train])
    cv2.waitKey()

    cv2.imshow(data['valid_labels'][valid], data['valid_data'][valid])
    cv2.waitKey()
