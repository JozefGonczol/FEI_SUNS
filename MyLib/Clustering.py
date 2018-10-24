import os
import pickle
import ntpath
import numpy as np
import matplotlib.pyplot as plt
import cv2



def load_fruit(pickle_path):
    fruits = pickle.load(open(pickle_path, 'rb'))
    category = ntpath.basename(pickle_path).split('.')[0]
    return category, fruits


def analyze_fruit(pickle_path):
    name, samples = load_fruit(pickle_path)
    mean = get_mean(samples)
    mOK = 0
    fOK = 0
    size = len(samples) // 4
    samples = samples[:size]
    test = 0
    print("Analyzing class {}. {} samples".format(name, size))
    for key, fruit1 in enumerate(samples):
        for fruit2 in samples[key+1:]:
            dist = np.linalg.norm(fruit1 - fruit2)
            if dist < 10:
                fOK += 1
            test += 1

    for fruit in samples:
        dist = np.linalg.norm(mean - fruit)
        if dist < 25:
            mOK += 1

    result1 = (fOK*100)/test
    result2 = (mOK*100)/size
    print("Done. Similar are {}%, {}% close to mean".format(result1,result2))
    return name, result1, result2


def get_mean(fruits):
    cnt = len(fruits)
    avg = 0
    for im in fruits:
        avg += im
    return avg / cnt

def get_results(set):

    ffStat = {}
    mfStat = {}

    for dir in os.listdir("data/{}".format(set)):
        if os.path.isdir("data/{}/{}".format(set, dir)):
            continue
        name, result1, result2 = analyze_fruit("data/{}/{}".format(set, dir))
        ffStat[name] = result1
        mfStat[name] = result2
    plt.bar(range(len(ffStat)), list(ffStat.values()))
    plt.xticks(range(len(ffStat)), list(ffStat.keys()), rotation='vertical')
    plt.show()

    plt.bar(range(len(mfStat)), list(mfStat.values()))
    plt.xticks(range(len(mfStat)), list(mfStat.keys()), rotation='vertical')
    plt.show()


def k_means():
    all_data = pickle.load(open('data/all_data.pickle', 'rb'))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.3)
    flags = cv2.KMEANS_RANDOM_CENTERS

    test_data = all_data['test_data']
    test_labels = all_data['test_labels']

    compactness, labels, centers = cv2.kmeans(test_data, 10, None, criteria, 10, flags)

