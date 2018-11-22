import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm

def mlp(data, data_size, hidden_lay_number):
    dim = len(data['train_data'][0])
    hidden_lay_size = data_size
    hidden_layers = ()

    for i in range(0, hidden_lay_number):
        hidden_layers += (hidden_lay_size,)
        i += 1

    train_data = np.reshape(data['train_data'][0:data_size], [len(data['train_data'][0:data_size]), dim * dim *3])
    train_labels = data['train_labels'][0:data_size]
    test_data = np.reshape(data['test_data'], [len(data['test_data']), dim * dim * 3])
    test_labels = data['test_labels']

    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=hidden_layers, random_state=1)
    clf.fit(train_data, train_labels)

    labels = clf.predict(test_data)
    correct = np.count_nonzero(labels == test_labels)

    print(correct / len(labels))


def my_svm(data, data_size, arch):
    dim = len(data['train_data'][0])

    train_data = np.reshape(data['train_data'][0:data_size], [len(data['train_data'][0:data_size]), dim * dim * 3])
    train_labels = data['train_labels'][0:data_size]
    test_data = np.reshape(data['test_data'], [len(data['test_data']), dim * dim * 3])
    test_labels = data['test_labels']

    clf = svm.SVC(kernel=arch)
    clf.fit(train_data, train_labels)
    labels = clf.predict(test_data)
    correct = np.count_nonzero(labels == test_labels)

    print(correct / len(labels))
