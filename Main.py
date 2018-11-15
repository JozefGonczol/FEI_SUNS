import pickle

import MyLib.Data as Data
import MyLib.Clustering as clu
import MyLib.Neuro as Neuro

# # preparing and normalizing data
# cache_test = Data.cache_images("Test")
# cache_train = Data.cache_images("Training")
# Data.show_cached(cache_test)
# Data.show_cached(cache_train)
# Data.normalize_and_pickle(cache_test, "Test", 255)
# Data.normalize_and_pickle(cache_train, "Training", 255)
#
#
# # Cache and equalize Test/Valid data
# cache_test = Data.cache_pickled("Test")
# Data.show_cached(cache_test)
# cache_test = Data.equalize_classes(cache_test)
# for i, key in enumerate(cache_test):
#     print("{}. {} - {}".format(i, key, len(cache_test[key])))
#
# # Cache and equalize Training data
# cache_train = Data.cache_pickled("Training")
# Data.show_cached(cache_train)
# cache_train = Data.equalize_classes(cache_train)
# for i, key in enumerate(cache_train):
#     print("{}. {} - {}".format(i, key, len(cache_train[key])))
#
# Data.split_data(cache_test, cache_train)
# Data.check_all_pickle()
#
# #clusterring
# clu.get_results("Test")
# clu.k_means()
# clu.db_scan()

all_data = pickle.load(open('data/all_data.pickle', 'rb'))

Neuro.mlp(all_data, 50, 10)
Neuro.mlp(all_data, 100, 10)
Neuro.mlp(all_data, 200, 10)
Neuro.mlp(all_data, 1000, 10)
Neuro.mlp(all_data, 5000, 10)


