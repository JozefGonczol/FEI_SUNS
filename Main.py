import MyLib.Data as Data

# preparing and normalizing data
cache_test = Data.cache_images("Test")
cache_train = Data.cache_images("Training")
Data.show_cached(cache_test)
Data.show_cached_2(cache_train)
Data.normalize_and_pickle(cache_test, "Test", 255)
Data.normalize_and_pickle(cache_train, "Training", 255)

# Cache and equalize Test/Valid data
cache_test = Data.cache_pickled("Test")
Data.show_cached(cache_test)
cache_test = Data.equalize_classes(cache_test)
for i, key in enumerate(cache_test):
    print("{}. {} - {}".format(i, key, len(cache_test[key])))

# Cache and equalize Training data
cache_train = Data.cache_pickled("Training")
Data.show_cached_2(cache_train)
cache_train = Data.equalize_classes(cache_train)
for i, key in enumerate(cache_train):
    print("{}. {} - {}".format(i, key, len(cache_train[key])))

Data.split_data(cache_test, cache_train)

Data.check_all_pickle()
