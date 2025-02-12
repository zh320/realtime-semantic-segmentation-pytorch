dataset_hub = {}


def register_dataset(dataset_class):
    dataset_hub[dataset_class.__name__.lower()] = dataset_class
    return dataset_class