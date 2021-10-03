from loaders import RolloutObservationDataset


dataset_train = RolloutObservationDataset('datasets/mpe', lambda x:x, train=True)
dataset_train.load_next_buffer()
#print(dataset_train._buffer)
print(type(dataset_train._buffer))
print(len(dataset_train._buffer))
#print(dataset_train._buffer[0])
print(type(dataset_train._buffer[0]))
#print(dataset_train._buffer[0]['observations'])
print(type(dataset_train._buffer[0]['observations'][0]))
print(dataset_train._buffer[0]['observations'][0].shape)
print(dataset_train._buffer[0]['observations'][1].shape)
print(dataset_train._buffer[0]['observations'][2].shape)