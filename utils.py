import numpy as np
import h5py
from soccer_data.convert import DatasetConverter
from soccer_data.convert import transform_and_load
from keras.utils.io_utils import HDF5Matrix

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]
    def reset(self):
        self.offset = 4

def load_yolov2_weights(model, path):
    reader = WeightReader(path)
    reader.reset()
    num_layers = 23

    for i in range(1, num_layers):
        conv_layer = model.get_layer('conv_' + str(i))
        print('Reading layer ', i, ' weights...')

        if i < num_layers:
            bn_layer = model.get_layer('bn_' + str(i))

            size = np.prod(bn_layer.get_weights()[0].shape)

            beta  = reader.read_bytes(size)
            gamma = reader.read_bytes(size)
            mean  = reader.read_bytes(size)
            var   = reader.read_bytes(size)

            weights = bn_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = reader.read_bytes(p.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    print('Generating random weights for layer 23...')

    yolo_layer = model.layers[-1]
    weights = yolo_layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape) / (13 * 13)
    new_bias = np.random.normal(size=weights[1].shape) / (13 * 13)

    yolo_layer.set_weights([new_kernel, new_bias])

    return model

def create_hdf5_data(filename, size=(416, 416, 3), chunk_size=(32, 416, 416, 3)):
    with h5py.File(filename, 'w') as file:
        maxshape = (None, ) + size
        dataset = file.create_dataset('images', shape=chunk_size, maxshape=maxshape, chunks=chunk_size)

        

def normalize_data(data):
    return data / 255

def load_data(data_path, start, num_examples):
    X = HDF5Matrix(datapath=data_path, dataset='images', start=start, end=start+num_examples, normalizer=normalize_data)
    Y = HDF5Matrix(datapath=data_path, dataset='labels', start=start, end=start+num_examples)

    return X[:], Y[:]
