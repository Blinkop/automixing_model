import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from soccer_data.convert import DatasetConverter, transform_and_load
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

def draw_result(img, x, y):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((x - 5, y - 5), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patches(rect)
    plt.show()

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
            kernel = reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
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

def create_hdf5_data(chunks_generator, filename, X_size=(416, 416, 3), 
                     Y_size=(13, 13, 5), chunk_size=32):
    with h5py.File(filename, 'w') as file:
        X_maxshape = (None, ) + X_size
        Y_maxshape = (None, ) + Y_size
        Y_chunk_size = (chunk_size, ) + Y_size
        X_chunk_size = (chunk_size, ) + X_size

        dataset_images = file.create_dataset('images', compression='gzip', shape=X_chunk_size, maxshape=X_maxshape, chunks=X_chunk_size)
        dataset_labels = file.create_dataset('labels', compression='gzip', shape=Y_chunk_size, maxshape=Y_maxshape, chunks=Y_chunk_size)

        chunk_offset = 0
        
        for x_chunk, y_chunk in chunks_generator:
            print(chunk_offset)
            dataset_images.resize(chunk_offset + x_chunk.shape[0], axis=0)
            dataset_labels.resize(chunk_offset + y_chunk.shape[0], axis=0)

            dataset_images[chunk_offset:] = x_chunk
            dataset_labels[chunk_offset:] = y_chunk

            chunk_offset += chunk_size

def normalize_data(data):
    return data / 255

def load_data(data_path, start, num_examples):
    X = HDF5Matrix(datapath=data_path, dataset='images', start=start, end=start+num_examples, normalizer=normalize_data)
    Y = HDF5Matrix(datapath=data_path, dataset='labels', start=start, end=start+num_examples)

    return (X[:], Y[:])

def BatchGenerator(data_path, batch_size):
    h5f = h5py.File(data_path, 'r')
    length, size, _, channels = h5f['images'].shape
    h5f.close()

    while True:
        file = h5py.File(data_path, 'r')
        num_entries = 0

        while num_entries < (length - batch_size):
            X = file['images'][num_entries : num_entries + batch_size] / 255
            Y = file['labels'][num_entries : num_entries + batch_size]

            num_entries += batch_size
            yield (X, Y)
        
        file.close()
