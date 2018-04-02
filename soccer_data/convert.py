from os.path import abspath, join
from os import listdir
from concurrent.futures import ThreadPoolExecutor
import pickle
import time

from scipy.sparse import coo_matrix
import pandas as pd
import imageio
import numpy as np

from utils import chunks

class DatasetConverter:
    """
    Outputs (image, yolo_label) pair, one at a time.

    # Usage:
    kek = DatasetConverter(csv_path, image_folder).load()
    for pic, label in kek:
        pass
    
    # or
    pic, label = kek.convert(row_number)
    """

    def __init__(self, csv_path, image_folder_path, 
                 image_size=(416, 416), grid_size=(13, 13)):
        self._csv_path = abspath(csv_path)
        self._image_folder_path = abspath(image_folder_path)
        self._image_size = tuple(image_size)
        self._grid_size = tuple(grid_size)

        self._cell_width = image_size[0] / grid_size[0]
        self._cell_height = image_size[1] / grid_size[1]
        self._records = None
        self.num_records = 0
        
    def load(self):
        self._records = pd.read_csv(self._csv_path) \
                          .drop(['fieldBallX', 'fieldBallY', 'fieldBallZ'], axis=1)
        to_keep = self._records['screenshotBallX'].dropna().keys()
        self._records = self._records.iloc[to_keep]
        self.num_records = len(self._records)
        return self

    def convert(self, row_number):
        row = self._records.iloc[row_number]
        image_path = join(self._image_folder_path, row['picName'])

        image = np.array(imageio.imread(image_path))
        label = self._build_label(row)
        return image, label

    def _build_label(self, row):
        label = np.zeros(shape=(*self._grid_size, 5), dtype=float)
        row = row.drop(['picName', 'cameraId']).dropna()

        ball_x = -1
        ball_y = -1
        for obj_x, obj_y in chunks(row.keys(), 2):
            cell_x, cell_y, offset_x, offset_y = self._find_cell(row[obj_x], row[obj_y])
            prob, ball_prob, player_prob = 1, 0, 0
            if obj_x == "screenshotBallX":
                ball_prob = 1
                ball_x = cell_x
                ball_y = cell_y
            else:
                if cell_x == ball_x and cell_y == ball_y:
                    # because ball is the cool kid on the block, 
                    # we want him to keep his cell
                    continue
                player_prob = 1
            label[cell_x, cell_y] = np.array([prob, offset_x, offset_y, 
                                              ball_prob, player_prob], dtype=float)
        return label

    def _find_cell(self, obj_x, obj_y):
        # image_size // cell_width = 13 while the last cell is 12
        obj_x = min(obj_x, self._image_size[0] - 1)
        obj_y = min(obj_y, self._image_size[1] - 1)
        cell_x = int(obj_x // self._cell_width)
        cell_y = int(obj_y // self._cell_height)
        offset_x = (obj_x - cell_x * self._cell_width) / self._cell_width
        offset_y = (obj_y - cell_y * self._cell_height) / self._cell_height
        return cell_x, cell_y, offset_x, offset_y
    
    def __iter__(self):
        for current_row in range(len(self._records)):
            yield self.convert(current_row)

def transform_and_load(labels_path='football-data.csv', images_path='screenshots',
                       test_size=30, storage_folder_path='pickled_data'):
    transform_data(labels_path, images_path, pick_path=storage_folder_path)
    return load_data(test_size, storage_folder_path=storage_folder_path)

def transform_data(labels_path, images_path, size=416, 
                   flush_threshold=16, writing_threads=4, pick_path=''): 
    converter = DatasetConverter(labels_path, images_path, 
                                 image_size=(size, size)).load()
    thread_pool = ThreadPoolExecutor(max_workers=writing_threads)
    X = np.zeros((0, size, size, 3))
    Y = np.zeros((0, size // 32, size // 32, 5))

    start = time.time()
    #TODO converter.num_records а не 200
    for i in range(100):
        x, y = converter.convert(i)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        print(i, X.shape)
        X = np.append(X, x, axis=0)
        Y = np.append(Y, y, axis=0)
        if (i+1) % flush_threshold == 0:
            # copy arrays to avoid race conditions
            thread_pool.submit(serialize, pick_path, np.copy(X), np.copy(Y), i, size)
            X = np.zeros((0, size, size, 3))
            Y = np.zeros((0, size // 32, size // 32, 5))

    print(i)
    if np.count_nonzero(X) or np.count_nonzero(Y):
        thread_pool.submit(serialize, pick_path, np.copy(X), np.copy(Y), i, size)
    thread_pool.shutdown(wait=True)
    print(time.time() - start)

def serialize(path, X, Y, i, size):
    with open(path + '/X-%d.pkl' % i, mode='wb') as x_file, \
         open(path + '/Y-%d.pkl' % i, mode='wb') as y_file:
        pickle.dump(X, x_file)
        pickle.dump(Y, y_file)

def load_data(test_size, storage_folder_path='pickled_data', size=416):
    start = time.time()
    X = np.zeros((0, size, size, 3))
    Y = np.zeros((0, size // 32, size // 32, 5))

    filenames = listdir(storage_folder_path)
    # sort to ensure X and Y go in the same order
    for filename in sorted(filenames):
        with open(join(storage_folder_path, filename), 'rb') as file:
            data = pickle.load(file)
            if filename.startswith('X'):
                X = np.append(X, data, axis=0)
            else:
                Y = np.append(Y, data, axis=0)
            print('loaded %s' % filename)

    X_train = X[0:-test_size, :, :, :]
    X_test = X[X_train.shape[0]:, :, :, :]
    Y_train = Y[0:-test_size, :, :, :]
    Y_test = Y[Y_train.shape[0]:, :, :, :]

    print(time.time() - start)
    return X_train, Y_train, X_test, Y_test
