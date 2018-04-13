import model as M
import mixing as mx
import yolo as Y
import utils
import keras.backend as K
from keras.optimizers import Adam
import numpy as np

from soccer_data.convert import DatasetConverter
import h5py

def main():
    model = M.yolo_darknet19(input_shape=(416, 416, 3), output_depth=5)
    yolo_outputs = Y.yolo_head(model.output)
    to_evaluate = Y.yolo_eval(yolo_outputs)

    sess = K.get_session()
    Y.predict(sess, data, to_evaluate)

if __name__ == '__main__':
    main()
