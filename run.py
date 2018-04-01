import model as M
import mixing as mx
import yolo as Y
import utils
from keras.optimizers import Adam
import numpy as np

def build_graph(model):
    yolo_outputs = Y.yolo_head(model.output)
    return yolo_outputs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    X_train, Y_train, X_test, Y_test = utils.load_data('soccer_data/football-data.csv',
                                                       'soccer_data/screenshots',
                                                       pick_path='soccer_data/pickled_data',
                                                       size=416,
                                                       test_size=30)

    X_train /= 255
    X_test /= 255
    model = mx.train(X_train, Y_train)

# def main():
#     model = M.yolo_darknet19(input_shape=(416, 416, 3), output_depth=5)
#     yolo_outputs = build_graph(model)
#     ball_prob, ball_xy = Y.yolo_eval(yolo_outputs)
    


if __name__ == '__main__':
    main()
