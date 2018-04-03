import keras.backend as K
import tensorflow as tf
import yolo as Y
import model as M
import utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

def train(data_file, epochs):
    EPOCHS_PERIOD = 1
    NUM_EXAMPLES = 1024
    model = M.yolo_darknet19(input_shape=(416, 416, 3), output_depth=5)
    model = utils.load_yolov2_weights(model, 'yolov2.weights') # DANGEROUS
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss=Y.yolo_loss)

    checkpoint = ModelCheckpoint('darknet19_weights.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 period=EPOCHS_PERIOD,
                                 save_weights_only=True,
                                 save_best_only=True)

    tensorboard = TensorBoard('logs',
                              histogram_freq=EPOCHS_PERIOD,
                              write_graph=True,
                              write_images=True,
                              write_grads=True)

    for e in range(epochs):
        print('Epoch ', e, ' ...')
        for X_train, Y_train in utils.getBatchGenerator(data_file, NUM_EXAMPLES):
            model.fit(x=X_train, y=Y_train, epochs=1, batch_size=16, 
                      callbacks=[checkpoint, tensorboard],
                      shuffle=True, validation_split=0.05)

    return model
    
