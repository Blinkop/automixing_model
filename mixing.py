import keras.backend as K
import tensorflow as tf
import yolo as Y
import model as M
import utils
from soccer_data.convert import DatasetConverter
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

def create_data(chunk_size, csv_path='soccer_data/football-data.csv', screenshots_path='soccer_data/screenshots', output_name='data.h5'):
    converter = DatasetConverter(csv_path, screenshots_path).load()
    gen = converter.get_chunks(chunk_size)
    utils.create_hdf5_data(gen, 'data.h5', chunk_size=chunk_size)

def train(data_file):
    EPOCHS_PERIOD = 1
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 41128 // BATCH_SIZE
    model = M.yolo_darknet19(input_shape=(416, 416, 3), output_depth=5)
    model.load_weights('darknet19_weights_full.h5')
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

    model.fit_generator(generator=utils.BatchGenerator('data.h5', BATCH_SIZE),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=128,
                        verbose=1,
                        shuffle=True,
                        max_queue_size=16,
                        validation_data=utils.load_data('data.h5', 300, 4),
                        callbacks=[checkpoint, tensorboard],
                        initial_epoch=3)

    return model
    
