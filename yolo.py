import numpy as np
import tensorflow as tf
import keras.backend as K

def predict(sess, data):
    pass

def yolo_eval(yolo_output):
    conf, xy, classes = yolo_output
    conf_shape = K.shape(conf)
    m, grid_size = conf_shape[0], conf_shape[1]
    grid_pix_size = 32

    class_probs = tf.multiply(conf, classes)
    ball_probs = K.reshape(class_probs[..., 0], (m, grid_size * grid_size)) # Mx(GRID_SIZE*GRID_SIZE)
    players_probs = K.reshape(class_probs[..., 1], (m, grid_size * grid_size)) # Mx(GRID_SIZE*GRID_SIZE)

    

    ball_index = K.argmax(ball_probs, axis=-1) # (M, )
    ball_x_index = xy[:, ball_index[0], 0]
    ball_y_index = xy[:, ball_index[0], 1]
    print(K.int_shape(ball_x_index))
    print(K.int_shape(ball_y_index))

    # ball_x = grid_pix_size * (ball_x_index + xy[ball_x_index, ball_y_index, 0])
    # ball_y = grid_pix_size * (ball_y_index + xy[ball_x_index, ball_y_index, 1])
    # ball_xy = (ball_x, ball_y)
    # ball_prob = class_probs[ball_x_index, ball_y_index, 0]

    # return (ball_prob, ball_xy)
    return 1, 1


def yolo_head(input):
    obj_prob = K.sigmoid(input[..., 0])
    box_xy = K.sigmoid(input[..., 1:3])
    classes_probs = K.softmax(input[..., 3:])

    obj_prob = K.expand_dims(obj_prob, axis=-1)

    return (obj_prob, box_xy, classes_probs)

def yolo_loss(y_ground, y_pred):
    (pred_conf, pred_xy, pred_classes) = yolo_head(y_pred)
    reverse = tf.constant(-1.0)
    conf_shape = K.shape(pred_conf)

    lambda_coord = 5
    lambda_class = 1
    lambda_noobj = 0.5
    lambda_conf  = 1

    obj_mask = y_ground[..., 0]
    noobj_mask = tf.multiply(reverse, obj_mask + reverse)

    true_conf = y_ground[..., 0]
    true_xy = y_ground[..., 1:3]
    true_classes = y_ground[..., 3:]

    obj_mask = K.expand_dims(obj_mask, axis=-1)
    noobj_mask = K.expand_dims(noobj_mask, axis=-1)
    true_conf = K.expand_dims(true_conf, axis=-1)

    true_probs = tf.multiply(true_conf, true_classes)
    pred_probs = tf.multiply(pred_conf, pred_classes)

    loss_xy = tf.multiply(obj_mask, tf.square(true_xy - pred_xy))
    loss_classes = tf.multiply(obj_mask, tf.square(true_classes - pred_classes))
    loss_noobj_classes = tf.multiply(noobj_mask, tf.square(true_classes - pred_classes))
    loss_conf = tf.multiply(obj_mask, tf.square(true_probs - pred_probs))

    total_loss = 0.5 * (lambda_coord * K.sum(loss_xy)
                        + lambda_class * K.sum(loss_classes)
                        + lambda_noobj * K.sum(loss_noobj_classes)
                        + lambda_conf * K.sum(loss_conf)) / K.cast(conf_shape[0], 'float32')

    return total_loss
