import numpy as np
import tensorflow as tf
import keras.backend as K

def predict(sess, data, yolo_output, model):
    bP, bX, bY = yolo_output

    out_bP, out_bX, out_bY = sess.run([bP, bX, bY], feed_dict={model.input : data, K.learning_phase() : 0})
    m = out_bP.shape[0]

    for i in range(m):
        print('Image_' + str(i) + ' : with probability of '
             + str(out_bP[i])
             + 'ball coords are (' + str(out_bX[i]) + str(out_bY[i]) + ')')
    
def yolo_eval(yolo_output):
    """
    Returns dictionary with top-1 ball probability
    and top-11 players probability
    ball : (prob, x, y)
    player1 : (prob, x, y)
    ...
    player11 : (prob, x, y)
    """
    conf, xy, classes = yolo_output
    conf_shape = K.shape(conf)
    m, grid_size = conf_shape[0], conf_shape[1]
    grid_pix_size = 32

    class_probs = tf.multiply(conf, classes) # conf_object * class_conf          (M, GRID_SIZE, GRID_SIZE, 2)
    ball_probs = K.reshape(class_probs[..., 0], (m, grid_size * grid_size)) #    (M, GRID_SIZE * GRID_SIZE)
    players_probs = K.reshape(class_probs[..., 1], (m, grid_size * grid_size)) # (M, GRID_SIZE * GRID_SIZE)    

    _ball_idx = K.cast(K.argmax(ball_probs, axis=-1), 'int32') # (M, )
    _ball_idx = tf.expand_dims(_ball_idx, axis=-1)             # (M, 1)
    ball_idx = tf.concat([tf.expand_dims(tf.range(m), axis=-1), _ball_idx // grid_size, _ball_idx % grid_size], axis=1) # (M, 3)

    ball_x = tf.gather_nd(xy[..., 0], ball_idx, name='ball_x_index')
    ball_y = tf.gather_nd(xy[..., 1], ball_idx, name='ball_y_index')

    ball_x = grid_pix_size * (K.cast(ball_idx[:, 1], 'float32') + ball_x)
    ball_y = grid_pix_size * (K.cast(ball_idx[:, 2], 'float32') + ball_y)
    ball_prob = tf.gather_nd(class_probs[..., 0], ball_idx, name='ball_prob')

    return (ball_prob, ball_x, ball_y)


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
