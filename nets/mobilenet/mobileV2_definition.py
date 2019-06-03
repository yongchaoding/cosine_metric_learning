# vim: expandtab:ts=4:sw=4
import tensorflow as tf
import tensorflow.contrib.slim as slim
import nets.mobilenet.mobilenet_v2 as base

def create_network(images, num_classes=None, add_logits=True, reuse=None,
                   create_summaries=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = images
    network, _, networkFirst = base.mobilenet_base(network)
    
    feature1_dim = networkFirst.get_shape().as_list()[-1]
    print("feature1 dimensionality: ", feature1_dim)
    feature1 = slim.flatten(networkFirst)
    print("Feature1 Size: ", network.get_shape().as_list())

    feature_dim = network.get_shape().as_list()[-1]
    print("feature2 dimensionality: ", feature_dim)
    network = slim.flatten(network)
    print("Feature2 Size: ", network.get_shape().as_list())

    network = tf.concat([network, feature1], 1)
    print("Total Feature Size: ", network.get_shape().as_list())

    feature_dim = 128
    network = slim.dropout(network, keep_prob=0.6)
    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,       ## feature_dim
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    # Features in rows, normalize axis 1.
    features = tf.nn.l2_normalize(features, dim=1)

    if add_logits:
        with slim.variable_scope.variable_scope("ball", reuse=reuse):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, int(num_classes)),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (), tf.float32,
                initializer=tf.constant_initializer(0., tf.float32),
                regularizer=slim.l2_regularizer(1e-1))
            if create_summaries:
                tf.summary.scalar("scale", scale)
            scale = tf.nn.softplus(scale)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        logits = scale * tf.matmul(features, weights_normed)
    else:
        logits = None
    return features, logits


def create_network_factory(is_training, num_classes, add_logits,
                           weight_decay=1e-8, reuse=None):

    def factory_fn(image):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = create_network(
                        image, num_classes=num_classes, add_logits=add_logits,
                        reuse=reuse, create_summaries=is_training,
                        weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def preprocess(image, is_training=False, input_is_bgr=False):
    if input_is_bgr:
        image = image[:, :, ::-1]  # BGR to RGB
    ## 取平均值
    image = tf.divide(tf.cast(image, tf.float32), 255.0)
    ## 增益
    if is_training:
        image = tf.image.random_flip_left_right(image)
    return image
