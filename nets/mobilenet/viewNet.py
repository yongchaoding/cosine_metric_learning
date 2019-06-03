import cv2
import numpy as np
from mobilenet_v2 import mobilenet_base
import tensorflow as tf

image = tf.image.decode_jpeg(tf.read_file("test.jpg"))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (128, 64))


net, end = mobilenet_base(images)
# print("========== net ============")
# for part in net:
print(end)
print("========== end ============")
for part in end:
    print(part)
x = end["Predictions"]
print(x)
