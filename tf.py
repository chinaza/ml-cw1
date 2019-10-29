import numpy as np
import tensorflow as tf

X_1 = tf.placeholder(tf.float32, name = "X_1")
X_2 = tf.placeholder(tf.float32, name = "X_2")
Y = tf.add(X_1, X_2, name = "Y")
with tf.Session() as session:
 result = session.run(Y, feed_dict={X_1:[1,2,3], X_2:[4,5,6]})
 print(result)
