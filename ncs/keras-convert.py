import tensorflow as tf
from tensorflow import keras
import sys

name = sys.argv[1]

keras.backend.set_learning_phase(0)
model =  keras.models.load_model(name)
print(model.summary)

saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
saver.save(sess, "/home/bcy/data/tf_model")

fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()