import tensorflow as tf
from tensorflow import keras
import sys
from os import mkdir, path
from tensorflow.python.tools import freeze_graph

# admin
model_name = sys.argv[1]
if not model_name.endswith('.h5'):
	print('Requires .h5 model')
	exit()
ndir = model_name[:-3]
mkdir(ndir)

# open model
keras.backend.set_learning_phase(0)
model =  keras.models.load_model(model_name)
print(model.summary())

# resave as graph
saver = tf.train.Saver(tf.global_variables())
sess = tf.keras.backend.get_session()
sess.run(tf.local_variables_initializer())
saver.save(sess, ndir+'/tf-ncs')
tf.train.write_graph(sess.graph, ndir, 'tf-ncs.pb', False)

# freeze graph
freeze_graph.freeze_graph(
    input_graph=path.join(ndir, 'tf-ncs.pb'),
    input_checkpoint=path.join(ndir, 'tf-ncs'),
    output_graph=path.join(ndir, 'tf-ncs-froze.pb'),
    output_node_names='output',
    input_binary=True,
    input_saver='',
    restore_op_name='save/restore_all',
    filename_tensor_name='save/Const:0',
    clear_devices=True,
    initializer_nodes=''
    )
'''
    path.join(ndir, 'tf-ncs.pb'),
    "", True, path.join(ndir, 'tf-ncs'),
    'output', "", "", 'tf-ncs-froze.pb', True, "")
'''
fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()