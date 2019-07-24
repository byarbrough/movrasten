from mvnc import mvncapi as mvnc
# get the first NCS device by its name.  For this program we will always open the first NCS device.
devices = mvnc.enumerate_devices()
# get the first NCS device by its name.  For this program we will always open the first NCS device.
dev = mvnc.Device(devices[0])
print('found', dev)
# Read a compiled network graph from file (set the graph_filepath correctly for your graph file)
with open('/home/bcy/Documents/ai/movrasten/models/mod-ncs/belg_traf.graph', mode='rb') as f:
    graphFileBuff = f.read()

graph = mvnc.Graph('/home/bcy/Documents/ai/movrasten/models/mod-ncs/belg_traf.graph')

# Allocate the graph on the device and create input and output Fifos
in_fifo, out_fifo = graph.allocate_with_fifos(dev, graphFileBuff)

# Write the input to the input_fifo buffer and queue an inference in one call
graph.queue_inference_with_fifo_elem(in_fifo, out_fifo, input_img.astype('float32'), 'user object')

# Read the result to the output Fifo
output, userobj = out_fifo.read_elem()
print('Predicted:',output.argmax())