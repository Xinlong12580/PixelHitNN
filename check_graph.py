import tensorflow as tf

graph_def = tf.compat.v1.GraphDef()

with tf.io.gfile.GFile("checkpoints/L1U_y_center.keras.pb", "rb") as f:
    graph_def.ParseFromString(f.read())

for node in graph_def.node:
    print(node.name, node.op)
