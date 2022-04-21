import tensorflow as tf
import numpy as np
import time

tf.config.list_physical_devices()

display(tf.config.experimental.get_device_details(tf.config.list_physical_devices()[1]))
display(tf.config.experimental.get_device_details(tf.config.list_physical_devices()[2]))


# generate random data

# first computation z1 = W1x1 + b1
x1 = tf.random.uniform(
    shape = (18000,1),
    maxval=500
)

w1 = tf.random.uniform(
    shape = (40000,18000),
    maxval=200
)

b1 = tf.random.uniform(
    shape = (40000,1),
    maxval=50
)

# second computation z2 = W2x2 + b2
x2 = tf.random.uniform(
    shape = (18000,1),
    minval=1000,
    maxval=1200
)

w2 = tf.random.uniform(
    shape = (40000,18000),
    maxval=400
)

b2 = tf.random.uniform(
    shape = (40000,1),
    maxval=75
)

# CPU COMPUTATION 
start = time.time()
z1 = tf.matmul(w1,x1) + b1
a1 = 1 / (1+tf.exp(-z1))
z2 = tf.matmul(w2,x2) + b2
a2 = 1 / (1+tf.exp(-z2))
end = time.time()

comp_time_cpu = end - start
comp_time_cpu

# SINGLE GPU COMPUTATION
start = time.time()
with tf.device('GPU:0'):
    z1 = tf.matmul(w1,x1) + b1
    a1 = 1 / (1+tf.exp(-z1))
    z2 = tf.matmul(w2,x2) + b2
    a2 = 1 / (1+tf.exp(-z2))
end = time.time()

comp_time_gpu = end - start
comp_time_gpu

# 2 GPUS COMPUTATION
start = time.time()
with tf.device('GPU:0'):
    z1 = tf.matmul(w1,x1) + b1
    a1 = 1 / (1+tf.exp(-z1))
with tf.device('GPU:1'):
    z2 = tf.matmul(w2,x2) + b2
    a2 = 1 / (1+tf.exp(-z2))
end = time.time()

comp_time_gpu2 = end - start
comp_time_gpu2

print('#### PRINTING RESULTS ####\n\n')
print(f'Time for computation with CPU: {comp_time_cpu} [s]\n')
print(f'Time for computation with GPU: {comp_time_gpu} [s]\n')
print(f'Time for computation with 2 GPUs: {comp_time_gpu2} [s]\n')