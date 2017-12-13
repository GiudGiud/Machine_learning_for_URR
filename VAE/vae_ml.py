import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data')
import scipy.io
mat = scipy.io.loadmat( '/Applications/Fall18Courses/6.867/project/complex_model/reformat_complex.mat' )

data = mat['results']
## data = mat['results_np']
res_dim = 160;
## full_data = ( np.real( data[ :, 0, 0:res_dim] ), np.imag( data[ :, 0, 0:res_dim] ), np.real( data[ :, 1, 0:res_dim] ), np.imag( data[ :, 1, 0:res_dim] ) )
## full_data_mat = np.concatenate( full_data, axis = 1 )
full_data_mat = data[ :, 0:res_dim ]
print( len( full_data_mat[ 0 ] ) )


tf.reset_default_graph()

batch_size = 64
sample_size = 10000

X_in = tf.placeholder(dtype=tf.float32, shape=[ None, res_dim ], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[ None, res_dim ], name='Y')
## Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 16
layer_num = 2


def res_batch( vec, seed, batch_size ):
    eff_start = ( seed*batch_size ) % len(vec)
    eff_end = ( (seed+1)*batch_size ) % len(vec)
    if eff_start < eff_end :
        return vec[eff_start:eff_end, : ]
    return np.concatenate( ( vec[ eff_start: len(vec), :], vec[ 0:eff_end, : ] ), axis = 0)

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def relu( x ):
    return tf.maximum(x, 0 )

activation = lrelu
learning_rate = .0005

def encoder(X_in, keep_prob):
    ##activation = sigmoid
    with tf.variable_scope("encoder", reuse=None):
        if layer_num == 0:
            x = X_in
        if layer_num == 1:
            x = tf.layers.dense( X_in, units = n_latent, activation = activation )
        if layer_num == 2:
            x = tf.layers.dense( X_in, units = n_latent, activation = activation )
            x = tf.layers.dense( x, units = n_latent, activation = activation )
    
        mn = tf.layers.dense(x, units=n_latent, activation = activation )
        sd       = 0.5 * tf.layers.dense(x, units=n_latent, activation = activation )
        epsilon = tf.random_normal(tf.stack([tf.shape(X_in)[0], n_latent]))
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        if layer_num == 0:
            x = sampled_z
        if layer_num == 1:
            x = tf.layers.dense( sampled_z, units = n_latent, activation = activation )
        if layer_num == 2:
            x = tf.layers.dense( sampled_z, units = n_latent, activation = activation )
            x = tf.layers.dense( x, units = n_latent, activation = activation )
        x = tf.layers.dense( x, units = res_dim, activation = activation )
        ## x = tf.layers.dense( x, units = res_dim, activation = lrelu )
        img = tf.reshape( x, shape = [-1, res_dim] )
        return img



sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

## unreshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(dec, Y), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 300000
meta_data = np.zeros( shape = ( epochs/200, 4 ) )
count = 0
for i in range(epochs):
    batch = res_batch( full_data_mat, i, batch_size )
    scipy.io.savemat('/Applications/Fall18Courses/6.867/project/complex_model/debug1.mat', mdict = {'results_np': batch})
    ## batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
    ## print( batch[0, 41] )
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
        meta_data[count, 0 ] = i
        meta_data[count, 1 ] = ls
        meta_data[count, 2 ] = np.mean( i_ls )
        meta_data[count, 3 ] = np.mean(d_ls)
        count = count+1
        print(i, ls, np.mean(i_ls), np.mean(d_ls))

scipy.io.savemat('/Applications/Fall18Courses/6.867/project/complex_model/losses.mat', mdict = {'losses': meta_data})
randoms = [np.random.normal(0, 1, n_latent) for _ in range(sample_size)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
scipy.io.savemat('/Applications/Fall18Courses/6.867/project/complex_model/test.mat', mdict = {'results_np': imgs})

