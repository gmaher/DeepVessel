import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def get_batch(X,Y,N,n=32, y_index='all'):
    inds = np.random.choice(range(N),size=n, replace=False)
    x = X[inds,:,:,:]
    y = Y[inds,:,:,:]
    if y_index != 'all':
        y = y[:,:,:,:,y_index]
        y = y[:,:,:,:,np.newaxis]
    x = x[:,:,:,:,np.newaxis]
    return x,y

def conv3D(x, activation=tf.nn.relu, shape=[3,3,3],nfilters=32, init=1e-3, scope='conv3d', reuse=False):
    #batch,W,H,D,C
    with tf.variable_scope(scope,reuse=reuse):
        s = x.get_shape()
        shape = shape +[int(s[4]),nfilters]
        W = tf.Variable(tf.random_normal(shape=shape,stddev=init),
            name='W')
        b = tf.Variable(tf.ones([nfilters]), name='b')

        h = tf.nn.convolution(x,W,padding='SAME',strides=[1,1,1], name='h')+b

        a = activation(h, name='a')

        return a

def conv3D_N(x, activation=tf.nn.relu, shape=[3,3,3],nfilters=32,init=1e-3,scope='conv3dn',N=2):
    o = x
    for i in range(N):
        s = scope +'_'+str(i)
        o = conv3D(o,activation,shape,nfilters,init,s)

    return o

def resnet_conv3D_N(x, activation=tf.nn.relu, shape=[3,3,3],nfilters=32,init=1e-3,scope='conv3dn',N=2):
    o = x
    for i in range(N-1):
        s = scope +'_'+str(i)
        o = conv3D(o,activation,shape,nfilters,init,s)

    s = scope +'_'+str(N)
    o = conv3D(o,tf.identity,shape,nfilters,init,s)
    y = activation(o+x)

    return y

def downsample3D_2(x, scope='downsample3d', reuse=False):
    pass
