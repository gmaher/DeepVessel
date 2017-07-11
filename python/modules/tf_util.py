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
        b = tf.Variable(tf.ones([nfilters])*init, name='b')

        h = tf.nn.convolution(x,W,padding='SAME',strides=[1,1,1], name='h')+b

        a = activation(h)

        return a

def repeat(x,axis,repeat):
    s = x.get_shape().as_list()
    splits = tf.split(value=x,num_or_size_splits=s[axis],axis=axis)
    rep = [s for s in splits for _ in range(repeat)]
    return tf.concat(rep,axis)

def resize_tensor(x,scales=[1,2,2,2,1]):
    out = x
    for i in range(1,len(scales)):
        out = repeat(out,i,scales[i])
    return out

def upsample3D(x, scope='upsample'):
    with tf.variable_scope(scope):
        o = resize_tensor(x)
        return o

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

def unetBlock(x,nfilters=32,scope='unet3d',init=1e-3,activation=tf.nn.relu):
    with tf.variable_scope(scope):
        o1 = conv3D(x,nfilters=nfilters,init=init,activation=activation)
        o2 = conv3D(o1,nfilters=2*nfilters,init=init,activation=activation)
        o3 = tf.nn.pool(o2,[2,2,2],strides=[2,2,2],pooling_type='MAX',padding='SAME',name='pool')
    return o3,o2

def unetUpsampleBlock(x,y,activation=tf.nn.relu,init=1e-3,nfilters=32,scope='unet3dupsample'):
    with tf.variable_scope(scope):
        o = upsample3D(y)
        o = tf.concat([o,x],axis=4)
        o = conv3D(o,nfilters=nfilters,init=init,activation=activation)
        o = conv3D(o,nfilters=nfilters,init=init,activation=activation)
        return o

def UNET3D(x,activation=tf.nn.relu,nfilters=32,init=1e-3):
    o1_down,o1 = unetBlock(x,nfilters=nfilters,scope='layer1',init=init,activation=activation)
    print o1,o1_down
    o2_down,o2 = unetBlock(o1_down,nfilters=2*nfilters,scope='layer2',init=init,activation=activation)
    print o2,o2_down
    o3_down,o3 = unetBlock(o2_down,nfilters=4*nfilters,scope='layer3',init=init,activation=activation)
    print o3,o3_down
    o4_down,o4 = unetBlock(o3_down,nfilters=8*nfilters,scope='layer4',init=init,activation=activation)
    print o4,o4_down

    a_3 = unetUpsampleBlock(o3,o4,nfilters=8*nfilters,scope='o_layer3',init=init,activation=activation)
    print a_3
    a_2 = unetUpsampleBlock(o2,a_3,nfilters=4*nfilters,scope='o_layer2',init=init,activation=activation)
    print a_2
    a_1 = unetUpsampleBlock(o1,a_2,nfilters=2*nfilters,scope='o_layer1',init=init,activation=activation)
    print a_1
    yhat = conv3D(a_1,tf.identity,shape=[1,1,1],nfilters=1,scope='yhat',init=init)
    print yhat
    yclass = tf.sigmoid(yhat)
    print yclass
    return yhat,yclass
