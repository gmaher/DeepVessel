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

        a = activation(h, name='a')

        return a

def upsample3D(x, scope='upsample'):
    with tf.variable_scope(scope):
        s = x.get_shape()
        output_shape = [int(s[0]),2*int(s[1]),
            2*int(s[2]),2*int(s[3]),int(s[4])]
        W = tf.ones([2,2,2,s[4],s[4]],name='W')
        o = tf.nn.conv3d_transpose(x,W,output_shape=output_shape,
            strides=[1,1,1,1,1],padding='VALID', name='upsample')
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

def unetBlock(x,nfilters=32,scope='unet3d'):
    with tf.variable_scope(scope):
        o1 = conv3D(x,nfilters=nfilters)
        o2 = conv3D(o1,nfilters=2*nfilters)
        o3 = tf.nn.pool(o2,[2,2,2],strides=[2,2,2],pooling_type='MAX',padding='SAME',name='pool')
    return o3,o2

def unetUpsampleBlock(x,y,nfilters=32,scope='unet3dupsample'):
    with tf.variable_scope(scope):
        o = upsample3D(y)
        o = tf.concat([o,x],axis=4)
        o = conv3D(o,nfilters=nfilters)
        o = conv3D(o,nfilters=nfilters)
        return o

def UNET3D(x,nfilters=32):
    o1_down,o1 = unetBlock(x,nfilters,'layer1')
    print o1,o1_down
    o2_down,o2 = unetBlock(o1_down,2*nfilters,'layer2')
    print o2,o2_down
    o3_down,o3 = unetBlock(o2_down,4*nfilters,'layer3')
    print o3,o3_down
    # o4_down,o4 = unetBlock(o3_down,8*nfilters,'layer4')
    # print o4,o4_down

    # a_3 = unetUpsampleBlock(o3,o4,8*nfilters,'o_layer3')
    # print a_3
    a_2 = unetUpsampleBlock(o2,o3,4*nfilters,'o_layer2')
    print a_2
    a_1 = unetUpsampleBlock(o1,a_2,2*nfilters,'o_layer1')
    print a_1
    yhat = conv3D(a_1,tf.identity,shape=[1,1,1],nfilters=1,scope='yhat')
    print yhat
    yclass = tf.sigmoid(yhat)
    print yclass
    return yhat,yclass
