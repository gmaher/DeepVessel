import tensorflow as tf
import modules.tf_util as tf_util
import numpy as np
import tables
import matplotlib.pyplot as plt

#######################################################
# Get data
#######################################################
data_path = '/media/marsdenlab/Data1/datasets/DeepVessel/'
train = data_path+'train_dist72_white_large_ct.hdf5'
val = data_path+'val_dist72_white_large_ct.hdf5'
test = data_path+'test_dist72_white_ct.hdf5'

neg_train = data_path+'train_dist72_white_negative_ct.hdf5'
neg_val = data_path+'val_dist72_white_negative_ct.hdf5'
f_train = tables.open_file(train)
f_val = tables.open_file(val)

f_neg_train = tables.open_file(neg_train)
f_neg_val = tables.open_file(neg_val)

input_shape = f_train.root.X.shape
output_shape = f_train.root.Y.shape
print input_shape, output_shape

print 'data shapes, train={},{}'.format(input_shape,output_shape)


input_shape = f_neg_train.root.X.shape
output_shape = f_neg_train.root.Y.shape
print input_shape, output_shape

print 'data shapes, train={},{}'.format(input_shape,output_shape)

######################################################
# Define variables
######################################################
N = f_train.root.X.shape[0]
N_neg = f_neg_train.root.X.shape[0]
Nval = f_val.root.X.shape[0]
W,H,D = f_train.root.X[0].shape
C = 1
Nbatch = 4
lr = 1e-5
Nsteps=15000
print_step=1000
init = 1.0e-3
Nlayers = 8
Nfilters = 32
VESSEL_SCALE = 1000
EPS=1e-4
leaky_relu = tf.contrib.keras.layers.LeakyReLU(0.2)
y_index=0
alph = 0.3
beta = 0.7
#########################################################
# Define graph
#########################################################
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x)-alpha)
#########################################################
# Define graph
#########################################################
x = tf.placeholder(shape=[None,W,H,D,C],dtype=tf.float32)
y = tf.placeholder(shape=[None,W,H,D,C],dtype=tf.float32)

yhat,yclass = tf_util.UNET3D(x,nfilters=Nfilters,activation=leaky_relu,init=init)

#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=yhat,name='loss'))

TP = tf.reduce_sum(yclass*y)
FP = tf.reduce_sum(yclass*(1-y))
FN = tf.reduce_sum((1-yclass)*y)
loss = -TP/(TP + alph*FP+beta*FN+EPS)

opt = tf.train.AdamOptimizer(lr)
train = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print yhat

#######################################################
# Train
#######################################################
train_hist = []
val_hist = []

for i in range(Nsteps+1):
    xb,yb = tf_util.get_batch(f_train.root.X,f_train.root.Y,N,n=3*Nbatch/4,y_index=y_index)
    xbn,ybn = tf_util.get_batch(f_neg_train.root.X,f_neg_train.root.Y,N_neg,n=Nbatch/2,y_index=y_index)
    xb = np.concatenate((xb,xbn),axis=0)
    yb = np.concatenate((yb,ybn),axis=0)
    l,_=sess.run([loss,train],{x:xb,y:yb})
    if i%(print_step/5)==0:
        print "iteration {}".format(i)
    if i%(print_step/5) == 0:
        xb,yb = tf_util.get_batch(f_val.root.X,f_val.root.Y,Nval,Nbatch,y_index=1)
        lval,ypred=sess.run([loss,yclass],{x:xb,y:yb})
        print "Train: {}, Val: {}, p var {}, p mean {}, p max {}, p min {}".format(l,lval,
                np.var(ypred),np.mean(ypred), np.amax(ypred),np.amin(ypred))

        train_hist.append(l)
        val_hist.append(lval)

saver = tf.train.Saver()
saver.save(sess,'./models/unet/unet')
