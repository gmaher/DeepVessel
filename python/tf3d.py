import tensorflow as tf
import modules.tf_util as tf_util
import numpy as np
import tables
import matplotlib.pyplot as plt

#######################################################
# Get data
#######################################################
data_path = '/home/marsdenlab/datasets/DeepVesselData/'
train = data_path+'train_dist72.hdf5'
val = data_path+'val_dist72.hdf5'
test = data_path+'test_dist72.hdf5'
save_dir = './models/cnn'
f_train = tables.open_file(train)
f_val = tables.open_file(val)

input_shape = f_train.root.X.shape
output_shape = f_train.root.Y.shape
print input_shape, output_shape

print 'data shapes, train={},{}'.format(input_shape,output_shape)

######################################################
# Define variables
######################################################
N = f_train.root.X.shape[0]
Nval = f_val.root.X.shape[0]
W,H,D = f_train.root.X[0].shape
C = 1
Nbatch = 16
lr = 5e-3
Nsteps=5000
print_step=100
init = 1e-3
Nlayers = 5
#########################################################
# Define graph
#########################################################
x = tf.placeholder(shape=[Nbatch,W,H,D,C],dtype=tf.float32)
y = tf.placeholder(shape=[Nbatch,W,H,D,C],dtype=tf.float32)

o_4 = tf_util.conv3D_N(x,N=Nlayers)

yhat = tf_util.conv3D(o_4,tf.identity,nfilters=1,scope='yhat',init=init)
yclass = tf.sigmoid(yhat)
# yhat,yclass = tf_util.UNET3D(x)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=yhat,name='loss'))

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
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
for i in range(Nsteps):
    xb,yb = tf_util.get_batch(f_train.root.X,f_train.root.Y,N,n=Nbatch,y_index=0)
    l,_=sess.run([loss,train],{x:xb,y:yb})
    if i%(print_step/5)==0:
        print "iteration {}".format(i)
    if i%print_step == 0:
        xb,yb = tf_util.get_batch(f_val.root.X,f_val.root.Y,Nval,Nbatch,y_index=0)
        lval=sess.run(loss,{x:xb,y:yb})
        print "Train: {}, Val: {}".format(l,lval)

saver.save(sess,'./models/cnn/cnn')
######################################################
# Plot
######################################################
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['figure.figsize'] = (20.0, 10.0)
def implot(mp,ax):
    im = ax.imshow(mp.astype(np.float32), cmap='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

for i in range(5):
    j = np.random.randint(len(f_val.root.X))
    xval = f_val.root.X[j:j+2][:,:,:,:,np.newaxis]
    yval = f_val.root.Y[j:j+2][:,:,:,:,0]
    yval = yval[:,:,:,:,np.newaxis]
    ypred = sess.run(yclass,{x:xval,y:yval})
    mpx = np.amax(xval[0,:,:,:,0],axis=(1))
    mpy = np.amax(yval[0,:,:,:,0],axis=(1))
    mpd = np.amax(ypred[0,:,:,:,0],axis=(1))

    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    implot(mpx,ax1)
    implot(mpy,ax2)
    implot(mpd,ax3)
    plt.tight_layout
    plt.savefig('{}.png'.format(i),dpi=400)

plt.close('all')
