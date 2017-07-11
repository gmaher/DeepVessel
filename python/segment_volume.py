import tensorflow as tf
import modules.tf_util as tf_util
import numpy as np
import tables
import matplotlib.pyplot as plt
import SimpleITK as sitk
import argparse
#######################################################
# Get data
#######################################################
parser = argparse.ArgumentParser()
parser.add_argument('volume')
args = parser.parse_args()
vol = args.volume

img = sitk.ReadImage(vol)
img_np = sitk.GetArrayFromImage(img)
#img_np = (1.0*img_np-np.amin(img_np))/(np.amax(img_np)-np.amin(img_np))
img_np = (1.0*img_np-np.mean(img_np))/(np.std(img_np))

save_dir = './models/unet/unet'

print 'image shape {}'.format(img_np.shape)

#######################################################
# Get data
#######################################################
# data_path = '/home/marsdenlab/datasets/DeepVesselData/'
# train = data_path+'train_dist72.hdf5'
# val = data_path+'val_dist72.hdf5'
# test = data_path+'test_dist72.hdf5'
# save_dir = './models/unet/unet'
# f_train = tables.open_file(train)
# f_val = tables.open_file(val)
#
# input_shape = f_train.root.X.shape
# output_shape = f_train.root.Y.shape
# print input_shape, output_shape
#
# print 'data shapes, train={},{}'.format(input_shape,output_shape)

######################################################
# Define variables
######################################################
W = 72
H = 72
D = 72
C = 1
init = 1e-3
Nlayers = 8
stride = 16
Nfilters=10
Nx,Ny,Nz = img_np.shape
leaky_relu = tf.contrib.keras.layers.LeakyReLU(0.2)
#########################################################
# Define graph
#########################################################
x = tf.placeholder(shape=[None,W,H,D,C],dtype=tf.float32)
y = tf.placeholder(shape=[None,W,H,D,C],dtype=tf.float32)

# o_4 = tf_util.conv3D_N(x,N=Nlayers,nfilters=Nfilters)
#
# yhat = tf_util.conv3D(o_4,tf.identity,nfilters=1,scope='yhat',init=init)
# yclass = tf.sigmoid(yhat)
yhat,yclass = tf_util.UNET3D(x,nfilters=Nfilters,activation=leaky_relu,init=init)

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess,save_dir)

######################################################
# Plot
######################################################
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# plt.rcParams['figure.figsize'] = (20.0, 10.0)
# def implot(mp,ax):
#     im = ax.imshow(mp.astype(np.float32), cmap='gray')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = plt.colorbar(im, cax=cax)
#
# for i in range(5):
#     j = np.random.randint(len(f_val.root.X))
#     xval = f_val.root.X[j:j+2][:,:,:,:,np.newaxis]
#     yval = f_val.root.Y[j:j+2][:,:,:,:,0]
#     yval = yval[:,:,:,:,np.newaxis]
#     ypred = sess.run(yclass,{x:xval,y:yval})
#     mpx = np.amax(xval[0,:,:,:,0],axis=(1))
#     mpy = np.amax(yval[0,:,:,:,0],axis=(1))
#     mpd = np.amax(ypred[0,:,:,:,0],axis=(1))
#
#     plt.figure()
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
#     implot(mpx,ax1)
#     implot(mpy,ax2)
#     implot(mpd,ax3)
#     plt.tight_layout
#     plt.savefig('{}.png'.format(i),dpi=400)
#
# plt.close('all')

##########################################################
# Segment volume
##########################################################
counts = np.zeros_like(img_np)
seg = np.zeros_like(img_np)
for i in range(0,Nx-W,stride):
    print float(i)/(Nx-W)
    for j in range(0,Ny-H,stride):
        for k in range(0,Nz-D,stride):
            print (i,j,k)
            counts[i:i+W,j:j+H,k:k+D] += 1.0
            x_ = img_np[i:i+W,j:j+H,k:k+D].reshape((1,W,H,D,C))
            out = sess.run(yclass,{x:x_})[0,:,:,:,0]
            seg[i:i+W,j:j+H,k:k+D] += 1.0/counts[i:i+W,j:j+H,k:k+D]*\
                (out-seg[i:i+W,j:j+H,k:k+D])

out_image = sitk.GetImageFromArray(seg)
sitk.WriteImage(out_image,'./output/out.mha')
