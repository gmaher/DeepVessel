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
img_np = (1.0*img_np-np.amin(img_np))/(np.amax(img_np)-np.amin(img_np))
save_dir = './models/cnn/cnn'

print 'image shape {}'.format(img_np.shape)

######################################################
# Define variables
######################################################
W = 72
H = 72
D = 72
C = 1
init = 1e-3
Nlayers = 6
stride = 16
Nx,Ny,Nz = img_np.shape
#########################################################
# Define graph
#########################################################
x = tf.placeholder(shape=[None,W,H,D,C],dtype=tf.float32)
y = tf.placeholder(shape=[None,W,H,D,C],dtype=tf.float32)

o_4 = tf_util.conv3D_N(x,N=Nlayers)

yhat = tf_util.conv3D(o_4,tf.identity,nfilters=1,scope='yhat',init=init)
yclass = tf.sigmoid(yhat)

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess,save_dir)

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
