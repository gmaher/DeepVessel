import numpy as np
import SimpleITK as sitk
import h5py
from skimage.morphology import skeletonize_3d
import matplotlib.pyplot as plt
from tqdm import tqdm

images = open('./data/images.txt').readlines()
images = [i.replace('\n','') for i in images]

truths = open('./data/truths.txt').readlines()
truths = [i.replace('\n','') for i in truths]

filt = sitk.SignedMaurerDistanceMapImageFilter()

imfile = open('processed_images.txt','w')
segfile = open('processed_truths.txt','w')
distfile = open('processed_distances.txt','w')
skelfile = open('processed_skeletons.txt','w')

for i in tqdm(range(len(images))):
    imgName = images[i].split('/')[-1].replace('-cm.mha','')
    imgName = imgName.replace('-image.mha','')
    img = sitk.ReadImage(images[i])
    truth = sitk.ReadImage(truths[i])

    imgnp = sitk.GetArrayFromImage(img)
    truthnp = sitk.GetArrayFromImage(truth)
    #insideIsPositive, squaredDistance, useImageSpacing

    skel = skeletonize_3d(sitk.GetArrayFromImage(truth))
    skel_img = sitk.GetImageFromArray(skel)

    dist = filt.Execute(skel_img, True, False, False)
    distnp = sitk.GetArrayFromImage(dist)

    sitk.WriteImage(img,'./data/mhas/{}.mha'.format(imgName))
    sitk.WriteImage(truth,'./data/mhas/{}_{}.mha'.format(imgName,'seg'))
    sitk.WriteImage(dist,'./data/mhas/{}_{}.mha'.format(imgName,'distance'))
    sitk.WriteImage(skel_img,'./data/mhas/{}_{}.mha'.format(imgName,'skeleton'))

    imfile.write('/home/marsdenlab/projects/DeepVessel/data/mhas/{}.mha\n'.format(imgName))
    segfile.write('/home/marsdenlab/projects/DeepVessel/data/mhas/{}_{}.mha\n'.format(imgName,'seg'))
    distfile.write('/home/marsdenlab/projects/DeepVessel/data/mhas/{}_{}.mha\n'.format(imgName, 'distance'))
    skelfile.write('/home/marsdenlab/projects/DeepVessel/data/mhas/{}_{}.mha\n'.format(imgName,'skel'))

    #plot maximum intensity projections
    axes = [0,1,2]
    for a in axes:
        d = np.amax(distnp,axis=a)
        s = np.amax(skel,axis=a)
        i = np.amax(imgnp,axis=a)
        t = np.amax(truthnp,axis=a)

        plt.figure()
        plt.imshow(i, cmap='gray')
        plt.colorbar()
        plt.savefig('./data/{}_img{}.png'.format(imgName,a),dpi=500)
        plt.close()

        plt.figure()
        plt.imshow(t, cmap='gray')
        plt.colorbar()
        plt.savefig('./data/{}_seg{}.png'.format(imgName,a), dpi=500)
        plt.close()

        plt.figure()
        plt.imshow(d, cmap='gray')
        plt.colorbar()
        plt.savefig('./data/{}_dist{}.png'.format(imgName,a),dpi=500)
        plt.close()

        plt.figure()
        plt.imshow(s, cmap='gray')
        plt.colorbar()
        plt.savefig('./data/{}_skel{}.png'.format(imgName,a), dpi=500)
        plt.close()
