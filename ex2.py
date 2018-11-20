import nibabel as nib
import numpy as np
import ntpath
from skimage import measure
import matplotlib.pyplot as plt

IMAX = 1300


def SegmentationByTH(path, Imin, Imax):
    """
    A function that segments the given nifti file using the given thresholds and saves the segmented image.
    :param path: The path to the nii.gz file that should be segmented
    :param Imin: The lower threshold for the segmentation
    :param Imax: The upper threshold for the segmentation
    """
    img = nib.load(path)
    img_data = img.get_data()
    file_name = ((ntpath.basename(path)).split('.'))[0]

    # segment the image using Imin as the lower threshold and Imax as the upper threshold
    img_data[img_data < Imin] = 0
    img_data[img_data > Imax] = 0
    img_data[img_data != 0] = 1

    # self.labels, self.connected_components_num = measure.label(img_data, return_num=True)
    # print(self.connected_components_num)

    # save the segmentation file
    nib.save(img, file_name + '_seg_' + str(Imin) + '_' + str(Imax) + '.nii.gz')

    return img_data




def SkeletonTHFinder(path):
    file_name = ((ntpath.basename(path)).split('.'))[0]   # TODO: explain usage of this library
    connected_components_list = []
    imin_list = []
    for imin in range(150, 500, 14):
        imin_list.append(imin)
        print(imin)
        segmentation = SegmentationByTH(path, imin, IMAX)
        labels, connected_components_num = measure.label(segmentation, return_num=True)
        connected_components_list.append(connected_components_num)

    plt.figure()
    plt.plot(imin_list, connected_components_list)
    plt.title('Number of connectivity components as function of Imin')
    plt.savefig(file_name + '_Graph.jpg')





if __name__ == '__main__':
    path = "C:\\Users\\pc\\Desktop\\Mathematic\\Medical-Image-Processing\\ex2\\data\\case1\\Case1_CT.nii.gz"
    SkeletonTHFinder(path)