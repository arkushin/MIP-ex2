import nibabel as nib
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


IMAX = 1300


def SegmentationByTH(path, Imin, Imax):
    """
    A function that segments the given nifti file using the given thresholds and saves the segmented image.
    :param path: The path to the nii.gz file that should be segmented
    :param Imin: The lower threshold for the segmentation
    :param Imax: The upper threshold for the segmentation
    """
    file_name = path.split('.')[0]
    img = nib.load(path)
    img_data = img.get_data()

    # segment the image using Imin as the lower threshold and Imax as the upper threshold
    img_data[img_data < Imin] = 0
    img_data[img_data > Imax] = 0
    img_data[img_data != 0] = 1

    # save the segmentation file
    nib.save(img, file_name + '_seg_' + str(Imin) + '_' + str(Imax) + '.nii.gz')

    return img_data


def SkeletonTHFinder(path):
    """
    A fucnction that iterates over different imin values, chooses the best one of them and then creates a skeleton
    segmentation of the given file using the best imin and morphological operations
    :param path: The path to the file that should be segmented
    :return: The function saves the segmentation of the skeleton and returns the best imin according to which the
    function produced the skeleton segmentation
    """
    file_name = path.split('.')[0]

    # segment the file with different Imin values between 150 and 500
    connected_components_list = []
    imin_list = []
    for imin in range(150, 500, 14):
        imin_list.append(imin)
        print(imin)
        segmentation = SegmentationByTH(path, imin, IMAX)
        labels, connected_components_num = measure.label(segmentation, return_num=True)
        connected_components_list.append(connected_components_num)

    # create and save the graph of Imin vs. number of connectivity components
    plt.figure()
    plt.plot(imin_list, connected_components_list)
    plt.title(file_name + ': Number of connectivity components as function of Imin')
    plt.savefig(file_name + '_Graph.jpg')

    # find best Imin:
    minimas_list = argrelextrema(np.array(connected_components_list), np.less)[0]
    best_imin = imin_list[minimas_list[0]]
    if len(minimas_list) > 1:
        second_minima = imin_list[minimas_list[1]]
        if second_minima < best_imin and second_minima < 300:
            best_imin = second_minima

    # load segmentation file of the best Imin
    best_segmentation = nib.load(file_name + '_seg_' + str(best_imin) + '_' + str(IMAX) + '.nii.gz')
    im_data = best_segmentation.get_data()
    selem = morphology.cube(7)

    # perform image closing in order to fill holes and connect related areas
    closing_flags = morphology.binary_closing(im_data, selem)
    im_data[::] = 0
    im_data[closing_flags] = 1

    # label the different connectivity components and remove all except the biggest component
    im_data[::], new_connected_components_num = measure.label(im_data, return_num=True)
    props = measure.regionprops(im_data.astype(np.uint16))
    max_area = props[0].area
    for i in range(1, new_connected_components_num):
        if props[i].area > max_area:
            max_area = props[i].area
    im_data[::] = morphology.remove_small_objects(im_data.astype(np.uint16), max_area)
    im_data[im_data != 0] = 1
    im_data[::] = morphology.remove_small_holes(im_data.astype(np.uint16), area_threshold=max_area)

    # save the resulting skeleton segmentation and return the best Imin
    nib.save(best_segmentation, file_name + '_SkeletonSegmentation.nii.gz')
    return best_imin


if __name__ == '__main__':
    path = "Case2_CT.nii.gz"
    print('best imin: ', SkeletonTHFinder(path))

