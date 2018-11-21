import nibabel as nib
import numpy as np
import ntpath
from skimage import measure, morphology
import matplotlib.pyplot as plt
import copy

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
    # file_name = ((ntpath.basename(path)).split('.'))[0]
    file_name = path.split('.')[0]

    # segment the image using Imin as the lower threshold and Imax as the upper threshold
    img_data[img_data < Imin] = 0
    img_data[img_data > Imax] = 0
    img_data[img_data != 0] = 1

    # save the segmentation file
    nib.save(img, file_name + '_seg_' + str(Imin) + '_' + str(Imax) + '.nii.gz')

    return img_data


def test(path):
    img = nib.load(path)
    img_data = img.get_data()
    img_data[::] = 0
    nib.save(img, 'test_seg.nii.gz')


def SkeletonTHFinder(path):
    file_name = path.split('.')[0]
    # connected_components_list = []
    # imin_list = []
    # for imin in range(150, 500, 14):
    #     imin_list.append(imin)
    #     print(imin)
    #     segmentation = SegmentationByTH(path, imin, IMAX)
    #     labels, connected_components_num = measure.label(segmentation, return_num=True)
    #     connected_components_list.append(connected_components_num)

    # create and save the graph of Imin vs. number of connectivity components
    # plt.figure()
    # plt.plot(imin_list, connected_components_list)
    # plt.title(file_name + ': Number of connectivity components as function of Imin')
    # plt.savefig(file_name + '_Graph.jpg')
    #
    # best_imin = imin_list[connected_components_list.index(min(connected_components_list))]
    # print(best_imin)

    best_imin = 248
    # load segmentation file of the best Imin
    best_segmentation = nib.load(file_name + '_seg_' + str(best_imin) + '_' + str(IMAX) + '.nii.gz')

    # best_segmentation = do_morphological_operations(best_segmentation)
    im_data = best_segmentation.get_data()
    print('start labels: ', np.unique(im_data))

    # img_data = do_morphological_operations(img_data)

    selem = morphology.cube(5)

    # img_data = segmented_image.get_data()
    # img_data = data

    # perform image closing in order to fill holes and connect related areas
    morphology_flags = morphology.binary_closing(im_data, selem)
    im_data[::] = 0
    im_data[morphology_flags] = 1

    print('after closing: ', np.unique(im_data))

    im_data, new_connected_components_num = measure.label(im_data, return_num=True)
    print('after labeling: ', new_connected_components_num)
    print('labels after labeling: ', np.unique(im_data))

    props = measure.regionprops(im_data)
    max_area = props[0].area
    print('searching for max area...')
    for i in range(1, new_connected_components_num):
        if props[i].area > max_area:
            max_area = props[i].area
    print('max area: ', max_area)
    im_data = morphology.remove_small_objects(im_data, 2 * max_area, in_place=True)
    print('after objects removal: ', np.unique(im_data))


    new_im_data = best_segmentation.get_data()
    print('new im data labels: ', np.unique(new_im_data))
    print(np.array_equal(new_im_data, im_data))

    # img_data[::] = 0
    # print('setting img_data to 0')
    # img_data[small_objects_flags == 1] = 1
    # print('done with setting values')
    # labels, new_connected_components_num = measure.label(img_data, return_num=True)
    # print('after removal: ', new_connected_components_num)

    # img_data[img_data != 0] = 1


    nib.save(best_segmentation, file_name + '_SkeletonSegmentation.nii.gz')


#
# def do_morphological_operations(data):
#     """
#     A function that gets an labeled image, and using morphological operations turns the image into one connectivity
#     component
#     :param segmented_image: The image that should be processed recieved as a nii.gz file
#     :return: The image with only one connectivity component
#     """
#     selem = morphology.cube(5)
#
#     # img_data = segmented_image.get_data()
#     # img_data = data
#
#     # perform image closing in order to fill holes and connect related areas
#     morphology_flags = morphology.binary_closing(img_data, selem)
#     img_data[::] = 0
#     img_data[morphology_flags] = 1
#
#     img_data, new_connected_components_num = measure.label(img_data, return_num=True)
#     print('after closing: ', new_connected_components_num)
#     props = measure.regionprops(img_data)
#     max_area = props[0].area
#     print('searching for max area...')
#     for i in range(1, new_connected_components_num):
#         if props[i].area > max_area:
#             max_area = props[i].area
#     print('max area: ', max_area)
#     small_objects_flags = morphology.remove_small_objects(img_data, 2*max_area)
#     print('after objects removal')
#     img_data[::] = 0
#     print('setting img_data to 0')
#     img_data[small_objects_flags == 1] = 1
#     print('done with setting values')
#     labels, new_connected_components_num = measure.label(img_data, return_num=True)
#     print('after removal: ', new_connected_components_num)
#     return img_data
#



if __name__ == '__main__':
    path = "Case1_CT.nii.gz"
    SkeletonTHFinder(path)
    # segmentation = SegmentationByTH(path, 234, IMAX)
    # labels, num = measure.label(segmentation, return_num=True)
    # print(num)