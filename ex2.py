import nibabel as nib
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import copy

IMAX = 1300


##############
# USER GUIDE #
##############
# All functions expect to get a path to the nii.gz file they are supposed to receive.
# Please provide either a full path or put the nii.gz files in the same directory as the ex2.py file for the functions
# to read the files properly
# main function can be activated in the end of this file to run the code


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

    # morphological operations:
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


def AortaSegmentation(CT_path, L1_path):
    """
    A function that produces the segmentation of the aorta in the slices where L1 is segmented and saves it as a new file
    :param CT_path: The CT scan in which the aorta should be segmented
    :param L1_path: The L1 segmentation that was provided
    """
    file_name = CT_path.split('.')[0]

    ct_scan = nib.load(CT_path)
    L1_segmentation = nib.load(L1_path)
    ct_data = ct_scan.get_data()
    L1_data = L1_segmentation.get_data()

    # find borders of L1:
    lower_border = 0
    while np.array_equal(L1_data[:, :, lower_border], np.zeros((L1_data.shape[0], L1_data.shape[1]))):
        lower_border += 1
    upper_border = lower_border
    while not np.array_equal(L1_data[:, :, upper_border], np.zeros((L1_data.shape[0], L1_data.shape[1]))):
        upper_border += 1

    # clear original CT scan in all slices except the slices that should be segmented:
    ct_data[:, :, :lower_border] = 0
    ct_data[:, :, upper_border:] = 0

    # find slice with best segmentation of L1
    best_segmentation_slice = L1_data[:, :, lower_border]
    best_segmentation_slice_area = L1_data[:, :, lower_border].sum()
    for slc in range(lower_border + 1, upper_border + 1):
        current_slice_area = L1_data[:, :, slc].sum()
        if best_segmentation_slice_area < current_slice_area:
            best_segmentation_slice_area = current_slice_area
            best_segmentation_slice = L1_data[:, :, slc]

    # find the ROI according to the best segmentation slice of L1
    ROI_borders = find_ROI_borders(best_segmentation_slice)

    # segment the aorta in all slices
    for slc in range(upper_border - 1, lower_border - 1, -1):
        if slc == upper_border - 1:
            ct_data[:, :, slc] = segment_aorta(ct_data[:, :, slc], ROI_borders)
            continue
        ct_data[:, :, slc] = segment_aorta(ct_data[:, :, slc], ROI_borders, copy.deepcopy(ct_data[:, :, slc + 1]))

    ct_data[ct_data != 0] = 1

    nib.save(ct_scan, file_name + '_aorta_segmentation.nii.gz')


def segment_aorta(axial_slice, ROI_borders, upper_slice=None):
    """
    A function that segments the aorta in the given slice
    :param axial_slice: The slice in which the aorta should be segmented
    :param ROI_borders: a list containing the borders of the desired ROI to look for the aorta in
    :param upper_slice: if the current slice that should be segmented isn't the upper slice of L1, the upper slice is
    provided in order to verify that the segmentation of the slice is accurate
    :return: The aorta segmentation of the given slice
    """
    min_row, max_row, min_col, max_col = ROI_borders
    ROI = copy.deepcopy(axial_slice[min_row:max_row, min_col:max_col])

    # create histogram of the ROI and define the thresholds for the aorta's gray levels:
    ROI_hist = np.histogram(ROI, 60, [0, 180], density=True)
    maximas_list = argrelextrema(ROI_hist[0], np.greater)[0]
    max_th = ROI_hist[0][maximas_list[-1]]
    i = 1
    while max_th < 0.004:  # check that the local maximum that was found is part of the aorta
        i += 1
        max_th = ROI_hist[0][maximas_list[-i]]
    lower_th = ROI_hist[1][maximas_list[-i]] - 20
    upper_th = ROI_hist[1][maximas_list[-i]] + 35

    # segment the aorta according to the thresholds that were found:
    ROI[ROI < lower_th] = 0
    ROI[ROI > upper_th] = 0
    ROI[ROI != 0] = 1

    # perform morphological operations on the segmented image:
    ROI[:, :] = morphology.closing(ROI, selem=morphology.disk(2))
    ROI[:, :], new_connected_components_num = measure.label(ROI, return_num=True)
    ROI[:, :] = morphology.remove_small_objects(ROI.astype(np.uint16))

    # find biggest connectivity component that is round and remove all others:
    ROI[:, :], new_connected_components_num = measure.label(ROI, return_num=True)
    props = measure.regionprops(ROI.astype(np.uint16), coordinates='rc')
    area_list = [region.area for region in props]
    eccentricity_list = [region.eccentricity for region in props]
    aorta_area = 0
    while len(area_list) > 0:
        current_max_area = area_list.index(max(area_list))
        current_optimal_circle = eccentricity_list.index(min(eccentricity_list))
        if current_max_area == current_optimal_circle:
            aorta_area = area_list[current_max_area]
            break
        area_list.pop(current_max_area)
        eccentricity_list.pop(current_max_area)
    for region in props:
        if region.area != aorta_area:
            ROI[ROI == region.label] = 0

    # apply the segmentation of the ROI on the given slice
    axial_slice[:, :] = 0
    axial_slice[min_row:max_row, min_col:max_col] = ROI

    # compare the segmentation of the current slice to the segmentation of the upper slice:
    if upper_slice is not None:
        intersection_flags = np.logical_and(axial_slice, upper_slice)
        union_flags = np.logical_or(axial_slice, upper_slice)
        # if there is no sufficient overlap between the two, take the upper slice segmentation
        if intersection_flags.sum() / union_flags.sum() < 0.45:
            axial_slice[:, :] = upper_slice

    return axial_slice


def find_ROI_borders(segmentation_slice):
    """
    A function that defines the borders of the ROI according to the given segmentation
    :param segmentation_slice: A segmented slice of L1 according to which the borders of the ROI should be defined
    :return: The borders for the ROI
    """
    max_col = 0
    while not segmentation_slice[:, max_col].any():
        max_col += 1
    max_col += 5
    min_col = max_col - 55

    min_row = 0
    while not segmentation_slice[min_row, :].any():
        min_row += 1
    max_row = min_row
    while segmentation_slice[max_row].any():
        max_row += 1
    max_row = min_row + int((max_row - min_row) * 0.6)

    return [min_row, max_row, min_col, max_col]


def evaluateSegmentation(ground_truth_segmentation_path, estimated_segmentation_path):
    """
    A function that evaluates the segmentation of the aorta using Dice coefficient and the Volume Overlap Difference
    :param ground_truth_segmentation_path: The path to the true segmentation of the aorta as provided in the exercise
    :param estimated_segmentation_path: The path to the segmentation created in the AortaSegmentation function
    :return: The VOD and dice coefficient values
    """
    true_seg = nib.load(ground_truth_segmentation_path)
    est_seg = nib.load(estimated_segmentation_path)
    header = true_seg.header

    true_seg_data = true_seg.get_data()
    est_seg_data = est_seg.get_data()

    # find borders of segmentation:
    lower_border = 0
    while not est_seg_data[:, :, lower_border].any():
        lower_border += 1
    upper_border = lower_border
    while est_seg_data[:, :, upper_border].any():
        upper_border += 1

    true_seg_data[:, :, :lower_border] = 0
    true_seg_data[:, :, upper_border:] = 0
    true_seg_data[true_seg_data != 0] = 1

    pixel_volume = header['pixdim'][1] * header['pixdim'][2] * header['pixdim'][3]

    true_seg_volume = true_seg_data.sum() * pixel_volume
    est_seg_volume = est_seg_data.sum() * pixel_volume
    intersection_volume = np.logical_and(true_seg_data, est_seg_data).sum() * pixel_volume
    union_volume = np.logical_or(true_seg_data, est_seg_data).sum() * pixel_volume

    VOD = 1 - (intersection_volume / union_volume)
    dice_coefficient = (2 * intersection_volume) / (true_seg_volume + est_seg_volume)

    return VOD, dice_coefficient



##############################################################################
# uncomment 'main' and fill case number instead '#' for running the program  #
##############################################################################
# if __name__ == '__main__':
    # path = "Case#_CT.nii.gz"
    # print('best imin: ', SkeletonTHFinder(path))
    # AortaSegmentation(path, 'Case#_L1.nii.gz')
    # print(evaluateSegmentation('Case#_Aorta.nii.gz', 'Case#_CT_aorta_segmentation.nii.gz'))
