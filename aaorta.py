import nibabel as nib
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
import copy
from scipy.signal import argrelextrema


def AortaSegmentation(CT_path, L1_path):
    """
    A function that produces the segmentation of the aorta in the slices where L1 is segmented and saves it in a new file
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
    # best_slice_index = lower_border
    best_segmentation_slice = L1_data[:, :, lower_border]
    best_segmentation_slice_area = L1_data[:, :, lower_border].sum()
    for slc in range(lower_border + 1, upper_border + 1):
        current_slice_area = L1_data[:, :, slc].sum()
        if best_segmentation_slice_area < current_slice_area:
            best_segmentation_slice_area = current_slice_area
            best_segmentation_slice = L1_data[:, :, slc]
            # best_slice_index = slc
    # print(best_slice_index)
    ROI_borders = find_ROI_borders(best_segmentation_slice)

    # segment the aorta in all slices
    for slc in range(upper_border-1, lower_border-1, -1):

        print('slice', slc)
        # min_row, max_row, min_col, max_col = ROI_borders
        # ROI = copy.deepcopy(ct_data[min_row:max_row, min_col:max_col,slc])
        # plt.imshow(ROI.T, cmap='gray')
        # plt.show()
        if slc == upper_border-1:
            ct_data[:,:,slc] = segment_aorta(ct_data[:, :, slc], ROI_borders)
            continue
        ct_data[:,:,slc] = segment_aorta(ct_data[:, :, slc], ROI_borders, copy.deepcopy(ct_data[:,:,slc+1]))

    ct_data[ct_data != 0] = 1
    nib.save(ct_scan, file_name + '_MYaorta.nii.gz')



def segment_aorta(axial_slice, ROI_borders, upper_slice=None):
    """
    A function that segments the aorta in the given slice
    :param axial_slice: The slice in which the aorta should be segmented
    :param ROI_borders: a list containing the borders of the desired ROI to look for the aorta in
    :return: The aorta segmentation of the given slice
    """
    orig = copy.deepcopy(axial_slice)
    min_row, max_row, min_col, max_col = ROI_borders
    # print('min_row', min_row)
    # print('max_row', max_row)
    # print('min_col', min_col)
    # print('max_col', max_col)
    ROI = copy.deepcopy(axial_slice[min_row:max_row, min_col:max_col])

    # plt.imshow(ROI.T, cmap='gray')
    # plt.show()

    ROI_hist = np.histogram(ROI, 60, [0, 180], density=True)
    maximas_list = argrelextrema(ROI_hist[0], np.greater)[0]
    max_th = ROI_hist[0][maximas_list[-1]]
    i = 1
    while max_th < 0.004:
        i += 1
        max_th = ROI_hist[0][maximas_list[-i]]

    lower_th = ROI_hist[1][maximas_list[-i]] - 20
    upper_th = ROI_hist[1][maximas_list[-i]] + 35

    # print('lower th', lower_th)
    # print('upper th', upper_th)

    # plt.hist(ROI.ravel(), 60, [0, 180], density=True)
    # plt.show()


    ROI[ROI < lower_th] = 0
    ROI[ROI > upper_th] = 0
    ROI[ROI != 0] = 1

    # ROI[::], new_connected_components_num = measure.label(ROI, return_num=True)
    # props = measure.regionprops(ROI.astype(np.uint16))
    # max_area = props[0].area
    # for i in range(1, new_connected_components_num):
        # if props[i].area > max_area:
        #     max_area = props[i].area
    # plt.imshow(ROI.T, cmap='gray')
    # plt.title('ROI segmentation')
    # plt.show()

    ROI[:, :] = morphology.closing(ROI, selem=morphology.disk(2))
    ROI[:,:], new_connected_components_num = measure.label(ROI, return_num=True)
    ROI[:,:] = morphology.remove_small_objects(ROI.astype(np.uint16))

    # plt.imshow(ROI.T, cmap='gray')
    # plt.title('ROI segmentation')
    # plt.show()

    ROI[:,:], new_connected_components_num = measure.label(ROI, return_num=True)
    props = measure.regionprops(ROI.astype(np.uint16), coordinates='rc')

    area_list = [region.area for region in props]
    # print(area_list)
    eccentricity_list = [region.eccentricity for region in props]
    # print(eccentricity_list)

    aorta_area = 0
    while len(area_list) > 0:
        current_max_area = area_list.index(max(area_list))
        # print(current_max_area)
        current_optimal_circle = eccentricity_list.index(min(eccentricity_list))
        # print(current_optimal_circle)
        if current_max_area == current_optimal_circle:
            aorta_area = area_list[current_max_area]
            break
        area_list.pop(current_max_area)
        eccentricity_list.pop(current_max_area)

    for region in props:
        if region.area != aorta_area:
            ROI[ROI == region.label] = 0



    # # ROI[:, :] = morphology.closing(ROI, selem=morphology.disk(2))
    # plt.imshow(ROI.T, cmap='gray')
    # plt.title('ROI segmentation')
    # plt.show()

    axial_slice[:, :] = 0
    axial_slice[min_row:max_row, min_col:max_col] = ROI





    if upper_slice is not None:

        # plt.imshow(axial_slice.T, cmap='gray')
        # plt.title('axial slice')
        # plt.show()
        #
        # plt.imshow(upper_slice.T, cmap='gray')
        # plt.title('upper slice')
        # plt.show()

        intersection_flags = np.logical_and(axial_slice, upper_slice)
        union_flags = np.logical_or(axial_slice, upper_slice)
        if intersection_flags.sum() / union_flags.sum() < 0.45:
            axial_slice[:,:] = upper_slice

    return axial_slice


    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(axial_slice.T, cmap='gray')
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(orig.T, cmap='gray')
    # plt.show()


def find_ROI_borders(segmentation_slice):
    """
    A function that defines the borders of the ROI according to the given segmentation
    :param segmentation_slice:
    :return:
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





if __name__ == '__main__':
    AortaSegmentation('Case4_CT.nii.gz', 'Case4_L1.nii.gz')
