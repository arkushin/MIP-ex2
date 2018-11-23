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

    ct_scan  = nib.load(CT_path)
    L1_segmentation = nib.load(L1_path)

    ct_data = ct_scan.get_data()
    L1_data = L1_segmentation.get_data()

    # slc = ct_data[:,:,220].T
    # plt.imshow(slc)
    # plt.show()

    # find borders of L1:
    lower_border = 0
    while np.array_equal(L1_data[:,:,lower_border], np.zeros((L1_data.shape[0], L1_data.shape[1]))):
        lower_border += 1
    upper_border = lower_border
    while not np.array_equal(L1_data[:,:,upper_border], np.zeros((L1_data.shape[0], L1_data.shape[1]))):
        upper_border += 1

    # clear original CT scan in all slices except the slices that should be segmented:
    ct_data[:,:,:lower_border] = 0
    ct_data[:,:,upper_border:] = 0

    segment_aorta(ct_data[:,:,206], L1_data[:,:,206])




    # segment the aorta in each slice independently
    # for axial_slice in range(lower_border, upper_border):
    #     disk = morphology.disk(5)
    #     ct_data[:,:,axial_slice][ct_data[:,:,axial_slice] < 150] = 0
    #     ct_data[:,:,axial_slice][ct_data[:,:,axial_slice] >= 150] = 1
    #     ct_data[:,:,axial_slice] = np.logical_or(morphology.binary_closing(ct_data[:,:,axial_slice], selem=disk), L1_data[:,:,axial_slice])
    #
    #     slc = ct_data[:, :, axial_slice].T
    #     plt.imshow(slc, cmap='gray')
    #     plt.title('slice: ' + str(axial_slice))
    #     plt.show()
    # #
    # #     ct_data[:,:,axial_slice] = segment_aorta(ct_data[:,:,slice])



def segment_aorta(axial_slice, segmentation_slice): #TODO: do transpose for the slices that are being received!!!
    """
    A function that segments the aorta in the given slice
    :param axial_slice: The slice in which the aorta should be segmented
    :return: The segmentation of the given slice
    """
    axial_slice = axial_slice.T
    segmentation_slice = segmentation_slice.T
    orig = copy.deepcopy(axial_slice)
    min_row, max_row, min_col, max_col = find_ROI_borders(copy.deepcopy(axial_slice),segmentation_slice)
    print('min_row', min_row)
    print('max_row', max_row)
    print('min_col', min_col)
    print('max_col', max_col)
    ROI = copy.deepcopy(axial_slice[min_col:max_col, min_row:max_row])
    plt.imshow(ROI.T, cmap='gray')
    plt.show()
    min_val = ROI.min()
    max_val = ROI.max()
    print('min gray level in ROI:', min_val)
    bins = (max_val  - min_val).astype(np.uint32)  #TODO: should be around 3000
    ROI_hist = np.histogram(ROI, int(bins / 4), [min_val, max_val])
    maximas_list = argrelextrema(ROI_hist[0], np.less)[0]
    lower_th = ROI_hist[1][maximas_list[-1]] - 20
    upper_th = ROI_hist[1][maximas_list[-1]] + 20
    print('lower th',lower_th)
    print('upper th', upper_th)



    plt.hist(ROI.ravel(), int(bins/4), [min_val, max_val])
    plt.show()
    ROI[ROI < lower_th] = 0
    ROI[ROI > upper_th] = 0
    ROI[ROI != 0] = 1

    ROI[::], new_connected_components_num = measure.label(ROI, return_num=True)
    props = measure.regionprops(ROI.astype(np.uint16))
    max_area = props[0].area
    for i in range(1, new_connected_components_num):
        if props[i].area > max_area:
            max_area = props[i].area
    ROI[::] = morphology.remove_small_objects(ROI.astype(np.uint16), max_area)
    # ROI[::] = morphology.remove_small_holes(ROI.astype(np.uint16), area_threshold=max_area)


    ROI[:,:] = morphology.closing(ROI, selem=morphology.disk(2))
    plt.imshow(ROI.T, cmap='gray')
    plt.title('ROI segmentation')
    plt.show()


    axial_slice[:,:] = 0
    axial_slice[min_row:max_row, min_col:max_col] = ROI
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(axial_slice.T, cmap='gray')
    fig.add_subplot(1,2,2)
    plt.imshow(orig.T, cmap='gray')
    plt.show()



def find_ROI_borders(segmentation_slice):
    """
    A function that defines the borders of the ROI according to the given segmentation
    :param segmentation_slice:
    :return:
    """
    plt.imshow(segmentation_slice, cmap='gray')
    plt.show()
    max_col = 0
    while not segmentation_slice[:, max_col].any():
        max_col += 1
    min_col = max_col - 50

    min_row = 0
    while not segmentation_slice[min_row,:].any():
        min_row += 1
    max_row = min_row
    while segmentation_slice[max_row].any():
        max_row += 1

    return min_row, max_row, min_col, max_col


#
# def find_ROI_borders(axial_slice, slice_segmentation):
#     slice_segmentation = slice_segmentation.T
#     orig_seg = copy.deepcopy(slice_segmentation)
#     disk = morphology.disk(5)
#     axial_slice[axial_slice < 200] = 0
#     axial_slice[axial_slice >= 200] = 1
#     axial_slice[:,:] = morphology.binary_closing(axial_slice, selem=disk)
#     axial_slice[:,:] = morphology.remove_small_objects(axial_slice.astype(np.uint16))
#     fig = plt.figure()
#     fig.add_subplot(1,3,1)
#     plt.imshow(axial_slice.T, cmap='gray')
#     plt.title('axial')
#     fig.add_subplot(1,3,2)
#     plt.imshow(orig_seg.T, cmap='gray')
#     plt.title('segmentation before')
#
#     # slice_segmentation[:,:] = np.logical_or(axial_slice, slice_segmentation)
#     fig.add_subplot(1, 3, 3)
#     plt.imshow(slice_segmentation.T, cmap='gray')
#     plt.title('segmentation after')
#
#     plt.show()
#
#     min_col = 0
#     while not slice_segmentation[min_col,:].any():
#         min_col += 1
#     max_col = min_col  # todo: check this number!
#     while not slice_segmentation[max_col,:].any():
#         max_col += 1
#     max_row = 0
#     while not slice_segmentation[:, max_row].any():
#         max_row += 1
#     min_row = max_row - 70  # todo: check this number!
#
#     return  min_row, max_row, min_col, max_col
#






if __name__ == '__main__':
    AortaSegmentation('Case1_CT.nii.gz', 'Case1_L1.nii.gz')



