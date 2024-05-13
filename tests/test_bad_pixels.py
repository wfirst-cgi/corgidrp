import os
import numpy as np
from corgidrp.data import Image, Dataset
from corgidrp.l2a_to_l2b import correct_bad_pixels
import pytest
from corgidrp.data import Dataset

def test_bad_pixels():
    print("test correct bad pixels pipeline step")
    fpaths = ['test_data/example_L1_input.fits']
    dataset = Dataset(fpaths) # makes the Dataset object from the fits file
    data_cube = dataset.all_data # 3D data cube for all frames in the dataset
    dq_cube = dataset.all_dq # 3D DQ array cube for all frames in the dataset
    # Add some CR
    col_cr = [12, 123, 234, 456, 678, 890]
    row_cr = [546, 789, 123, 43, 547, 675]
    for i_col in col_cr:
        for i_row in row_cr:
            dq_cube[0, i_col, i_row] += 128

    history_msg = "Pixels affected by CR added"
    dataset.update_after_processing_step(history_msg, new_all_data=data_cube,
        new_all_dq=dq_cube)

    # Generate bad pixel detector mask
    dq_mask = Dataset(fpaths).all_dq 
    bp_mask = dq_mask[0,:,:]
    # Add some Bad Detector Pixels
    col_bp = [12, 120, 234, 450, 678, 990]
    row_bp = [546, 89, 123, 243, 447, 675]
    for i_col in col_bp:
        for i_row in row_bp:
            bp_mask[i_col, i_row] += 4

    correct_bad_pixels(dataset, bp_mask)


if __name__ == '__main__':
    test_bad_pixels()
