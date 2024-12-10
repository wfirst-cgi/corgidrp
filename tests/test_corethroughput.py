# Use
# assert A == pytest.approx(min, max)

import os
import pytest
import numpy as np
from astropy.io import fits
from skimage.measure import block_reduce

from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
import corgidrp.corethroughput as corethroughput

ct_filepath = os.path.join(os.path.dirname(__file__), 'test_data')
# Mock error
err = np.ones([1,1024,1024]) * 0.5
# Default headers
prhd, exthd = create_default_headers()

def setup_module():
    """
    Create a dataset with some representative psf responses. 
    """
    # corgidrp dataset
    global dataset_psf, dataset_psf_rev
    # arbitrary set of PSF positions to be tested in EXCAM pixels referred to (0,0)
    global psf_position_x, psf_position_y
    psf_position_x = [512, 522, 532, 542, 552, 562, 522, 532, 542, 552, 562]
    psf_position_y = [512, 522, 532, 542, 552, 562, 502, 492, 482, 472, 462]
    # fsm positions for the off-axis psfs
    global fsm_pos
    fsm_pos = [[1,1]]*len(psf_position_x[1:])

    data_unocc = np.zeros([1024, 1024])
    # unocculted PSF
    unocc_psf_filepath = os.path.join(ct_filepath, 'hlc_os11_no_fpm.fits')
    # os11 unocculted PSF is sampled at the same pixel pitch as EXCAM
    unocc_psf = fits.getdata(unocc_psf_filepath)
    # Insert PSF at its location
    idx_0_0 = psf_position_x[0] - unocc_psf.shape[0]
    idx_0_1 = idx_0_0 + unocc_psf.shape[0]
    idx_1_0 = psf_position_y[0] - unocc_psf.shape[1]
    idx_1_1 = idx_1_0 + unocc_psf.shape[1]
    data_unocc[idx_0_0:idx_0_1, idx_1_0:idx_1_1] = unocc_psf
    
    data_psf = [Image(data_unocc,pri_hdr = prhd, ext_hdr = exthd, err = err)]
    # oversampled os11 psfs
    occ_psf_filepath = os.path.join(ct_filepath, 'hlc_os11_psfs_oversampled.fits')
    occ_psf = fits.getdata(occ_psf_filepath)
    for i_psf, _ in enumerate(psf_position_x[1:]):
        psf_tmp = occ_psf[0, i_psf]
        # re-sample to EXCAM's pixel pitch: os11 off-axis psf is 5x oversampled
        psf_tmp_red = block_reduce(psf_tmp, block_size=(5,5), func=np.mean)
        data_tmp = np.zeros([1024, 1024])
        idx_0_0 = psf_position_x[i_psf] - psf_tmp_red.shape[0]
        idx_0_1 = idx_0_0 + psf_tmp_red.shape[0]
        idx_1_0 = psf_position_y[i_psf] - psf_tmp_red.shape[1]
        idx_1_1 = idx_1_0 + psf_tmp_red.shape[1]
        data_tmp[idx_0_0:idx_0_1, idx_1_0:idx_1_1] = psf_tmp_red
        data_psf += [Image(data_tmp,pri_hdr = prhd, ext_hdr = exthd, err = err)]

    dataset_psf = Dataset(data_psf)
    data_psf.reverse()
    dataset_psf_rev = Dataset(data_psf)

def test_fsm_pos():
    """ Test FSM positions are a list of N pairs of values, where N is the number
        of off-axis psfs.
    """
    # do not pass if fsm_pos is not a list
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, np.array(fsm_pos))
    # do not pass if fsm_pos has less elements
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos[1:])
    # do not pass if fsm_pos has more elements
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos+[[1,1]])
    # Do not pass if fsm_pos is not a list of paired values
    fsm_pos_bad = [[1,1]]*(len(psf_position_x[1:]) - 1)
    fsm_pos_bad += [[1]]
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos_bad)

def test_unocc():
    """ Test array position of the unocculted PSF in the data array """

    # do not pass if the order is different (swap first two)
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf_rev, fsm_pos)

def test_psf_pix_and_ct():
    """
    Test 1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.
    """

    psf_pix, ct = corethroughput.estimate_psf_pix_and_ct(dataset_psf, fsm_pos)
    
if __name__ == '__main__':

    test_fsm_pos()
    test_unocc()   
    test_psf_pix_and_ct()


