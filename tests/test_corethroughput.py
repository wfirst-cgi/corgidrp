import os
import pytest
import numpy as np
from astropy.io import fits
from skimage.measure import block_reduce

from corgidrp.mocks import create_default_headers
from corgidrp.data import Image, Dataset
from corgidrp import corethroughput

ct_filepath = os.path.join(os.path.dirname(__file__), 'test_data')
# Mock error
err = np.ones([1,1024,1024]) * 0.5
# Default headers
prhd, exthd = create_default_headers()

def setup_module():
    """
    Create a dataset with some representative psf responses. 
    """
    global dataset_psf
    # arbitrary set of PSF positions to be tested in EXCAM pixels referred to (0,0)
    global psf_position_x, psf_position_y
    # Some arbitrary shifts
    psf_position_x = [512, 522, 532, 542, 552, 562, 522, 532, 542, 552, 562]
    psf_position_y = [512, 522, 532, 542, 552, 562, 502, 492, 482, 472, 462]
    global idx_os11
    idx_os11 = 8
    global ct_os11
    ct_os11 = []

    data_psf = []
    # add pupil image(s) of the unocculted source's observation
    data_pupil = fits.getdata(os.path.join(ct_filepath, 'pupil_image_0000094916.fits'))
    # Normalize to 1 since OS11 off-axis PSFs are already normalized to the
    # unocculted response
    data_pupil /= np.sum(data_pupil)
    # Add some noise (pupil images are high SNR)
    data_pupil_1 = data_pupil.copy()
    rng = np.random.default_rng(seed=0)
    data_pupil_1 += rng.normal(0, data_pupil.std()/10, data_pupil_1.shape)
    data_pupil_2 = data_pupil.copy()
    data_pupil_2 += rng.normal(0, data_pupil.std()/10, data_pupil_1.shape)
    data_psf += [Image(data_pupil_1,pri_hdr = prhd, ext_hdr = exthd, err = err)]
    data_psf += [Image(data_pupil_2,pri_hdr = prhd, ext_hdr = exthd, err = err)]

    # add os11 psfs
    occ_psf_filepath = os.path.join(ct_filepath, 'hlc_os11_psfs_oversampled.fits')
    occ_psf = fits.getdata(occ_psf_filepath)
    for i_psf, _ in enumerate(psf_position_x):
        psf_tmp = occ_psf[0, idx_os11+i_psf]
        # re-sample to EXCAM's pixel pitch: os11 off-axis psf is 5x oversampled
        psf_tmp_red = 25*block_reduce(psf_tmp, block_size=(5,5), func=np.mean)
        data_tmp = np.zeros([1024, 1024])
        idx_0_0 = psf_position_x[i_psf] - psf_tmp_red.shape[0] // 2
        idx_0_1 = idx_0_0 + psf_tmp_red.shape[0]
        idx_1_0 = psf_position_y[i_psf] - psf_tmp_red.shape[1] // 2
        idx_1_1 = idx_1_0 + psf_tmp_red.shape[1]
        data_tmp[idx_0_0:idx_0_1, idx_1_0:idx_1_1] = psf_tmp_red
        data_psf += [Image(data_tmp,pri_hdr = prhd, ext_hdr = exthd, err = err)]
        # normalized to 1 if there were no masks
        ct_os11 += [psf_tmp[psf_tmp > psf_tmp.max()/2].sum()]

    dataset_psf = Dataset(data_psf)

def test_psf_pix_and_ct():
    """
    Test 1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.
    """

    # do not pass if setting a method that does not exist
    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, pix_method='bad')

    with pytest.raises(Exception):
        corethroughput.estimate_psf_pix_and_ct(dataset_psf, ct_method='bad')

    psf_pix_est, ct_est = corethroughput.estimate_psf_pix_and_ct(dataset_psf)

    # Read OS11 PSF offsets (l/D=50.19mas=2.3 EXCAM pix, 1 EXCAM pix=0.4347825 l/D, 1 EXCAM pix=21.8213 mas)
    r_off = fits.getdata(os.path.join(ct_filepath, 'hlc_os11_psfs_radial_offsets.fits'))
    r_off_pix = r_off[idx_os11:idx_os11+len(psf_pix_est)] * 2.3
    # Difference between expected and retrieved positions
    diff_pix_x = psf_position_x - psf_pix_est[:,0]
    # os11 azimuthal axis
    assert diff_pix_x == pytest.approx(0)
    # os11 radial axis
    diff_pix_y = psf_position_y + r_off_pix - psf_pix_est[:,1] 
    assert diff_pix_y == pytest.approx(0, abs=0.75)

    # core throughput in [0,1]
    assert np.all(ct_est) >= 0
    assert np.all(ct_est) <= 1

    # Some tolerance for comparison between I/O values. CT in [0,1]
    assert ct_est == pytest.approx(np.array(ct_os11), abs=0.005)

if __name__ == '__main__':

    test_psf_pix_and_ct()


