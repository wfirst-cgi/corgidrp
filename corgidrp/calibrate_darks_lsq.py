import os
from pathlib import Path
import numpy as np
import warnings
from astropy.io import fits

from corgidrp.detector import Metadata
import corgidrp.util.check as check
from corgidrp.util.mean_combine import mean_combine
from corgidrp.data import NoiseMap


class CalDarksLSQException(Exception):
    """Exception class for calibrate_darks_lsq."""

def calibrate_darks_lsq(datasets, meta_path=None):
    """The input datasets represents a collection of frame stacks of the
    same number of dark frames (in e- units), where the stacks are for various
    EM gain values and exposure times.  The frames in each stack should be
    SCI full frames that:

    - have had their bias subtracted (assuming 0 bias offset and full frame;
    this function calibrates bias offset)
    - have had masks made for cosmic rays
    - have been corrected for nonlinearity
    - have been converted from DN to e-
    - have had the cosmic ray masks combined with any bad pixel masks which may
    have come from pre-processing if there are any (because creation of the
    fixed bad pixel mask containing warm/hot pixels and pixels with sub-optimal
    functionality requires a master dark, which requires this function first)
    - have been desmeared if desmearing is appropriate.  Under normal
    circumstances, darks should not be desmeared.  The only time desmearing
    would be useful is in the unexpected case that, for example,
    dark current is so high that it stands far above other noise that is
    not smeared upon readout, such as clock-induced charge
    and fixed-pattern noise.

    The steps shown above are a subset of the total number of steps
    involved in going from L1 to L2b.  This function averages
    each stack (which minimizes read noise since it has a mean of 0) while
    accounting for masks.  It then computes a per-pixel map of fixed-pattern
    noise (due to electromagnetic pick-up before going through the amplifier),
    dark current, and the clock-induced charge (CIC), and it also returns the
    bias offset value.  The function assumes the stacks have the same
    noise profile (at least the same CIC, fixed-pattern noise, and dark
    current).

    There are rows of each frame that are used for telemetry and are irrelevant
    for making a master dark.  The count values on those rows are such that
    this function may process the values as saturated, and the function would
    then fail if enough frames in a stack suffer from this apparent saturation.
    So this function disregards telemetry rows and does not do any fitting for
    master dark for those rows.

    Args:
    datasets (list, corgidrp.data.Dataset):
        This is a list of instances of corgidrp.data.Dataset.  Each instance
        should be for a stack of dark frames (counts in DN), and each stack is
        for a unique EM gain and frame time combination.
        Each sub-stack should have the same number of frames.
        Each frame should accord with the EDU
        science frame (specified by corgidrp.util.metadata.yaml).
        We recommend  >= 1300 frames for each sub-stack if calibrating
        darks for analog frames,
        thousands for photon counting depending on the maximum number of
        frames that will be used for photon counting.
    meta_path (str):
        Full path of .yaml file from which to draw detector parameters.
        For format and names of keys, see corgidrp.util.metadata.yaml.
        If None, uses that file.

    Returns:
    F_map : array-like (full frame)
        A per-pixel map of fixed-pattern noise (in e-).  Any negative values
        from the fit are made positive in the end.
    C_map : array-like (full frame)
        A per-pixel map of EXCAM clock-induced charge (in e-). Any negative
        values from the fit are made positive in the end.
    D_map : array-like (full frame)
        A per-pixel map of dark current (in e-/s). Any negative values
        from the fit are made positive in the end.
    bias_offset : float
        The median for the residual FPN+CIC in the region where bias was
        calculated (i.e., prescan). In DN.
    bias_offset_up : float
        The upper bound of bias offset, accounting for error in input datasets
        and the fit.
    bias_offset_low : float
        The lower bound of bias offset, accounting for error in input datasets
        and the fit.
    F_image_map : array-like (image area)
        A per-pixel map of fixed-pattern noise in the image area (in e-).
        Any negative values from the fit are made positive in the end.
    C_image_map : array-like (image area)
        A per-pixel map of EXCAM clock-induced charge in the image area
        (in e-). Any negative values from the fit are made positive in the end.
    D_image_map : array-like (image area)
        A per-pixel map of dark current in the image area (in e-/s).
        Any negative values from the fit are made positive in the end.
    Fvar : float
        Variance of fixed-pattern noise map (in e-).
    Cvar : float
        Variance of clock-induced charge map (in e-).
    Dvar : float
        Variance of dark current map (in e-).
    read_noise : float
        Read noise estimate from the noise profile of a mean frame (in e-).
        It's read off from the sub-stack with the lowest product of EM gain and
        frame time so that the gained variance of C and D is comparable to or
        lower than read noise variance, thus making reading it off doable.
        If read_noise is returned as NaN, the read noise estimate is not
        trustworthy, possibly because not enough frames were used per substack
        for that or because the next lowest gain setting is much larger than
        the gain used in the sub-stack.  The official calibrated read noise
        comes from the k gain calibration, and this is just a rough estimate
        that can be used as a sanity check, for checking agreement with the
        official calibrated value.
    R_map : array-like
        A per-pixel map of the adjusted coefficient of determination
        (adjusted R^2) value for the fit.
    F_image_mean : float
        F averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of F_image_map.  This is just for comparison.
    C_image_mean : float
        C averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of C_image_map.  This is just for comparison.
    D_image_mean : float
        D averaged over all pixels,
        before any negative ones are made positive.  Should be roughly the same
        as taking the mean of D_image_map.  This is just for comparison.
    unreliable_pix_map : array-like (full frame)
        A pixel value in this array indicates how many sub-stacks are usable
        for a fit for that pixel.  For each sub-stack for which
        a pixel is masked for more than half of
        the frames in the sub-stack, 1 is added to that pixel's value
        in unreliable_pix_map.  Since the least-squares fit function has 3
        parameters, at least 4 sub-stacks are needed for a given pixel in order
        to perform a fit for that pixel.  The pixels in unreliable_pix_map that
        are >= len(stack_arr)-3 cannot be fit.  NOTE:  This uses a flag value
        of 1, which falls under the category of
        "Bad pixel - unspecified reason".  Can be changed if necessary.
    F_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated FPN.
    C_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated CIC.
    D_std_map : array-like (full frame)
        The standard deviation per pixel for the calibrated dark current.
    stacks_err : array-like (full frame)
        Standard error per pixel coming from the frames in datasets used to
        calibrate the noise maps.
    F_noise_map : corgidrp.data.NoiseMap instance
        Includes the FPN noise map for the data, F_std_map for the err, and
        unreliable_pix_map for the dq.  The header info is taken from that of
        one of the frames from the input datasets and can be changed via a call
        to the NoiseMaps class if necessary.
    C_noise_map : corgidrp.data.NoiseMap instance
        Includes the CIC noise map for the data, C_std_map combined with
        stacks_err (since CIC is not scaled by exposure time or EM gain when
        making the master dark from the calibrated noise maps) for the err, and
        unreliable_pix_map for the dq.  The header info is taken from that of
        one of the frames from the input datasets and can be changed via a call
        to the NoiseMaps class if necessary.
    D_noise_map : corgidrp.data.NoiseMap instance
        Includes the dark current noise map for the data, D_std_map
        for the err, and unreliable_pix_map for the dq.
        The header info is taken from that of
        one of the frames from the input datasets and can be changed via a call
        to the NoiseMaps class if necessary.
    """

    if len(datasets) <= 3:
        raise CalDarksLSQException('Number of sub-stacks in datasets must '
                'be more than 3 for proper curve fit.')
    # getting telemetry rows to ignore in fit
    if meta_path is None:
        meta = Metadata()
    else:
        meta = Metadata(meta_path)
    metadata = meta.get_data()
    telem_rows_start = metadata['telem_rows_start']
    telem_rows_end = metadata['telem_rows_end']
    telem_rows = slice(telem_rows_start, telem_rows_end)

    g_arr = np.array([])
    t_arr = np.array([])
    k_arr = np.array([])
    mean_frames = []
    mean_num_good_fr = []
    unreliable_pix_map = np.zeros((meta.frame_rows,
                                   meta.frame_cols)).astype(int)
    for i in range(len(datasets)):
        frames = []
        bpmaps = []
        errs = []
        check.threeD_array(datasets[i].all_data,
                           'datasets['+str(i)+'].all_data', TypeError)
        if len(datasets[i].all_data) < 1300:
            warnings.warn('A sub-stack was found with less than 1300 frames, '
            'which is the recommended number per sub-stack for an analog '
            'master dark')
        if i > 0:
            if np.shape(datasets[i-1].all_data) != np.shape(datasets[i].all_data):
                raise CalDarksLSQException('All sub-stacks must have the '
                            'same number of frames and frame shape.')
        try: # if EM gain measured directly from frame TODO change hdr name if necessary
            g_arr = np.append(g_arr, datasets[i].frames[0].ext_hdr['EMGAIN_M'])
        except: # use commanded gain otherwise TODO change hdr name if necessary
            g_arr = np.append(g_arr, datasets[i].frames[0].ext_hdr['CMDGAIN'])
        exptime = datasets[i].frames[0].ext_hdr['EXPTIME']
        cmdgain = datasets[i].frames[0].ext_hdr['CMDGAIN']
        kgain = datasets[i].frames[0].ext_hdr['KGAIN']
        t_arr = np.append(t_arr, exptime)
        k_arr = np.append(k_arr, kgain)
        # check that all frames in sub-stack have same exposure time
        for fr in datasets[i].frames:
            if fr.ext_hdr['EXPTIME'] != exptime:
                raise CalDarksLSQException('The exposure time must be the '
                                           'same for all frames per '
                                           'sub-stack.')
            if fr.ext_hdr['CMDGAIN'] != cmdgain:
                raise CalDarksLSQException('The commanded gain must be the '
                                           'same for all frames per '
                                           'sub-stack.')
            if fr.ext_hdr['KGAIN'] != kgain:
                raise CalDarksLSQException('The k gain must be the '
                                           'same for all frames per '
                                           'sub-stack.')
            # ensure frame is in float so nan can be assigned, though it should
            # already be float
            frame = fr.data.astype(float)
            # For the fit, all types of bad pixels should be masked:
            b1 = fr.dq.astype(bool).astype(int)
            err = fr.err[0]
            frame[telem_rows] = np.nan
            i0 = meta.slice_section(frame, 'image')
            if np.isnan(i0).any():
                raise ValueError('telem_rows cannot be in image area.')
            # setting to 0 prevents failure of mean_combine
            # b0: didn't mask telem_rows b/c they weren't saturated but nan'ed
            frame[telem_rows] = 0
            frames.append(frame)
            bpmaps.append(b1)
            errs.append(err)
        mean_frame, _, map_im, _ = mean_combine(frames, bpmaps)
        mean_err, _, _, _ = mean_combine(errs, bpmaps, err=True)
        pixel_mask = (map_im < len(datasets[i].frames)/2).astype(int)
        mean_num = np.mean(map_im)
        mean_frame[telem_rows] = np.nan
        mean_frames.append(mean_frame)
        mean_num_good_fr.append(mean_num)
        unreliable_pix_map += pixel_mask
    mean_stack = np.stack(mean_frames)
    mean_err_stack = np.stack(mean_err)
    if (unreliable_pix_map >= len(datasets)-3).any():
        warnings.warn('At least one pixel was masked for more than half of '
                      'the frames in some sub-stacks, leaving 3 or fewer '
                      'sub-stacks that did not suffer this masking for these '
                      'pixels, which means the fit was unreliable for '
                      'these pixels.  These are the pixels in the output '
                      'unreliable_pixel_map that are >= len(datasets)-3.')

    if len(np.unique(g_arr)) < 2:
        raise CalDarksLSQException("Must have at least 2 unique EM gains "
                                   'represented by the sub-stacks in '
                                   'datasets.')
    if len(g_arr[g_arr<=1]) != 0:
        raise CalDarksLSQException('Each EM gain must be greater '
            'than 1.')
    if len(np.unique(t_arr)) < 2:
        raise CalDarksLSQException("Must have at 2 unique exposure times.")
    if len(t_arr[t_arr<=0]) != 0:
        raise CalDarksLSQException('Each exposure time must be greater '
            'than 0.')
    if len(k_arr[k_arr<=0]) != 0:
        raise CalDarksLSQException('Each element of k_arr must be greater '
            'than 0.')
    unique_sub_stacks = list(zip(g_arr, t_arr))
    for el in unique_sub_stacks:
        if unique_sub_stacks.count(el) > 1:
            raise CalDarksLSQException('The EM gain and frame time '
            'combinations for the sub-stacks must be unique.')

    # need the correlation coefficient for FPN for read noise estimate later;
    # other noise sources aren't correlated frame to frame
    # Use correlation b/w mean stacks since read noise is negligible (along
    # with dark current and CIC); correlation b/w mean stacks then
    # approximately equal to the correlation b/w FPN from stack to stack
    # this is the stack that will be used later for estimating read noise:
    min1 = np.argmin(g_arr*t_arr)
    # getting next "closest" stack for finding correlation b/w FPN maps:
    # same time, next gain up from least gain (close in time*gain but different
    # gain so that effective read noise values are more uncorrelated)
    tinds = np.where(t_arr == t_arr[min1])
    nextg = g_arr[g_arr > g_arr[min1]].min()
    ginds = np.where(g_arr == nextg)
    intersect = np.intersect1d(tinds, ginds)
    if intersect.size > 0:
        min2 = intersect[0]
    else: # just get next smallest g_arr*t_arr
        min2 = np.where(np.argsort(g_arr*t_arr) == 1)[0][0]
    msi = meta.imaging_slice(mean_stack[min1])
    msi2 = meta.imaging_slice(mean_stack[min2])
    avg_corr = np.corrcoef(msi.ravel(), msi2.ravel())[0, 1]

    # number of observations (i.e., # of averaged stacks provided for fit)
    M = len(g_arr)
    F_map = np.zeros_like(mean_stack[0])
    C_map = np.zeros_like(mean_stack[0])
    D_map = np.zeros_like(mean_stack[0])

    # input data error comes from .err arrays; could use this for error bars
    # in input data for weighted least squares, but we'll just simply get the
    # std error and add it in quadrature to least squares fit standard dev
    stacks_err = np.sqrt(np.sum(mean_err_stack**2, axis=0))/len(mean_err_stack)

    # matrix to be used for least squares and covariance matrix
    X = np.array([np.ones([len(g_arr)]), g_arr, g_arr*t_arr]).T
    mean_stack_Y = np.transpose(mean_stack, (2,0,1))
    params_Y = np.linalg.pinv(X)@mean_stack_Y
    #next line: checked with KKT method for including bounds
    #actually, do this after determining everything else so that
    # bias_offset, etc is accurate
    #params_Y[params_Y < 0] = 0
    params = np.transpose(params_Y, (1,2,0))
    F_map = params[0]
    C_map = params[1]
    D_map = params[2]
    # using chi squared for ordinary least squares (OLS) variance estiamate
    # This is OLS since the parameters to fit are linear in fit function
    # 3: number of fitted params
    residual_stack = mean_stack - np.transpose(X@params_Y, (1,2,0))
    sigma2_frame = np.sum(residual_stack**2, axis=0)/(M - 3)
    # average sigma2 for image area and use that for all three vars since
    # that's only place where all 3 (F, C, and D) should be present
    sigma2_image = meta.imaging_slice(sigma2_frame)
    sigma2 = np.mean(sigma2_image)
    cov_matrix = np.linalg.inv(X.T@X)

    # For full frame map of standard deviation:
    F_std_map = np.sqrt(sigma2_frame*cov_matrix[0,0])
    C_std_map = np.sqrt(sigma2_frame*cov_matrix[1,1])
    # D_std_map here used only for bias_offset error estimate
    D_std_map = np.sqrt(sigma2_frame*cov_matrix[2,2])
    # Dark current should only be in image area, so error only for that area:
    D_std_map_im = np.sqrt(sigma2_image*cov_matrix[2,2])

    var_matrix = sigma2*cov_matrix
    # variances here would naturally account for masked pixels due to cosmics
    # since mean_combine does this
    Fvar = var_matrix[0,0]
    Cvar = var_matrix[1,1]
    Dvar = var_matrix[2,2]

    ss_r = np.sum((np.mean(mean_stack, axis=0) -
        np.transpose(X@params_Y, (1,2,0)))**2, axis=0)
    ss_e = np.sum(residual_stack**2, axis=0)
    Rsq = ss_r/(ss_e+ss_r)
    # adjusted coefficient of determination, adjusted R^2:
    # R^2adj = 1 - (1-R^2)*(n-1)/(n-p), p: # of fitted params
    # The closer to 1, the better the fit.
    # Can have negative values. Can never be above 1.
    # If R_map has nan or inf values, then something is probably wrong;
    # this is good feedback to the user. However, nans in the telemetry rows
    # are expected.
    R_map = 1 - (1 - Rsq)*(M - 1)/(M - 3)

    # doesn't matter which k used for just slicing
    D_image_map = meta.imaging_slice(D_map)
    C_image_map = meta.imaging_slice(C_map)
    F_image_map = meta.imaging_slice(F_map)

    # res: should subtract D_map, too.  D should be zero there (in prescan),
    # but fitting errors may lead to non-zero values there.
    bias_offset = np.zeros([len(mean_stack)])
    bias_offset_up = np.zeros([len(mean_stack)])
    bias_offset_low = np.zeros([len(mean_stack)])
    for i in range(len(mean_stack)):
        # NOTE assume no error in g_arr values (or t_arr or k_arr)
        res = mean_stack[i] - g_arr[i]*C_map - F_map - g_arr[i]*t_arr[i]*D_map
        # upper and lower bounds
        res_up = ((mean_stack[i]+np.abs(stacks_err)) -
                  g_arr[i]*(C_map-C_std_map) - (F_map-F_std_map)
                  - g_arr[i]*t_arr[i]*(D_map-D_std_map))
        res_low = ((mean_stack[i]-np.abs(stacks_err)) -
                   g_arr[i]*(C_map+C_std_map) - (F_map+F_std_map)
                  - g_arr[i]*t_arr[i]*(D_map+D_std_map))
        res_prescan = meta.slice_section(res, 'prescan')
        res_up_prescan = meta.slice_section(res_up, 'prescan')
        res_low_prescan = meta.slice_section(res_low, 'prescan')
        # prescan may contain NaN'ed telemetry rows, so use nanmedian
        bias_offset[i] = np.nanmedian(res_prescan)/k_arr[i] # in DN
        bias_offset_up[i] = np.nanmedian(res_up_prescan)/k_arr[i] # in DN
        bias_offset_low[i] = np.nanmedian(res_low_prescan)/k_arr[i] # in DN
    bias_offset = np.mean(bias_offset)
    bias_offset_up = np.mean(bias_offset_up)
    bias_offset_low = np.mean(bias_offset_low)

    # don't average read noise for all frames since reading off read noise
    # from a frame with gained variance of C and D much higher than read
    # noise variance is not doable
    # read noise must be comparable to gained variance of D and C
    # so we read it off for lowest-gain-time frame
    l = np.argmin((g_arr*t_arr))
    # Num below accounts for pixels lost to cosmic rays
    Num = mean_num_good_fr[l]
    # take std of just image area; more variance if image and different regions
    # included; below assumes no variance inherent in FPN

    mean_stack_image = meta.imaging_slice(mean_stack[l])
    read_noise2 = (np.var(mean_stack_image)*Num -
        g_arr[l]**2*
        np.var(D_image_map*t_arr[l]+C_image_map) -
        ((Num-1)*avg_corr+1)*np.var(F_image_map))
    if read_noise2 >= 0:
        read_noise = np.sqrt(read_noise2)
    else:
        read_noise = np.nan
        warnings.warn('read_noise is NaN.  The number of frames per substack '
                     'should be higher in order for this read noise estimate '
                     'to be reliable. However, if the lowest gain setting '
                      'is much larger than the gain used in the substack, '
                      'the best estimate for read noise may not be good.')

    # actual dark current should only be present in CCD pixels (image area),
    # even if we get erroneous non-zero D values in non-CCD pixels.  Let D_map
    # be zeros everywhere except for the image area
    D_map = np.zeros((meta.frame_rows, meta.frame_cols))
    D_std_map = np.zeros((meta.frame_rows, meta.frame_cols))
    # and reset the telemetry rows to NaN
    D_map[telem_rows] = np.nan

    im_rows, im_cols, r0c0 = meta._imaging_area_geom()
    D_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = D_image_map
    D_std_map[r0c0[0]:r0c0[0]+im_rows,
                    r0c0[1]:r0c0[1]+im_cols] = D_std_map_im

    # now catch any elements that were negative for C and D:
    D_map[D_map < 0] = 0
    C_map[C_map < 0] = 0
    #mean taken before zeroing out the negatives for C and D
    F_image_mean = np.mean(F_image_map)
    C_image_mean = np.mean(C_image_map)
    D_image_mean = np.mean(D_image_map)
    C_image_map[C_image_map < 0] = 0
    D_image_map[D_image_map < 0] = 0

    # Since CIC will not scaled by gain or exptime when master dark created
    # using these noise maps, just bundle stacks_err in with C_std_map
    C_std_map_combo = np.sqrt(C_std_map**2 + stacks_err**2)
    # assume headers from a dataset frame for headers of calibrated noise map
    prihdr = datasets[0].frames[0].pri_hdr
    exthdr = datasets[0].frames[0].ext_hdr
    exthdr['EXPTIME'] = None
    if 'EMGAIN_M' in exthdr.keys():
        exthdr['EMGAIN_M'] = None
    exthdr['CMDGAIN'] = None
    exthdr['KGAIN'] = None
    exthdr['BUNIT'] = 'detected electrons'
    exthdr['HIERARCH DATA_LEVEL'] = None

    err_hdr = fits.Header()
    err_hdr['BUNIT'] = 'detected electrons'

    # make one big dataset of all the input datasets solely for the purpose of
    # input_dataset going into NoiseMap and the recording of filenames
    total_data = np.array([])
    for ds in datasets:
        total_data = np.append(total_data, ds.frames)

    exthdr['DATATYPE'] = 'NoiseMap'

    F_noise_map = NoiseMap(F_map, 'FPN', prihdr.copy(), exthdr.copy(), total_data,
                           F_std_map,
                           unreliable_pix_map, err_hdr=err_hdr)

    C_noise_map = NoiseMap(C_map, 'CIC', prihdr.copy(), exthdr.copy(), total_data,
                           C_std_map_combo,
                           unreliable_pix_map, err_hdr=err_hdr)

    D_noise_map = NoiseMap(D_map, 'DC', prihdr.copy(), exthdr.copy(), total_data,
                           D_std_map,
                           unreliable_pix_map, err_hdr=err_hdr)

    return (F_map, C_map, D_map, bias_offset, bias_offset_up, bias_offset_low,
            F_image_map, C_image_map,
            D_image_map, Fvar, Cvar, Dvar, read_noise, R_map, F_image_mean,
            C_image_mean, D_image_mean, unreliable_pix_map, F_std_map,
            C_std_map, D_std_map, stacks_err, F_noise_map, C_noise_map,
            D_noise_map)