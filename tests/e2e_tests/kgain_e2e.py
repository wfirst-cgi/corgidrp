import argparse
import os, shutil
import glob
import pytest
import numpy as np
import astropy.time as time
import astropy.io.fits as fits
import corgidrp
import corgidrp.data as data
import corgidrp.walker as walker
import corgidrp.caldb as caldb

try:
    from cal.kgain.calibrate_kgain import calibrate_kgain
    import cal
except:
    pass

thisfile_dir = os.path.dirname(__file__) # this file's folder

# tvac_kgain: 8.49404981510777 e-/DN, result from new iit code with specified file input order; used to be #8.8145 #e/DN,
# tvac_readnoise: 121.76070832489948 e-, result from new iit code with specified file input order; used to be 130.12 e-

@pytest.mark.e2e
def test_l1_to_kgain(tvacdata_path, e2eoutput_path):

    # sort and prepare raw files to run through both II&T and DRP
    default_config_file = os.path.join(cal.lib_dir, 'kgain', 'config_files', 'kgain_parms.yaml')
    stack_arr2_f = []
    stack_arr_f = []
    box_data = os.path.join(tvacdata_path, 'TV-20_EXCAM_noise_characterization', 'kgain') 
    for f in os.listdir(box_data):
        file = os.path.join(box_data, f)
        if not file.endswith('.fits'):
            continue
        for i in range(51841, 51871):
            if str(i) in file:
                stack_arr2_f.append(file)
                with fits.open(file, mode='update') as hdus:
                    try:
                        hdus[0].header['OBSTYPE'] = 'MNFRAME'
                    except:
                        pass
                    try:
                        hdus[1].header['OBSTYPE'] = 'MNFRAME'
                    except:
                        pass
                exit
        for i in range(51731, 51841):
            if str(i) in file:
                stack_arr_f.append(file)
                with fits.open(file, mode='update') as hdus:
                    try:
                        hdus[0].header['OBSTYPE'] = 'KGAIN'
                    except:
                        pass
                    try:
                        hdus[1].header['OBSTYPE'] = 'KGAIN'
                    except:
                        pass
                exit
    #stack_arr2 = np.stack(stack_arr2)
    # fileorder_filepath = os.path.join(os.path.split(box_data)[0], 'results', 'TVAC_kgain_file_order.npy')
    #np.save(fileorder_filepath, stack_arr_f+stack_arr2_f)
    stack_arr_f = sorted(stack_arr_f)
    stack_dat = data.Dataset(stack_arr_f)
    stack2_dat = data.Dataset(stack_arr2_f)
    stack_arr2 = stack2_dat.all_data

    split, _ = stack_dat.split_dataset(exthdr_keywords=['EXPTIME'])
    stack_arr = []
    for dset in split:
        # Breaking up the one set that has 10 frames at the same exptime (instead of 5 like all the rest);
        # Making the set 2 separate sets of 5 each perhaps unfairly weights this exptime 
        # which is doubly represented now, but doing this results in the same 8.8145 number from before
        if dset.all_data.shape[0] == 10:
            stack_arr.append(dset.all_data[:5])
            # for ind in [5,6,7,8,9]:
            #     fp = dset.frames[ind].filepath
            #     stack_arr_f.remove(fp)
            
            stack_arr.append(dset.all_data[5:])
            continue
        stack_arr.append(dset.all_data)
    stack_arr = np.stack(stack_arr)
    pass
    #### Done ordering files for II&T and DRP

    ####### ordered_filelist is simply the combination of the the two ordered stacks that are II&T inputs is the input needed for the DRP calibration
    ordered_filelist = stack_arr_f+stack_arr2_f

    ########## Calling II&T code
    (tvac_kgain, tvac_readnoise, mean_rn_std_e, ptc) = calibrate_kgain(stack_arr, stack_arr2, emgain=1, min_val=800, max_val=3000, 
                    binwidth=68, config_file=default_config_file, 
                    mkplot=None, verbose=None)
    
    ########### Now run the DRP

    # make DRP output directory if needed
    kgain_outputdir = os.path.join(e2eoutput_path, "l1_to_kgain_output")
    if os.path.exists(kgain_outputdir):
        shutil.rmtree(kgain_outputdir)
    os.mkdir(kgain_outputdir)

    ####### Run the DRP walker
    walker.walk_corgidrp(ordered_filelist, "", kgain_outputdir, template="l1_to_kgain.json")
    kgain_file = os.path.join(kgain_outputdir, os.path.split(ordered_filelist[0])[1][:-5]+'_kgain.fits') #"CGI_EXCAM_L1_0000051731_kgain.fits")

    kgain = data.KGain(kgain_file)
    
    ##### compare II&T ("TVAC") results with DRP results
    new_kgain = kgain.value
    new_readnoise = kgain.ext_hdr["RN"]
    print("determined kgain:", new_kgain)
    print("determined read noise", new_readnoise)    
    
    diff_kgain = new_kgain - tvac_kgain
    diff_readnoise = new_readnoise - tvac_readnoise
    print ("difference to TVAC kgain:", diff_kgain)
    print ("difference to TVAC read noise:", diff_readnoise)
    print ("error of kgain:", kgain.error)
    print ("error of readnoise:", kgain.ext_hdr["RN_ERR"])

    assert np.abs(diff_kgain) == 0
    assert np.abs(diff_readnoise) == 0 

    this_caldb = caldb.CalDB()
    this_caldb.remove_entry(kgain)

    
if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    tvacdata_dir = '/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder'  #"/home/schreiber/DataCopy/corgi/CGI_TVAC_Data/"
    outputdir = thisfile_dir

    ap = argparse.ArgumentParser(description="run the l1->kgain end-to-end test")
    ap.add_argument("-tvac", "--tvacdata_dir", default=tvacdata_dir,
                    help="Path to CGI_TVAC_Data Folder [%(default)s]")
    ap.add_argument("-o", "--outputdir", default=outputdir,
                    help="directory to write results to [%(default)s]")
    args = ap.parse_args()
    tvacdata_dir = args.tvacdata_dir
    outputdir = args.outputdir
    test_l1_to_kgain(tvacdata_dir, outputdir)
