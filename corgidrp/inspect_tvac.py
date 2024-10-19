# If data files are to be copied over another directory for the e2e NL comparison
test_e2e = True
data_dir_test_e2e = '/Users/srhildeb/Documents/GitHub/CGI_TVAC_Data/TV-36_Coronagraphic_Data/Cals/'

import shutil
import numpy as np
from astropy.io import fits

# From Guillermo Gonzalez
# Nonlin 382: 51841-51870 (30), 51731-51840 (110), 51941-51984 (44),
# 51986-52051 (66), 55122-55187 (66), 55191-55256 (66)

# Additional information:
# L1 frame numbers for calibration frames collected during TV-20:
# Calibration product total num frames L1 range
# CIC map 80 50765-50796, 52125-52136, 56128-56133,
# 64348-64377
# Dark map 48 52077-52088, 56104-56109, 64228-64257

# Detailed information from Guillermo Gonzalez (ggonzo@tellus1.com)
# Frames 51841-51870 are used to generate the mean frame
# The EM gain values in the frame headers are 'commanded gain'. The values I
# listed in 'nonlin_table_240322.txt' are 'actual gain' values calculated
# from the EM gain calibration equation (Peter Williams can give you more details
# about that).
#
# norm_val=2020 DN
# NOTE: This and the following pixel values are from our Matlab script.
# To apply this to Python, subtract 1 from each.
# offset_colroi1=800 (-1)
# offset_colroi2=1000  (-1)
# offset_rowroi1=100  (-1)
# offset_rowroi2=1000  (-1)
# rowroi1=305  (-1)
# rowroi2=735  (-1)
# colroi1=1385  (-1)
# colroi2=1845  (-1)
# rowback11=20  (-1)
# rowback12=300  (-1)
# rowback21=740  (-1)
# rowback22=1000  (-1)
# colback11=1200  (-1)
# colback12=2000  (-1)
# colback21=1200  (-1)
# colback22= 2000  (-1)
# min_exp_time= 0,
# num_bins=didn't use this parameter in the original analysis (adopted default)
# min_bin= didn't use this parameter in the original analysis
# min_mask_factor= didn't use this parameter in the original analysis
# NOTE: I hard-wired 'min_mask' to be 1500 DN in the original analysis

# The parameters that were not used were substituted by the hard-wired value.

# Steps to be done:
# 1/ Write a recipe and double check all the necessary steps in the FDD.
#    1.1/ When does ERR field get generated if we start with L1 data?
#    1.2/ When does DQ field get generated if we start with L1 data?
#    1.3/ Apply bias subtraction: do we have to use return_full_frame=True?
#    1.3/ Apply CR detection
#    1.4/ (?) Apply NL correction (needs to generate the NL correction first)
#    1.5/ (?) Update to L2a
# 2/ Modify the corgidrp.walker.guess_template() function to add logic for
# determining when to use your recipe based on header keywords (e.g., OBSTYPE)
# 3/ Create them as part of the setup process during the script
# (see tests/e2e_tests/l1_to_l2b_e2e.py for examples of how to do this for each
# type of calibration)
# 4/ Write a single e2e script. See examples in tests/e2e_tests/.
# 5/ Test e2e script
# 6/ run /usr/bin/time -v python your_e2e_test.py (%CPU, elapsed time, Maximum
# resident set size (KB)) [w/o -v option:       243.31 real        65.56 user       121.30 sys]
# 7/ Document your recipe on https://collaboration.ipac.caltech.edu/display/romancoronagraph/Corgi-DRP+Implementation+Document#CorgiDRPImplementationDocument-2.0Implementation
# 8/ Submit PR!

data_dir = 'TV-20_EXCAM_noise_characterization/nonlin/'

# TVAC files
tvac_file_0 = [
'CGI_EXCAM_L1_0000051841.fits',
'CGI_EXCAM_L1_0000051731.fits',
'CGI_EXCAM_L1_0000051941.fits',
'CGI_EXCAM_L1_0000051986.fits',
'CGI_EXCAM_L1_0000055122.fits',
'CGI_EXCAM_L1_0000055191.fits']

n_files = [30, 110, 44, 66, 66, 66]
if len(tvac_file_0) != len(n_files):
    raise Exception('Inconsistent number of files and stacks')

frame_list = []
total_exp_time_s = 0
for i_group, file in enumerate(tvac_file_0):
    l1_number = int(file[file.find('L1_')+3:file.find('L1_')+13])
    print(f'Group of {n_files[i_group]} files starting with {file}')
    for i_file in range(n_files[i_group]):
        file_name = f'CGI_EXCAM_L1_00000{l1_number+i_file}.fits'
        fits_file = fits.open(data_dir+file_name)
        prihdr = fits_file[0].header
        exthdr = fits_file[1].header
        # Add OBSTYPE
        if n_files[i_group] == 30:
            prihdr['OBSTYPE'] = 'MNFRAME'
        else:
            prihdr['OBSTYPE'] = 'NONLIN'
        data = fits_file[1].data * 1.
        # Generate CORGIDRP image
        frame_list.append(Image(data, pri_hdr = prihdr, ext_hdr = exthdr))
        print(f"File {file_name} is type {exthdr['ARRTYPE']} {exthdr['HIERARCH DATA_LEVEL']}, EM={exthdr['CMDGAIN']}, EXP TIME={exthdr['EXPTIME']} seconds, OBSTYPE={prihdr['OBSTYPE']}, ARRTYPE={exthdr['ARRTYPE']}")#, DATETIME {exthdr['DATETIME']}")
        total_exp_time_s += float(exthdr['EXPTIME'])
        # Optionally copy data for the e2e test comparison
        # Following folder naming convention from Guillermo Gonzalez/tellus1 ('Comparing NL calibration' 9/6/24)
        if test_e2e:
            # Mean frame: 51841-51870
            if 51841 <= l1_number+i_file <= 51870:
                data_e2e = 'data_30_frames/'
            else:
                approx_gain = round(exthdr['CMDGAIN']*10)/10
                if approx_gain == 1:
                    data_e2e = 'G_1/data_all/'
                elif approx_gain == 1.6:
                    data_e2e = 'G_2/data_all/'
                elif approx_gain == 5.2:
                    data_e2e = 'G_10/data_all/'
                elif approx_gain == 8.6:
                    data_e2e = 'G_20/data_all/'
                elif approx_gain == 16.7:
                    data_e2e = 'G_50/data_all/'
                elif approx_gain == 27.5:
                    data_e2e = 'G_100/data_all/'
                elif approx_gain == 45.3:
                    data_e2e = 'G_200/data_all/'
                elif approx_gain == 87.5:
                    data_e2e = 'G_500/data_all/'
                elif approx_gain == 144.1:
                    data_e2e = 'G_1000/data_all/'
                elif approx_gain == 237.3:
                    data_e2e = 'G_2000/data_all/'
                elif approx_gain == 458.7:
                    data_e2e = 'G_5000/data_all/'
                elif approx_gain == 584.4:
                    data_e2e = 'G_7000/data_all/'
                else:
                    raise ValueError(f"EM gain {exthdr['CMDGAIN']} not in the list")
            # Copy data_dir+file_name to data_dir_test_e2e + data_e2e
            shutil.copy(data_dir+file_name, data_dir_test_e2e+data_e2e) 
    print('****************************************************************')
# Print total exposure time
print(f'The total exposure time in all the frames is {total_exp_time_s} sec') 

