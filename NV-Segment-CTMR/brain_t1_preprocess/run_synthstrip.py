import glob
import subprocess


img_dir = 'MRI/Image/Directory/'

for img_i in glob.glob(img_dir+"*.nii.gz"):
    img_name = img_i.split('/')[-1].split('_')[0]
    print(img_name)
    subprocess.call('python3.8 ./synthstrip-docker -i {} -o {} -m {}'.format(img_i, img_dir+'t1w_stripped/'+img_name+'_t1w.nii.gz', img_dir+'t1w_mask/'+img_name+'_t1w_mask.nii.gz'), shell=True)