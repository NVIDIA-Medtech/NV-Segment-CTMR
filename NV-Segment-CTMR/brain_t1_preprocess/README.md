# Brain MRI preprocessing pipeline

## STEP 1 Skull stripping with SynthStrip

Visit the [official website](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) of SynthStrip to download Docker image. Then run:

`python3.8 -u run_synthstrip.py`

## STEP 2 MRI preprocessing
Affine align to the MNI template\
`python3.8 preprocess.py sub-01_T1w.nii.gz mni_icbm152_2009c_t1_1mm_masked_img.nii.gz output.nii.gz --save-preprocess output.preprocess.json`

Affine alignment to the LUMIR template for consistency with the LUMIR dataset\
`python3.8 preprocess.py sub-01_T1w.nii.gz LUMIR_template.nii.gz output.nii.gz --save-preprocess output.preprocess.json`

## STEP 3 Revert to original space
Revert a processed image back to the original space (uses saved metadata)\
`python3.8 revert_preprocess.py output.nii.gz --out output_revert.nii.gz --meta output.preprocess.json`

Revert a processed mask back to the original space\
`python3.8 revert_preprocess.py output.nii.gz --out output_revert.nii.gz --mask output_lbl.nii.gz --mask-out output_lbl_revert.nii.gz --meta output.preprocess.json`