from intensity_normalization.normalize.kde import KDENormalize
from intensity_normalization.typing import Modality, TissueType
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import ants
import argparse
import json
import os
import shutil
import base64

def reorient_image_to_match(reference_nii, target_nii):
    reference_ornt = nib.aff2axcodes(reference_nii.affine)
    target_reoriented = nib.as_closest_canonical(target_nii, enforce_diag=False)
    target_ornt = nib.aff2axcodes(target_reoriented.affine)
    
    # If orientations don't match, perform reorientation
    if target_ornt != reference_ornt:
        # Calculate the transformation matrix to match the reference orientation
        ornt_trans = nib.orientations.ornt_transform(nib.io_orientation(target_reoriented.affine),
                                                     nib.io_orientation(reference_nii.affine))
        target_reoriented = target_reoriented.as_reoriented(ornt_trans)
    return target_reoriented


def resampling(img_npy, img_pixdim, tar_pixdim, order, mode='constant'):
    if order == 0:
        img_npy = img_npy.astype(np.uint16)
    img_npy = zoom(img_npy, ((img_pixdim[0] / tar_pixdim[0]), (img_pixdim[1] / tar_pixdim[1]), (img_pixdim[2] / tar_pixdim[2])), order=order, prefilter=False, mode=mode)
    return img_npy

def intensity_norm(img_npy: np.ndarray, mod: Modality) -> np.ndarray:
    kde_norm = KDENormalize(norm_value=110)
    out = kde_norm(img_npy.astype(np.float32), modality=mod)
    out[out < 0] = 0
    vmax = np.percentile(out[out > 0], 99.9) if np.any(out > 0) else 1.0
    out = np.clip(out / max(vmax, 1e-6), 0, 1).astype(np.float32)
    return out

def make_affine_from_pixdim(pixdim):
    # Create a 4x4 affine with spacing along the diagonal
    affine = np.eye(4)
    affine[0, 0] = pixdim[0]
    affine[1, 1] = pixdim[1]
    affine[2, 2] = pixdim[2]
    return affine

def main():
    parser = argparse.ArgumentParser(description="Affine register and normalize image to template space.")
    parser.add_argument("img_path", help="Path to moving image (NIfTI)")
    parser.add_argument("template_path", help="Path to template image (NIfTI)")
    parser.add_argument("output_path", help="Path to save output image (NIfTI)")
    parser.add_argument("-m", "--mask", dest="mask_path", default=None,
                    help="Path to brain mask of the moving image (NIfTI)")
    parser.add_argument("-s", "--save-preprocess", dest="meta_path", default=None,
                    help="Path to save preprocessing metadata (JSON). Defaults to output_path + '.preprocess.json'")
    args = parser.parse_args()

    img_modality = Modality.T1
    img_nib = nib.load(args.img_path)
    template_nib = nib.load(args.template_path)
    original_affine = img_nib.affine.copy()
    original_shape = img_nib.shape
    original_pixdim = img_nib.header.structarr['pixdim'][1:-4]
    if args.mask_path is not None:
        mask_nib = nib.load(args.mask_path)
        mask_nib = reorient_image_to_match(template_nib, mask_nib)

    img_nib = reorient_image_to_match(template_nib, img_nib)
    affine_type = 'Affine'
    affine_metric = 'meanSquares'
    tar_pixdim = [1.0, 1.0, 1.0]  # Target pixel dimensions
    img_pixdim = img_nib.header.structarr['pixdim'][1:-4]
    img_npy = img_nib.get_fdata()
    if args.mask_path is not None:
        mask_npy = mask_nib.get_fdata()
        print(img_npy.shape, mask_npy.shape)
        img_npy = img_npy * (mask_npy > 0)
        
    # N4 bias field correction
    img_ants = ants.from_numpy(img_npy)
    img_ants = ants.n4_bias_field_correction(img_ants)
    img_npy = img_ants.numpy()
    # Intensity normalization
    img_npy = intensity_norm(img_npy, img_modality)
    img_npy = resampling(img_npy, img_pixdim, tar_pixdim, order=2)
    
    # Affine registration
    tmp_npy = template_nib.get_fdata()
    tmp_npy = intensity_norm(tmp_npy, img_modality)
    
    tmp_ants = ants.from_numpy(tmp_npy)
    img_ants = ants.from_numpy(img_npy)
    regMovTmp = ants.registration(fixed=tmp_ants, moving=img_ants, type_of_transform=affine_type, aff_metric=affine_metric)
    img_ants = ants.apply_transforms(fixed=tmp_ants, moving=img_ants, transformlist=regMovTmp['fwdtransforms'],)
    
    img_npy = img_ants.numpy()
    nib_img = nib.Nifti1Image(img_npy, template_nib.affine, header=template_nib.header)
    nib.save(nib_img, args.output_path)

    output_root = args.output_path
    if output_root.endswith('.nii.gz'):
        output_root = output_root[:-7]
    else:
        output_root = os.path.splitext(output_root)[0]
    
    meta_path = args.meta_path or f"{output_root}.preprocess.json"
    saved_transforms = []
    for idx, tfm_path in enumerate(regMovTmp['fwdtransforms']):
        tfm_name = os.path.basename(tfm_path)
        tfm_out = f"{output_root}.{idx}.{tfm_name}"
        shutil.copy(tfm_path, tfm_out)
        saved_transforms.append(tfm_out)

    try:
        header_bytes = img_nib.header.binaryblock
    except Exception:
        header_bytes = img_nib.header.as_bytes()

    meta = {
        "template_path": os.path.abspath(args.template_path),
        "template_affine": template_nib.affine.tolist(),
        "template_shape": list(template_nib.shape),
        "original_affine": original_affine.tolist(),
        "original_shape": list(original_shape),
        "original_pixdim": original_pixdim.tolist(),
        "original_header": base64.b64encode(header_bytes).decode('ascii'),
        "target_pixdim": tar_pixdim,
        "transformlist": saved_transforms,
        "output_path": os.path.abspath(args.output_path),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    
if __name__ == '__main__':
    main()