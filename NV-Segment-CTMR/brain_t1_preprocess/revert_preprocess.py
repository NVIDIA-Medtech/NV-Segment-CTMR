import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import ants
import argparse
import json
import os
import base64
import io

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


def main():
    parser = argparse.ArgumentParser(description="Revert preprocessing back to original image space (except intensity normalization).")
    parser.add_argument("processed_path", help="Path to processed image in template space (NIfTI)")
    parser.add_argument("--out", dest="output_path", nargs='?', default=None,
                        help="Path to save reverted image (NIfTI)")
    parser.add_argument("--mask", dest="mask_path", nargs='?', default=None,
                        help="Path to mask associated with processed image (NIfTI)")
    parser.add_argument("--mask-out", dest="mask_output_path", nargs='?', default=None,
                        help="Path to save reverted mask (NIfTI)")
    parser.add_argument("-t", "--transform", dest="transformlist", nargs='+', default=None,
                        help="Forward transforms from preprocessing (moving->template). Will be inverted.")
    parser.add_argument("--meta", dest="meta_path", default=None,
                        help="Path to preprocessing metadata JSON saved by preprocess.py")
    parser.add_argument("--interp", default="linear", choices=["linear", "nearestNeighbor", "bspline"],
                        help="Interpolator for inverse transform")
    parser.add_argument("--resample-order", type=int, default=2,
                        help="Interpolation order for resampling back to original spacing")
    parser.add_argument("--org", dest="original_path", nargs='?', default=None,
                        help="Path to original image (NIfTI). Optional if --meta includes original geometry")
    parser.add_argument("--template", dest="template_path", nargs='?', default=None,
                        help="Path to template image used in preprocessing (NIfTI). Optional if --meta includes template geometry")
    args = parser.parse_args()

    meta = None
    if args.meta_path is not None:
        with open(args.meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if args.template_path is None:
            args.template_path = meta.get("template_path")
        if args.transformlist is None:
            args.transformlist = meta.get("transformlist")

    if args.template_path is None or args.transformlist is None:
        raise ValueError("template_path and transformlist are required unless provided by --meta")

    processed_nib = nib.load(args.processed_path)
    template_nib = nib.load(args.template_path) if os.path.exists(args.template_path) else None

    original_header = None
    original_affine = None
    if args.original_path is not None:
        original_nib = nib.load(args.original_path)
        original_header = original_nib.header.copy()
        original_affine = original_nib.affine.copy()
    else:
        if meta is None:
            raise ValueError("original_path is required unless --meta includes original geometry")
        original_shape = meta.get("original_shape")
        original_affine = np.array(meta.get("original_affine"), dtype=np.float64)
        original_pixdim = meta.get("original_pixdim")
        header_b64 = meta.get("original_header")
        if original_shape is None or original_affine is None or original_pixdim is None:
            raise ValueError("--meta must include original_shape, original_affine, original_pixdim")
        if header_b64 is not None:
            header_bytes = base64.b64decode(header_b64)
            original_header = nib.Nifti1Header.from_fileobj(io.BytesIO(header_bytes))
        original_nib = nib.Nifti1Image(np.zeros(original_shape, dtype=np.float32), original_affine, header=original_header)
        original_nib.header.set_zooms(original_pixdim)

    if template_nib is None:
        if meta is None:
            raise ValueError("template_path must exist or --meta must include template geometry")
        template_shape = meta.get("template_shape")
        template_affine = np.array(meta.get("template_affine"), dtype=np.float64)
        if template_shape is None or template_affine is None:
            raise ValueError("--meta must include template_shape and template_affine")
        template_nib = nib.Nifti1Image(np.zeros(template_shape, dtype=np.float32), template_affine)

    # Match preprocessing orientation (original -> template orientation)
    original_reoriented = reorient_image_to_match(template_nib, original_nib)

    # Recreate the 1mm resampled grid used in preprocessing
    tar_pixdim = [1.0, 1.0, 1.0]
    if meta is not None and meta.get("target_pixdim") is not None:
        tar_pixdim = meta.get("target_pixdim")
    original_pixdim = original_reoriented.header.structarr['pixdim'][1:-4]
    original_reoriented_npy = original_reoriented.get_fdata()
    ref_resampled_npy = resampling(original_reoriented_npy, original_pixdim, tar_pixdim, order=2)

    # Invert affine registration (template -> resampled original space)
    fixed_ants = ants.from_numpy(ref_resampled_npy)
    moving_ants = ants.from_numpy(processed_nib.get_fdata())
    transformlist = list(args.transformlist)
    invert_flags = [True] * len(transformlist)
    if len(transformlist) > 1:
        transformlist = transformlist[::-1]
        invert_flags = invert_flags[::-1]
    inv_ants = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=transformlist,
        whichtoinvert=invert_flags,
        interpolator=args.interp,
    )

    # Resample back to original spacing
    inv_npy = inv_ants.numpy()
    reverted_npy = resampling(inv_npy, tar_pixdim, original_pixdim, order=args.resample_order)

    
    # Reorient back to original orientation
    reverted_nib = nib.Nifti1Image(reverted_npy, original_reoriented.affine, header=original_reoriented.header)
    reverted_nib = reorient_image_to_match(original_nib, reverted_nib)

    # Save with original header/affine
    out_header = original_header if original_header is not None else original_nib.header.copy()
    qform_code = out_header['qform_code']
    sform_code = out_header['sform_code']
    if hasattr(qform_code, '__len__'):
        qform_code = int(np.array(qform_code).ravel()[0])
    if hasattr(sform_code, '__len__'):
        sform_code = int(np.array(sform_code).ravel()[0])
    out_header.set_qform(original_affine, code=int(qform_code))
    out_header.set_sform(original_affine, code=int(sform_code))
    out_nib = nib.Nifti1Image(reverted_nib.get_fdata(), original_affine, header=out_header)
    nib.save(out_nib, args.output_path)

    if args.mask_path is not None:
        mask_nib = nib.load(args.mask_path)
        mask_npy = mask_nib.get_fdata()
        mask_moving_ants = ants.from_numpy(mask_npy)
        inv_mask_ants = ants.apply_transforms(
            fixed=fixed_ants,
            moving=mask_moving_ants,
            transformlist=transformlist,
            whichtoinvert=invert_flags,
            interpolator="nearestNeighbor",
        )
        inv_mask_npy = inv_mask_ants.numpy()
        reverted_mask_npy = resampling(inv_mask_npy, tar_pixdim, original_pixdim, order=0)
        
        reverted_mask_nib = nib.Nifti1Image(reverted_mask_npy, original_reoriented.affine, header=original_reoriented.header)
        reverted_mask_nib = reorient_image_to_match(original_nib, reverted_mask_nib)

        mask_out_header = out_header.copy()
        mask_out_header.set_qform(original_affine, code=int(qform_code))
        mask_out_header.set_sform(original_affine, code=int(sform_code))

        mask_output_path = args.mask_output_path
        if mask_output_path is None:
            if args.output_path is None:
                raise ValueError("mask_out is required when output_path is not provided")
            mask_output_path = args.output_path.replace('.nii.gz', '.mask.nii.gz')
            if mask_output_path == args.output_path:
                mask_output_path = f"{args.output_path}.mask.nii.gz"

        out_mask_nib = nib.Nifti1Image(reverted_mask_nib.get_fdata(), original_affine, header=mask_out_header)
        nib.save(out_mask_nib, mask_output_path)

if __name__ == '__main__':
    main()
