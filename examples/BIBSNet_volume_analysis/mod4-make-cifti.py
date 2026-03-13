import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import os

# Define file paths
template_files = "data/template-files"
src_files = "data/src"
out_dir = "data/outputs"

# df = pd.read_csv('img_bibsnet_space-T1w_desc-aseg_volumes.tsv', sep='\t')
# for sandbox, this file should be pulled from local dir instead and will be a tsv
df = pd.read_csv("../../local/data/img_bibsnet_space-T1w_desc-aseg_volumes.csv")

# Define age column and ROI columns (exclude age columns and Unknown ROI)
age_col='img_bibsnet_space-T1w_desc-aseg_volumes_candidate_age'
prefix='img_bibsnet_space-T1w_desc-aseg_volumes_'
roi_cols = [col for col in df.columns if col.startswith(prefix) and 'age' not in col and 'Unknown' not in col]

# Compute model weights for each ROI using a polynomial fit of age to volume, and store in a new dataframe
roi_weights = []
    
for roi in roi_cols:
    roi_name = roi.replace(prefix, '') # strip prefix for ROIs 
    
    # Flatten into one array (note: ignores repeated participants) and drop NaNs to avoid issues with polynomial fit
    valid = df[[age_col, roi]].dropna()
    x = valid[age_col].to_numpy().ravel()
    y = valid[roi].to_numpy().ravel()

    # Polynomial fit
    coefs = Polynomial.fit(x, y, deg=2).convert().coef

    roi_weights.append({
        "roi": roi_name,
        "intercept": coefs[0],
        "age_linear": coefs[1],
        "age_quadratic": coefs[2]
    })

weights_df = pd.DataFrame(roi_weights)
print(weights_df)

# Can plot or save to tsv file if wanted, but likely not needed in this case
weights_df.to_csv('data/outputs/model-weights.tsv', sep='\t', index=False)
# plt.bar(weights_df['roi'], weights_df['age_linear'])
# plt.xticks(rotation=90)
# plt.ylabel("Age coefficient")
# plt.title("Linear age effect by ROI")
# plt.tight_layout()
# plt.show()

# GENERATE CIFTI FILE
# Read in FreeSurfer LUT to get label IDs for each ROI (this file only includes the ROIs defined for HBCD data) and merge with weights DF on the roi column to integrate model weights
lut_df = pd.read_csv(f"{src_files}/FreeSurfer-LUT-short.csv")
rois_df = pd.merge(lut_df, weights_df, on="roi", how="inner")
print(rois_df)

# Convert aseg labels to weights (output as file roi_weights_volume.nii.gz) using wb_command - uses src aseg file for mapping
expr = " + ".join([f"(aseg=={row.label})*{row.age_linear}" for _, row in rois_df.iterrows()])
cmd = f"wb_command -volume-math '{expr}' {out_dir}/roi_weights_volume.nii.gz -var aseg {template_files}/aseg_dseg.nii.gz"
os.system(cmd)

# Define paths to template files
atlasroi = f"{template_files}/91282_Greyordinates/L.atlasroi.32k_fs_LR.shape.gii"

hemis = ["L", "R"]

for hemi in hemis:
    # Surface files
    midthickness = f"{template_files}/hemi-{hemi}_space-fsLR_den-32k_desc-hcp_midthickness.surf.gii"
    wm = f"{template_files}/hemi-{hemi}_space-fsLR_den-32k_white.surf.gii"
    pial = f"{template_files}/hemi-{hemi}_space-fsLR_den-32k_pial.surf.gii"

    # Output files
    output_shape = f"{out_dir}/{hemi}_linear_age_weights_per_ROI.shape.gii"
    output_shape_dilate = f"{out_dir}/{hemi}_linear_age_weights_per_ROI-dilate.shape.gii"
    output_shape_mask = f"{out_dir}/{hemi}_linear_age_weights_per_ROI-dilate-mask.shape.gii"

    # Commands
    cmds = [
        f"wb_command -volume-to-surface-mapping {out_dir}/roi_weights_volume.nii.gz {midthickness} {output_shape} "
        f"-ribbon-constrained {wm} {pial} -interpolate ENCLOSING_VOXEL",
        f"wb_command -metric-dilate {output_shape} {midthickness} 10 {output_shape_dilate} -nearest",
        f"wb_command -metric-mask {output_shape_dilate} {atlasroi} {output_shape_mask}"
    ]

    for cmd in cmds:
        os.system(cmd)