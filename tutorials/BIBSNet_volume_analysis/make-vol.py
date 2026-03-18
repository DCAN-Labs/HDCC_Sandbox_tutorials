import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Define file paths
src = "data/src"
out_dir = "data/outputs"

## CAN USE T1 OR T2 RESULTS HERE
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
# print(weights_df)

# Can plot or save to tsv file if wanted, but likely not needed in this case
weights_df.to_csv(f'{out_dir}/linear-model-weights.tsv', sep='\t', index=False)
plt.bar(weights_df['roi'], weights_df['age_linear'])
plt.xticks(rotation=90)
plt.ylabel("Age coefficient")
plt.title("Linear age effect by ROI")
plt.tight_layout()
plt.savefig(f'{out_dir}/linear_age_effects_by_roi.png')
# plt.show()

# GENERATE CIFTI FILE
# Read in FreeSurfer LUT to get label IDs for each ROI (this file only includes the ROIs defined for HBCD data) and merge with weights DF on the roi column to integrate model weights
lut_df = pd.read_csv(f"{src}/FreeSurfer-LUT-short.csv")
roi_df = pd.merge(lut_df, weights_df, on="roi", how="inner")
roi_df.to_csv(f'{out_dir}/model-weights-roi-labels.tsv', sep='\t', index=False)
print(roi_df)

# Convert aseg ROI labels to model weight values (output as file roi_weights_volume.nii.gz)
template_aseg = f'{src}/sub-380056/ses-7mo/anat/sub-380056_ses-7mo_space-INFANTMNIacpc_desc-aseg_dseg.nii.gz'
out_vol = f'{out_dir}/roi_weights_volume.nii.gz'

# Load aseg
aseg_img = nib.load(template_aseg)
aseg_data = aseg_img.get_fdata().astype(int)

# Create output array (float for weights)
weights_data = np.zeros_like(aseg_data, dtype=float)

# Map labels > weights and apply mapping
label_to_weight = dict(zip(roi_df["label"], roi_df["age_linear"]))

for label, weight in label_to_weight.items():
    weights_data[aseg_data == label] = weight

# Save new NIfTI
out_img = nib.Nifti1Image(weights_data, affine=aseg_img.affine, header=aseg_img.header)
nib.save(out_img, out_vol)

