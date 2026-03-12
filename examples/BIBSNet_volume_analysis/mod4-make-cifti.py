import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# df = pd.read_csv('img_bibsnet_space-T1w_desc-aseg_volumes.tsv', sep='\t')
df = pd.read_csv('../../local/img_bibsnet_space-T1w_desc-aseg_volumes.csv')

age_col='img_bibsnet_space-T1w_desc-aseg_volumes_candidate_age'
prefix='img_bibsnet_space-T1w_desc-aseg_volumes_'
region_cols = [col for col in df.columns if col.startswith(prefix) and 'age' not in col]
roi_weights = []
    
for region_col in region_cols:
    region_name = region_col.replace(prefix, '') # strip prefix for ROIs 
    
    # Flatten into one array (note: ignores repeated participants)
    valid = df[[age_col, region_col]].dropna()
    x = valid[age_col].to_numpy().ravel()
    y = valid[region_col].to_numpy().ravel()

    # Polynomial fit
    coefs = Polynomial.fit(x, y, deg=2).convert().coef

    roi_weights.append({
        "roi": region_name,
        "intercept": coefs[0],
        "age_linear": coefs[1],
        "age_quadratic": coefs[2]
    })

weights_df = pd.DataFrame(roi_weights)
weights_df = weights_df[weights_df['roi'] != 'Unknown'] # remove Unknown ROI 
weights_df.to_csv('../../local/linear-weights.tsv', sep='\t')

plt.bar(weights_df['roi'], weights_df['age_linear'])
plt.xticks(rotation=90)
plt.ylabel("Age coefficient")
plt.title("Linear age effect by ROI")
plt.savefig('linear_age_effects.png')

# plt.bar(weights_df['roi'], weights_df['age_quadratic'])
# plt.xticks(rotation=90)
# plt.ylabel("Age coefficient")
# plt.title("Quadratic age effect by ROI")
# plt.show()

# GENERATE CIFTI FILE - NOT WORKING YET
# Load atlas file (NEED TO FIND ONE TO USE)
atlas = nib.load("atlas.dlabel.nii")
atlas_data = atlas.get_fdata().squeeze()
output = np.zeros_like(atlas_data)

# this might work? need to make sure the ROI names in df match the atlas labels - define label_lookup 
for roi_name, weight in zip(weights_df["roi"], weights_df["linear"]):
    roi_id = label_lookup[roi_name]
    output[atlas_data == roi_id] = weight

# Save CIFTI file using atlas file header info
new_cifti = nib.Cifti2Image(
    output[np.newaxis, :],
    header=atlas.header,
    nifti_header=atlas.nifti_header
)

nib.save(new_cifti, "linear_age_weights_per_ROI.dscalar.nii")

   