import numpy as np
import pandas as pd
import os
from nilearn import datasets, image, plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.reporting import get_clusters_table

# ------------------------
# STEP 1: Fetch Dataset
# ------------------------
dataset = datasets.fetch_openneuro_dataset(
    dataset_version='ds000114',
    data_dir='.',
    include=['sub-01', 'sub-02', 'sub-03'],
    types=["func", "anat", "events"]
)

func_files = sorted(dataset.func)
anat_files = sorted(dataset.anat)
events_files = sorted(dataset.events)

# ------------------------
# STEP 2: First-Level GLM
# ------------------------
first_level_models = []
contrast_maps = []

for i, (func, anat, events) in enumerate(zip(func_files, anat_files, events_files)):
    print(f"Running subject {i+1}")
    
    # Load events
    events_df = pd.read_csv(events, sep='\t')
    events_df = events_df[events_df['trial_type'].isin(['left_hand', 'right_hand'])]

    # Set up GLM
    model = FirstLevelModel(t_r=2.5, slice_time_ref=0.5, hrf_model='spm',
                            mask_img=None, noise_model='ar1', standardize=True,
                            minimize_memory=True)
    
    model.fit(run_imgs=func, events=events_df)

    # Compute contrast
    design = model.design_matrices_[0].columns
    print(f"Design matrix columns: {design}")
    
    # 'left_hand' vs 'right_hand'
    contrast = model.compute_contrast('left_hand - right_hand', output_type='z_score')
    
    contrast.to_filename(f"sub-{i+1}_contrast.nii.gz")
    contrast_maps.append(contrast)
    first_level_models.append(model)

# ------------------------
# STEP 3: Second-Level GLM
# ------------------------
design_matrix = pd.DataFrame([1] * len(contrast_maps), columns=["intercept"])

second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(contrast_maps, design_matrix=design_matrix)

z_map = second_level_model.compute_contrast(output_type='z_score')

# Save and plot
z_map.to_filename("group_z_map.nii.gz")

plotting.plot_stat_map(
    z_map,
    threshold=3.1,
    display_mode="z",
    cut_coords=7,
    title="Group-level contrast: Left > Right Hand"
)

# ------------------------
# STEP 4: Cluster Table
# ------------------------
table = get_clusters_table(z_map, stat_threshold=3.1, cluster_threshold=10)
print(table)
