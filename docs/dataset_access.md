# Dataset access and privacy

The MRI dataset is not included in this repository. 



The repository's manifest uses relative image and mask paths. 

`subject_id,image_path,mask_path,split,spacing_y_mm,spacing_x_mm,slice_id`

Use subject-level assignments so all slices from one subject remain in the same
split. Keep the independent test subjects separate from training, validation,
and checkpoint selection.
