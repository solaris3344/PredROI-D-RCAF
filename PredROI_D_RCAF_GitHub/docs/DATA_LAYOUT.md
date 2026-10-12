# Data layout

The repository does **not** redistribute the Cheng CE-MRI or BDNeuro-MRI datasets.

## Cheng CE-MRI

The main scripts expect:

1. A CSV manifest with one row per MRI slice.
2. A patient identifier for every slice.
3. Either `class_name` (`glioma`, `meningioma`, `pituitary`) or an integer `label`.
4. A valid PNG path (`png_path`, `path`, `image_path`, or `filepath`).
5. A MAT/HDF5 directory containing the corresponding mask file for each PNG stem.

Example:

```text
png_path,patient_id,class_name
C:\DATA\Cheng\png\1.png,P001,glioma
```

The primary patient-disjoint pipeline uses tumor masks only for localizer fitting on training patients. Classifier-training crops are generated out of fold, and outer-validation crops are image-only.

## BDNeuro-MRI v8

External evaluation scripts accept a dataset root through `--dataset_root`. The Cheng-trained models are evaluated without BDNeuro fine-tuning. Because patient identifiers are unavailable in the external set used in the paper, external endpoints are image-level.

## Checkpoints and generated outputs

Checkpoints and run outputs are intentionally excluded from the repository ZIP. They can be placed anywhere and passed through the relevant command-line arguments.
