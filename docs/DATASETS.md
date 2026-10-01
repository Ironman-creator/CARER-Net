# Dataset preparation

This repository does not redistribute fundus images or vessel annotations. Obtain each dataset from its original provider and comply with its terms of use.

- DRIVE: https://drive.grand-challenge.org/
- STARE: https://cecas.clemson.edu/~ahoover/stare/
- CHASE_DB1: use the official CHASE_DB1 distribution associated with the dataset publication.
- FIVES: https://doi.org/10.6084/m9.figshare.19688169.v1

After download, prepare the following directory layout. Image and mask filenames may retain their official names; `data_utils.py` pairs them through stable dataset-specific keys.

```text
data/
  DRIVE/
    train_images/
    train_masks/
    test_images/
    test_masks/
  STARE/
    train_images/
    train_masks/
    test_images/
    test_masks/
  CHASE/
    train_images/
    train_masks/
    test_images/
    test_masks/
  FIVES/
    train_images/
    train_masks/
    test_images/
    test_masks/
```

For FIVES, `prepare_fives.py` can copy the official `train/Original`, `train/Ground truth`, `test/Original`, and `test/Ground truth` folders into the normalized layout:

```bash
python prepare_fives.py --source "/path/to/FIVES_official" --destination data/FIVES
```

Do not commit `data/` to Git. The repository `.gitignore` excludes it.

## Data statement

The associated study used only publicly available, de-identified datasets and did not recruit participants or collect new patient data. The original dataset providers remain responsible for their access conditions and licenses.
