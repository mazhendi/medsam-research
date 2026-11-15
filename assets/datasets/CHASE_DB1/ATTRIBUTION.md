# CHASE_DB1 retinal vessel reference dataset

This directory contains the unmodified `CHASEDB1.zip` data archive (28 fundus photographs,
two independent observer vessel annotations per photograph).

Creators/contributors: Sarah Barman, Andreas Hoppe, Paolo Remagnino, Muhammad Moazam Fraz,
Bunyarit Uyyanonvara, Alicja R. Rudnicka, and Christopher G. Owen. Publisher: Kingston
University, in collaboration with St. George's, University of London. Dataset year: 2012.

**Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).**

- Publisher's record and license statement:
  https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
- License: https://creativecommons.org/licenses/by/4.0/
- Legal code: https://creativecommons.org/licenses/by/4.0/legalcode
- Publication: Fraz et al., "An ensemble classification-based approach applied to retinal
  blood vessel segmentation," IEEE Transactions on Biomedical Engineering 59(9),
  2538–2548, 2012. DOI: https://doi.org/10.1109/TBME.2012.2205687

The publisher's archive endpoint returned HTTP 403 in this execution environment.
The archive was obtained from a **third-party mirror**, not directly from the publisher:

https://raw.githubusercontent.com/playyalp1/CHASE_DB1/1d1e9beb61f5e63fce3e3de3954ddcbf5db00a38/CHASEDB1.zip

SHA-256: `bb750b73633492d700e56d91578ee3cdb7e4dbd08279e7465583e9efd26790b6`

Archive bytes are unchanged. Conversion by `medsam prepare-retina` writes lossless RGB PNGs,
binarizes each observer annotation independently at >0, and does not resize or enhance images.
The generated 16/4/8 image split is our custom child-disjoint development protocol,
**not** an official CHASE_DB1 split. No synthetic field-of-view mask is provided.

Retain this attribution and license link when redistributing this archive or derived data.
The dataset license applies to this data asset independently of the surrounding software.
