# FACS Data

### Binary decision 1

- `facs_dec1_v1`: 

Data source: [https://github.com/AddisonHowe/mescs-invitro-facs](https://github.com/AddisonHowe/mescs-invitro-facs). \
Copied from `out/4a_dimred_pca/facs_v3/dec1_fitonsubset/transition1_subset_epi_tr_ce_an_pc12`. \
Includes training, validation, and testing split of experimental conditions. \
Cells are labeled, and those involved in the first binary decision are isolated. \
PCA is performed on the combined subset of cells collected at days 2 and 3.5. \
Note that PCA, as applied via sklearn, performs mean-centering but not variance normalization.
Therefore, the cell subset is mean-centered, and any subsequent transformation using the PCA object will first translate the cells according to this mean.
All cells, spanning all timepoints, are projected onto the plane spanned by the first two PCs.

- `facs_dec1_v2`: 

Data source: [https://github.com/AddisonHowe/mescs-invitro-facs](https://github.com/AddisonHowe/mescs-invitro-facs). \
Copied from `out/4a_dimred_pca/facs_v5/dec1_fitonsubset_lognorm/transition1_subset_epi_tr_ce_an_pc12`. \
Includes training, validation, and testing split of experimental conditions. \
Cells are labeled, and those involved in the first binary decision are isolated. \
FACS data is log normalized using $y=\log_{10}(1+x)$, where $x$ is the fluorescence level provided. \
PCA is performed on the combined subset of cells collected at days 2 and 3.5. \
Note that PCA, as applied via sklearn, performs mean-centering but not variance normalization.
Therefore, the cell subset is mean-centered, and any subsequent transformation using the PCA object will first translate the cells according to this mean.
All cells, spanning all timepoints between D2.0 and D3.5, are projected onto the plane spanned by the first two PCs.

- `facs_dec1_v3`: 

Data source: [https://github.com/AddisonHowe/mescs-invitro-facs](https://github.com/AddisonHowe/mescs-invitro-facs). \
Copied from `out/4a_dimred_pca/facs_v5/dec1_fitonsubset_logicle/transition1_subset_epi_tr_ce_an_pc12`. \
Includes training, validation, and testing split of experimental conditions. \
Cells are labeled, and those involved in the first binary decision are isolated. \
FACS data is normalized using logicle scaling. \
PCA is performed on the combined subset of cells collected at days 2 and 3.5. \
Note that PCA, as applied via sklearn, performs mean-centering but not variance normalization.
Therefore, the cell subset is mean-centered, and any subsequent transformation using the PCA object will first translate the cells according to this mean.
All cells, spanning all timepoints, are projected onto the plane spanned by the first two PCs.

- `facs_dec1_v4`: 

Data source: [https://github.com/AddisonHowe/mescs-invitro-facs](https://github.com/AddisonHowe/mescs-invitro-facs). \
Copied from `out/4a_dimred_pca/facs_v5/dec1_fitonsubset/transition1_subset_epi_tr_ce_an_pc12`. \
Includes training, validation, and testing split of experimental conditions. \
Cells are labeled, and those involved in the first binary decision are isolated. \
PCA is performed on the combined subset of cells collected at days 2 and 3.5. \
Note that PCA, as applied via sklearn, performs mean-centering but not variance normalization.
Therefore, the cell subset is mean-centered, and any subsequent transformation using the PCA object will first translate the cells according to this mean.
All cells, spanning all timepoints between D2.0 and D3.5, are projected onto the plane spanned by the first two PCs.

### Binary decision 2

- `facs_dec2_v1`: 

Data source: [https://github.com/AddisonHowe/mescs-invitro-facs](https://github.com/AddisonHowe/mescs-invitro-facs). \
Copied from `out/4a_dimred_pca/facs_v4/dec2_fitonsubset/transition2_subset_ce_pn_m_pc12`. \
Includes training, validation, and testing split of experimental conditions. \
Cells are labeled, and those involved in the second binary decision are isolated. \
PCA is performed on the combined subset of cells collected at days 3 and 5. \
Therefore, the cell subset is mean-centered, and any subsequent transformation using the PCA object will first translate the cells according to this mean.
All cells, spanning all timepoints between D3.0 and D5.0, are projected onto the plane spanned by the first two PCs.

- `facs_dec2_v2`: 

Data source: [https://github.com/AddisonHowe/mescs-invitro-facs](https://github.com/AddisonHowe/mescs-invitro-facs). \
Copied from `out/4a_dimred_pca/facs_v5/dec2_fitonsubset_lognorm/transition2_subset_ce_pn_m_pc12`. \
Includes training, validation, and testing split of experimental conditions. \
Cells are labeled, and those involved in the second binary decision are isolated. \
FACS data is log normalized using $y=\log_{10}(1+x)$, where $x$ is the fluorescence level provided. \
PCA is performed on the combined subset of cells collected at days 3 and 5. \
Therefore, the cell subset is mean-centered, and any subsequent transformation using the PCA object will first translate the cells according to this mean.
All cells, spanning all timepoints between D3.0 and D5.0, are projected onto the plane spanned by the first two PCs.

- `facs_dec2_v3`: 

Data source: [https://github.com/AddisonHowe/mescs-invitro-facs](https://github.com/AddisonHowe/mescs-invitro-facs). \
Copied from `out/4a_dimred_pca/facs_v5/dec2_fitonsubset_logicle/transition2_subset_ce_pn_m_pc12`. \
Includes training, validation, and testing split of experimental conditions. \
Cells are labeled, and those involved in the second binary decision are isolated. \
FACS data is normalized using logicle scaling. \
PCA is performed on the combined subset of cells collected at days 3 and 5. \
Therefore, the cell subset is mean-centered, and any subsequent transformation using the PCA object will first translate the cells according to this mean.
All cells, spanning all timepoints between D3.0 and D5.0, are projected onto the plane spanned by the first two PCs.
