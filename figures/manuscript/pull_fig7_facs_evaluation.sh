#!/bin/bash
#=============================================================================
#
# FILE: pull_fig7_facs_evaluation.sh
#
# USAGE: pull_fig7_facs_evaluation.sh [mesc-proj-path]
#
# DESCRIPTION: Copy files from the specified mESC FACS Data project.
#
# EXAMPLE: sh pull_fig7_facs_evaluation.sh $MESC_PROJ_PATH
#=============================================================================

MESC_PROJ_PATH=$1

basedir=$MESC_PROJ_PATH/out
outdir=figures/manuscript/out/fig7_facs_evaluation/facs

mkdir -p $outdir

# Copy images from facs_v3 pca1b
datdir=$basedir/4a_dimred_pca/facs_v3/dec1_fitonsubset/images/pc1pc2

timepoints=(2.0 2.5 3.0 3.5 4.0)
conditions=(
    "CHIR 2-4"
    "CHIR 2-5 FGF 2-4"
)

for t in ${timepoints[@]}; do
    for cond in "${conditions[@]}"; do
        # Copy density plot
        f=$datdir/$t/dec1_density_$cond.pdf;
        fname=dec1_density_"$cond"_$t.pdf
        cp "$f" $outdir/"$fname"
        # Copy scatter/kde plot
        f=$datdir/$t/dec1_scatter_$cond.pdf;
        fname=dec1_scatter_"$cond"_$t.pdf
        cp "$f" $outdir/"$fname"
    done
done

cp $basedir/4a_dimred_pca/facs_v3/dec1_fitonsubset/images/"pca_dec1_CHIR 2-4_sig_hist".pdf $outdir
cp $basedir/4a_dimred_pca/facs_v3/dec1_fitonsubset/images/"pca_dec1_CHIR 2-5 FGF 2-4_sig_hist".pdf $outdir

cp $basedir/1b_signal_plots/"eff_signal_cond_CHIR 2-4.pdf" $outdir
cp $basedir/1b_signal_plots/"eff_signal_cond_CHIR 2-5 FGF 2-4.pdf" $outdir

cp $basedir/4a_dimred_pca/facs_v3/dec1_fitonsubset/images/"pca_decision_1_CHIR 2-4_histograms".pdf $outdir
cp $basedir/4a_dimred_pca/facs_v3/dec1_fitonsubset/images/"pca_decision_1_CHIR 2-5 FGF 2-4_histograms".pdf $outdir

# cp $basedir/2_clustering/images/"corrected_clustering_counts_condition_CHIR 2-4.pdf" $outdir
# cp $basedir/2_clustering/images/"corrected_clustering_counts_condition_CHIR 2-5 FGF 2-4.pdf" $outdir

cp $basedir/2_clustering/images/legend.pdf $outdir
