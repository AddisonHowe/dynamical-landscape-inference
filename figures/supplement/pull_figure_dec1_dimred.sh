#!/bin/bash
#=============================================================================
#
# FILE: pull_figure_dec1_dimred.sh
#
# USAGE: pull_figure_dec1_dimred.sh [mesc-proj-path] [dimred-subdir] [sigs-subdir]
#
# DESCRIPTION: Copy files from the specified mESC FACS Data project.
#
# EXAMPLE: sh pull_figure_dec1_dimred.sh MESC_PROJ_PATH DIMRED_SUBDIR SIGS_SUBDIR
#=============================================================================

MESC_PROJ_PATH=$1
DIMRED_SUBDIR=$2
SIGS_SUBDIR=$3

basedir=$MESC_PROJ_PATH/out
outdir=figures/supplement/out/fig_dec1_dimred/facs

mkdir -p $outdir

# Copy images from directory
datdir=$basedir/${DIMRED_SUBDIR}/images/pc1pc2

timepoints=(2.0 2.5 3.0 3.5)
conditions=(
    "NO CHIR"
    "CHIR 2-3"
    "CHIR 2-5"
)

for t in ${timepoints[@]}; do
    for cond in "${conditions[@]}"; do
        # Copy density plot
        f=$datdir/$t/dec1_density_$cond.pdf;
        fname=density_"$cond"_$t.pdf
        cp "$f" $outdir/"$fname"
        # Copy scatter/kde plot
        f=$datdir/$t/dec1_scatter_$cond.pdf;
        fname=scatter_"$cond"_$t.pdf
        cp "$f" $outdir/"$fname"
    done
done

cp $basedir/${DIMRED_SUBDIR}/images/pca_screeplot.pdf $outdir
cp $basedir/${DIMRED_SUBDIR}/images/pca_loadings.pdf $outdir

cp $basedir/2_clustering/images/legend.pdf $outdir

# cp $basedir/${SIGS_SUBDIR}/"eff_signal_cond_NO CHIR.pdf" $outdir
# cp $basedir/${SIGS_SUBDIR}/"eff_signal_cond_CHIR 2-3.pdf" $outdir
# cp $basedir/${SIGS_SUBDIR}/"eff_signal_cond_CHIR 2-5.pdf" $outdir
