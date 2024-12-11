#!/bin/bash
#=============================================================================
#
# FILE: pull_figure_dec2_dimred.sh
#
# USAGE: pull_figure_dec2_dimred.sh [mesc-proj-path] [dimred-subdir] [sigs-subdir]
#
# DESCRIPTION: Copy files from the specified mESC FACS Data project.
#
# EXAMPLE: sh pull_figure_dec2_dimred.sh MESC_PROJ_PATH DIMRED_SUBDIR SIGS_SUBDIR
#=============================================================================

MESC_PROJ_PATH=$1
DIMRED_SUBDIR=$2
SIGS_SUBDIR=$3

basedir=$MESC_PROJ_PATH/out
outdir=figures/supplement/out/fig_dec2_dimred/facs

mkdir -p $outdir

# Copy images from directory
datdir=$basedir/${DIMRED_SUBDIR}/images/pc1pc2

timepoints=(3.0 3.5 4.0 4.5 5.0)
conditions=(
    "CHIR 2-3"
    "CHIR 2-5"
    "CHIR 2-5 FGF 2-3"
)

for t in ${timepoints[@]}; do
    for cond in "${conditions[@]}"; do
        # Copy density plot
        f=$datdir/$t/dec2_density_$cond.pdf;
        fname=density_"$cond"_$t.pdf
        cp "$f" $outdir/"$fname"
        # Copy scatter/kde plot
        f=$datdir/$t/dec2_scatter_$cond.pdf;
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
