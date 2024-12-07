#!/bin/bash
#=============================================================================
#
# FILE: make_all_supplement_plots.sh
#
# USAGE: make_all_supplement_plots.sh
#
# DESCRIPTION: Generate all plots appearing in supplemental figures.
#
# EXAMPLE: sh make_all_supplement_plots.sh
#=============================================================================

runs=(
    run_make_figure_binary_choice
    run_make_figure_binary_flip
    run_make_figure_phi1_training
    run_make_figure_phi2_training
    run_make_figure_phi1_results
    run_make_figure_phi2_results
    run_make_figure_phi1_fixed_point_comparison
    run_make_figure_phi2_fixed_point_comparison
    run_make_figure_phi1_fixed_point_history
    run_make_figure_phi2_fixed_point_history
)

for runname in "${runs[@]}"; do
    echo Running ${runname}.sh
    echo "####################################################"
    sh figures/supplement/${runname}.sh
    echo "####################################################"
done
