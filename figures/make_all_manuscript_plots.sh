#!/bin/bash
#=============================================================================
#
# FILE: make_all_manuscript_plots.sh
#
# USAGE: make_all_manuscript_plots.sh
#
# DESCRIPTION: Generate all plots appearing in primary figures.
#
# EXAMPLE: sh make_all_manuscript_plots.sh
#=============================================================================

echo Running run_make_fig0_visual_abstract.sh
echo "####################################################"
sh figures/manuscript/run_make_fig0_visual_abstract.sh
echo "####################################################"

echo Running run_make_fig1_landscape_models.sh
echo "####################################################"
sh figures/manuscript/run_make_fig1_landscape_models.sh
echo "####################################################"

echo Running run_make_fig3_synthetic_training.sh
echo "####################################################"
sh figures/manuscript/run_make_fig3_synthetic_training.sh
echo "####################################################"

echo Running run_make_fig4_sampling_rate_sensitivity.sh
echo "####################################################"
sh figures/manuscript/run_make_fig4_sampling_rate_sensitivity.sh
echo "####################################################"

echo Running run_make_fig5_dimred_schematic.sh
echo "####################################################"
sh figures/manuscript/run_make_fig5_dimred_schematic.sh
echo "####################################################"

echo Running run_make_fig6_facs_training.sh
echo "####################################################"
sh figures/manuscript/run_make_fig6_facs_training.sh
echo "####################################################"

echo Running run_make_fig7_facs_evaluation.sh
echo "####################################################"
sh figures/manuscript/run_make_fig7_facs_evaluation.sh
echo "####################################################"

echo Running run_make_run_make_fig8_saddle_landscapes.sh
echo "####################################################"
sh figures/manuscript/run_make_run_make_fig8_saddle_landscapes.sh
echo "####################################################"
