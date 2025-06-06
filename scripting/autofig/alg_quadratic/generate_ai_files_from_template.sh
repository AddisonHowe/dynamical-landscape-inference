#!/bin/bash

# Source location of Adobe Illustrator and project directory.
source .env

ILLUSTRATOR_PATH=$ILLUSTRATOR_PATH
PROJ_DIR_TILDE=$PROJ_DIR_TILDE

# Base project directory, using tilde without expansion as explained below.
PROJ_DIR="${PROJ_DIR_TILDE/#\~/$HOME}"

# Output filename suffix, to append to the model name
IMAGE_SUFFIX=image1

sleeptime=3

# Template .ai file, and the folder containing images linked in the template.
# Note that the folder name will be replaced, and therefore needs to use the
# tilde explicitly, without substitution, for the filename.
template_fpath=$PROJ_DIR/scripting/autofig/alg_quadratic/alg_quadratic_template1.ai
template_linkdir=$PROJ_DIR_TILDE/scripting/autofig/alg_quadratic/template1_images

# Log file
logfpath=$PROJ_DIR/scripting/logs/log_alg_quadratic.txt
echo "Log" > $logfpath

# This is where all generated ai files will be stored, one for every run below.
aioutdir=${PROJ_DIR}/figures/autofig/out/alg_quadratic
mkdir -p $aioutdir

# Script to modify the links in an .ai file, with placeholder in/out files,
# and temporary generated script, with placeholder in/out files replaced
scriptfpath=$PROJ_DIR/scripting/autofig/modify_links.jsx
tmp_script_fpath=$PROJ_DIR/scripting/autofig/alg_quadratic/_tmp_modify_links.jsx

# Directories containing images corresponding to trained models.
RUNDIRBASE=data/model_evaluation/eval_models_alg_quadratic
rundirs=$(ls $RUNDIRBASE)
# rundirs=$(ls $RUNDIRBASE | grep subsetkey)  # To only apply on subset

# Main Loop
for modelname in ${rundirs[@]}; do
    rd=$PROJ_DIR/$RUNDIRBASE/$modelname
    echo $modelname
    echo $modelname >> $logfpath
    fname=${modelname}_${IMAGE_SUFFIX}
    cp $template_fpath $aioutdir/$fname.ai
    open -g -a "$ILLUSTRATOR_PATH" $aioutdir/$fname.ai
    sleep $sleeptime
    sed -e "s|<OLD_FOLDER_PATH>|$template_linkdir|" \
        -e "s|<LOGFPATH>|$logfpath|" \
        -e "s|<NEW_FOLDER_PATH>|$rd|" $scriptfpath > $tmp_script_fpath
    osascript -e 'tell application "Adobe Illustrator" to do javascript file "'"$tmp_script_fpath"'"' && >> $logfpath;
    rm $tmp_script_fpath
    rm $aioutdir/$fname.ai  # remove .ai file, keeping only the pdf
done
echo Done!
