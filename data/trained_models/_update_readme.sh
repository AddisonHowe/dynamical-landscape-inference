#!/bin/bash

file="data/trained_models/README.md"

echo_data_string () {
    modelgroup=$1
    datdir=$2
    if [[ $modelgroup == "facs" ]]; then
        echo '- Data source: `'${datdir#data/training_data/facs/}'`\n'
    else 
        echo '- Data source: `'${datdir#data/training_data/}'`\n'
    fi
}

echo_arglist_string () {
    modelgroup=$1
    arglist_fname=$2 
    arglist_fpath=$3
    echo \\t'- ['$arglist_fname']('${arglist_fpath/data/..}')\n'
}

echo_model_string () {
    modelgroup=$1
    modelname=$2
    modeldir=./$modelgroup/$modelname
    echo \\t\\t'- ['$modelname']('${modeldir}')\n'
}

echo_trained_models () {
    basemodeldir=$1
    modelgroup=$(basename $basemodeldir)  # plnn_synbindec, facs, etc.
    # Get list of models in the directory, and the corresponding argument file
    modelnames=()
    arglistfnames=()
    datdirs=()
    errors=()
    for modeldir in $basemodeldir/model_*; do
        modelname=$(basename $modeldir)
        modelnames+=($modelname)
        # Remove prefix model_ and suffix timestamp from to get arglist filename
        arglistfname=${modelname#model_}
        arglistfname=run_${arglistfname::-16}
        arglistfnames+=($arglistfname)
        # Get the filepath to the argument file
        arglistfpath=data/model_training_args/$modelgroup/$arglistfname.tsv
        # If the argument file exists, get the data directory
        if [[ -f "$arglistfpath" ]]; then
            # Get the data directory used via the argument file
            datdir=$(awk -F'\t+' '$1 == "training_data" {print $2}' $arglistfpath)
            datdir=${datdir%/training}
            datdirs+=($datdir)
            errors+=(0)
        else
            datdirs+=("DNE")
            errors+=(1)
        fi
    done
    
    # Get list of unique data directories and argument files
    datadirsunique=$(printf "%s\n" "${datdirs[@]}" | sort | uniq)
    argfilesunique=$(printf "%s\n" "${arglistfnames[@]}" | sort | uniq)
    
    # Process each unique data directory
    for d in ${datadirsunique[@]}; do
        if [[ $d == "DNE" ]]; then
            continue
        fi
        echo_data_string $modelgroup $d

        # Process each unique argument file matching the data directory
        for f in ${argfilesunique[@]}; do
            fpath=data/model_training_args/$modelgroup/$f.tsv
            if [[ -f "$fpath" ]]; then
                d2=$(awk -F'\t+' '$1 == "training_data" {print $2}' $fpath)
                d2=${d2%/training}
                if [[ $d2 == $d ]]; then
                    echo_arglist_string $modelgroup $f $fpath
                    # Process each model matching the argfile
                    for m in ${modelnames[@]}; do
                        f2=${m#model_}
                        f2=run_${f2::-16}
                        if [[ "$f2" == "$f" ]]; then
                            echo_model_string $modelgroup $m
                        fi
                    done
                fi
            fi
        done
    done
}

replacement="$(echo_trained_models data/trained_models/plnn_synbindec)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g;s|\\n |\\n|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[PLNN_SYNBINDEC\] -->/,/<!-- REPLACEMENT END KEY \[PLNN_SYNBINDEC\] -->/c\<!-- REPLACEMENT START KEY \[PLNN_SYNBINDEC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[PLNN_SYNBINDEC\] -->' $file
rm _tmp.txt

replacement="$(echo_trained_models data/trained_models/plnn_quadratic)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g;s|\\n |\\n|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[PLNN_QUADRATIC\] -->/,/<!-- REPLACEMENT END KEY \[PLNN_QUADRATIC\] -->/c\<!-- REPLACEMENT START KEY \[PLNN_QUADRATIC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[PLNN_QUADRATIC\] -->' $file
rm _tmp.txt

replacement="$(echo_trained_models data/trained_models/alg_synbindec)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g;s|\\n |\\n|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[ALG_SYNBINDEC\] -->/,/<!-- REPLACEMENT END KEY \[ALG_SYNBINDEC\] -->/c\<!-- REPLACEMENT START KEY \[ALG_SYNBINDEC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[ALG_SYNBINDEC\] -->' $file
rm _tmp.txt

replacement="$(echo_trained_models data/trained_models/alg_quadratic)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g;s|\\n |\\n|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[ALG_QUADRATIC\] -->/,/<!-- REPLACEMENT END KEY \[ALG_QUADRATIC\] -->/c\<!-- REPLACEMENT START KEY \[ALG_QUADRATIC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[ALG_QUADRATIC\] -->' $file
rm _tmp.txt

replacement="$(echo_trained_models data/trained_models/facs)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g;s|\\n |\\n|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[FACS\] -->/,/<!-- REPLACEMENT END KEY \[FACS\] -->/c\<!-- REPLACEMENT START KEY \[FACS\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[FACS\] -->' $file
rm _tmp.txt

# replacement="$(echo_trained_models data/trained_models/misc)"
# echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g;s|\\n |\\n|g' > _tmp.txt
# replacement=$(cat _tmp.txt)
# sed -i '/<!-- REPLACEMENT START KEY \[MISC\] -->/,/<!-- REPLACEMENT END KEY \[MISC\] -->/c\<!-- REPLACEMENT START KEY \[MISC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[MISC\] -->' $file
# rm _tmp.txt
