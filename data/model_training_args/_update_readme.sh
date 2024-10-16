#!/bin/bash

file="data/model_training_args/README.md"

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/plnn_synbindec)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[PLNN_SYNBINDEC\] -->/,/<!-- REPLACEMENT END KEY \[PLNN_SYNBINDEC\] -->/c\<!-- REPLACEMENT START KEY \[PLNN_SYNBINDEC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[PLNN_SYNBINDEC\] -->' $file
rm _tmp.txt

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/plnn_quadratic)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[PLNN_QUADRATIC\] -->/,/<!-- REPLACEMENT END KEY \[PLNN_QUADRATIC\] -->/c\<!-- REPLACEMENT START KEY \[PLNN_QUADRATIC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[PLNN_QUADRATIC\] -->' $file
rm _tmp.txt

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/alg_synbindec)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[ALG_SYNBINDEC\] -->/,/<!-- REPLACEMENT END KEY \[ALG_SYNBINDEC\] -->/c\<!-- REPLACEMENT START KEY \[ALG_SYNBINDEC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[ALG_SYNBINDEC\] -->' $file
rm _tmp.txt

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/alg_quadratic)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[ALG_QUADRATIC\] -->/,/<!-- REPLACEMENT END KEY \[ALG_QUADRATIC\] -->/c\<!-- REPLACEMENT START KEY \[ALG_QUADRATIC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[ALG_QUADRATIC\] -->' $file
rm _tmp.txt

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/facs)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[FACS\] -->/,/<!-- REPLACEMENT END KEY \[FACS\] -->/c\<!-- REPLACEMENT START KEY \[FACS\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[FACS\] -->' $file
rm _tmp.txt

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/misc)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[MISC\] -->/,/<!-- REPLACEMENT END KEY \[MISC\] -->/c\<!-- REPLACEMENT START KEY \[MISC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[MISC\] -->' $file
rm _tmp.txt
