#!/bin/bash

file="data/model_training_args/README.md"

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/synbindec)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[SYNBINDEC\] -->/,/<!-- REPLACEMENT END KEY \[SYNBINDEC\] -->/c\<!-- REPLACEMENT START KEY \[SYNBINDEC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[SYNBINDEC\] -->' $file
rm _tmp.txt

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/quadratic)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[QUADRATIC\] -->/,/<!-- REPLACEMENT END KEY \[QUADRATIC\] -->/c\<!-- REPLACEMENT START KEY \[QUADRATIC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[QUADRATIC\] -->' $file
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

replacement="$(sh scripting/echo_runarg_table.sh data/model_training_args/algbindec)"
echo $replacement | sed 's|data/model_training_args|../model_training_args|g;s|data/training_data|../training_data|g' > _tmp.txt
replacement=$(cat _tmp.txt)
sed -i '/<!-- REPLACEMENT START KEY \[ALGBINDEC\] -->/,/<!-- REPLACEMENT END KEY \[ALGBINDEC\] -->/c\<!-- REPLACEMENT START KEY \[ALGBINDEC\] -->\n'"$replacement"'\n<!-- REPLACEMENT END KEY \[ALGBINDEC\] -->' $file
rm _tmp.txt