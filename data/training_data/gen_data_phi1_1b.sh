#!/bin/bash

runname="data_phi1_1b"

landscape="phi1"
s10_range="-0.5 0.5"
s20_range="0.5 1.5"
s11_range="-1 1"
s21_range="-0.5 0.5"
logr1_range="-3 2"
logr2_range="-3 2"

nsims_train=500
nsims_valid=150
nsims_test=150

seed_train=3763487
seed_valid=9875
seed_test=345642

sigma=0.1
tfin=100
dt=0.001
dt_save=50.0
ncells=500
burnin=0.1
nsignals=2
signal_schedule=sigmoid
param_func=identity
noise_schedule=constant
x0="0 -0.5"

animation_dt=1.0
sims_to_animate="0 1 2 3"
animation_duration=10

mkdir -p logs/gen_training_data

logfile=logs/gen_training_data/${runname}.o
echo Logging information to ${logfile}
echo "Generating training data..."

generate_data \
    -o data/training_data/${runname}/training \
    --nsims $nsims_train \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
    --param_func $param_func \
    --noise_schedule $noise_schedule --noise_args ${sigma} \
    --x0 $x0 \
    --seed $seed_train \
    --animate \
    --duration $animation_duration \
    --animation_dt $animation_dt \
    --sims_to_animate $sims_to_animate \
> $logfile 

echo "Generating validation data..."

generate_data \
    -o data/training_data/${runname}/validation \
    --nsims $nsims_valid \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
    --param_func $param_func \
    --noise_schedule $noise_schedule --noise_args ${sigma} \
    --x0 $x0 \
    --seed $seed_valid \
    --animate \
    --duration $animation_duration \
    --animation_dt $animation_dt \
    --sims_to_animate $sims_to_animate \
>> $logfile 

echo "Generating testing data..."

generate_data \
    -o data/training_data/${runname}/testing \
    --nsims $nsims_test \
    --tfin $tfin --dt $dt --dt_save $dt_save --ncells $ncells --burnin $burnin \
    --landscape_name ${landscape} \
    --nsignals $nsignals \
    --signal_schedule $signal_schedule \
    --s10_range ${s10_range} \
    --s20_range ${s20_range} \
    --s11_range ${s11_range} \
    --s21_range ${s21_range} \
    --logr1_range ${logr1_range} \
    --logr2_range ${logr2_range} \
    --param_func $param_func \
    --noise_schedule $noise_schedule --noise_args ${sigma} \
    --x0 $x0 \
    --seed $seed_test \
    --animate \
    --duration $animation_duration \
    --animation_dt $animation_dt \
    --sims_to_animate $sims_to_animate \
>> $logfile 

echo "Done!"
