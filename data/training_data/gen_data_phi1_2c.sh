#!/bin/bash

runname="data_phi1_2c"

landscape="phi1"
s10_range="-0.5 0.5"
s20_range="0.5 1.5"
s11_range="-1 1"
s21_range="-0.5 0.5"
logr1_range="1.5 2"
logr2_range="1.5 2"

nsims_train=400
nsims_valid=80
nsims_test=80

seed_train=3225280143
seed_valid=3573406061
seed_test=533767190

sigma=0.1
tfin=20
dt=0.001
dt_save=20.0
ncells=500
burnin=0.1
nsignals=2
signal_schedule=sigmoid
param_func=identity
noise_schedule=constant
x0="0 -0.5"

animation_dt=1.0
sims_to_animate="0 1 2"
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
