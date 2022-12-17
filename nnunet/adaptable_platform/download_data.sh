#!/bin/bash

abdel_folder=$nnUNet_preprocessed/Task127_experts_227/nnUNetData_plans_v2.1_stage0_Abdel/
mkdir -p $abdel_folder
gcloud storage cp gs://task127_experts_227_train_abdel/* $abdel_folder

ben_folder=$nnUNet_preprocessed/Task127_experts_227/nnUNetData_plans_v2.1_stage0_Ben/
mkdir -p $ben_folder
gcloud storage cp gs://task127_experts_227_train_ben/* $ben_folder

jeremy_folder=$nnUNet_preprocessed/Task127_experts_227/nnUNetData_plans_v2.1_stage0_Jeremy/
mkdir -p $jeremy_folder
gcloud storage cp gs://task127_experts_227_train_jeremy/* $jeremy_folder

plan_folder=/home/ostmeiersophie/NCCTfolders/nnUNet_preprocessed/Task127_experts_227/
gcloud storage cp gs://task127_experts_227/* $plan_folder
