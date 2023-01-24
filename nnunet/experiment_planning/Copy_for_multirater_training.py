import os
import re
import time

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunet.paths import preprocessing_output_dir


def multi_rater_preprocess(task_ids: list):
    task_names = []
    for i in task_ids:
        for t in os.listdir(preprocessing_output_dir):
            i = str(i)
            if i in t:
                task_names.append(t)

    # built name for multi_rater preprocess folder
    multi_tasks = ""
    for i in task_ids:
        multi_tasks += str(i) + "_"

    dir_multirater = os.path.join(preprocessing_output_dir, f"Task000_{multi_tasks}raters")
    for i in range(len(task_names)):
        dir_task = join(preprocessing_output_dir, task_names[i])
        gt_folder = join(dir_task, "gt_segmentations")
        data_folder = join(dir_task, "nnUNetData_plans_v2.1_stage0")
        gt_folder_multi_rater = join(dir_multirater, f"gt_segmentations_rater{i+1}")
        data_folder_multi_rater = join(dir_multirater, f"nnUNetData_plans_v2.1_stage0_rater{i+1}")
        if os.path.exists(dir_multirater) and os.path.exists(gt_folder_multi_rater) and os.path.exists(data_folder_multi_rater):
            print("There is already a multi rater file preprocessed, please check!")
            pass
        if i == 0:
            shutil.copytree(dir_task, dir_multirater, dirs_exist_ok=True)
            if os.path.exists( join(dir_multirater, "gt_segmentations")):
                shutil.move(join(dir_multirater, "gt_segmentations"), gt_folder_multi_rater)
                shutil.move(join(dir_multirater, "nnUNetData_plans_v2.1_stage0"), data_folder_multi_rater)
            else:
                print("Waiting for file to copy...")
                time.sleep(5)
                if os.path.exists(join(dir_multirater, "gt_segmentations")):
                    shutil.move(join(dir_multirater, "gt_segmentations"), gt_folder_multi_rater)
                    shutil.move(join(dir_multirater, "nnUNetData_plans_v2.1_stage0"), data_folder_multi_rater)
        else:
            shutil.copytree(gt_folder, gt_folder_multi_rater)
            shutil.copytree(data_folder, data_folder_multi_rater)
            print("Done. Folder for multi_rater training ready")

if __name__ == "__main__":
    multi_rater_preprocess([801,802,803])