DATASETS = {
    "val_seen" : "tasks/R2R/data/r2r_val_seen.json",
    "val_unseen" : "tasks/R2R/data/r2r_val_unseen.json"

}

inst = 0
split = "val_seen" 

config = {
    "instruction_index" : inst,
    "dataset" : f"./tasks/R2R/data/r2r_{split}.json",
    "simulator" : "v3",
    "save_path" : f"./model_paths/full-data/low-level/qwen2_5_non-panoramic_full_{split}_inst_{inst}.json",
    "model_action_path" : f"./model_paths/full-data/low-level/actions_qwen2_5_non-panoramic_full_{split}_inst_{inst}.json",
    "submission_path" : f"./model_paths/full-data/low-level/r2r_{split}_submission_non-panoramic_full_inst_{inst}.json",
}