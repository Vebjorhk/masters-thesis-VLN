import os
import random
import json
from collections import defaultdict

import torch
import numpy as np
import transformers
from datasets import Dataset, Features, Sequence, Value, Image

def set_seed(seed: int):
    random.seed(seed)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    transformers.set_seed(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    return device


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        system_prompt = json.load(file)["system_prompt"]

    return system_prompt

def load_images(batch):
    # Convert each file path to the format expected by the `Image` feature
    batch["images"] = [{"path": fp, "bytes": None} for fp in batch["images"]]
    return batch


def format_validation(data, processor, system_prompt, instruction_index, format_function):
    grouped_data = defaultdict(list)

    for entry in data:
        grouped_data[entry["path_id"]].append(entry)

    
    result = list(grouped_data.values())
    random.shuffle(result)

    # 20 environments
    validation = [result[i] for i in range(20)]
    tmp = []
    for ls in validation:
        for itm in ls:
            tmp.append(itm)

    validation = tmp

    #test = [result[i] for i in range(20, 40)]

    # only look at the first route instruction in this case, så remember to change this
    validation = format_function(validation, processor, system_prompt, instruction_index, data_type="test")
    #print(len(validation))
    #test = format_prompts(test, processor, system_prompt, instruction_index=0)

    # around 299 examples in validation
    return validation

def grouped_paths(data):
    grouped_data = defaultdict(list)
    for sample in data:
        grouped_data[sample["path_id"]].append(sample)

    return grouped_data


def sorting_for_curriculum_learning(data):
    grouped_data = defaultdict(list)

    for sample in data:
        grouped_data[sample["path_id"]].append(sample)
        
    sorted_ids = sorted(grouped_data, key=lambda x: len(grouped_data[x]))
    
    sorted_paths = []
    
    for ix in sorted_ids:
        for sample in grouped_data[ix]:
            sorted_paths.append(sample)


    return sorted_paths

def change_labels(data):
    action_dict = {
        "Move Forward" : "Move", 
        "Turn Left" : "Left", 
        "Turn Right" : "Right", 
        "Stop Navigation" : "Stop", 
        "Automatically Turn Towards Node" : "Automatically Turn Towards Node"
    }

    for sample in data:
        sample["gold_label"] = action_dict[sample["gold_label"]]
        sample["previous_actions"] = [action_dict[i] for i in sample["previous_actions"]]
        sample["possible_actions"] = [action_dict[i] for i in sample["possible_actions"]]

    return data


def format_phi_prompts(data, processor, system_prompt, instruction_index, data_type="train"):
    formatted_data = []
    for example in data:
        images = []
        user_prompt = f"Route instruction: {example['instructions'][instruction_index]}\n"
        user_prompt += f"Current step: {example['step_id']}\n"
        user_prompt += "Previous images:\n"

        for i, img in enumerate(example["past_images"]):
            user_prompt += f"<|image_{i+1}|>,\n" 
            images.append(f"dataset/{data_type}/{img}")

        if len(example["past_images"]) == 0:
            user_prompt += "None\n"

        user_prompt += "Previous actions:"
        for action in example["previous_actions"]:
            user_prompt += f" {action},"
            
        if len(example["previous_actions"]) == 0:
            user_prompt += "\nNone"

        user_prompt += f"\nCurrent image: <|image_{len(example['past_images'])+1}|>\n"
        images.append(f"dataset/{data_type}/{example['current_image']}")

        user_prompt += "Possible actions: ["
        for action in example["possible_actions"]:
            user_prompt += f"{action}, "

        user_prompt += "]\nNow, predict the next action based on the input you have received:"

        messages = [
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : user_prompt},
        ]

        formatted = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generate_prompt=True
        )

        messages = [
            {"role" : "assistant", "content" : example["gold_label"]}
        ]

        labels = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generate_prompt=True
        )

        new_example = {}

        new_example["text"] = formatted
        new_example["labels"] = labels
        new_example["images"] = images
        formatted_data.append(new_example)

    formatted_data = Dataset.from_list(formatted_data)

    return formatted_data


def format_qwen2_prompts(dataset, processor, system_prompt, instruction_index, path, data_type="train"):
    root_path = os.path.join(path, data_type)
    formatted_data = []
    
    for sample in dataset:
        content = [
            {
                "type" : "text", 
                #"text" : f"Route instruction: {sample['instructions'][instruction_index]}\nPrevious images: "
                "text" : f"Route instruction: {sample['instructions'][instruction_index]}\nCurrent step: {sample['step_id']}\nPrevious images: " #removed for curriculum learning
            },
        ]
        
        images = sample["past_images"]
        images = [os.path.join(root_path, i) for i in sample["past_images"]]

        # HUSK DENNE HVIS DU SKAL PRØVE NOE ANNET
        #if len(images) > 5:
        #    images = images[-6:]
        #    for img in images:
        #        content.append({"type" : "image", "image" : img})

        #else:
        for img in images:
            content.append({"type" : "image", "image" : img}) 

        if len(images) == 0:
            content[0]["text"] += f"[]"

        content.append(
            {
                "type" : "text", 
                "text" : f"\nPrevious actions: {sample['previous_actions'].__str__()}\nCurrent image:"
            }
        )
        content.append(
            {
                "type" : "image", 
                "image" : os.path.join(root_path, sample["current_image"])
            }
        )
        content.append(
            {
                "type" : "text", 
                "text" : f"\nPossible actions: {sample['possible_actions'].__str__()}\nNow predict the next action based on the input you have recived:"
            }
        )

        messages = [
            {"role" : "system", "content" : [{"type" : "text", "text" : system_prompt}]},
            {"role" : "user", "content" : content},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        #labels = f"<|im_start|>assistant\n{sample['gold_label']}<|im_end|>\n<|endoftext|>"
        labels = f"<|im_start|>assistant\n{sample['gold_label']}<|im_end|>\n"# might be some cause of the problem not adding the end of text
        images.extend([os.path.join(root_path, sample["current_image"])])
        
        formatted_sample = {}
        formatted_sample["text"] = text
        formatted_sample["labels"] = labels
        formatted_sample["images"] = images
        formatted_sample["gold_actions"] = sample["gold_label"]
        # maybe remove for later
        formatted_sample["path_id"] = sample["path_id"]
        formatted_sample["step_id"] = sample["step_id"]

        formatted_data.append(formatted_sample)

    formatted_data = Dataset.from_list(formatted_data)
    return formatted_data



