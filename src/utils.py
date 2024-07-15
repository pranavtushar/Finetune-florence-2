import torch
import yaml
import os

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_checkpoint_dir(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)

def collate_fn(batch, processor):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), 
                       images=list(images),
                        return_tensors="pt", 
                        padding=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return inputs, answers