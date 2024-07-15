
from torch.utils.data import Dataset
from src.utils import load_config
from PIL import Image
import pandas as pd
import os

class ImgDescriptionDataset(Dataset):
    def __init__(self, dataset_name_or_path: str, image_dir: str, split: str = "train"):
        super().__init__()
        self.dataset = pd.read_csv(dataset_name_or_path)
        train_size = int(0.8 * len(self.dataset))

        if split == "train":
            self.dataset = self.dataset[:train_size]
        elif split == "valid":
            self.dataset = self.dataset[train_size:]

        self.ground_truth = self.dataset['image_caption'].to_list()
        self.image_list = self.dataset['image_name'].to_list()
        self.image_dir = image_dir
        self.config = load_config()
        self.question_prompt = self.config['prompt']['question']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image = Image.open(os.path.join(self.image_dir, self.image_list[idx]))
        if image.mode != "RGB":
            image = image.convert("RGB")
        target_sequence = self.ground_truth[idx]
        return self.question_prompt, target_sequence, image
