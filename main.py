from src.setup import setup_model_and_processor
from src.utils import load_config, collate_fn
from src.dataset import ImgDescriptionDataset
from torch.utils.data import DataLoader
from src.train import train_model
from datetime import datetime
import functools

def main():
    config = load_config('config.yaml')
    model, processor, device = setup_model_and_processor()

    train_dataset = ImgDescriptionDataset(config['paths']['dataset_csv'], 
                                 config['paths']['image_dir'],
                                 split="train")
    val_dataset = ImgDescriptionDataset(config['paths']['dataset_csv'], 
                               config['paths']['image_dir'],
                               split="valid")
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['training']['batch_size'], 
                              collate_fn=functools.partial(collate_fn, processor=processor), 
                              num_workers=config['training']['num_workers'],
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                             batch_size=config['training']['batch_size'], 
                             collate_fn=functools.partial(collate_fn, processor=processor), 
                            num_workers=config['training']['num_workers'])

    train_model(train_loader, val_loader,
                 model, processor,
                   int(config['training']['epochs']), float(config['training']['learning_rate']), 
                   int(config['training']['validation_interval']), 
                   config['paths']['checkpoint_dir'],device=device)


    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    push_model_name = config['model']['name_template'].format(timestamp=current_time)
    push_processor_name = f"{push_model_name}-processor"

    model.push_to_hub(push_model_name)
    processor.push_to_hub(push_processor_name)

    print('Success')

if __name__ == "__main__":
    main()