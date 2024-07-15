
from transformers import AutoModelForCausalLM, AutoProcessor
from src.utils import load_config
import torch

def setup_model_and_processor():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'], trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(config['model']['name'], trust_remote_code=True)
    
    return model, processor, device
