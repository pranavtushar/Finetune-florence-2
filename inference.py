from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import argparse
import torch

class ImageCaptioning:
    def __init__(self, model_id, device):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise RuntimeError(f"Error in loading image: {e}")

    def generate_caption(self, task_prompt, text_input, image):
        prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        try:
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=2000,
                num_beams=3
            )
        except Exception as e:
            raise RuntimeError(f"Error in generating caption: {e}")
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer

def main():
    parser = argparse.ArgumentParser(description="Generate image captions using a fine-tuned model.")
    parser.add_argument('--model_id', type=str, required=True, help='ID of the fine-tuned model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--task_prompt', type=str, required=True, help='Task prompt for the model')
    parser.add_argument('--text_input', type=str, required=True, help='Text input for the model')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_captioning = ImageCaptioning(args.model_id, device)
    image = image_captioning.preprocess_image(args.image_path)
    
    result = image_captioning.generate_caption(args.task_prompt, args.text_input, image)

    print(result)

if __name__ == "__main__":
    main()

# python3 inference.py --model_id microsoft/Florence-2-large-ft  --image_path inference.jpg --task_prompt '<image_description_ft>' --text_input 'Describe the activities of each individuals in the image?'
