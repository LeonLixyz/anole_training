import os
import json
import torch
import random
import copy

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from utils.evaluator import is_not_similar_to_white_or_light_blue

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# from utils.loss_funcs.cliploss import _convert_image_to_rgb

def change_bg_color(img):
    np_img = np.transpose(np.array(img), (2, 0, 1))
    not_similar_mask = is_not_similar_to_white_or_light_blue(np_img, 200, 200)
    np_img[:, ~not_similar_mask] = 255
    np_img[:, not_similar_mask] = 0
    return Image.fromarray(np.transpose(np_img, (1, 2, 0)))


class InterleaveAnoleTokenizedDataset(Dataset):
    def __init__(self,
                 dataset,
                 model,
                 processor,
                 split,
                 input_max_length,
                 label_max_length,
                 input_format,
                 **kwargs):
        self.model = model
        self.processor = processor
        self.split = split
        self.dataset = dataset

        self.input_max_length = input_max_length
        self.label_max_length = label_max_length
        
        self.label_processor = copy.deepcopy(self.processor)
        self.label_processor.tokenizer.padding_side = "right"

        self.processor.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.dataset)

    def load_image_from_path(self, img_path):
        try:
            if os.path.exists(img_path):
                return Image.open(img_path).convert('RGB')
            else:
                print(f"Warning: Image path {img_path} does not exist")
                return None
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def __getitem__(self, idx):
        item = self.dataset[idx]

        input_text = item['input_text']
        
        # Load images from paths if input_imgs is not available
        if 'input_imgs' not in item or not item['input_imgs']:
            input_imgs = []
            if 'input_img_paths' in item and item['input_img_paths']:
                for img_path in item['input_img_paths']:
                    img = self.load_image_from_path(img_path)
                    if img:
                        input_imgs.append(img)
        else:
            input_imgs = item['input_imgs']

        label_text = item['label_text']
        
        # Load images from paths if label_imgs is not available
        if 'label_imgs' not in item or not item['label_imgs']:
            label_imgs = []
            if 'label_img_paths' in item and item['label_img_paths']:
                for img_path in item['label_img_paths']:
                    img = self.load_image_from_path(img_path)
                    if img:
                        label_imgs.append(img)
        else:
            label_imgs = item['label_imgs']

        input_imgs = [self.augment_image(img, random.randint(0, 8)) for img in input_imgs]

        if self.split in ['train']:
            tokenized_input = self.processor(
                [input_text],
                images=input_imgs if input_imgs else None,
                padding="max_length",
                return_tensors="pt",
                max_length=self.input_max_length
            )
            
            tokenized_label = self.label_processor(
                [label_text],          # for padding
                images=label_imgs if label_imgs else None,
                padding="max_length",
                return_tensors="pt",
                max_length=self.label_max_length,
            )
            tokenized_label = {k: v[:, 1:] if k in ['input_ids', 'attention_mask'] else v for k, v in tokenized_label.items()}     # omit <s> starting token
            if label_imgs:
                # Get exactly 1024 image tokens regardless of how many tokens the model expects
                image_tokens = self.model.model.model.get_image_tokens(tokenized_label['pixel_values'].to(self.model.device).to(torch.bfloat16))
                # Create a mask for the image token positions
                image_token_mask = tokenized_label['input_ids'] == self.model.config.image_token_id
                # Count how many image tokens we need
                num_image_tokens = image_token_mask.sum().item()
                # If we have more than 1024 placeholders, we need to truncate
                if num_image_tokens > 1024:
                    # Find the indices of the image tokens
                    image_token_indices = torch.where(image_token_mask)[1]
                    # Keep only the first 1024 indices
                    keep_indices = image_token_indices[:1024]
                    # Update the mask to only include the first 1024 image tokens
                    new_mask = torch.zeros_like(image_token_mask).bool()
                    new_mask[0, keep_indices] = True
                    image_token_mask = new_mask
                    num_image_tokens = 1024
                
                # Reshape tokens to exactly match the size needed for assignment
                # This ensures the shape is correct for the masked assignment
                if num_image_tokens <= 1024:
                    # Just use the tokens directly, with flattening
                    shaped_tokens = image_tokens.to(torch.int64).reshape(-1)[:num_image_tokens]
                else:
                    # Repeat tokens to fill all needed positions, then flatten
                    repeats = (num_image_tokens + 1023) // 1024  # ceiling division
                    shaped_tokens = image_tokens.to(torch.int64).repeat(repeats)[:num_image_tokens]
                
                # Assign the tokens to the masked positions
                tokenized_label['input_ids'][image_token_mask] = shaped_tokens.to(tokenized_label['input_ids'].device)

            if input_imgs:
                # Do the same for input images
                image_tokens = self.model.model.model.get_image_tokens(tokenized_input['pixel_values'].to(self.model.device).to(torch.bfloat16))
                image_token_mask = tokenized_input['input_ids'] == self.model.config.image_token_id
                num_image_tokens = image_token_mask.sum().item()
                
                if num_image_tokens > 1024:
                    image_token_indices = torch.where(image_token_mask)[1]
                    keep_indices = image_token_indices[:1024]
                    new_mask = torch.zeros_like(image_token_mask).bool()
                    new_mask[0, keep_indices] = True
                    image_token_mask = new_mask
                    num_image_tokens = 1024
                
                # Reshape tokens to exactly match the size needed
                if num_image_tokens <= 1024:
                    shaped_tokens = image_tokens.to(torch.int64).reshape(-1)[:num_image_tokens]
                else:
                    repeats = (num_image_tokens + 1023) // 1024  # ceiling division
                    shaped_tokens = image_tokens.to(torch.int64).repeat(repeats)[:num_image_tokens]
                
                tokenized_input['input_ids'][image_token_mask] = shaped_tokens.to(tokenized_input['input_ids'].device)
                _ = tokenized_input.pop("pixel_values")

            label_ids = torch.cat((torch.full(tokenized_input['input_ids'].shape, -100), tokenized_label['input_ids']), 1)
            label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

            tokenized_input['input_ids'] = torch.cat((tokenized_input['input_ids'], tokenized_label['input_ids']), 1)
            tokenized_input['attention_mask'] = torch.cat([tokenized_input.pop('attention_mask'), tokenized_label["attention_mask"]], 1)

            return {
                **tokenized_input,
                "labels": label_ids,
            }
        
        else:
            tokenized_input = self.processor(
                text=input_text,
                images=input_imgs if input_imgs else None,
                padding="max_length",
                return_tensors="pt",
                max_length=self.input_max_length
            )

            if input_imgs:
                image_tokens = self.model.model.model.get_image_tokens(tokenized_input['pixel_values'].to(self.model.device).to(torch.bfloat16))
                image_token_count = tokenized_input['input_ids'][tokenized_input['input_ids'] == self.model.config.image_token_id].shape[0] 
                tokenized_input['input_ids'][tokenized_input['input_ids'] == self.model.config.image_token_id] = image_tokens.to(torch.int64).to(tokenized_input['input_ids'].device).reshape(-1)[:image_token_count]
                _ = tokenized_input.pop("pixel_values")
            return {
                **tokenized_input,
            }

    def augment_image(self, reconstructed_img, k):
        for i in range(k):
            tokenized_image = self.processor(
                text="<image>", 
                images=reconstructed_img,
                return_tensors='pt'
            )['pixel_values'].to(torch.bfloat16).to(self.model.device)
            img_tokens = self.model.model.model.get_image_tokens(tokenized_image)
            reconstructed_img_pixels = self.model.model.decode_image_tokens(img_tokens)
            reconstructed_img = self.processor.postprocess_pixel_values(reconstructed_img_pixels).squeeze()
            reconstructed_img = Image.fromarray(reconstructed_img.permute(1, 2, 0).detach().cpu().numpy())
        return reconstructed_img