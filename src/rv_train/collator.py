"""Data collator for VLA training with Qwen2.5-VL"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from qwen_vl_utils import process_vision_info

@dataclass
class VLACollator:
    """
    Collator for VLA training that handles image + text batching

    This collator:
    1. Applies chat template to format messages
    2, Processes images through Qwen's vision encoder
    3. creates labels with masked system/user token
    4. Applies action mask augmentation (masking random action tokens)
    """

    processor: Any # Qwen2_5_VLProcessor
    act_mask_aug_act: float = 0.4

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)

        # Apply chat template
        texts = []
        image_inputs = []
        action_texts = []

        for example in examples:
            messages = example['messages']
            images = example['images']

            # Extract action text(assistant contents is [{"type": "text", "text":...}])
            action_text = messages[-1]['content'][0]['text']
            action_texts.append(action_text)

            # Format for Qwen Processor - inject acutal images
            formatted = []
            for msg in messages:
                if msg['role'] == 'user':
                    content = []
                    for item in msg['content']:
                        if item['type'] == 'image':
                            content.append({'type': 'image', 'image': images[0]})
                        else:
                            content.append(item)
                        formatted.append({'role': 'user', 'content': content})
                else:
                    #system and assistant aleady in correct format
                    formatted.append(msg)

            text = self.processor.tokenizer.apply_chat_template(
                formatted, add_generation_prompt=False, add_vision_id=False
            )
            texts.append(text)
            image_inputs.append(process_vision_info(formatted)[0])

        
        # Tokenize batch
        model_inputs = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        # create labels (mask system + user tokens)
        labels = model_inputs['input_ids'].clone()

        for i in range(batch_size):
            # Compute length of system + user tokens
            action_tokens = self.processor.tokenizer(action_texts[i], add_special_tokens=False)['input_ids']
            action_len = len(action_tokens)

            # Total non-pad tokens
            nonpad_len = model_inputs['attention_mask'][i].sum().item()
            # System + user len = total - action - 2 (assistant end token)
            sysuser_len = int(nonpad_len - action_len - 2)

            # Mask system + user tokens
            labels[i, :sysuser_len] = -100

            # Apply action mask augmentation (mathces originam QwenActor logic)
            # BUG: original implementation mask not only action token but also spacebar but I maintain the original logic for now.
            seq_len = labels.size(1)
            if random.random() < 0.1:
                aug_pct = 0.0
            else:
                aug_pct = random.uniform(0, self.act_mask_aug_act)

            mask_len = int(len(action_text[i]) * aug_pct)
            if mask_len > 0:
                mask_indices = random.sample(range(len(action_text[i])), mask_len)
                mask_indice = [x + sysuser_len for x in mask_indices]
                mask_indice = [idx for idx in mask_indice if idx < seq_len]
                if mask_indice:
                    labels[i, mask_indice] = -100
                    model_inputs['input_ids'][i, mask_indice] = 30 ## '?' token

        # Mask pad tokens (151643 = <|endoftext|>)
        # Note: EOS is 151645 (<|im_end|>), which should NOT be masked for training
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        model_inputs['labels'] = labels
        return model_inputs