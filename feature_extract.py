# Import libraries
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import json
from tqdm import tqdm
import re
import pandas as pd
sys.path.append('./models/LLaVA')

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

# Load model and tokenizer
disable_torch_init()
model_path = "/workspace/data/pretrained/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
print(f"Loading model: {model_name}")
device = "cuda:0"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,
    model_name,
    device_map=device
)

# Data paths
train_data_path = "/workspace/data/Detection/train_migrate_2.jsonl"
test_data_path = "/workspace/data/Detection/test_migrate_2.jsonl"
image_dir = "/workspace/data/COCODataset/val2014/val2014"
results_dir = "/workspace/data/Detection/overthink/data_pp"

def get_image_attention_map(attentions, layer_idx, image_token_idx, image_token_length=576, mode='max'):
    """
    Description:
        - Extracts and aggregates attention scores directed towards image tokens by the last token from a specific layer.
    Parameters:
        - attentions: list of attention tensors from each layer
        - layer_idx: index of the layer to extract attention from 
        - image_token_idx: starting index of image tokens in the sequence
        - image_token_length: number of image tokens
        - mode: 'max' or 'mean' to aggregate attention scores
    Returns:
        - attn: aggregated attention scores towards image tokens
    """
    # layers, 1xheadsxqxk
    layer_attn = attentions[layer_idx]
    attn_image = layer_attn[-1, :, -1, image_token_idx:image_token_idx+image_token_length]
    if mode == 'max':
        attn = attn_image.max(dim=0)[0]
    elif mode == 'mean':
        attn = attn_image.mean(dim=0)
    else:
        raise ValueError("Invalid mode")
    patches = int(np.sqrt(image_token_length))
    attn = attn.reshape((patches, patches))
    return attn.detach().cpu().numpy().astype(np.float32)

def get_non_image_attention(attentions, layer_idx, image_token_idx, image_token_length=576, mode='max'):
    """
    Description:
        - Extracts and aggregates attention scores directed towards non-image tokens by the last token from a specific layer.
    Parameters:
        - attentions: list of attention tensors from each layer
        - layer_idx: index of the layer to extract attention from 
        - image_token_idx: starting index of image tokens in the sequence
        - image_token_length: number of image tokens
        - mode: 'max' or 'mean' to aggregate attention scores
    Returns:
        - attn: aggregated attention scores towards non-image tokens
    """
    layer_attn = attentions[layer_idx]
    exclude_idx = torch.arange(image_token_idx, image_token_idx + image_token_length)
    seq_len = layer_attn.shape[-1]
    include_idx = torch.tensor([i for i in range(seq_len) if i not in exclude_idx])
    
    attn_non_image = layer_attn[-1, :, -1, include_idx]
    if mode == 'max':
        attn = attn_non_image.max(dim=0)[0]
    elif mode == 'mean':
        attn = attn_non_image.mean(dim=0)
    else:
        raise ValueError("Invalid mode")
    
    return attn.detach().cpu().numpy().astype(np.float32)

def get_probs_entropies(hidden_states, mapping_layer, norm_layer):
    """
    Description:
        - Computes the probabilities and entropies of the next token predictions from hidden states in each layer.
    Parameters:
        - hidden_states: list of hidden states from each layer
        - mapping_layer: the output embedding layer to map hidden states to logits
        - norm_layer: normalization layer to apply to hidden states
    Returns:
        - probs_list: list of probability distributions for the next token from each layer
        - entropies: list of entropies for the next token predictions from each layer
    """
    num_layers = len(hidden_states)
    probs_list = []
    entropies = []
    for i in range(1, num_layers):
        if i == num_layers-1:
            out = mapping_layer(hidden_states[i][:,-1,:])
        else:
            out = mapping_layer(norm_layer(hidden_states[i][:,-1,:]))
        next_probs = F.softmax(out, dim=-1)
        probs_list.append(next_probs.squeeze(0))
        log_probs = F.log_softmax(out, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        entropies.append(entropy.item())
    probs_list = torch.stack(probs_list).detach().cpu()
    entropies = torch.tensor(entropies).detach().cpu()
    return probs_list, entropies

def compute_overthinking(row):
    # No of layers
    layers = len([x for x in row.keys() if x.startswith('H_')])
    layers_list = list(range(layers)) 
    tokens_seen = [row[f'token_id_L{L}_R0'] for L in layers_list if f'token_id_L{L}_R0' in row.keys()]
    unique_tokens = len(set(tokens_seen))  # number of unique tokens
    # Mean entropy across these layers
    mean_entropy = np.mean([row[f'H_{L}'] for L in layers_list])
    # Overthinking score: mean_entropy * (unique tokens / total layers)
    score = mean_entropy * (unique_tokens / len(layers_list))
    return score


def extract_features(data_path, output_dir="./data", output_filename="./train.csv"):
    features_list = []
    target_list = []

    with open(data_path,'r') as f:
        data_lines = f.readlines()
        num_image_tokens = model.get_vision_tower().num_patches
        with torch.no_grad():
            mapping_layer = model.get_output_embeddings() # lm head
            norm_layer = model.model.norm # final norm layer
            
        for line in tqdm(data_lines):
            data = json.loads(line)
            path = os.path.join(image_dir, data['image'])
            try:
                image = Image.open(path).convert("RGB")
            except:
                continue
            
            # Prepare image tensor
            image = Image.open(path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_sizes = [image.size]

            query = "Describe this image."
            qs = DEFAULT_IMAGE_TOKEN + "\n" + query

            # Create conversation and format prompt
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Tokenize input
            input_ids = tokenizer_image_token(
                prompt, 
                tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors="pt",
            ).unsqueeze(0).to(model.device)
            
            img_idx = input_ids[0].tolist().index(IMAGE_TOKEN_INDEX)

            # Generate description
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0),
                    image_sizes=image_sizes,
                    do_sample=False,
                    output_scores = True,
                    output_hidden_states = True,
                    output_attentions = True,
                    return_dict_in_generate = True,
                    max_new_tokens=2048
                )
                
            description = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

            # Find token positions
            token_positions = {}
            if not description or len(description.strip())==0:
                description = data['description']

            # Track positions of grounded and hallucinated tokens
            for token in data['grounded_tokens']+data['hallucinated_tokens']:
                pattern = r'\b{}\b'.format(re.escape(token))  # exact whole-word match
                matches = list(re.finditer(pattern, description))
                
                if matches:
                    token_positions[token] = [m.start() for m in matches]

            # Extract features for each token position
            for token, positions in token_positions.items():
                for tp in positions:
                    # Prefix prompt
                    prefix = torch.tensor(tokenizer.encode(description[:tp-1])).to(model.device)
                    with torch.inference_mode():
                        # Obtain hidden states and attentions until the next token
                        new_output = model.single_forward_with_prefix(
                                        input_ids,
                                        prefix = prefix,
                                        images=image_tensor.unsqueeze(0),
                                        image_sizes=image_sizes,
                                    )
                        probs, entropies = get_probs_entropies(new_output.hidden_states, mapping_layer, norm_layer)

                        tk = 10
                        text_attns = []
                        img_attns = []
                        entropies = entropies.numpy()
                        n_layers = len(entropies)
                        top_token_ids = []
                        top_token_probs = []
                        for ll in range(n_layers):
                            # Get image and text attentions, top token ids and probs for each layer
                            img_attn = get_image_attention_map(new_output.attentions, ll, img_idx, image_token_length=num_image_tokens, mode='max').flatten()
                            text_attn = get_non_image_attention(new_output.attentions, ll, img_idx, image_token_length=num_image_tokens, mode='max').flatten()
                            text_attns.append(text_attn.mean())
                            img_attns.append(img_attn.mean())
                            top_prob, top_idx = torch.topk(probs[ll], k=tk, dim=-1)
                            top_token_ids.append(top_idx.cpu().numpy())
                            top_token_probs.append(top_prob.cpu().numpy())
                    
                        f1 = entropies
                        f2 = np.array(img_attns)
                        f3 = np.array(text_attns)

                        # Prepare dataframe
                        new_row = {
                            "image_id": data['image_id'],
                            "image": data['image'], 
                            "description": description, 
                            "annotations": data['annotations'],
                            "captions": data['captions'], 
                            "prefix_prompt": description[:tp-1], 
                            "next_token": token,
                            "grounded_tokens": data['grounded_tokens'], 
                            "hallucinated_tokens": data['hallucinated_tokens']
                        }
                        for ll in range(n_layers):
                            new_row[f"H_{ll}"] = f1[ll]
                            new_row[f"IA_{ll}"] = f2[ll]
                            new_row[f"TA_{ll}"] = f3[ll]
                            for tt in range(tk):
                                new_row[f"token_id_L{ll}_R{tt}"] = top_token_ids[ll][tt]
                                new_row[f"token_prob_L{ll}_R{tt}"] = top_token_probs[ll][tt]
                        new_row["overthinking_score"] = compute_overthinking(new_row)
                        features_list.append(new_row)
                        # Assign target label
                        if token in data['grounded_tokens']:
                            target_list.append(0)
                        else:
                            target_list.append(1)
            df = pd.DataFrame(features_list)
            df['target'] = target_list
            # Save to CSV
            df.to_csv(os.path.join(output_dir, output_filename), index=False)

if __name__ == "__main__":
    # Extract features for training and testing data
    extract_features(train_data_path, output_dir=results_dir, output_filename="train.csv")
    extract_features(test_data_path, output_dir=results_dir, output_filename="test.csv")
       
