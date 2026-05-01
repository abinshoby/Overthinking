# Import libraries
import argparse
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

def load_model(model_path, device="cuda:0"):
    """
    Description:
        - Loads the specified model and its associated tokenizer and image processor based on the model type (LLaVA, Qwen3, or Gemma3).
    Parameters:
        - model_path (str): The path to the model.
        - device (str): The device to load the model on.
    Returns:
        - A dictionary containing the loaded model, tokenizer, image processor, context length, and model type.
    """
    if "llava" in model_path.lower():
        print("Loading LLaVA model...")
        # Load model and tokenizer
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        print(f"Loading model: {model_name}")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map=device
        )
        return {
            'tokenizer': tokenizer, 
            'model': model, 
            'image_processor': image_processor, 
            'processor': None,
            'context_len': context_len, 
            'model_type': "llava"
        }
    elif "qwen" in model_path.lower():
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError("Please install the latest transformers library or use Dockerfile_new_transformer to use Qwen3 and Gemma3 models.")
        print("Loading Qwen3 model...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, device_map=device,  dtype="auto").eval()
        model.set_attn_implementation('eager')
        return {
            'processor': processor,
            'model': model,
            'image_processor': None,
            'context_len': None,
            'tokenizer': None,
            'model_type': "qwen"
        }
    elif "gemma" in model_path.lower():
        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except ImportError:
            raise ImportError("Please install the latest transformers library or use Dockerfile_new_transformer to use Qwen3 and Gemma3 models.")
        print("Loading Gemma3 model...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map=device).eval()
        model.set_attn_implementation('eager')
        return {
            'processor': processor,
            'model': model,
            'image_processor': None,
            'context_len': None,
            'tokenizer': None,
            'model_type': "gemma"
        }
    else:
        raise ValueError("Unsupported model type")

def get_prefix_response(model, model_type, processor, image_path, query="Describe this image", prefix_prompt="", max_new_tokens=1024):
    """
    Description:
    - Generates the model's response, hidden states, and attentions for a given image and query, using a specified prefix prompt to condition the generation.
    Parameters:
    - model: The loaded language model.
    - model_type: The type of the model (e.g., "qwen", "gemma").
    - processor: The processor associated with the model for preparing inputs.
    - image_path: The path to the input image.
    - query: The text query to accompany the image.
    - prefix_prompt: The prefix text to condition the model's generation.
    - max_new_tokens: The maximum number of new tokens to generate.
    Returns:
    - response: The generated response from the model.
    - hidden_states: The hidden states from the model.
    - attentions: The attentions from the model.
    - img_idx: The index of the image token in the input.
    - num_image_tokens: The number of image tokens in the input.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": prefix_prompt}
            ]
        }
    ]
    if model_type == "qwen":
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, return_tensors="pt", 
            return_dict=True, 
            skip_special_tokens=True
        ).to(model.device, dtype=torch.bfloat16)
    elif model_type == "gemma":
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_tensors="pt", 
            return_dict=True
        ).to(model.device, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"][0].tolist()
    if model_type == "qwen":
        eos_id = model.config.text_config.eos_token_id
    elif model_type == "gemma":
        eos_id = model.config.eos_token_id[1]
    
    # find last occurrence of EOS
    last_eos_idx = None
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] == eos_id:
            last_eos_idx = i
            break
    
    # truncate everything after the last EOS
    if last_eos_idx is not None:
        input_ids = input_ids[: last_eos_idx]  # include the EOS itself
    
    # convert back to tensor
    input_ids = torch.tensor([input_ids], device=inputs["input_ids"].device)
    attention_mask = torch.ones_like(input_ids)
    
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    
    # print(processor.decode(inputs["input_ids"][0]))
    input_len = inputs["input_ids"].shape[-1]
    if model_type == "qwen":
        img_idx = inputs["input_ids"][0].tolist().index(model.config.image_token_id)
        num_image_tokens = inputs["input_ids"][0].tolist().count(model.config.image_token_id)
    elif model_type == "gemma":
        img_idx = inputs["input_ids"][0].tolist().index(model.config.image_token_index)
        num_image_tokens = model.config.mm_tokens_per_image

        
    with torch.no_grad():
        out = model(
            **inputs, 
            do_sample=False, 
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
        hidden_states = out.hidden_states
        attentions = out.attentions
    
        past = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)[0]


    response = processor.decode(next_token, skip_special_tokens=True)
    
    return response, hidden_states, attentions, img_idx, num_image_tokens


def get_response(model, model_type, processor, image_path, query="Describe this image.", max_new_tokens=1024):
    """
    Description:
    - Generates the model's response for a given image and query.
    Parameters:
    - model: The loaded language model.
    - model_type: The type of the model (e.g., "qwen", "gemma").
    - processor: The processor associated with the model for preparing inputs.
    - image_path: The path to the input image.
    - query: The text query to accompany the image.
    - max_new_tokens: The maximum number of new tokens to generate.
    Returns:
    - response: The generated response from the model.
    - img_idx: The index of the image token in the input.
    - num_image_tokens: The number of image tokens in the input.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    if model_type == "qwen":
        num_image_tokens = inputs["input_ids"][0].tolist().count(model.config.image_token_id)
        img_idx = inputs["input_ids"][0].tolist().index(model.config.image_token_id)
    elif model_type == "gemma":
        num_image_tokens = model.config.mm_tokens_per_image
        img_idx = inputs["input_ids"][0].tolist().index(model.config.image_token_index)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]
    
    response = processor.decode(generation, skip_special_tokens=True)
    return response, img_idx, num_image_tokens  
    

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
    # patches = int(np.sqrt(image_token_length))
    # attn = attn.reshape((patches, patches))
    return attn.detach().cpu().float().numpy().astype(np.float32)

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
    
    return attn.detach().cpu().float().numpy().astype(np.float32)

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
        h = hidden_states[i][:, -1, :].clone().detach()
        if i == num_layers-1:
            out = mapping_layer(h)
        else:
            out = mapping_layer(norm_layer(h))
        next_probs = F.softmax(out, dim=-1)
        probs_list.append(next_probs.squeeze(0))
        log_probs = F.log_softmax(out, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        entropies.append(entropy.item())
    probs_list = torch.stack(probs_list).detach().cpu().float()
    entropies = torch.tensor(entropies).detach().cpu().float()
    return probs_list, entropies


def compute_overthinking(row):
    """
    Description:
    - Computes an overthinking score for a given token based on the mean entropy of the model's predictions across layers and the diversity of the top predicted tokens.
    Parameters:
    - row: A dictionary containing the features for a specific token, including entropy values and top predicted tokens from each layer.
    Returns:
    - score: The computed overthinking score, which is a product of the mean entropy and the ratio of unique top predicted tokens to total layers.
    """
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


def extract_features(input_data_path, output_data_path="./data.csv", image_dir="/workspace/data/COCODataset/val2014/val2014", model_path="/workspace/data/pretrained/llava-v1.5-7b"):
    """
    Description:
        - Extracts features from a specified model for each token in the descriptions of a dataset, and saves these features along with target labels indicating whether each token is grounded or hallucinated.
    Parameters:
        - input_data_path (str): Path to the input dataset in JSONL format, where each line contains an image, its description, and annotations of grounded and hallucinated tokens.
        - output_data_path (str): Path to save the output CSV file containing the extracted features and target labels.
        - image_dir (str): Directory containing the images referenced in the dataset.
        - model_path (str): Path to the pre-trained model.
    """
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    features_list = []
    target_list = []

    # Load model and tokenizer
    
    model_info = load_model(model_path)
    tokenizer = model_info['tokenizer']
    model = model_info['model']
    image_processor = model_info['image_processor']
    context_len = model_info['context_len']
    model_type = model_info['model_type']
    processor= model_info['processor']

    if model_type == "llava":
        num_image_tokens = model.get_vision_tower().num_patches

    with open(input_data_path,'r') as f:
        data_lines = f.readlines()
        
        with torch.no_grad():
            mapping_layer = model.get_output_embeddings() # lm head
            if model_type == "llava":
                norm_layer = model.model.norm # final norm layer
            elif model_type in ["qwen", "gemma"]:
                norm_layer = model.language_model.norm
            
        for line in tqdm(data_lines):
            data = json.loads(line)
            path = os.path.join(image_dir, data['image'])
            try:
                image = Image.open(path).convert("RGB")
            except:
                continue
            
            query = "Describe this image."
            # Prepare image tensor
            if model_type=='llava':
                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                image_sizes = [image.size]

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
            elif model_type in ["qwen", "gemma"]:
                description, img_idx, num_image_tokens = get_response(model, model_type, processor, path, query=query)

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
                    if model_type == "llava":
                        prefix = torch.tensor(tokenizer.encode(description[:tp-1])).to(model.device)
                    elif model_type in ["qwen", "gemma"]:
                        prefix = description[:tp-1]
                    with torch.inference_mode():
                        # Obtain hidden states and attentions until the next token
                        if model_type == "llava":
                            new_output = model.single_forward_with_prefix(
                                            input_ids,
                                            prefix = prefix,
                                            images=image_tensor.unsqueeze(0),
                                            image_sizes=image_sizes,
                                        )
                            hidden_states = new_output.hidden_states
                            attentions = new_output.attentions
                        elif model_type in ["qwen", "gemma"]:
                            next_token_, hidden_states, attentions, _,_ = get_prefix_response(model, model_type, processor, path, query=query, prefix_prompt=prefix)
                            
                        probs, entropies = get_probs_entropies(hidden_states, mapping_layer, norm_layer)


                        tk = 10
                        text_attns = []
                        img_attns = []
                        entropies = entropies.numpy()
                        n_layers = len(entropies)
                        top_token_ids = []
                        top_token_probs = []
                        for ll in range(n_layers):
                            # Get image and text attentions, top token ids and probs for each layer
                            img_attn = get_image_attention_map(attentions, ll, img_idx, image_token_length=num_image_tokens, mode='max').flatten()
                            text_attn = get_non_image_attention(attentions, ll, img_idx, image_token_length=num_image_tokens, mode='max').flatten()
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
            df.to_csv(output_data_path, index=False)

if __name__ == "__main__":
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="Path to the input training/test data json file", default="./data/LLaVA-1.5/train_migrate_2.jsonl")
    parser.add_argument("--output_data_path", type=str, help="Save path of the output training/test data CSV file", default="./data/LLaVA-1.5/train_features.csv")
    parser.add_argument("--image_dir", type=str, help="Path to the COCO image directory", default="/workspace/data/COCODataset/val2014/val2014")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint to extract features from", default="/workspace/data/pretrained/llava-v1.5-7b")
    args = parser.parse_args()

    # Extract features for training and testing data
    extract_features(
        args.input_data_path, 
        output_data_path=args.output_data_path, 
        image_dir=args.image_dir, 
        model_path=args.model_path
    )
       

    