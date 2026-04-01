#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..mask_utils import remove_singletons, get_kept_lh, OUTLIER_NOUNS
import re
from scipy.ndimage import binary_closing
import nltk
import numpy as np

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    @torch.no_grad()    
    def single_forward_with_prefix(
        self,
        inputs: Optional[torch.Tensor] = None,
        prefix: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        get_last_candidates = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # first pass
        input_ids = prefix[0].unsqueeze(0).unsqueeze(0) #BOS token
        model_kwargs = {'position_id': position_ids,
                        'attention_mask': attention_mask,
                        'inputs_embeds': inputs_embeds,
                        'use_cache': True,
                        }
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_embeds, self.generation_config.pad_token_id, self.generation_config.eos_token_id
            )
        model_inputs = self.prepare_inputs_for_generation(input_ids = input_ids,**model_kwargs)

        next_token_id = 1
        candidates = None
        while True:
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            if next_token_id < len(prefix):
                next_token = prefix[next_token_id].unsqueeze(0)
                if get_last_candidates:
                    if input_ids[0,-1].item() in [29889,13]:
                        candidates = input_ids, torch.topk(torch.softmax(next_token_logits, dim=-1), k=5, dim=-1)
            else:
                if get_last_candidates:
                    return self(**model_inputs, output_attentions=True, output_hidden_states=True, return_dict=True), candidates
                else:
                    return self(**model_inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=input_ids,
                **model_kwargs,
            )
            next_token_id += 1
            
    @torch.no_grad()
    def generate_with_prefix(
        self,
        inputs: Optional[torch.Tensor] = None,
        prefix: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # first pass
        input_ids = prefix[0].unsqueeze(0).unsqueeze(0) #BOS token
        model_kwargs = {'position_id': position_ids,
                        'attention_mask': attention_mask,
                        'inputs_embeds': inputs_embeds,
                        'use_cache': True,
                        }
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_embeds, self.generation_config.pad_token_id, self.generation_config.eos_token_id
            )
        model_inputs = self.prepare_inputs_for_generation(input_ids = input_ids,**model_kwargs)

        next_token_id = 1
        while True:
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            if next_token_id < len(prefix):
                next_token = prefix[next_token_id].unsqueeze(0)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)

            if next_token.item() == self.config.eos_token_id:
                return input_ids
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=input_ids,
                **model_kwargs,
            )
            next_token_id += 1
            
    @torch.no_grad()
    def generate_with_prefix_until_candidate(
        self,
        inputs: Optional[torch.Tensor] = None,
        prefix: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        top_k: int = 3,
        accumulate_prob: float = 0.5,
        debug: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        def dprint(*a, **k):
            if debug:
                print(*a, **k)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # first pass
        input_ids = prefix[0].unsqueeze(0).unsqueeze(0) #BOS token
        model_kwargs = {'position_id': position_ids,
                        'attention_mask': attention_mask,
                        'inputs_embeds': inputs_embeds,
                        'use_cache': True,
                        }
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_embeds, self.generation_config.pad_token_id, self.generation_config.eos_token_id
            )
        model_inputs = self.prepare_inputs_for_generation(input_ids = input_ids,**model_kwargs)

        attentions = []
        next_token_id = 1
        while True:
            outputs = self(**model_inputs, return_dict=True, output_attentions=True)
            attentions.append(outputs.attentions)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            if next_token_id < len(prefix):
                next_token = prefix[next_token_id].unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            else:
                top_k_probs, top_k_indices = torch.topk(next_token_probs, k=top_k, dim=-1)
                # keep top-k tokens with accumulate prob >= accumulate_prob
                cumsum_probs = torch.cumsum(top_k_probs, dim=-1)
                cumsum_ids =  (cumsum_probs >= accumulate_prob).nonzero(as_tuple=True)[1]
                if len(cumsum_ids) > 0:
                    min_k = cumsum_ids[0].item() + 1
                else:
                    min_k = top_k
                if min_k == 1:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
                else:
                    return {'input_ids':input_ids, 'top_k_indices':top_k_indices[0,:min_k].tolist(), 'attentions':attentions}
                    dprint(f"[uncertainty] top-{min_k} tokens with total prob ≥ {accumulate_prob} @pos={next_token_id}:{self.tokenizer.decode(next_token.item())}")
            if next_token.item() == self.config.eos_token_id:
                return {'input_ids':input_ids, 'top_k_indices':None, 'attentions':attentions}
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=input_ids,
                **model_kwargs,
            )
            next_token_id += 1
            
    @torch.no_grad()
    def generate_with_img_correction(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        iou_thresh = 0.5,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        debug = bool(kwargs.pop("debug", False))
        def dprint(*a, **k):
            if debug:
                print(*a, **k)
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generate_with_img_correction, Please set it first by `model.tokenizer = your_tokenizer`")
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        # ==================== start of logic ====================
        objects = []                     # accepted nouns
        masks = []                       # corresponding masks
        
        def detect_nouns(text,joiner =' '):
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)  # English
            keep = {"NN", "NNS", "NNP", "NNPS"}
            merged = []
            i = 0
            while i < len(tokens):
                if tags[i][1] in keep:
                    # gom các noun liên tiếp
                    j = i + 1
                    phrase = [tokens[i]]
                    while j < len(tokens) and tags[j][1] in keep:
                        phrase.append(tokens[j])
                        j += 1
                    merged.append(joiner.join(phrase))
                    i = j
                else:
                    i += 1
            return merged
        
        def find_sublist_start(a, b): #for finding token index of noun in input_ids
            if not b:
                return None
            n, m = len(a), len(b)
            for i in range(n - m + 1):
                if a[i:i+m] == b:
                    return i
            return None

        # Helpers for mask (objects)
        def _extract_last_query_attn_matrix(attentions) -> torch.Tensor:
            begin_vis_pos = 35; vis_len = 576
            L = 32  # num layers
            mats = []
            for l in range(L):
                # shape: (B, H, Q, K)
                att = attentions[l]  # (1, H, q=1, K_total) trong generate step
                mats.append(att[0, :, -1, begin_vis_pos:begin_vis_pos + vis_len])  # (H, vis_len)
            return torch.stack(mats, dim=0)  # (L, H, vis_len)
            
        
        def _get_object_mask(attn_lh: torch.Tensor) -> np.ndarray:
            """Check if token_id do not overlap with any of current_masks."""
            kept = get_kept_lh(attn_lh)
            if len(kept) < 5:
                return None
            binaries = []
            for i in range(5):
                r = kept[i]
                l, h = r["layer"], r["head"]
                a2d = attn_lh[l, h, :].reshape(24, 24).detach().cpu().numpy()
                
                a2d = (a2d - a2d.min()) / (a2d.max() - a2d.min() + 1e-8)
                
                S = a2d
                mean_val = np.mean(S)
                B = np.maximum(S - mean_val*2, 0)
                binary = (B > 1e-8).astype(np.int32)
                binary = remove_singletons(binary)
                binary = binary_closing(binary, structure=np.ones((3,3))).astype(np.int32)
                binaries.append(binary)       
            return np.median(binaries, axis=0).astype(np.uint8)
        
        def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
            a = a.astype(bool)
            b = b.astype(bool)
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            return float(inter) / float(union + 1e-8)
        
        def _mask_is_compatible(new_mask: np.ndarray, masks_list: list, iou_thresh: float) -> bool:
            """True if the new_mask do not overlap too much with any of masks_list."""
            if new_mask is None:
                return False
            area = int(new_mask.astype(bool).sum())
            if area == 0:
                return False
            for old in masks_list:
                if old is None:
                    continue
                iou = _compute_iou(new_mask, old)
                if iou > iou_thresh:
                    return False
            return True

        # ====== MAIN LOOP ======
        current_output = {'input_ids':None, 'top_k_indices':[1], 'attentions':[]}
        while current_output['top_k_indices'] is not None:
            top_k_indices = current_output['top_k_indices']
            ranks = [0 for _ in range(len(top_k_indices))]
            
            candidate_outputs = []
            hallus = [0 for _ in range(len(top_k_indices))]
            temp_objects = [[0] for _ in range(len(top_k_indices))]
            temp_masks = [[0] for _ in range(len(top_k_indices))]
            
            
            for i in range(len(top_k_indices)):
                candidate_id = top_k_indices[i]
                if current_output['input_ids'] is not None:
                    candidate_prefix = torch.cat([current_output['input_ids'], torch.tensor([[candidate_id]], device=self.device)], dim=-1)[0]
                    dprint(f'[try] {self.tokenizer.decode(candidate_prefix)}')
                else:
                    candidate_prefix = torch.tensor([candidate_id], device=self.device)
                    dprint('default start')
                next_break = self.generate_with_prefix_until_candidate(
                    inputs,
                    prefix=candidate_prefix,
                    images=images,
                    image_sizes=image_sizes,
                    debug=debug,
                )
                candidate_outputs.append(next_break)
                
                # new_tokens = next_break['input_ids'][0,len(candidate_prefix)-1:].tolist()
                # new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                all_tokens = next_break['input_ids'][0].tolist()
                last_end = len(all_tokens) - 1 - all_tokens[::-1].index(29889) if 29889 in all_tokens else -1
                new_tokens = all_tokens[last_end+1:]
                new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                dprint(f'  [new] "{new_text}"')
                
                for noun in detect_nouns(new_text):
                    if noun in OUTLIER_NOUNS or noun in objects:
                        continue
                    else:
                        token_noun = self.tokenizer.encode(noun, add_special_tokens=False)
                        noun_id = find_sublist_start(new_tokens, token_noun)
                        try:
                            attn = next_break['attentions'][len(next_break["attentions"]) - len(new_tokens) + noun_id]
                        except:
                            continue #incompleted words
                        noun_mask = _get_object_mask(_extract_last_query_attn_matrix(attn))
                        if not _mask_is_compatible(noun_mask, masks, iou_thresh):
                            hallus[i] += 1
                            dprint(f'  [hallucination] "{noun}" has no good mask or overlaps too much with existing ones.')
                        else:
                            dprint(f'  [non hallucination] "{noun}" is good to go.')
                            temp_objects[i].append(noun)
                            temp_masks[i].append(noun_mask)
                # sort by number of hallucinations (less is better), then by rank (lower is better)
                ranks[i] = (hallus[i], i)
            selected = sorted(range(len(top_k_indices)), key=lambda x: (ranks[x][0], ranks[x][1]))[0]
            selected_id = top_k_indices[selected]
            objects += temp_objects[selected][1:]
            masks += temp_masks[selected][1:]
            dprint(f'[select] {self.tokenizer.decode(selected_id)} with {hallus[selected]} hallucinations.')
            current_output = candidate_outputs[selected]
        
        if debug:
            return current_output['input_ids'], objects, masks
        return current_output['input_ids']
    
    @torch.no_grad()
    def generate_sentence_with_prefix(
        self,
        inputs: Optional[torch.Tensor] = None,
        prefix: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        top_k: int = 5,
        accumulate_prob: float = 0.6,
        debug: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        def dprint(*a, **k):
            if debug:
                print(*a, **k)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # first pass
        input_ids = prefix[0].unsqueeze(0).unsqueeze(0) #BOS token
        model_kwargs = {'position_id': position_ids,
                        'attention_mask': attention_mask,
                        'inputs_embeds': inputs_embeds,
                        'use_cache': True,
                        }
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_embeds, self.generation_config.pad_token_id, self.generation_config.eos_token_id
            )
        model_inputs = self.prepare_inputs_for_generation(input_ids = input_ids,**model_kwargs)

        attentions = []
        next_token_id = 1
        checkpointed = False
        candidates = None
        checkpoint_input_ids = None
        while True:
            outputs = self(**model_inputs, return_dict=True, output_attentions=True)
            attentions.append(outputs.attentions)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            if next_token_id < len(prefix):
                next_token = prefix[next_token_id].unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
                if not checkpointed and torch.max(next_token_probs) < accumulate_prob:
                    top_k_probs, top_k_indices = torch.topk(next_token_probs, k=top_k, dim=-1)
                    # keep top-k tokens with accumulate prob >= accumulate_prob
                    cumsum_probs = torch.cumsum(top_k_probs, dim=-1)
                    cumsum_ids =  (cumsum_probs >= accumulate_prob).nonzero(as_tuple=True)[1]
                    if len(cumsum_ids) > 0:
                        min_k = cumsum_ids[0].item() + 1
                    else:
                        min_k = top_k
                    candidates = top_k_indices[0,:min_k].tolist()
                    checkpoint_input_ids = input_ids.clone()
                    checkpointed = True
                    
                    # early termination if "." or EOS in candidates
                    if 29889 in candidates:
                        input_ids = torch.cat([input_ids, torch.tensor([[29889]], device=self.device)], dim=-1)
                        return {'input_ids':input_ids, 'attentions':attentions, 'candidates':None, 'checkpoint_input_ids':input_ids}
                    if self.config.eos_token_id in candidates:
                        input_ids = torch.cat([input_ids, torch.tensor([[self.config.eos_token_id]], device=self.device)], dim=-1)
                        return {'input_ids':input_ids, 'attentions':attentions, 'candidates':None, 'checkpoint_input_ids':None}
                    
                input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
                if next_token.item() in [869,29889]: # 869 is the token id for ".", and 29889 for "_."
                    if checkpoint_input_ids is None:
                        checkpoint_input_ids = input_ids.clone()
                    return {'input_ids':input_ids, 'attentions':attentions, 'candidates':candidates, 'checkpoint_input_ids':checkpoint_input_ids}
                if next_token.item() == self.config.eos_token_id:
                    return {'input_ids':input_ids, 'attentions':attentions, 'candidates':None, 'checkpoint_input_ids':None}
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=input_ids,
                **model_kwargs,
            )
            next_token_id += 1
            
    @torch.no_grad()
    def generate_with_img_correction_2(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        iou_thresh = 0.5,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        debug = bool(kwargs.pop("debug", False))
        def dprint(*a, **k):
            if debug:
                print(*a, **k)
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generate_with_img_correction, Please set it first by `model.tokenizer = your_tokenizer`")
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        # ==================== start of logic ====================        
        def detect_nouns(text,joiner =' '):
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)  # English
            keep = {"NN", "NNS", "NNP", "NNPS"}
            merged = []
            i = 0
            while i < len(tokens):
                if tags[i][1] in keep:
                    j = i + 1
                    phrase = [tokens[i]]
                    while j < len(tokens) and tags[j][1] in keep:
                        phrase.append(tokens[j])
                        j += 1
                    merged.append(joiner.join(phrase))
                    i = j
                else:
                    i += 1
            return merged
        
        def find_sublist_start(a, b): #for finding token index of noun in input_ids
            if not b:
                return None
            n, m = len(a), len(b)
            for i in range(n - m + 1):
                if a[i:i+m] == b:
                    return i
            return None

        # Helpers for mask (objects)
        def _extract_last_query_attn_matrix(attentions) -> torch.Tensor:
            begin_vis_pos = 35; vis_len = 576
            L = 32  # num layers
            mats = []
            for l in range(L):
                # shape: (B, H, Q, K)
                att = attentions[l]  # (1, H, q=1, K_total) trong generate step
                mats.append(att[0, :, -1, begin_vis_pos:begin_vis_pos + vis_len])  # (H, vis_len)
            return torch.stack(mats, dim=0)  # (L, H, vis_len)
            
        
        def _get_object_mask(attn_lh: torch.Tensor) -> np.ndarray:
            """Check if token_id do not overlap with any of current_masks."""
            kept = get_kept_lh(attn_lh)
            if len(kept) < 5:
                return None
            binaries = []
            for i in range(5):
                r = kept[i]
                l, h = r["layer"], r["head"]
                a2d = attn_lh[l, h, :].reshape(24, 24).detach().cpu().numpy()
                
                a2d = (a2d - a2d.min()) / (a2d.max() - a2d.min() + 1e-8)
                
                S = a2d
                mean_val = np.mean(S)
                B = np.maximum(S - mean_val*2, 0)
                binary = (B > 1e-8).astype(np.int32)
                binary = remove_singletons(binary)
                binary = binary_closing(binary, structure=np.ones((3,3))).astype(np.int32)
                binaries.append(binary)       
            return np.median(binaries, axis=0).astype(np.uint8)
        
        def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
            a = a.astype(bool)
            b = b.astype(bool)
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            return float(inter) / float(union + 1e-8)
        
        def _mask_is_compatible(new_mask: np.ndarray, masks_list: list, iou_thresh: float) -> bool:
            """True if the new_mask do not overlap too much with any of masks_list."""
            if new_mask is None:
                return False
            area = int(new_mask.astype(bool).sum())
            if area == 0:
                return False
            for old in masks_list:
                if old is None:
                    continue
                iou = _compute_iou(new_mask, old)
                if iou > iou_thresh:
                    return False
            return True

        # ====== MAIN LOOP ======
        current_output = {'input_ids':[[1]], 'attentions':None, 'candidates':None, 'checkpoint_input_ids':[1]}
        # first loop:
        current_output = self.generate_sentence_with_prefix(
            inputs,
            prefix=torch.tensor([1], device=self.device), #BOS token
            images=images,
            image_sizes=image_sizes
        )
        
        objects = []                     # accepted nouns
        masks = []                       # corresponding masks
        
        sentence_objects = []
        sentence_masks = []
               
        while current_output.get('checkpoint_input_ids',None) is not None:
            if current_output['candidates'] is None:
                current_output = self.generate_sentence_with_prefix(
                    inputs,
                    prefix=current_output['input_ids'][0],
                    images=images,
                    image_sizes=image_sizes
                )
                objects += sentence_objects
                masks += sentence_masks
                sentence_objects = []
                sentence_masks = []
                dprint(f'[accept sentence] {self.tokenizer.decode(current_output["input_ids"][0])}')
            else:
                top_k_indices = current_output['candidates']
                ranks = [0 for _ in range(len(top_k_indices))]
                
                candidate_outputs = []
                hallus = [0 for _ in range(len(top_k_indices))]
                counts = [0 for _ in range(len(top_k_indices))]
                temp_objects = [[0] for _ in range(len(top_k_indices))]
                temp_masks = [[0] for _ in range(len(top_k_indices))]            
            
                for i in range(len(top_k_indices)):
                    candidate_id = top_k_indices[i]
                    dprint(f'[try] {self.tokenizer.decode(candidate_id)}')
                    candidate_prefix = torch.cat([current_output['checkpoint_input_ids'], torch.tensor([[candidate_id]], device=self.device)], dim=-1)[0]

                    next_break = self.generate_sentence_with_prefix(
                        inputs,
                        prefix=candidate_prefix,
                        images=images,
                        image_sizes=image_sizes,
                        debug=debug,
                    )
                    
                    candidate_outputs.append(next_break)
                    all_tokens = next_break['input_ids'][0].tolist()
                    last_end = len(all_tokens) - 2 - all_tokens[len(all_tokens) - 2::-1].index(29889) if 29889 in all_tokens[:-1] else -1
                    new_tokens = all_tokens[last_end+1:]
                    new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    dprint(f'  [new] "{new_text}"')
                
                    for noun in set(detect_nouns(new_text)):
                        if noun in OUTLIER_NOUNS:
                            continue
                        if noun in objects:
                            counts[i] += 1
                        else:
                            counts[i] += 1
                            token_noun = self.tokenizer.encode(noun, add_special_tokens=False)
                            noun_id = find_sublist_start(new_tokens, token_noun) #sometimes broken because in quote -> not a noun
                            try:
                                attn = next_break['attentions'][len(next_break["attentions"]) - len(new_tokens) + noun_id]
                                noun_mask = _get_object_mask(_extract_last_query_attn_matrix(attn))
                                if not _mask_is_compatible(noun_mask, masks, iou_thresh):
                                    hallus[i] += 1
                                    dprint(f'\t[hallucination] "{noun}" has no good mask or overlaps too much with existing ones.')
                                else:
                                    dprint(f'\t[non hallucination] "{noun}" is good to go.')
                                    temp_objects[i].append(noun)
                                    temp_masks[i].append(noun_mask)
                            except:
                                counts[i] -= 1
                    # sort by number of hallucinations (less is better), then by rank (lower is better)
                    ranks[i] = (hallus[i],counts[i], i)
                selected = sorted(range(len(top_k_indices)), key=lambda x: (ranks[x][0], ranks[x][1], ranks[x][2]))[0]
                selected_id = top_k_indices[selected]
                current_output = candidate_outputs[selected]
                if current_output['candidates'] is None or 29889 in current_output['candidates']:
                    sentence_objects += temp_objects[selected][1:]
                    sentence_masks += temp_masks[selected][1:]
                dprint(f'[select] {self.tokenizer.decode(selected_id)} with {hallus[selected]} hallucinations.')
                
        
        if debug:
            return current_output['input_ids'], objects, masks
        return current_output['input_ids']
            
            # if next_break['top_k_indices'] is None:
            #     return next_break
            # return next_break

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
