import logging
import math
import os
from datetime import datetime
from importlib import import_module
from typing import List, Union, Callable, Optional, Dict, Tuple

import PIL.Image
import deepspeed
import numpy as np
import torch
from torch import Tensor
from torch.nn import init
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from transformers.generation.utils import GenerateOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, deepspeed_config

from ovis.model.configuration_ovis import OvisConfig
from ovis.model.conversation_formatter import ConversationFormatter
from ovis.util.constants import IGNORE_ID, BEGIN_LINE, END_LINE, VISUAL_ATOM_ID, INDICATOR_IDS, \
    IMAGE_TOKEN_ID, VIDEO_TOKEN_ID
from ovis.util.utils import rank0_print


class VisualEmbedding(torch.nn.Embedding):
    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()


class VisualTokenizer(torch.nn.Module):
    def __init__(self, vit, visual_vocab_size, image_processor_name_or_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit = vit
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name_or_path, do_center_crop=False)
        head_dim = visual_vocab_size - len(INDICATOR_IDS)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.vit.config.hidden_size * self.vit.config.hidden_stride ** 2, head_dim, bias=False),
            torch.nn.LayerNorm(head_dim)
        )

    def _get_last_block(self):
        return self.vit._get_block(-1)

    def _encode(self, pixel_values, grid_thws):
        output = self.vit(pixel_values, grid_thws, output_hidden_states=True, return_dict=True)
        features = output.hidden_states[-1]
        seq_len, _ = features.shape
        features = features.reshape(seq_len // (self.vit.config.hidden_stride ** 2), -1)
        return features

    # Adapted from qwen2_vl
    @staticmethod
    def smart_resize(
        height: int, width: int, factor: int = 28, min_pixels: int = 448 * 448, max_pixels: int = 1344 * 1792
    ):
        """Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """

        if height < factor or width < factor:
            logging.warning(
                f"Resizing image from ({height=}, {width=}) because a dimension is smaller than {factor}."
            )
            if height < width:
                width = round(factor / height * width)
                height = factor
            else:
                height = round(factor / width * height)
                width = factor

        elif max(height, width) / min(height, width) > 200:
            logging.warning(
                f"Resizing image from ({height=}, {width=}) because the aspect ratio is larger than 200"
            )
            if height > width:
                height = 200 * width
            else:
                width = 200 * height

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def preprocess(
        self,
        image: Optional[PIL.Image.Image] = None,
        video: Optional[List[PIL.Image.Image]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None
    ):
        patch_size = self.vit.config.patch_size
        temporal_patch_size = self.vit.config.temporal_patch_size
        hidden_stride = self.vit.config.hidden_stride
        assert (image is None) ^ (video is None), "Invalid input: expect either image or video"
        if image is not None:
            images = [image]
        else:
            images = video
        images = [image.convert("RGB") if image.mode != 'RGB' else image for image in images]
        width, height = images[0].size
        processed_images = []
        for image in images:
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=patch_size * hidden_stride,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            new_size = dict(height=resized_height, width=resized_width)
            new_image = self.image_processor.preprocess(image, size=new_size, return_tensors="np")['pixel_values'][0]
            processed_images.append(new_image)

        patches = np.array(processed_images)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(patches[-1][np.newaxis], temporal_patch_size - 1, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])

        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // hidden_stride,
            hidden_stride,
            patch_size,
            grid_w // hidden_stride,
            hidden_stride,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
        flatten_patches = torch.tensor(flatten_patches)

        return flatten_patches, grid_thw

    def get_dummy_visual_inputs(self):
        pixel_values = torch.zeros((2 * 2, 3 * self.vit.config.patch_size ** 2), dtype=self.vit.dtype,
                                   device=self.vit.device)
        grid_thws = torch.tensor([[1, 2, 2]], dtype=torch.long, device=self.vit.device)
        return pixel_values, grid_thws

    def forward(
        self, pixel_values, grid_thws
    ) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self._encode(pixel_values, grid_thws)
        logits = self.head(features)
        tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        # tokens' shape is [#Token, VocabSize-2], so padding with [#Token, 2], after
        # which, tokens' shape should become [#Token, VocabSize];
        token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(token_len, len(INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=1)
        return tokens

    def get_monitor_tensors(self):
        monitor_tensors = dict(
            vit_bottom=self.vit._get_attn_weight(0),
            vit_top=self.vit._get_attn_weight(-1),
            head=self.head[0].weight,
            pos_embed=self.vit._get_pose_embed()
        )
        return monitor_tensors


class OvisPreTrainedModel(PreTrainedModel):
    config_class = OvisConfig
    base_model_prefix = "ovis"


class Ovis(OvisPreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, config: OvisConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if kwargs.get('train_from_scratch'):
            self.llm = kwargs['llm']
            self.text_tokenizer = kwargs['text_tokenizer']
            self.visual_tokenizer = kwargs['visual_tokenizer']
        else:
            self.llm = AutoModelForCausalLM.from_config(self.config.llm_config)
            assert self.config.hidden_size == self.llm.config.hidden_size, "hidden size mismatch"
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
            self.visual_tokenizer = VisualTokenizer(vit=AutoModel.from_config(self.config.vit_config),
                                                    visual_vocab_size=self.config.visual_vocab_size,
                                                    image_processor_name_or_path=self.config.name_or_path)
        # initialize vte
        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                self.vte = VisualEmbedding(self.config.visual_vocab_size, self.config.hidden_size)
        else:
            self.vte = VisualEmbedding(self.config.visual_vocab_size, self.config.hidden_size,
                                       device=self.visual_tokenizer.vit.device, dtype=self.visual_tokenizer.vit.dtype)

        def _merge_modules(modules_list: tuple):
            merged_modules = []
            for modules in modules_list:
                merged_modules.extend(modules if modules else [])
            return merged_modules

        self._no_split_modules = _merge_modules(
            (self.llm._no_split_modules, self.visual_tokenizer.vit._no_split_modules))
        self._skip_keys_device_placement = self.llm._skip_keys_device_placement
        self._keep_in_fp32_modules = _merge_modules(
            (self.llm._keep_in_fp32_modules, self.visual_tokenizer.vit._keep_in_fp32_modules))
        self.is_parallelizable = all((self.llm.is_parallelizable, self.visual_tokenizer.vit.is_parallelizable))
        self.supports_gradient_checkpointing = True

    def tie_weights(self):
        self.llm.tie_weights()

    def re_init_vte(self, mean, std):
        vte = self.get_vte()
        rank0_print(BEGIN_LINE)
        rank0_print(f'[{datetime.now()}] Before re-initialization of vte: ')
        with deepspeed.zero.GatheredParameters([vte.weight]):
            rank0_print(f'vte.weight: {vte.weight}')
        with deepspeed.zero.GatheredParameters([vte.weight], modifier_rank=0):
            if not is_deepspeed_zero3_enabled() or deepspeed.comm.get_rank() == 0:
                vte.reset_parameters(mean, std)
        rank0_print(f'[{datetime.now()}] After re-initialization of vte:')
        with deepspeed.zero.GatheredParameters([vte.weight]):
            rank0_print(f'vte.weight: {vte.weight}')
        rank0_print(END_LINE)

    def get_monitor_tensors(self):
        monitor_tensors = dict(
            wte=self.get_wte().weight,
            lm_head=self.llm.get_output_embeddings().weight,
            vte=self.vte.weight
        )
        monitor_tensors.update(
            {f'visual_tokenizer_{k}': v for k, v in self.visual_tokenizer.get_monitor_tensors().items()})
        return monitor_tensors

    def get_wte(self):
        return self.llm.get_input_embeddings()

    def get_conversation_formatter(self) -> ConversationFormatter:
        if getattr(self, 'conversation_formatter', None) is None:
            self.conversation_formatter = getattr(import_module(".conversation_formatter", __package__),
                                                  self.config.conversation_formatter_class)(self.text_tokenizer)
        return self.conversation_formatter

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        grid_thws: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        return_token_info: bool = False,
        original_input_length: Optional[int] = None,
        use_mas_loss: bool = False,
        mas_loss_weight: float = 1.0,
        mas_tau: float = 0.5,
        mas_only: bool = False,  # New: use MAS loss only (disable standard loss)
        **kwargs
    ):
        # Always get token_info for MAS loss if needed
        need_token_info = use_mas_loss and labels is not None and pixel_values is not None
        actual_return_token_info = return_token_info or need_token_info
        
        merge_result = self.merge_multimodal(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            return_token_info=actual_return_token_info,
            original_input_length=original_input_length,
        )
        
        if actual_return_token_info:
            inputs_embeds, token_info = merge_result
        else:
            inputs_embeds = merge_result
            token_info = None
        
        # Determine if we need attention weights for MAS loss
        need_attention = use_mas_loss and labels is not None and pixel_values is not None
        if need_attention:
            kwargs['output_attentions'] = True
            kwargs['return_dict_in_generate'] = True
        
        # If using MAS loss only, disable standard loss computation
        if mas_only:
            llm_output = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=None, **kwargs)
        else:
            llm_output = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
        
        # Apply MAS loss if requested
        if use_mas_loss and labels is not None and pixel_values is not None:
            try:
                from .mas_loss import create_mas_loss
                
                # Initialize MAS loss if not exists
                if not hasattr(self, '_mas_loss_fn'):
                    self._mas_loss_fn = create_mas_loss(
                        tau_mas=mas_tau, 
                        alpha=mas_loss_weight, 
                        task_type="counting"  # Set for counting tasks
                    )
                
                # Check if we have attention weights
                if hasattr(llm_output, 'attentions') and llm_output.attentions is not None:
                    # Get attention weights (token_info should already be available)
                    attention_weights = llm_output.attentions
                    
                    # Determine generated steps (positions where labels != -100)
                    if labels is not None:
                        generated_mask = (labels != -100)
                        if generated_mask.any():
                            generated_steps = []
                            for batch_idx in range(labels.size(0)):
                                batch_generated = torch.where(generated_mask[batch_idx])[0].tolist()
                                generated_steps.extend(batch_generated)
                            generated_steps = list(set(generated_steps))  # Remove duplicates
                        else:
                            generated_steps = None
                    else:
                        generated_steps = None
                    
                    # Get predictions from logits
                    predictions = None
                    if hasattr(llm_output, 'logits') and llm_output.logits is not None:
                        predictions = llm_output.logits  # [batch, seq_len, vocab_size]
                    
                    # Compute MAS loss (only for wrong predictions)
                    mas_loss = self._mas_loss_fn(
                        attention_weights, 
                        token_info, 
                        generated_steps,
                        predictions=predictions,
                        labels=labels,
                        tokenizer=self.text_tokenizer  # Pass tokenizer for counting validation
                    )
                    
                    # Set loss based on mode
                    if mas_only:
                        # Use only MAS loss - set both as attribute and dict key
                        llm_output.loss = mas_loss
                        llm_output['loss'] = mas_loss  # Also add to OrderedDict for trainer
                    else:
                        # Add MAS loss to the main loss
                        if hasattr(llm_output, 'loss') and llm_output.loss is not None:
                            total_loss = llm_output.loss + mas_loss
                        else:
                            total_loss = mas_loss
                        
                        # Set combined loss - both as attribute and dict key
                        llm_output.loss = total_loss
                        llm_output['loss'] = total_loss  # Also add to OrderedDict for trainer
                else:
                    # No attention weights available, but mas_only mode requires a loss
                    if mas_only:
                        # Create a dummy loss to prevent training failure
                        batch_size = input_ids.size(0)
                        device = input_ids.device
                        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        
                        # Set dummy loss - both as attribute and dict key
                        llm_output.loss = dummy_loss
                        llm_output['loss'] = dummy_loss  # Also add to OrderedDict for trainer
                        print("Warning: No attention weights available for MAS loss, using dummy loss")
                    
            except Exception as e:
                # If MAS loss computation fails, handle gracefully
                print(f"Warning: MAS loss computation failed: {e}")
                if mas_only:
                    # If MAS-only mode fails, create a dummy loss or fallback
                    batch_size = input_ids.size(0)
                    device = input_ids.device
                    dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    
                    # Set dummy loss - both as attribute and dict key
                    llm_output.loss = dummy_loss
                    llm_output['loss'] = dummy_loss  # Also add to OrderedDict for trainer
                    print("Warning: MAS loss failed, using dummy loss for mas_only mode")
        elif mas_only:
            # mas_only is True but conditions not met, create dummy loss
            print("DEBUG: mas_only=True but MAS conditions not met, creating dummy loss")
            batch_size = input_ids.size(0)
            device = input_ids.device
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Create new output with dummy loss
            from transformers.modeling_outputs import CausalLMOutputWithPast
            llm_output = CausalLMOutputWithPast(
                loss=dummy_loss,
                logits=llm_output.logits,
                past_key_values=getattr(llm_output, 'past_key_values', None),
                hidden_states=getattr(llm_output, 'hidden_states', None),
                attentions=getattr(llm_output, 'attentions', None),
            )
        
        # Final check: ensure loss exists when mas_only=True
        if mas_only and (not hasattr(llm_output, 'loss') or llm_output.loss is None):
            print("DEBUG: EMERGENCY - Creating emergency dummy loss")
            batch_size = input_ids.size(0)
            device = input_ids.device
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Create new output with emergency dummy loss
            from transformers.modeling_outputs import CausalLMOutputWithPast
            llm_output = CausalLMOutputWithPast(
                loss=dummy_loss,
                logits=llm_output.logits,
                past_key_values=getattr(llm_output, 'past_key_values', None),
                hidden_states=getattr(llm_output, 'hidden_states', None),
                attentions=getattr(llm_output, 'attentions', None),
            )
        
        # Ensure loss is accessible to trainer when mas_only=True
        if mas_only and hasattr(llm_output, 'loss') and llm_output.loss is not None:
            # Since CausalLMOutputWithPast inherits from OrderedDict, 
            # we need to add the loss as a dict item for the trainer to see it
            llm_output['loss'] = llm_output.loss
        
        if return_token_info:
            # Attach token_info to the output
            if isinstance(llm_output, tuple):
                return llm_output + (token_info,)
            else:
                return llm_output, token_info
        
        return llm_output

    def merge_multimodal(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        grid_thws: Optional[torch.Tensor],
        return_token_info: bool = False,
        original_input_length: Optional[int] = None,
    ):
        placeholder_token_mask = torch.lt(input_ids, 0)
        multimodal_embeds = self.get_wte()(torch.masked_fill(input_ids, placeholder_token_mask, 0))
        
        # Initialize token info
        token_info = {
            'vision_tokens': 0,
            'system_instruction_tokens': 0,
            'total_tokens': input_ids.shape[1],
            'vision_positions': [],
            'system_instruction_positions': [],
            'indicator_positions': []
        }
        
        # We need to create a dummy visual input in two cases:
        # 1. During training in a distributed setup (e.g., DDP), to ensure that gradients
        #    for the visual encoder are synchronized, even if the real input is missing.
        #    This prevents the backward pass from hanging.
        # 2. When using DeepSpeed ZeRO-3, which shards model parameters. A dummy input
        #    is required even during evaluation to ensure all model parameters are correctly
        #    gathered and the forward pass can complete.
        need_dummy_visual_input = pixel_values is None and (self.training or is_deepspeed_zero3_enabled())
        if need_dummy_visual_input:
            pixel_values, grid_thws = self.visual_tokenizer.get_dummy_visual_inputs()
        
        if pixel_values is not None:
            # Count vision tokens
            if grid_thws is not None:
                num_image_atoms = grid_thws.prod().item()
                num_image_atoms //= self.visual_tokenizer.vit.config.hidden_stride ** 2
                num_image_atoms //= self.visual_tokenizer.vit.config.temporal_patch_size
                token_info['vision_tokens'] = num_image_atoms
            
            visual_indicator_embeds = self.vte(torch.tensor(
                list(range(self.config.visual_vocab_size - len(INDICATOR_IDS), self.config.visual_vocab_size)),
                dtype=torch.long,
                device=self.vte.weight.device
            )).to(dtype=multimodal_embeds.dtype, device=multimodal_embeds.device)
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = self.vte(visual_tokens).to(dtype=multimodal_embeds.dtype, device=multimodal_embeds.device)
            
            # Find positions of different token types
            for i, indicator_id in enumerate(INDICATOR_IDS):
                indicator_positions = (input_ids == indicator_id).nonzero(as_tuple=True)[1]
                token_info['indicator_positions'].extend(indicator_positions.tolist())
                multimodal_embeds[input_ids == indicator_id] = visual_indicator_embeds[i]
            
            # Find visual atom positions
            visual_positions = (input_ids == VISUAL_ATOM_ID).nonzero(as_tuple=True)[1]
            token_info['vision_positions'] = visual_positions.tolist()
            multimodal_embeds[input_ids == VISUAL_ATOM_ID] = visual_embeds
            
            # Find system instruction token positions
            # System instruction tokens: từ <|im_end|> trở đi (chỉ tính tokens ban đầu, không tính generated tokens)
            if len(token_info['indicator_positions']) >= 2:
                im_start_pos = min(token_info['indicator_positions'])  # <|im_start|>
                im_end_pos = max(token_info['indicator_positions'])    # <|im_end|>
                
                # Calculate the expected system instruction length based on the input
                # The input format is: "<image>\n[system_instruction]"
                # We need to count tokens in the system instruction part only
                
                # For now, let's use a simpler approach: count tokens from <|im_end|> to the end
                # but limit it to a reasonable number (the actual prompt length)
                
                # Find all positions after <|im_end|> that are not vision or indicator
                vision_positions_set = set(token_info['vision_positions'])
                indicator_positions_set = set(token_info['indicator_positions'])
                
                after_im_end_positions = []
                for pos in range(im_end_pos + 1, token_info['total_tokens']):
                    if pos not in vision_positions_set and pos not in indicator_positions_set:
                        after_im_end_positions.append(pos)
                
                # The system instruction should be the tokens that were in the original input
                # We need to determine this from the input_ids, not from the final sequence
                # For now, let's assume the system instruction is the first few tokens after <|im_end|>
                # and the rest are generated tokens
                
                # Count ALL tokens after <|im_end|> as system instruction + user input tokens
                # This includes both the system instruction and the user's input prompt
                system_positions = after_im_end_positions
                
                token_info['system_instruction_positions'] = system_positions
                token_info['system_instruction_tokens'] = len(token_info['system_instruction_positions'])
                
                # Debug info removed for cleaner output
            else:
                # Fallback: treat all text as system instruction tokens
                all_special_positions = set(token_info['indicator_positions'] + token_info['vision_positions'])
                token_info['system_instruction_positions'] = [i for i in range(token_info['total_tokens']) if i not in all_special_positions]
                token_info['system_instruction_tokens'] = len(token_info['system_instruction_positions'])
        
        if need_dummy_visual_input:
            multimodal_embeds += visual_embeds.sum() * 0.0 + visual_indicator_embeds.sum() * 0.0
        
        if return_token_info:
            return multimodal_embeds, token_info
        return multimodal_embeds

    def _merge_inputs(
        self, raw_input_ids, raw_labels, placeholder_indexes, grid_thws, indicator_begin_id, indicator_end_id
    ):
        input_ids = []
        labels = []
        prev_index = 0
        for placeholder_index, grid_thw in zip(placeholder_indexes, grid_thws):
            input_ids.extend(raw_input_ids[prev_index:placeholder_index])
            labels.extend(raw_labels[prev_index:placeholder_index])
            num_image_atoms = grid_thw.prod().item()
            num_image_atoms //= self.visual_tokenizer.vit.config.hidden_stride ** 2
            num_image_atoms //= self.visual_tokenizer.vit.config.temporal_patch_size
            input_ids.extend([indicator_begin_id] + [VISUAL_ATOM_ID] * num_image_atoms + [indicator_end_id])
            labels.extend([IGNORE_ID] * (num_image_atoms + 2))
            prev_index = placeholder_index + 1
        input_ids.extend(raw_input_ids[prev_index:])
        labels.extend(raw_labels[prev_index:])
        return input_ids, labels

    def preprocess_inputs(
        self,
        text_or_conversations: Union[List[Dict], str],
        images: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        videos: Optional[Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
        min_pixels=448 * 448,
        max_pixels=1344 * 1792,
        generation_preface='',
        return_labels=False,
        frame_selector=None,
        # enable_thinking=False,
    ):
        # convert text to conversations
        if isinstance(text_or_conversations, str):
            conversations = [{
                "from": "human",
                "value": text_or_conversations
            }]
        elif isinstance(text_or_conversations, list):
            conversations = text_or_conversations
        else:
            raise ValueError(
                f'[{datetime.now()}] Invalid type of `text_or_conversations`, expected `List[Dict]` or `str`,'
                f' but got {type(text_or_conversations)}')

        # select frame
        if frame_selector is not None:
            conversations, videos = frame_selector(conversations=conversations, frames=videos, clear_prompt=True)

        # format conversations
        prompt, raw_input_ids, raw_labels = self.get_conversation_formatter().format(
            conversations, generation_preface=generation_preface)
        image_token_indexes = [i for i, v in enumerate(raw_input_ids) if v == IMAGE_TOKEN_ID]
        video_token_indexes = [i for i, v in enumerate(raw_input_ids) if v == VIDEO_TOKEN_ID]

        # merge inputs
        input_ids, labels = raw_input_ids, raw_labels
        pixel_values, grid_thws = None, None
        if images is not None and videos is not None:
            raise ValueError(
                "Multiple visual input data types detected (both `images` and `videos` provided). "
                "This model supports only one type of visual input data at a time. "
                "Please provide either `images` or `videos`, but not both."
            )
        if min(len(image_token_indexes), len(video_token_indexes)) > 0:
            raise ValueError(
                "Multiple visual modality placeholders detected in text (`<image>` and `<video>`). "
                "The input text can contain placeholders for only one type of visual modality at a time. "
                "Please use either `<image>` or `<video>` placeholders, but not both."
            )
        if images is None and videos is None and max(len(image_token_indexes), len(video_token_indexes)) > 0:
            raise ValueError(
                "Visual modality placeholder(s) detected in the input text "
                "(e.g., `<image>` or `<video>`), but no corresponding visual data (`images` or `videos`) was supplied. "
                "A visual placeholder requires the corresponding data to be processed. "
                "To resolve this issue, please either: "
                "1. Remove the visual placeholder(s) from your input text, OR "
                "2. Provide the appropriate `images` or `videos` data alongside the text."
            )

        if images is not None:
            images = images if isinstance(images, list) else [images]
            pixel_values, grid_thws = zip(
                *(self.visual_tokenizer.preprocess(image=image, min_pixels=min_pixels, max_pixels=max_pixels)
                  for image in images)
            )
            assert len(image_token_indexes) == len(pixel_values), f"Mismatch in number of image {len(pixel_values)} and `<image>` {len(image_token_indexes)}"
            input_ids, labels = self._merge_inputs(
                raw_input_ids, raw_labels, image_token_indexes, grid_thws, INDICATOR_IDS[0], INDICATOR_IDS[1]
            )
            pixel_values = torch.cat(pixel_values, dim=0)
            grid_thws = torch.cat(grid_thws, dim=0)
        elif videos is not None:
            videos = videos if isinstance(videos[0], list) else [videos]
            assert len(videos) == 1, "only support single video"
            pixel_values, grid_thws = self.visual_tokenizer.preprocess(
                video=videos[0], min_pixels=min_pixels, max_pixels=max_pixels
            )
            assert len(video_token_indexes) == len(videos), f"Mismatch in number of video {len(video_token_indexes)} and `<video>` {len(videos)}"
            input_ids, labels = self._merge_inputs(
                raw_input_ids, raw_labels, video_token_indexes, grid_thws, INDICATOR_IDS[2], INDICATOR_IDS[3]
            )

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        if return_labels:
            assert all([label == IGNORE_ID or label >= 0 for label in labels]), "Invalid labels"
            labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
            return prompt, input_ids, pixel_values, grid_thws, labels
        else:
            return prompt, input_ids, pixel_values, grid_thws

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs
    ):
        super().save_pretrained(save_directory,
                                is_main_process=is_main_process,
                                state_dict=state_dict,
                                save_function=save_function,
                                safe_serialization=safe_serialization)
        self.text_tokenizer.save_pretrained(save_directory)
        self.visual_tokenizer.image_processor.save_pretrained(save_directory)

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        attention_mask = torch.ne(inputs, self.text_tokenizer.pad_token_id).to(device=inputs.device)
        
        # Get original input length before any generation
        original_input_length = inputs.shape[1] if inputs is not None else None
        
        # Get token info from merge_multimodal
        merge_result = self.merge_multimodal(
            input_ids=inputs,
            pixel_values=kwargs.pop('pixel_values', None),
            grid_thws=kwargs.pop('grid_thws', None),
            return_token_info=True,
            original_input_length=original_input_length
        )
        
        if isinstance(merge_result, tuple):
            inputs_embeds, token_info = merge_result
        else:
            inputs_embeds = merge_result
            token_info = None
        inputs_embeds = inputs_embeds.detach()
        torch.cuda.empty_cache()
        
        # Extract output_attentions from kwargs
        output_attentions = kwargs.pop('output_attentions', False)
        
        # Set return_dict_in_generate=True when output_attentions=True
        if output_attentions:
            kwargs['return_dict_in_generate'] = True
        
        # Call LLM generate with output_attentions
        llm_output = self.llm.generate(
            inputs=None, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            output_attentions=output_attentions,
            **kwargs
        )
        
        # Handle attention weights return
        if output_attentions:
            # When output_attentions=True, llm_output is GenerateDecoderOnlyOutput object
            if hasattr(llm_output, 'sequences'):
                generated_ids = llm_output.sequences
            else:
                generated_ids = llm_output
                
            if hasattr(llm_output, 'attentions'):
                attention_weights = llm_output.attentions
                
                # Return token_info if available
                if token_info is not None:
                    return generated_ids, attention_weights, token_info
                else:
                    return generated_ids, attention_weights
            else:
                if token_info is not None:
                    return generated_ids, token_info
                else:
                    return generated_ids
        else:
            # No attention requested, return normally
            generated_ids = llm_output
            # Return token_info if available (even without attention)
            if token_info is not None:
                return generated_ids, token_info
            else:
                return generated_ids

    @torch.no_grad()
    def chat(
        self,
        prompt: str,
        images: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        videos: Optional[Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
        do_sample: bool = False,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        min_pixels: int = 448 * 448,  # Parameter for image preprocessing
        max_pixels: int = 1792 * 1792,  # Parameter for image preprocessing
        history: Optional[Dict] = None,
        **generate_kwargs,  # Allows passing other generation arguments
    ):
        """
        Performs a single turn of conversation, optionally including visual input.
        Supports a two-phase generation process with a "thinking_budget" for complex reasoning.
        Args:
            prompt (str): The user's input prompt.
            images (Optional): Optional single image or list of images.
            videos (Optional): Optional single video (list of frames) or list of videos.
            do_sample (bool): Whether to use sampling during generation.
            max_new_tokens (int): The maximum number of new tokens to generate in total.
            enable_thinking (bool): If True, enables the model's Chain-of-Thought process.
            thinking_budget (Optional[int]): The maximum number of tokens for the "thinking" phase.
                                            If the model doesn't finish thinking within this budget,
                                            it will be forced to start generating the final answer.
            min_pixels (int): Minimum total pixels for image processing.
            max_pixels (int): Maximum total pixels for image processing.
            history (Optional[Dict]): Conversation history.
            **generate_kwargs: Additional arguments for the generation method.

        Returns:
            Tuple[str, str, Dict]: A tuple containing:
                - response (str): The final, user-facing response.
                - thinking (str): The model's internal thought process (if enable_thinking=True).
                - updated_history (Dict): The updated conversation history.
        """
        # Initialize history if starting a new conversation
        if history is None:
            history = {"conversations": [], "images": None, "videos": None}
        conversations = history["conversations"] + [{"from": "human", "value": prompt}]
        current_images = (images if isinstance(images, list) else [images]) if images is not None else []
        combined_images = (history["images"] or []) + current_images
        combined_images = combined_images or None
        current_videos = (videos if isinstance(videos[0], list) else [videos]) if videos is not None else []
        combined_videos = (history["videos"] or []) + current_videos
        combined_videos = combined_videos or None
    
        _, initial_input_ids, pixel_values, grid_thws = self.preprocess_inputs(
            conversations,
            images=combined_images,
            videos=combined_videos,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            generation_preface="<think>\n\n</think>\n\n" if not enable_thinking else ''
        )
        initial_input_ids = initial_input_ids.to(device=self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        if grid_thws is not None:
            grid_thws = grid_thws.to(device=self.device)
            
        THINK_END_TOKEN_ID = 151668  # </think>
        IM_END_TOKEN_ID = 151645  # <|im_end|>
        # Extract output_attentions early to ensure it's passed correctly
        output_attentions = generate_kwargs.get("output_attentions", False)
        # Debug message removed for cleaner output
        
        common_generate_args = {
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "do_sample": do_sample,
            "pad_token_id": self.text_tokenizer.pad_token_id,
            "output_attentions": output_attentions,
            **generate_kwargs,
        }
        # Debug message removed for cleaner output
        use_thinking_phase = enable_thinking and thinking_budget is not None and thinking_budget > 0
        attention_weights = None
        if not use_thinking_phase:
            generate_output = self.generate(
                initial_input_ids,
                max_new_tokens=max_new_tokens,
                **common_generate_args
            )
            if isinstance(generate_output, tuple):
                if len(generate_output) == 3:
                    generated_ids, attention_weights, token_info = generate_output
                elif len(generate_output) == 2:
                    # Check if second item is attention_weights or token_info
                    if isinstance(generate_output[1], (list, tuple)) and len(generate_output[1]) > 0:
                        # Second item is attention_weights
                        generated_ids, attention_weights = generate_output
                        token_info = None
                    else:
                        # Second item is token_info
                        generated_ids, token_info = generate_output
                        attention_weights = None
                else:
                    generated_ids = generate_output[0]
                    attention_weights = None
                    token_info = None
            else:
                generated_ids = generate_output
                attention_weights = None
                token_info = None
        else:
            # stage1: thinking_budget
            phase1_output = self.generate(
                initial_input_ids,
                max_new_tokens=thinking_budget,
                **common_generate_args
            )
            if isinstance(phase1_output, tuple):
                if len(phase1_output) == 3:
                    phase1_output_ids, phase1_attention, phase1_token_info = phase1_output
                elif len(phase1_output) == 2:
                    # Check if second item is attention_weights or token_info
                    if isinstance(phase1_output[1], (list, tuple)) and len(phase1_output[1]) > 0:
                        # Second item is attention_weights
                        phase1_output_ids, phase1_attention = phase1_output
                        phase1_token_info = None
                    else:
                        # Second item is token_info
                        phase1_output_ids, phase1_token_info = phase1_output
                        phase1_attention = None
                else:
                    phase1_output_ids = phase1_output[0]
                    phase1_attention = None
                    phase1_token_info = None
            else:
                phase1_output_ids = phase1_output
                phase1_attention = None
                phase1_token_info = None
                
            if IM_END_TOKEN_ID in phase1_output_ids[0]:
                generated_ids = phase1_output_ids
                attention_weights = phase1_attention
            else:
                intermediate_ids = phase1_output_ids
                if THINK_END_TOKEN_ID not in intermediate_ids[0]:
                    early_stop_text = (
                        "\n\nConsidering the limited time by the user, I have to give the solution "
                        "based on the thinking directly now.\n</think>\n\n"
                    )
                    early_stop_ids = self.text_tokenizer(
                        early_stop_text, return_tensors="pt", add_special_tokens=False
                    ).input_ids.to(self.device)
                    intermediate_ids = torch.cat([intermediate_ids, early_stop_ids], dim=1)
                # stage2: complete the generation
                phase1_tokens_consumed = intermediate_ids.shape[1]
                remaining_tokens = max_new_tokens - phase1_tokens_consumed
                if remaining_tokens > 0:
                    combined_input_ids = torch.cat([initial_input_ids, intermediate_ids], dim=1)
                    phase2_output = self.generate(
                        combined_input_ids,
                        max_new_tokens=remaining_tokens,
                        **common_generate_args
                    )
                    if isinstance(phase2_output, tuple):
                        if len(phase2_output) == 3:
                            phase2_output_ids, phase2_attention, phase2_token_info = phase2_output
                        elif len(phase2_output) == 2:
                            # Check if second item is attention_weights or token_info
                            if isinstance(phase2_output[1], (list, tuple)) and len(phase2_output[1]) > 0:
                                # Second item is attention_weights
                                phase2_output_ids, phase2_attention = phase2_output
                                phase2_token_info = None
                            else:
                                # Second item is token_info
                                phase2_output_ids, phase2_token_info = phase2_output
                                phase2_attention = None
                        else:
                            phase2_output_ids = phase2_output[0]
                            phase2_attention = None
                            phase2_token_info = None
                    else:
                        phase2_output_ids = phase2_output
                        phase2_attention = None
                        phase2_token_info = None
                    generated_ids = torch.cat([intermediate_ids, phase2_output_ids], dim=1)
                    # Combine attention weights if available
                    if phase1_attention is not None and phase2_attention is not None:
                        attention_weights = (phase1_attention, phase2_attention)
                    elif phase1_attention is not None:
                        attention_weights = phase1_attention
                    elif phase2_attention is not None:
                        attention_weights = phase2_attention
                else:
                    generated_ids = intermediate_ids
                    attention_weights = phase1_attention
        full_generated_ids_list = generated_ids[0].tolist()
        thinking, response = "", ""
        if enable_thinking:
            try:
                think_end_idx = full_generated_ids_list.index(THINK_END_TOKEN_ID) + 1
                thinking_ids = full_generated_ids_list[:think_end_idx]
                response_ids = full_generated_ids_list[think_end_idx:]
                thinking = self.text_tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
                response = self.text_tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            except ValueError:
                response = self.text_tokenizer.decode(full_generated_ids_list, skip_special_tokens=True).strip()
        else:
            response = self.text_tokenizer.decode(full_generated_ids_list, skip_special_tokens=True).strip()
        updated_history = {
            "conversations": conversations + [{"from": "gpt", "value": response}],
            "images": combined_images,
            "videos": combined_videos
        }

        # Return attention weights and token info if available
        if attention_weights is not None:
            # Get token_info from the most recent generation
            final_token_info = token_info
            if enable_thinking and 'phase2_token_info' in locals():
                final_token_info = phase2_token_info
            
            if final_token_info is not None:
                return response, thinking, updated_history, attention_weights, final_token_info
            else:
                return response, thinking, updated_history, attention_weights
        return response, thinking, updated_history

AutoConfig.register("ovis", OvisConfig)
AutoModelForCausalLM.register(OvisConfig, Ovis)
