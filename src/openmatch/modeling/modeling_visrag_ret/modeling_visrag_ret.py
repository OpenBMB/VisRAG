import torch
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional
from ..modeling_minicpmv.modeling_minicpmv import MiniCPMV
from concurrent.futures import ThreadPoolExecutor


def transform_image_mp(img_list, transform, device, max_workers=None):
    pixel_values = []
    

    
    # 使用ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for img_batch in img_list:
            img_inps = list(executor.map(transform, img_batch))
            for i in range(len(img_inps)):
                img_inps[i] = img_inps[i].to(device)
            pixel_values.append(img_inps if img_inps else [])

    return pixel_values


@dataclass
class BaseModelOutputWithAttentionMask(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.Tensor] = None

class VisRAG_Ret(MiniCPMV): # -> MiniCPMV ->  Ultimately a CausalLM
    def fused_tokenize(
        self,
        data_list=None, # List[str] 
        img_list=None, # List[List[PIL.Image]]
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None, # default None
        return_vision_hidden_states=False,
        **kwargs):
        
        assert data_list is not None
        bs = len(data_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = self._process_list(tokenizer, data_list, max_inp_length, padding_side="right")
        
        if vision_hidden_states is None:
            pixel_values = transform_image_mp(img_list, self.transform, self.device, max_workers=8)
            model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        return model_inputs
    
    def prepare_context(self, inputs, tokenizer):
        text_, image_ = inputs
        if not isinstance(text_, str):
            raise NotImplementedError(f"chatml format expected, expect outmost type to be str but got {type(text_)}")
        
        # 1.add text
        content = text_ 
        
        # 2. add image
        if image_:
            if self.config.slice_mode:
                images, final_placeholder = self.get_slice_image_placeholder(
                    image_, tokenizer
                ) # crop one image into multiple sub images -> List[Image]
                content = final_placeholder + "\n" + content
            else:
                images = [image_] # only keep one image without cropping -> List[Image]
                content = (
                    tokenizer.im_start
                    + tokenizer.unk_token * self.config.query_num
                    + tokenizer.im_end
                    + "\n"
                    + content
                )
        else:
            images = []
        
        return content, images
    
    def forward(
        self,
        text, # List[str] B*str
        image, # List[ PIL.Image ] B*PIL.Image, one image for each data
        tokenizer,
        vision_hidden_states=None,
        max_inp_length=2048,
        **kwargs):
        
        processed_image = []
        processed_text = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            contexts = list(executor.map(lambda inputs: self.prepare_context(inputs, tokenizer), zip(text, image)))
        
        for context in contexts:
            content_, image_ = context
            processed_text.append(content_)
            processed_image.append(image_)
        
        model_inputs = self.fused_tokenize(
            data_list=processed_text, # List[str]
            img_list=processed_image, # List[List[PIL.Image]]
            tokenizer=tokenizer,
            max_inp_length=max_inp_length
        )
        
        # this is vision encoder forward.
        model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
        vlm_outputs = self.llm.model(
            input_ids=None, # because image and text have been merged into model_inputs["inputs_embeds"] here, we don't give input_ids
            position_ids=None,
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            return_dict=True
        )
        
        return BaseModelOutputWithAttentionMask(
            last_hidden_state=vlm_outputs.last_hidden_state,
            attention_mask=model_inputs.attention_mask
        )
        
