from typing import Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import requests
import torch
from torch.nn import CrossEntropyLoss

from lavis.models import load_model_and_preprocess
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
import pdb

pretrain_flant5xl_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xl.pth'
pretrain_vicuna7b_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_vicuna7b.pth'


class BLIP2_VQA(Blip2Base):
    def __init__(self,                 
                    ckpt_path=None,
                    evaluate=False,
                    model_type="blip2_vicuna",
                    train_llm=False,
                    train_qformer=True,
                    doremi_train=True,
                 ):
        """
        Args:
        """               
        super().__init__()

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        if ckpt_path:
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=evaluate, device='cpu')
            state_dict = torch.load(ckpt_path, map_location='cpu')['model']
            for key in list(state_dict.keys()):
                state_dict[key[6:]] = state_dict.pop(key)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(device)
        else:
            if model_type == "blip2_t5":
                self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=evaluate, device=device)
            elif model_type == "blip2_vicuna":
                self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=evaluate, device=device)  # hard-coded in lavis to use pretrain ckpt

        if train_llm and model_type == "blip2_vicuna":
            for name, param in self.model.llm_model.named_parameters():
                param.requires_grad = True
        elif train_llm and model_type == "blip2_t5":
            for name, param in self.model.t5_model.named_parameters():
                param.requires_grad = True
        if not train_qformer:
            for name, param in self.model.Qformer.named_parameters():
                param.requires_grad = False

        self.doremi = doremi_train


    
    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128,
            domain_ids=None, return_pertoken_losses=None):
        if train:
            samples = {
                "image": image,
                "text_input": question,
                "text_output": answer,
                "prompt": None,
            }
            if self.doremi:
                output = self.model(samples)
                loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
                logits = output.logits
                labels = output.labels
                # pdb.set_trace()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss.view(image.shape[0],-1)
                loss = torch.mean(loss, -1)
            else:
                loss = self.model(samples).loss
            return loss
        else:
            samples = {
                "image": image,
                "prompt": question,
            }
            return self.model.generate(samples)

def blip2_vqa(args):
    model = BLIP2_VQA(
        ckpt_path=args.pretrained, 
        evaluate=args.evaluate,
        model_type=args.model_type,
        train_llm=args.train_llm,
        train_qformer=args.train_qformer,
    )
    return model  
