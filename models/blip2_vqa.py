from lavis.models import load_model_and_preprocess
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

import torch
from PIL import Image
import requests


pretrain_flant5xl_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xl.pth'
pretrain_vicuna7b_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_vicuna7b.pth'

class BLIP2_VQA(Blip2Base):
    def __init__(self,                 
                    ckpt_path=None,
                    evaluate=False,
                    model_type="blip2_t5",
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
                self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=evaluate, device=device)
            elif model_type == "blip2_vicuna":
                self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=evaluate, device=device)  # hard-coded in lavis to use pretrain ckpt

    
    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128):
        # image = self.vis_processors["eval"](image).unsqueeze(0).to(device)
        if train:
            samples = {
                "image": image,
                "text_input": question,
                "text_output": answer,
                "prompt": None,
            }
            return self.model(samples)['loss']
        else:
            samples = {
                "image": image,
                "prompt": question,
            }
            return self.model.generate(samples)

def blip2_vqa(args):
    model = BLIP2_VQA(ckpt_path=args.pretrained, evaluate=args.evaluate)
    # if pretrained:
    #     model,msg = load_checkpoint(model,pretrained)
    return model  
