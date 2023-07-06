from io import BytesIO

import re
import copy

import logging
import pdb
import warnings
import random
import numpy as np
import torch
import base64
from importlib import import_module
import utils.transforms as T
from torchvision import transforms
from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# TODO: fill with task type


OUTPUT_REGION_TASK = {
    "detection", "VG", "object_region_selection", "region_generation","pointing_grounded_VQA","descriptive_object_region_generate", "text_localization", "visual_object_region", "visual_subject_region", "descriptive_object_region_select", "VG_selection", "grounded_VQA", "select_overlap_most_region", "select_overlap_least_region", "select_overlaped_region", "select_nonoverlaped_region"
}

OUTPUT_IMAGE_CODE_TASK = {"image_generation", "infilling", "im_region_extraction", "im_descriptive_infilling", "image_completion",  "image_completion_w_image_caption", "image_completion_w_region_caption", "im_descriptive_extraction"}

NO_IMAGE_AS_INPUT = {'image_generation'}

OPTIONS_REGION_TASK = {
    "object_region_selection","pointing_grounded_VQA", "text_localization", "descriptive_object_region_select","VG_selection", "grounded_VQA", "select_overlap_most_region", "select_overlap_least_region","select_overlaped_region", "select_nonoverlaped_region"
}

META_REGION_TASK = {
    "visual_answer_justification", "commonsense_VQA", "visual_object_region", "visual_subject_region", "select_overlap_most_region", "select_overlap_least_region", "select_overlaped_region", "select_nonoverlaped_region", "if_region_overlap"
}


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])
    
    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])
    code_masks = None
    if samples[0].get("code_mask", None) is not None:
        code_masks = torch.cat([sample['code_mask'] for sample in samples])
    
    w = [s["w"] for s in samples]
    h = [s["h"] for s in samples]
    image_path = [s["image_path"] for s in samples]
    options = [s["options"] for s in samples]
    task = [s["task"] for s in samples]
    if 'region_info' in samples[0]:
        region_info = [s["region_info"] for s in samples]
    else:
        region_info = None
    if "caption" in samples[0]:
        captions = [s["caption"] for s in samples]
    else:
        captions = None

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()
        
    # TODO: replace attributes
    attribute_len = samples[0].get("attribute_len", None) 
    if attribute_len is not None:
        attributes = torch.tensor([ s['attributes'] + [s['common_attribute']] * (src_tokens.shape[1] - len(s['attributes'])) for s in samples ] ).half()
        common_attributes = torch.tensor( [ s['common_attribute'] for s in samples ] ).half()
    else:
        attributes = None

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "code_masks": code_masks,
            "prev_output_tokens": prev_output_tokens,
            "attributes": attributes,
            "common_attributes": common_attributes
        },
        "target": target,
        "w": w,
        "h": h,
        "image_path": image_path,
        "options": options,
        "task": task,
    }
    if captions is not None:
        batch['captions'] = captions
    if region_info is not None:
        batch['region_info'] = region_info

    return batch


class InstructionDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=512,
        num_bins=1000,
        max_image_size=512,
        code_image_size=256,
        code_dict_size=8192,
        instruction_template='templates_5',
        use_natural=False,
        instruction_id=-1,
        image_gen=0,
        use_instruction_loss=False,
        attribute_len=9
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.build_instruction = import_module(instruction_template).build_instruction
        self.use_natural = use_natural
        self.instruction_id = instruction_id
        self.use_instruction_loss = use_instruction_loss
        if self.use_instruction_loss:
           self.instr_item = self.src_dict.encode_line('<instr>', append_eos=False).long()
        self.batch_size = 8
        self.same_task_num = 4
        self.instruction_num = 5
        self.attribute_len = attribute_len
        
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins
        self.code_image_size = code_image_size
        self.code_dict_size = code_dict_size
        self.image_gen = image_gen
        self.split = split

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        # for image infilling
        self.code_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # for positioning
        self.positioning_transform = {
            "imagenet": T.Compose([
                T.RandomResize([patch_image_size], max_size=patch_image_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, max_image_size=max_image_size)
            ]),
            "other": T.Compose([
                T.RandomResize([patch_image_size], max_size=patch_image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
            ])    
        }

        self.patch_resize_transform = {
            "imagenet": transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]),
            "other": transforms.Compose([
                lambda image: image.convert("RGB"),
                transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        }
    
    def get_random_masked_image(self, crop_loc, code_image_size):
        y, x, h, w = crop_loc # top, left, height, width
        scale = self.patch_image_size / code_image_size
        y, x, h, w = y*scale, x*scale, h*scale, w*scale
        mask_top, mask_left = int(y), int(x)
        mask_right, mask_bottom = int(x + w), int(y + h)
        # mask_ids = [
        #     i*self.code_image_size*2+j
        #     for i in range(self.code_image_size*2) for j in range(self.code_image_size*2)
        #     if not (mask_left <= i < mask_right and mask_top <= j < mask_bottom)
        # ]
        return mask_top, mask_left, mask_right, mask_bottom
        
    
    def get_region_coords(self, region_coords, w, h, image_src, image):
        if type(region_coords[0]) is not list:
            region_coords = [region_coords]
        region_coords_txt = []
        for region_coord in region_coords:
            boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
            x0, y0, x1, y1 = region_coord
            region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
            boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
            boxes_target["labels"] = np.array([0])
            boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

            patch_image, patch_boxes = self.positioning_transform[image_src](image, boxes_target)          
            resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
            quant_x0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][0] * (self.num_bins - 1)).round()))
            quant_y0 = "<bin_{}>".format(int((patch_boxes["boxes"][0][1] * (self.num_bins - 1)).round()))
            quant_x1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][2] * (self.num_bins - 1)).round()))
            quant_y1 = "<bin_{}>".format(int((patch_boxes["boxes"][0][3] * (self.num_bins - 1)).round()))
            region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
            region_coords_txt.append(region_coord)
        region_coords = region_coords_txt
    
        return region_coords, patch_image

    def __getitem__(self, index):
        if self.use_instruction_loss and self.instruction_id == -1:
            
            batch_index = index % self.batch_size
            if batch_index == 0:
                self.instruction_set = random.sample(range(self.instruction_num), self.same_task_num)
            if batch_index <  self.same_task_num:
                instruction_id = self.instruction_set[batch_index]
            else:
                instruction_id = self.instruction_id
        else:
            instruction_id = self.instruction_id
            
        example  = self.dataset[index]

        uniq_id = example.get("unique_id")
        task = example.get("task_name")
        image_path = example.get("image_path") 
        image_code = example.get("image_code")
        region_coords = example.get("region")
        options = example.get("options")
        image_src = "other" if example.get("image_source") != "imagenet" else example.get("image_source")
        crop_loc = example.get("crop_loc")
        code_image_size = example.get("code_image_size", self.code_image_size)
        target = example.get("target_txt")
        meta_data = example.get("meta_data")
        modality_feat = [0]*9 # initial modality feature
        """
        0 img in input
        1 text in input
        2 region in input
        3 img in tar
        4 text in tar
        5 region in tar
        6 token is text
        7 token is img
        8 token is region
        """
        if not meta_data: # TODO: legal problem, unify the key
            meta_data = example.get("meta")
        
        logger.debug(f"Task name: {task}")
        
        code_mask = torch.tensor([False])
        if image_path and task not in NO_IMAGE_AS_INPUT:
            # image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            modality_feat[0] = 1 # has image as input
        else:
            w, h = 0, 0

        if region_coords:
            region_coords, patch_image = self.get_region_coords(region_coords, w, h, image_src, image)
            patch_mask = torch.tensor([True])
            modality_feat[2] = 1 # has region as input
            if task in OUTPUT_IMAGE_CODE_TASK:
                if crop_loc:
                    mask_top, mask_left, mask_right, mask_bottom = self.get_random_masked_image(crop_loc, code_image_size)
                    patch_image[:, mask_top:mask_bottom, mask_left:mask_right] = 0
                image_code = torch.LongTensor(image_code)
                code_mask = torch.tensor([True])
                modality_feat[3] = 1 # has image as output
        elif image_path:
            if task in OUTPUT_IMAGE_CODE_TASK:
                modality_feat[3] = 1 # has image as output
                if task not in NO_IMAGE_AS_INPUT:
                    patch_image = self.code_resize_transform(image)
                    if crop_loc:
                        mask_top, mask_left, mask_right, mask_bottom = self.get_random_masked_image(crop_loc, code_image_size)
                        patch_image[:, mask_top:mask_bottom, mask_left:mask_right] = 0
                else: # no image as input
                    patch_image = torch.zeros(3, self.patch_image_size, self.patch_image_size)
                    patch_mask = torch.tensor([False])
                if not image_code:
                    image_code = [int(1000)] * 128
                image_code = torch.LongTensor(image_code)
                code_mask = torch.tensor([True])
            else:    
                patch_image = self.patch_resize_transform[image_src](image)
            patch_mask = torch.tensor([True])
        else:
            # init empty patch, empty region
            patch_image = torch.zeros(3, self.patch_image_size, self.patch_image_size)
            patch_mask = torch.tensor([False])
        
        if task in OPTIONS_REGION_TASK:
            options, patch_image = self.get_region_coords(options, w, h, image_src, image)
            patch_mask = torch.tensor([True])
            modality_feat[2] = 1 # has region as input
        
        if task in META_REGION_TASK:
            for k, v in meta_data['object_regions'].items():
                obj_region, _ = self.get_region_coords(v, w, h, image_src, image)
                meta_data['object_regions'][k] = obj_region
                modality_feat[2] = 1 # has region as input
        
        instruction_text, target = self.build_instruction(task, text=example.get('text'), options=options, region=region_coords, context=example.get('context'), question=example.get('question'), explanation=example.get('explanation'), response=example.get('response'), premise=example.get('premise'), hypothesis=example.get('hypothesis'),answer=example.get('answer'), meta_data=meta_data, target=target, use_natural=self.use_natural, instruction_id=instruction_id)
        logger.debug(instruction_text)
        
        # src_item = self.encode_text(instruction_text,length=self.max_src_length)
        # src_item = self.encode_text(' {}'.format(instruction_text.lower().strip()), length=self.max_src_length)
        
        
        if task in OUTPUT_REGION_TASK:
            modality_feat[5] = 1 # has region as output
            if type(target) is list:
                target_txt = ' '.join([region_coord + ' ' + self.bpe.encode(f'{tgt.lower()}') for region_coord, tgt in zip(region_coords, target)])
                target_debug = ' '.join([region_coord + f' {tgt}' for region_coord, tgt in zip(region_coords, target)])
                logger.debug(f'Target: {target_debug}')
            else:
                target_txt = ' '.join(region_coords)
                logger.debug(f'Target: {target_txt}')
            tgt_item = self.encode_text(' ' + target_txt, use_bpe=False, length=self.max_tgt_length)
        elif task in OUTPUT_IMAGE_CODE_TASK:
            modality_feat[3] = 1 # has image output
            tgt_item = image_code + len(self.src_dict) - self.code_dict_size - self.num_bins
        else:
            modality_feat[4] = 1 # has text output
            tgt_item = self.encode_text(' ' + target.lower(), length=self.max_tgt_length)
            logger.debug(f'Target: {target}')
            
        src_item, token_modality_feats = self.encode_text_with_region(' {}'.format(instruction_text.lower().strip()), modality_feat, length=self.max_src_length, use_bpe=True)
        
        if self.use_instruction_loss:
            src_item = torch.cat([self.bos_item, self.instr_item, src_item, self.eos_item])
            token_modality_feats = [modality_feat, modality_feat] + token_modality_feats + [modality_feat]
        else:
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            token_modality_feats = [modality_feat] + token_modality_feats + [modality_feat]
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        if self.image_gen:
            caption = example.get('text').strip() if example.get('text') else ''
        
        
        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "w": w,
            "h": h,
            "image_path": image_path if image_path else '',
            "options": options,
            "task": task,
            "attribute_len": self.attribute_len,
            "attributes": token_modality_feats,
            'common_attribute': modality_feat
            # "w_resize_ratio": resize_w / w if region_coord else None,
            # "h_resize_ratio": resize_h / h if region_coord else None,
            # "region_coord": region if region_coord else None
        }
        if self.image_gen:
            example['caption'] = caption
        if task == 'im_descriptive_extraction':
            example['region_info'] = meta_data['region'][0]
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
