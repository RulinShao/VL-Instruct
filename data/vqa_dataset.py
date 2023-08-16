import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question
import pandas as pd

from torchvision.datasets.utils import download_url
from .templates_5 import build_instruction

SUPPORT_TASK_LIST = ['open-domain_VQA', 'text_legibility', 'image_quality', 'if_region_overlap', 'visual_object_region', 'region_generation', 'descriptive_object_region_select', 'visual_subject_region', 'object_grounding', 'object_region_selection', 'object_region_match', 'visual_object_identification', 'VQA_attribute', 'image_text_selection', 'GC_selection', 'VQA_counting', 'region_area', 'wikihow_immediate_next_step_selection', 'VQA_scene_recognition', 'descriptive_object_region_generate', 'text_type', 'VQA_object_presence', 'wikihow_next_step', 'object_relationship', 'VQA_positional_reasoning', 'wikihow_text_image_step_order', 'question_image_match', 'region_text_match', 'GC', 'ITM', 'VQA', 'object_match', 'VQA_object_recognition', 'text_localization', 'object_description_generate', 'wikihow_image_text_step_order', 'select_overlap_least_region', 'visual_attribute', 'VQA_color', 'missing_object_selection', 'visual_subject_identification', 'select_overlap_most_region', 'VQA_sport_recognition', 'multimodal_factual_checking', 'VG', 'region_caption_match', 'VQA_activity_recognition', 'region_object_selection', 'select_overlaped_region', 'image_caption', 'VQA_sentiment_understanding', 'select_nonoverlaped_region', 'VG_selection', 'VQA_utility_affordance']
NOT_SUPPORT_TASK_LIST = ['region_object_selection', 'select_overlaped_region', 'descriptive_object_region_select', 'missing_object_selection']
SUPPORT_TASK_LIST = [x for x in SUPPORT_TASK_LIST if not x in NOT_SUPPORT_TASK_LIST]

# Define domain id mapping for doremi
VL_DOMAINS = ['llava', 'mini-gpt4', 'macaw', 'lamm', ]
DOMAIN_TO_IDX = {
    name: idx for idx, name in enumerate(VL_DOMAINS)}

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train"):
        self.split = split        

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        
        if self.split == "train":
            self.data_type = ['llava', 'mini-gpt4', 'macaw', 'lamm']
        elif self.split == "test":
            self.data_type = ['aok_vqa', 'aok_vqa_rationale', 'memecaps', 'ok_vqa', 'science_qa', 'text_vqa', 'textcaps', 'visdial', 'winoground']
        self.annotations = []

        if 'llava' in self.data_type:
            ann_paths = '/mnt_out/rlshao/data/llava/llava_instruct_150k.json'
            img_path = '/mnt_out/rlshao/data/coco/train2017'
            
            with open(ann_paths, 'r') as f:
                annotation = json.load(f)
            for ann in annotation:
                ann.update({'data_type': 'llava', 'img_dir': img_path})

            self.annotations += annotation
            
        elif 'mini-gpt4' in self.data_type:
            ann_paths = '/mnt_out/rlshao/data/mini-gpt4/filter_cap.json'
            img_path = '/mnt_out/rlshao/data/mini-gpt4/image'

            with open(ann_paths, 'r') as f:
                annotation = json.load(f)['annotations']
            for ann in annotation:
                ann.update({'data_type': 'mini-gpt4', 'img_dir': img_path})

            self.annotations += annotation
        elif 'macaw' in self.data_type:
            ann_paths = '/mnt_out/rlshao/data/macaw/generated_examples_coco.json'
            img_path = '/mnt_out/rlshao/data/coco/train2014'
    
            with open(ann_paths, 'r') as f:
                annotation = json.load(f)['data']
            for ann in annotation:
                ann.update({'data_type': 'macaw', 'img_dir': img_path})

            self.annotations += annotation
        elif 'lamm' in self.data_type:
            ann_paths = '/mnt_out/rlshao/data/lamm/LAMM_instruct_186k.json'
            img_path = '/mnt_out/rlshao/data/lamm/'

            with open(ann_paths, 'r') as f:
                annotation = json.load(f)
            for ann in annotation:    
                ann.update({'data_type': 'lamm', 'img_dir': img_path})
            annotation = [ann for ann in annotation if ann['image'].split('/')[0] != 'bamboo_images']

            self.annotations += annotation
        elif 'vl-instruct' in self.data_type:
            ann_paths = '/mnt_out/rlshao/vl-data/train_group_4.jsonl'
            img_path = '/mnt_out/rlshao/vl-data/images'

            jsonObj = pd.read_json(path_or_buf=ann_paths, lines=True)
            jsonObj.columns = jsonObj.columns.str.replace("target_txt", "target")
            jsonObj.columns = jsonObj.columns.str.replace("task_name", "task")
            valid_anns = []
            for task in SUPPORT_TASK_LIST:
                valid_anns.append(jsonObj[jsonObj['task']==task])
            annotation = pd.concat(valid_anns).sample(frac=1).to_dict('records') 
            
            for ann in annotation:
                ann.update({'data_type': 'my-dataset', 'img_dir': img_path})

            self.annotations += annotation
        else:
            assert self.split == 'test'
            data_dir = '/mnt_out/rlshao/data/multiInstruct_v1.0/eval'

            for task_name in self.data_type:
                ann_path = f'{data_dir}/eval_jsonl/{task_name}.jsonl'
                annotation = pd.read_json(path_or_buf=ann_path, lines=True).to_dict('records')
                for ann in annotation:
                    ann.update({'image_path': os.path.join(f'{data_dir}/eval_images', ann['image_path'].strip('./')), 'data_type': task_name})
                self.annotations += annotation
       

        random.shuffle(self.annotations)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):    
        ann = self.annotations[index]
        data_type = ann['data_type']
        
        if data_type == 'llava' or data_type == 'lamm':
            if data_type == 'llava':
                image_path = os.path.join(ann['img_dir'], ann['image'])
                unique_id = ann['id']
            elif data_type == 'lamm':
                image_path = os.path.join(ann['img_dir'], ann['image'])
                unique_id = ann['image']
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            num_conversations = len(ann['conversations'])//2
            chosen_idx = 2 * random.randint(0, num_conversations-1)
            assert ann['conversations'][chosen_idx]['from'] == 'human'
            instruction = ann['conversations'][chosen_idx]['value']
            target = ann['conversations'][chosen_idx + 1]['value']
        elif data_type == 'mini-gpt4':
            image_path = os.path.join(ann['img_dir'], f"{ann['image_id']}.jpg")
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            instruction = 'Describe this image in detail. Give as many details as possible. Say everything you see.'
            target = ann['caption']
            unique_id = ann['image_id']
        elif data_type == 'macaw':
            image_path = os.path.join(ann['img_dir'], ann['id'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            instruction = ann['instruction']
            target = ann['response']
            unique_id = ann['id']
        elif data_type == 'vl-instruct':
            image_path = os.path.join(ann['img_dir'], ann['image_path'].split('/')[-1])
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)

            instruction, target = build_instruction(**ann)
            unique_id = ann['unique_id']
        else:
            image_path = ann['image_path']
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            instruction = ann['prompt']
            target = ann['target']
            unique_id = ann['unique_id']

        if self.split == 'test':
            #question = pre_question(instruction)   
            question = instruction
            question_id = unique_id           
            return image, question, question_id


        elif self.split=='train':                                  
            #question = pre_question(instruction)        
            question = instruction
            answers = [target]
            weights = [0.2]
            domain_id = DOMAIN_TO_IDX[data_type]

            return image, question, answers, weights, domain_id
        
        
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n, domain_ids = [], [], [], [], [], []
    for image, question, answer, weights, domain_id in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
        domain_ids.append(domain_id)
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n, domian_ids        

def check_lamm_image():
    bad_images = ['bamboo_images/2880700382_2c2817c6c5_c.jpg']
    ann_paths = '/mnt_out/rlshao/data/lamm/LAMM_instruct_186k.json'
    img_path = '/mnt_out/rlshao/data/lamm/'
    with open(ann_paths, 'r') as f:
        annotation = json.load(f)
    annotation = [ann for ann in annotation if ann['image'] not in bad_images]
    for ann in annotation:
        try:
            image_path = os.path.join(img_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
        except:
            print(f"bad image: {image_path}")

