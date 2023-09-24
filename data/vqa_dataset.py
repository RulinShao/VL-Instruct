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
from .variables import *
import pdb



class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train"):
        self.split = split        

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        
        if self.split == "train":
            self.data_type = ['vison-flan']
        elif self.split == "test":
            self.data_type = []
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
        elif 'vison-flan' in self.data_type:
            ann_paths = '/projects/nlp_lab/zhiyang/phd4_projects/dataset/VL-Instruct/vision-flan_unique_id.json'
            vision_flan_img_path = '/projects/nlp_lab/zhiyang/phd4_projects/vison-FLAN/dataset/'
            multiinstruct_img_path = '/projects/nlp_lab/zhiyang/phd4_projects/vison-FLAN/images/'
            raw_annotation = json.load(open(ann_paths,'r'))
            annotation = []
            for ann in raw_annotation:    
                ann['data_type'] = 'vision-flan'
                ann['img_dir'] = vision_flan_img_path if ann['id'].startswith('vision-flan') else multiinstruct_img_path
                annotation.append(ann)
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
        # pdb.set_trace()
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
        elif data_type == 'vision-flan':
            """
            {"id": "vision-flan_coco+image_classification_appliance+13368", "image": "alok_checked/sampled_dataset/coco+image_classification_appliance/images/coco+image_classification_appliance_728_000000124349.jpg", "task_name": "coco+image_classification_appliance", "conversations": [{"from": "human", "value": "Given an image of a common electronic appliance from around the house, identify the type of object it is. It could be an appliance that is commonly used in the kitchen to cook or store food.\nOptions: (a) This image contains an oven (b) This image contains a refrigerator (c) This image contains a sink (d) This image contains a toaster (e) This image contains a microwave\n<image>"}, {"from": "gpt", "value": "(e) This image contains a microwave"}]}
            """
            instance_id = ann['id']
            image_path = os.path.join(ann['img_dir'], ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            instruction = ann['conversations'][0]['value']
            target = ann['conversations'][1]['value']
            unique_id = ann['id']
            task_name = ann['task_name']
            
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
            reference_loss = [0.1]
            try:
                domain_id = TASK_TO_IDX[task_name]
            except:
                pdb.set_trace()

            return instance_id, image, question, answers, weights, reference_loss, domain_id
        
        
def vqa_collate_fn(batch):
    id_list, image_list, question_list, answer_list, reference_loss_list, domain_ids, weight_list, n = [], [], [], [], [], [], [], []
    for _id, image, question, answer, weights, reference_loss, domain_id in batch:
        id_list.append(_id)
        image_list.append(image)
        question_list.append(question)
        # weight_list += weights
        answer_list += answer
        reference_loss_list += reference_loss
        # n.append(len(answer))
        domain_ids.append(domain_id)
    return id_list, torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(domain_ids), torch.Tensor(reference_loss_list)        

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

