import sys
import json
import pdb
import glob
from datasets import load_metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import textdistance
import os
import csv
from openpyxl import Workbook
from pathlib import Path

GEN_TASKS = ['aokvqa_rational', 'scienceqa_exp', 'textcaps', 'visit', 'text_vqa']
ALL_TASKS = ['textcaps','scienceqa_exp','aok_vqa', 'science_qa', 'visit', 'text_vqa', 'visual_spatial_reasoning', 'natural_language_visual_reasoning', 'winoground', 'medic_damage_severity', 'medic_informative', 'medic_disaster_types', 'medic_humanitarian','aokvqa_rational']
METRICS = ['accu','lose_accu','meteor', 'ROUGE/rougeL_mid/fmeasure', 'ent_score', 'str_dist']


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ent_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ent_tokenizer = AutoTokenizer.from_pretrained(ent_model_name)
ent_model = AutoModelForSequenceClassification.from_pretrained(ent_model_name).to(device)

def entailment_score(targets, responses):
    # premise = "I first thought that I liked the movie, but upon second thought it was actually disappointing."
    # hypothesis = "The movie was not good."
    avg_ent_score = 0.0
    for target, res in zip(targets, responses):
        # pdb.set_trace()
        try:
            input = ent_tokenizer(target, res, truncation=True, return_tensors="pt")
        except:
            print(f'warning invalid target: {target} response: {res}')
            continue
        output = ent_model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
        avg_ent_score += prediction['entailment']
    return avg_ent_score/len(targets)/100.0

def generation(predictions, references):
    meteor_metric = load_metric('meteor')
    rouge_metric = load_metric('rouge')

    gen_results = {}
    
    # METEOR
    meteor = meteor_metric.compute(predictions=predictions, references=references)
    gen_results.update(meteor)

    # ROUGE
    rouge = rouge_metric.compute(predictions=predictions, references=references, rouge_types=['rougeL'])
    rouge = {f'ROUGE/{k}_mid/fmeasure': v.mid.fmeasure for k, v in rouge.items()}
    gen_results.update(rouge)

    return gen_results

def normalize_option(pred, ref):
    # pred = pred.replace(';','').strip()
    pred = pred.lower()
    ref = ref.lower()
    if pred.endswith(';'):
        pred = pred[:-1]
    if ref.endswith(';'):
        pred = pred[:-1]
    return pred, ref
    
def string_dist(predictions, references):
    avg_dist = 0.0
    for pred, ref in zip(predictions, references):
        try:
            forward_dist = textdistance.levenshtein.normalized_distance(pred, ref)
            backward_dist = textdistance.levenshtein.normalized_distance(ref, pred)
        except:
            print(f'warning invalid target: {ref} response: {pred}')
            continue
        dist = (forward_dist + backward_dist)/2
        avg_dist += dist
    return avg_dist/max(len(references), 1.0)
    
def classification(predictions, references):
    accu = 0.0
    for pred, ref in zip(predictions, references):
        pred, ref = normalize_option(pred, ref)
        if pred == ref:
            accu+=1.0
    return {'accu': accu/len(predictions)}

def very_lose_classification(predictions, references):
    accu = 0.0
    for pred, ref in zip(predictions, references):
        pred, ref = normalize_option(pred, ref)
        ref_1, ref_2 = ref.split(' ',1)
        if ref_1 in pred or ref_2 in pred:
            accu+=1.0
    return accu/len(predictions)
    
# def get_performance(input_file):
#     with open(input_file,'r') as fin:
#         for line in fin:
#             line = json.loads(line)
#             if line['predict'] == line['target']:
#                 accu+=1
#                 continue
#             # try:
#             #     target1, target2 = line['target'].split(' ',1)
#             # except:
#             #     pdb.set_trace()
#             # if target1.lower() in line['predict'].lower() or target2.lower() in line['predict'].lower():
#             #     accu+=1
            

def evaluate(input_file):
    predictions = []
    references = []
    prompt_predictions = []
    prompt_references = []
    with open(input_file,'r') as fin:
        print(input_file)
        task_name = input_file.split('/')[-1]
        task_name = task_name.replace('.jsonl','')
        print(f'#### now evaluating {task_name}')
        for line in fin:
            line = json.loads(line)
            predictions.append(line['predict'])
            references.append(line['target'])
            prompt_predictions.append(f"{line['prompt']} {line['predict']}")
            prompt_references.append(f"{line['prompt']} {line['target']}")
        if task_name in GEN_TASKS:
            # ent_score = entailment_score(references, predictions)
            results = generation(predictions, references)
            # results['ent_score'] = ent_score
        else:
            # ent_score = entailment_score(references, predictions)
            results = classification(predictions, references)
            results['lose_accu'] = very_lose_classification(predictions, references)
            # results['ent_score'] = ent_score
        ent_score = entailment_score(prompt_references, prompt_predictions)
        results['ent_score'] = ent_score
        results['str_dist'] = string_dist(references, predictions)
        
        
    return results, task_name

if __name__ == '__main__':
    model_path = sys.argv[1]
    # folder = sys.argv[2]
    result_folder = os.path.join(model_path, 'results')
    performance_folder = os.path.join(model_path, 'performance')
    performance_folder = Path(performance_folder)
    performance_folder.mkdir(parents=True, exist_ok=True)
    # model_name = model_name.replace('/','_')
    # llava-vicuna-v1-3-7b-pretrain_aokvqa_rational_answer.jsonl
    wb = Workbook()
    ws = wb.active
    # Define rows
    dataset_names_row = []
    metric_names_row = []
    metric_values_row = []
    all_results = []
    performance_list = []
    # with open(f"/projects/nlp_lab/zhiyang/phd4_projects/CMOA/performance/{model_name}.csv",'w') as fout:
    for task_name in ALL_TASKS:
        file_path = f'{result_folder}/{task_name}.jsonl'
        if not os.path.exists(file_path):
            print(f"the result file: {file_path} does not exist, will skip {task_name} !!!!!!!!")
            performance_list.append([None]*len(METRICS))
            continue
        # for file_path in tqdm(glob.glob(f'{folder}/{model_name}_*.jsonl')):
        try:
            results, _ = evaluate(file_path)
        except:
            print(f"{task_name} failed !!!!!!!!!")
            performance_list.append([None]*len(METRICS))
            continue
        # fout.write(task+'\n')
        # for k, v in results.items():
        #     fout.write(f"{k}\t{v}\n")
        # fout.write(f"\n")
        print(results)
        
        performance_list.append([results.get(_metric,None) for _metric in METRICS ])
        dataset_names_row.extend([task_name] * len(results))
        for metric_name in METRICS:
            if metric_name in results:
                metric_value = results[metric_name]
                metric_names_row.append(metric_name)
                metric_values_row.append(metric_value)
        all_results.append((task_name, results))
        
    ws.append(dataset_names_row)
    ws.append(metric_names_row)
    ws.append(metric_values_row)
    
    # Merge dataset name cells
    col_index = 1
    for dataset_dict in all_results:
        dataset_name, metrics = dataset_dict
        ws.merge_cells(start_row=1, start_column=col_index, end_row=1, end_column=col_index+len(metrics)-1)
        col_index += len(metrics)

    wb.save(f'{performance_folder}/performance.xlsx')
    json.dump(performance_list, open(f'{performance_folder}/performance.json','w'))
                
            
        
            
        
            
        