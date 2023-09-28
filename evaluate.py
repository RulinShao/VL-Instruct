import torch
from PIL import Image
from collections import OrderedDict
import json
from tqdm import tqdm
from models.blip2_vqa import blip2_eval
import argparse
import os
import shortuuid
from pathlib import Path
from train_doremi import Doremi

def eval_model(args):
    # Model
    checkpoint_path = os.path.join('/projects/nlp_lab/zhiyang/phd4_projects/VL-Instruct/checkpoints',args.model_name, args.checkpoint)
    model = blip2_eval(model_type=args.model_type, ckpt_path=checkpoint_path+'.pth')
    
    checkpoint_path = Path(checkpoint_path)
    save_path = checkpoint_path / 'results'
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_list = ['textcaps','scienceqa_exp','aok_vqa', 'science_qa', 'visit', 'text_vqa', 'visual_spatial_reasoning', 'natural_language_visual_reasoning', 'winoground', 'medic_damage_severity', 'medic_informative', 'medic_disaster_types','medic_humanitarian', 'aokvqa_rational']
    question_dir = "/projects/nlp_lab/zhiyang/phd4_projects/CMOA/eval_files/"
    for question_file in file_list:
        args.question_file = question_file
        answers_file = os.path.join(save_path, args.question_file +'.jsonl')
        args.question_file = os.path.join(question_dir , args.question_file)
        questions = [json.loads(q) for q in open(f"{args.question_file}.jsonl", "r")]
        if os.path.exists(answers_file):
            try:
                answers = [json.loads(q) for q in open(f"{answers_file}", "r")]
                if len(answers) == len(questions):
                    print(f"Predictions at {answers_file} exists and has the same length as number of questions. skip it.")
                    continue
            except:
                print(f'regenerate predictions at {answers_file}')
            # spdb.set_trace()
        print(f"Testing {args.question_file}.jsonl")
        print(f"Save predictions at {answers_file}")

        print(f'Totally {len(questions)} testing instances')

        print(f"a sample instance look like this:\n\n{questions[0]['prompt']}\n\nAnswer: {questions[0]['target']}")
        print(f"\nIt's image is at {os.path.join(args.image_folder, questions[0]['image_path'])}")
        # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        for line in tqdm(questions):
            idx = line["unique_id"]
            image_file = os.path.join(args.image_folder, line["image_path"])
            assert os.path.exists(image_file)
            image = Image.open(image_file).convert('RGB')
            image = model.vis_processors["eval"](image).unsqueeze(0).to(model.model.device).to(torch.float16)
            text_input = line["prompt"]
            with torch.inference_mode():
                result = model.model.generate(
                    samples={"image": image, "prompt": text_input}, max_length=128, repetition_penalty=1.5)
            # example["result"] = result[0]
            # fout.write(json.dumps(example)+'\n')
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": text_input,
                                       "predict": result[0],
                                       "target": line['target'],
                                       "image_path": image_file,
                                       "answer_id": ans_id,
                                       "model_id": args.model_name,
                                       "metadata": {}}) + "\n")
            ans_file.flush()
        ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="pretrain_flant5xl", type=str, choices=["pretrain_flant5xxl" "pretrain_flant5xl" "vicuna7b"])
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default="checkpoint_02")
    parser.add_argument("--image-folder", type=str, default="/projects/nlp_lab/zhiyang/phd4_projects/CMOA/eval_images")
    parser.add_argument("--question-dir", type=str,
                        default="/projects/nlp_lab/zhiyang/phd4_projects/CMOA/eval_files/")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    
    args = parser.parse_args()
    
    eval_model(args)
        