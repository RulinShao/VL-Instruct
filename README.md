# VL-Instruct

### Setup
Hard-coded the vicuna7b checkpoint in the LAVIS pkg. Install from the local source
```
cd LAVIS
pip install -e .
```

### Prepare the Data
Modify the data paths in the personalized dataloader in [data/vqa_dataset.py](https://github.com/RulinShao/VL-Instruct/blob/main/data/vqa_dataset.py)


### Train BLIP2-FlanT5-xxl
```
python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --model_type blip2_t5 --train_qformer
```

### Train BLIP2-Vicuna-7b 
TODO: double check if the model is loaded properly. Pay attention to `qformer_text_input` according to [this issue](https://github.com/salesforce/LAVIS/issues/344).
```
python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --model_type blip2_vicuna --train_qformer
```

### To also finetune the LLM 
```
python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --model_type blip2_vicuna --train_qformer --train_llm
```
