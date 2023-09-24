import json
import pdb

task_id = set([])

with open('/projects/nlp_lab/zhiyang/phd4_projects/vison-FLAN/vision-flan.json','r') as fin:
    inputs = json.load(fin)
    
outputs = []  
with open('/projects/nlp_lab/zhiyang/phd4_projects/VL-Instruct/vision-flan_unique_id.json','w') as fout:
    for i, line in enumerate(inputs):
        line['id'] += f"_{i+1}"
        outputs.append(line)
    json.dump(outputs, fout)
        
        