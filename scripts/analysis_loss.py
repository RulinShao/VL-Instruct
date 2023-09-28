import json
import pdb

reference_loss = json.load(open('/projects/nlp_lab/zhiyang/phd4_projects/VL-Instruct/checkpoints/vision-flan_flan-t5xl_old/reference_loss.json','r'))
vision_flan = json.load(open('/projects/nlp_lab/zhiyang/phd4_projects/VL-Instruct/dataset/vision-flan_unique_id.json', 'r'))

vision_flan_dict = {}
for line in vision_flan:
    vision_flan_dict[line['id']] = line
    
task_loss = {}
num_task_instance = {}
for id_, loss in reference_loss.items():
    # pdb.set_trace()
    # if 'multiinstruct' in id_:
    #     continue
    # try:
    #     task_name = id_.split('+',2)[0]
    # except:
    #     pdb.set_trace()
    task_name = vision_flan_dict[id_]['task_name']
    if not task_name in task_loss:
        task_loss[task_name] = 0.0
        num_task_instance[task_name] = 0.0
    num_task_instance[task_name]+=1
    task_loss[task_name] += sum(loss)/(32- loss.count(0.0))
for task_name in task_loss:
    task_loss[task_name] /= num_task_instance[task_name]
pdb.set_trace()
# missing_instance = 0
# for line in vision_flan:
#     if not line['id'] in reference_loss:
#         missing_instance+=1
        
# pdb.set_trace()