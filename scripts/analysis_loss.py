import json
import pdb

reference_loss = json.load(open('/projects/nlp_lab/zhiyang/phd4_projects/VL-Instruct/checkpoints/temp/reference_loss.json','r'))


task_loss = {}
num_task_instance = {}
for id_, loss in reference_loss.items():
    # pdb.set_trace()
    if 'multiinstruct' in id_:
        continue
    try:
        task_name = id_.split('+',2)[0]
    except:
        pdb.set_trace()
    if not task_name in task_loss:
        task_loss[task_name] = 0.0
        num_task_instance[task_name] = 0.0
    num_task_instance[task_name]+=1
    task_loss[task_name] += sum(loss)/(32- loss.count(0.0))
for task_name in task_loss:
    task_loss[task_name] /= num_task_instance[task_name]
pdb.set_trace()