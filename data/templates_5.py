#from email.mime import image
import random
#from util import produce_options
#import pdb

random.seed(7)

def build_instruction(task, text=None, options=None, region=None, context=None, question=None, explanation=None, response=None, premise=None, hypothesis=None, answer=None, meta_data=None, target=None, use_natural=False, instruction_id=-1, unique_id=None, image_source=None, image_path=None):
    if isinstance(region, list):
        if len(region) == 1:
            region = [str(x) for x in region[0]]
        else:
            for i in range(len(region)):
                region[i] = [str(x) for x in region[0]]
    if isinstance(options, list):
        #print(f'*** current options is {options}, task is {task}')
        if isinstance(options[0], list):
            options = options[0]
        random.shuffle(options)
        # options,target = produce_options(options,target)
    image_token = "image" # this show only appear before the output token
    options_token = "\n\n[Options]:"
    region_token = "Regions: "
    split_token = "||||" # or 
    region_split_token = "||||" # or 
    # multi-modal tasks 1
    if task == 'image_caption':
        instructs=[
            f"""In this task, you will look at the image and briefly describe the image.""",
            f"""What is the caption of the image?""",
            f"""Generate some text to describe the image.""",
            f"""Look at image and tell me what is the content.""",
            f"""In this task, you are given an image and you will need to generate some text to describe it.""",
            # f"""Tell me what this image is about?""",
            # f"""What is the content of the picture?"""   
        ]
    elif task == 'open-domain_VQA':
        instructs = [
            f"""{question}""",
            f"""{question}""",
            f"""{question}""",
            f"""{question}""",
            f"""{question}"""
        ]
    elif task == 'VQA':
        instructs = [
            f"{question}{options_token} {split_token. join(options)}",
            f"{question}{options_token} {split_token. join(options)}",
            f"{question}{options_token} {split_token. join(options)}",
            f"{question}{options_token} {split_token. join(options)}",
            f"{question}{options_token} {split_token. join(options)}"]
    elif task == 'GC': # same as region_caption
        instructs = [
            f"""The goal of this task is to generate description for one part of the image. The part is specified by {region_split_token.join(region)}.""",
            f"""What is the content of {region_split_token.join(region)}?""",
            # f"""{region_split_token.join(region)} sepcifies a region in image. What is {region_split_token.join(region)} about?""",
            f"""Describe the content of {region_split_token.join(region)} in image.""",
            # f"""What is the part of the image specified by {region_split_token.join(region)} about?""",
            f"""Generate a caption for {region_split_token.join(region)}.""",
            f"""{region_split_token.join(region)} is a region in image. Locate the region first and generate a description for that part of image.""",
            # f"""In this task, you will be given an image and the coordinates of a rectangular region in the image. Your goal is to generate some text to describe the rectangular region in the image. The Rectangular Region is {region_split_token.join(region)}.""",
        ]
    
    elif task == 'GC_selection': # same as region_caption
        instructs = [
            f"""Select the description for one part of the image. The part is specified by {region_split_token.join(region)}.{options_token} {split_token.join(options)}""",
            f"""What is the content of {region_split_token.join(region)}?{options_token} {split_token.join(options)}""",
            # f"""{region_split_token.join(region)} sepcifies a region in image. What is {region_split_token.join(region)} about?""",
            f"""Select the content of {region_split_token.join(region)} from options.{options_token} {split_token.join(options)}""",
            # f"""What is the part of the image specified by {region_split_token.join(region)} about?""",
            f"""What is the caption for {region_split_token.join(region)}?{options_token} {split_token.join(options)}""",
            f"""{region_split_token.join(region)} is a region in image. Select a description for that part of image.{options_token} {split_token.join(options)}""",
            # f"""In this task, you will be given an image and the coordinates of a rectangular region in the image. Your goal is to generate some text to describe the rectangular region in the image. The Rectangular Region is {region_split_token.join(region)}.""",
        ]
    
    elif task == 'VG':
        instructs = [
            # f"""What region in image does \"{text}\" describes?""",
            f"""The region in image that \"{text}\" describes is""",
            f"""Find the region in image that \"{text}\" describes.""",
            # f"""Which part of the image does \"{text}\" describe?""",
            f"""The goal of this task is to find the part of the image with the description: \"{text}\"""",
            f""" \"{text}\" describes part of the image. Find the part.""",
            f"""In this task, you are asked to localize the region in image that is described by the given text. The text is \"{text}\""""
        ]
        
    elif task == 'VG_selection': # same as region_caption
        instructs = [
            f"""Select region in the image that \"{text}\" describes.{options_token} {split_token.join(options)}""",
            f"""What is the region in the image that \"{text}\" describes?{options_token} {split_token.join(options)}""",
            f"""The goal of this task is to select the region of the image with the description: \"{text}\"{options_token} {split_token.join(options)}""",
            f""" \"{text}\" describes part of the image. Find the part.{options_token} {split_token.join(options)}""",
            f"""In this task, you are asked to localize the region in image that is described by the given text. The text is \"{text}\"{options_token} {split_token.join(options)}"""
        ]
    
    elif task == 'object_grounding':
        instructs = [
            f"""What is the object in {region_split_token.join(region)}""",
            f"""Identify the object in {region_split_token.join(region)}.""",
            f"""The goal of this task is to identify the object in given regions in image. The region is {region_split_token.join(region)}. What is the object?""",
            # f"""What object does region: {region_split_token.join(region)} contain?""",
            f"""The object contained in {region_split_token.join(region)} is""",
            f"""In this task, you are given the coordinates of some rectangular region in the image. You need to first localize each rectangular region and then identify what is the object in the region. The region is {region_split_token.join(region)}."""
        ]
    elif task == 'object_region_match':
        instructs = [
            f"""Is the object \"{text}\" in {region_split_token.join(region)}? {options_token} {split_token.join(options)}""",
            f"""Does the region {region_split_token.join(region)} contain \"{text}\"? {options_token} {split_token.join(options)}""",
            f"""Answer if the region {region_split_token.join(region)} contains \"{text}\". {options_token} {split_token.join(options)}""",
            f"""In this task, you will need to decide if the object in {region_split_token.join(region)} is \"{text}\". {options_token} {split_token.join(options)}""",
            f"""Decide if the object in {region_split_token.join(region)} matches \"{text}\". {options_token} {split_token.join(options)}"""
        ]
    elif task == 'object_match':
        instructs = [
            f"""Are the object in {region[0]} and object in {region[1]} the same type? {options_token} {split_token.join(options)}""",
            f"""In this task you are given two objects. Each object is specified by its location in the image. One object is in {region[0]} and another object is in {region[1]}. Decide if two objects have the same type. {options_token} {split_token.join(options)}""",
            f"""The goal of this task is to check if two regions contain the same type of object in the image. The two regions are {region_split_token.join(region)}. {options_token} {split_token.join(options)}""",
            f"""Do objects in {region_split_token.join(region)} have the same type? {options_token} {split_token.join(options)}""",
            f"""Determine whether the same kind of object is present in both given regions of the image. The two regions are {region_split_token.join(region)}. {options_token} {split_token.join(options)}"""
        ]    
    elif task == 'question_image_match':
        instructs = [
            f"""In this task, you need to decide if the image has enough information to answer \"{question}\" {options_token} {split_token.join(options)}""",
            f"""Given content of image, do you have enough information to answer \"{question}\" {options_token} {split_token.join(options)}""",
            f"""In this task, you are given the question \"{question}\" and you need to decide if the image provide you enough info to answer the question. {options_token} {split_token.join(options)}""",
            f"""Is it possible to answer \"{question}\" given the content of image? {options_token} {split_token.join(options)}""",
            f"""Does the image contain the answer to \"{question}\"? {options_token} {split_token.join(options)}"""
        ]
    elif task == 'object_region_selection':
        instructs = [
            f"""Select the region containing \"{text}\".{options_token} {split_token.join(options)}""",
            f"""What is the regions in the options that contain \"{text}\"?{options_token} {split_token.join(options)}""",
            f"""Which option contains \"{text}\"?{options_token} {split_token.join(options)}""",
            f"""Select the option that contains the object \"{text}\".{options_token} {split_token.join(options)}""",
            f"""You are given regions as options and select the option that contains the object \"{text}\".{options_token} {split_token.join(options)}""",
        ]
        
    # modify
    elif task == 'missing_object_selection':
        instructs = [f"""Select objects that do not appear in any of {region_split_token.join(region)}. Select "None" if you can't find any.{options_token} {split_token.join(options)}""",
                     f"""Select options that do not appear in any of {region_split_token.join(region)}.{options_token} {split_token.join(options)}""",
                     f"""Given {region_split_token.join(region)}, select objects that do not appear in any of the regions. Select "None" if you can't find it.{options_token} {split_token.join(options)}""",
                     f"""Which objects in options do not in appear in any of {region_split_token.join(region)}? Select "None" if you can't find it.{options_token} {split_token.join(options)}""",
                     f"""In this task, you are given some regions {region_split_token.join(region)}. Decide which object in options that do not appear in any of the given region.{options_token} {split_token.join(options)}"""
        ]
    elif task == 'ITM':
        instructs = [f"""Does \"{text}\" describes image? {options_token} {split_token.join(options)}""",
                     f"""Does the text: \"{text}\" and the content of image match? {options_token} {split_token.join(options)}""",
                     f"""Is the text: \"{text}\" the caption of image? {options_token} {split_token.join(options)}""",
                     f"""In this task you are given some text and you need to decide if the text describe the image. {options_token} {split_token.join(options)}""",
                     f"""Is the caption of image \"{text}\"? {options_token} {split_token.join(options)}""",
        ]
        
    # modify    
    elif task == 'region_object_selection': 
        instructs = [f"""Select objects from the options that appear in at least one of the regions. Select "None" if you can't find it.{region_token} {region_split_token.join(region)}. {options_token} {split_token.join(options)}""",
                     f"""Given objects in the options, select options that appear in at least one of {region_split_token.join(region)}.Select "None" if you can't find any.{options_token} {split_token.join(options)}""",
                     f"""What are the objects in the options that appear in at least one of the regions: {region_split_token.join(region)}?{options_token} {split_token.join(options)}""",
                     f"""Given {region_token} {region_split_token.join(region)}, decide which object appears in at least one of the region.{options_token} {split_token.join(options)}""",
                     f"""Given some regions, select object that appears in at least one of the region. {region_token} {region_split_token.join(region)}{options_token} {split_token.join(options)}"""
        ]
    elif task == 'region_generation': # mscoco
        instructs = [f"""What are the regions contain the object \"{text}\"?""",
                     f"""Given object: \"{text}\", what are the regions that contain this objects?""",
                     f"""The regions that contain \"{text}\" are""",
                     f"""The parts of image that have \"{text}\" are""",
                     f"""Identify the regions that contain \"{text}\".""",
                     f"""In this task, you are asked to identify all the regions in the image that contain the object \"{text}\".""",
                     f"""Which parts of image contain \"{text}\"?"""
                     ]
        
    elif task == 'region_caption_match':
        instructs = [f"""Decide if \"{text}\" is the description of {region_split_token.join(region)}. {options_token} {split_token.join(options)}""",
                     f"""Does \"{text}\" matches the content of {region_split_token.join(region)}. {options_token} {split_token.join(options)}""",
                     f"""In this task, you need to decide if \"{text}\" is a caption of {region_split_token.join(region)} in the image. {options_token} {split_token.join(options)}""",
                     f"""Can \"{text}\" describe {region_split_token.join(region)}? {options_token} {split_token.join(options)}""",
                     f"""Does {region_split_token.join(region)} and given text match? Text: {text} {options_token} {split_token.join(options)}"""
                     ]
    elif task == 'object_relationship':
        instructs = [
            f"""In this task, you are given the regions of a subject A and an object B in the image. Determine what is their relationship. The relationship can be the position of subject A relative to object B or what is subject A doing to object B. Region of subject A is {region[0]} and region of object B is {region[1]}.""",
            f"""What is the relationship between the subject in {region[0]} and object in {region[1]}?""",
            f"""Given a subject in {region[0]} and an object in {region[1]}, what's their relationship?""",
            f"""Subject A: {region[0]} Object B: {region[1]} and their relationship is""",
            f"""Tell me the relationship between the subject in {region[0]} and the object in {region[1]}."""
        ]
    elif task == 'visual_object_identification':
        instructs = [
            f"""Given the image, the subject in {region[0]} {meta_data['relation']} what?""",
            f"""Given the image, the subject in {region[0]} {meta_data['relation']} an object. What is the object?""",
            f"""Given the subject in {region[0]} and relationship \"{meta_data['relation']}\". What is the object?""",
            f"""Identify the name of the object, given the subject in {region[0]} and relationship: {meta_data['relation']}. """,
            f"""In this task, you are asked to identify the object given tne region of the subject in the image and their relationship. The subject is in {region[0]} and relationship is {meta_data['relation']}. The object is""",
        ]
    elif task == 'visual_subject_identification':
        instructs = [
            f"""Given the image and the object in {region[1]}, predict what is the subject {meta_data['relation']} the object?""",
            f"""Given the object in {region[1]}, and the relationship {meta_data['relation']}. What is the subject.""",
            f"""Identify the subject that {meta_data['relation']} the object.\nThe object is in {region[1]}""",
            f"""Which subject in the image that has {meta_data['relation']} with the object in {region[1]}""",
            f"""In this task, you are given the region of the object and the relation. What is the name of the subject? \n\nRelationship: {meta_data['relation']}\nObject: {region[1]}""",
            
        ]
    elif task == 'visual_object_region':
        region=  region_split_token.join(meta_data['object_regions']['subject'])
        instructs = [
            f"""Which object has the relationship \"{meta_data['relation']}\" with the subject in {region}? Answer the question by generating the region of the object.""",
            f"""Find the region of the object that has the relationship \"{meta_data['relation']}\" with the subject in {region}.""",
            f"""Given the image, where is the object that has the relatipnship \"{meta_data['relation']}\" with the the subject in {region}?""",
            f"""Identify the region of the object given the subject in {region} and relationship \"{meta_data['relation']}\".""",
            f"""What is the object region, given subject in {region} and relationship \"{meta_data['relation']}\"?""",
            f"""What is the object region, given the subject region and the relationship?\n\nSubject region: {region} Relationship: \"{meta_data['relation']}\"?"""
        ]
    elif task == 'visual_subject_region':
        region=  region_split_token.join(meta_data['object_regions']['object'])
        instructs = [
            f"""Given the object in {region}, where is the subject in the image that has relationship: \"{meta_data['relation']}\" with the object?""",
            f"""The object is in {region}. Identify the region of the subject that has relationship: {meta_data['relation']} with the object.""",
            f"""What is the region of the object, given subject in {region} and relationship \"{meta_data['relation']}\"?""",
            f"""Subject is in {region} and relationship is \"{meta_data['relation']}\". Generate the region of the object.""",
            f"""Based on the relationship and the subject, identify the object region. Subject region: {region} Relationship: {meta_data['relation']}"""
        ]
    elif task == 'descriptive_object_region_generate':
        instructs = [f"""Given the description of an object, generate the region that contains this object. The description is: \"{text}\"""",
                    f"""In this task, you are required to identify the object that is described by \"{text}\" and output the region of that object.""",
                    f"""What is the region of the object described by \"{text}\" in image?""",
                    f"""Where is the object described by \"{text}\"?""",
                    f"""Find the region of {text}""",
        ]
    elif task == 'descriptive_object_region_select':
        instructs = [
                    f"""Given the description of an object, select the region that contains this object.\n\nThe description is: \"{text}\"{options_token} {split_token.join(options)}""",
                    f"""In this task, you are required to identify the object that is described by \"{text}\" and select the region of that object from options.{options_token} {split_token.join(options)}""",
                    f"""What is the region of the object described by \"{text}\" in the picture?{options_token} {split_token.join(options)}""",
                    f"""Select the region of the object described by \"{text}\".{options_token} {split_token.join(options)}""",
                    f"""Given the image, select the region of {text}.{options_token} {split_token.join(options)}"""
        ]
    elif task == 'object_description_generate':
        instructs = [f"Generate a sentence to describe the object in the given bounding box. The description should help people to distinguish the object from other objects in the image.\n\nBounding box: {region_split_token.join(region)}",
                     f"Describe the object in the given region {region_split_token.join(region)}. The description should be about the location and appearance of the object so it can be distinguished from other object in the image.",
                     f"Given the object in {region_split_token.join(region)}, write a sentence to describe it. So it can be easily identified by people.",
                     f"Write a sentence to describe the object in the given region.\n\nRegion: {region_split_token.join(region)}",
                     f"Write a description of the object in region: {region_split_token.join(region)}. The description should help people to locate the object without causing confusion."
        ]
    # elif task == 'object_description_selection':
    #     instructs = [
    #     ]
    # elif task == 'object_description_match':
    #     instructs = [
    #     ]
    
    elif task == 'image_quality':
        instructs = [f"Select the reason from options to explain why the image quality is bad. {options_token} {split_token.join(options)}",
                     f"Explain why the image quality is bad. {options_token} {split_token.join(options)}",
                     f"Tell me what is wrong with the image. {options_token} {split_token.join(options)}",
                     f"The image quality might be low. Tell me why. {options_token} {split_token.join(options)}",
                     f"Select a reason for the bad quality of the image. {options_token} {split_token.join(options)}"
                     ]
    elif task == 'text_localization':
        instructs = [
            f"""Select the region from options that contains the given letters: \"{text}\". {options_token} {split_token.join(options)}""",
            f"""Determine which region contains the letters: \"{text}\"? {options_token} {split_token.join(options)}""",
            f"""Select the region that contains the text \"{text}\" {options_token} {split_token.join(options)}""",
            f"""Which region contains \"{text}\" {options_token} {split_token.join(options)}""",
            f"""Identify the region that has \"{text}\" written on. {options_token} {split_token.join(options)}"""
        ]
    elif task == 'text_legibility':
        instructs = [
            f"""Look at the given text region of the {image_token} and decide whether the text in the region is clear and complete. {region_token} {split_token.join(region)} {options_token} {split_token.join(options)}""",
            f"""Decide if the text in {split_token.join(region)} is clear and complete. {options_token} {split_token.join(options)}""",
            f"""Decide if the text in the given region is legible. Region {split_token.join(region)} {options_token} {split_token.join(options)}""",
            f"""In this task, you are given a region which has some text written on it. Tell me if the text on that region is clear. Region {split_token.join(region)} {options_token} {split_token.join(options)}""",
            f"""Tell me if the text on {split_token.join(region)} is clear and readable. {options_token} {split_token.join(options)}"""
        ]
        # f"""Instruction: Given the {image_token} and the visual region, determine whether the text in the visual regions is legible or not.\nInput: {options_token} {split_token.join(options)} {region_token} {split_token.join(region)} {output_token} The answer is """,
        #     f"""Instruction: Given the {image_token} and text region, please tell me if the text information in the region is legible or not.\nInput: {options_token} {split_token.join(options)} {region_token} {split_token.join(region)} {output_token} The answer is """
    elif task == 'text_type':
        instructs = [
            f"""Look at the text in the given region of the {image_token} and determine the type of text in the region from options. {region_token} {split_token.join(region)} {options_token} {split_token.join(options)}""",
            f"""Read the text in {split_token.join(region)} of the {image_token} and select the type of text from options. {options_token} {split_token.join(options)}""",
            f"""What type is the text in {split_token.join(region)}? {options_token} {split_token.join(options)}""",
            f"""The type of the text in {split_token.join(region)} is {options_token} {split_token.join(options)}""",
            f"""look at the text in {split_token.join(region)} and tell me it's type. {options_token} {split_token.join(options)}"""
        ]
        # f"""Instruction: Given the {image_token} and the text region, classify the text in the regions into one of the categories in {options_token}.\nInput: {options_token} {split_token.join(options)} {region_token} {split_token.join(region)} {output_token} The answer is """,
        #     f"""Instruction: Given the text region in {image_token}, please tell me if the text information in the region is \"{options[0]}\", \"{options[1]}\" or {options[2]}.\nInput: {options_token} {split_token.join(options)} {region_token} {split_token.join(region)} {output_token} The answer is """
    elif task == 'region_text_match':
        instructs = [
            f"""Look at the letters in {region_token} {split_token.join(region)} and determine if the letters in the region are the same as \"{text}\". {options_token} {split_token.join(options)}""",
            f"""Is the text \"{text}\" in {split_token.join(region)}? {options_token} {split_token.join(options)}""",
            f"""Does {split_token.join(region)} have the letters \"{text}\"? {options_token} {split_token.join(options)}""",
            f"""Is the text in {split_token.join(region)} the same as \"{text}\"? {options_token} {split_token.join(options)}""",
            f"""Do the letters in {split_token.join(region)} match \"{text}\"? {options_token} {split_token.join(options)}"""
        ]
        # f"""Instruction: Given the text region and the text information "\"{text}\"", decide if the text in the regions matches the text information.\nInput: {text_token} Text: \"{text}\" {region_token} {split_token.join(region)}{options_token} {split_token.join(options)} {output_token} The answer is """,
        #     f"""Instruction: Given the text region in {image_token} and the extra text information "\"{text}\"", please tell me if the text information matches text appeared in the region.\nInput: {text_token} Text: \"{text}\" {region_token} {split_token.join(region)} {options_token} {split_token.join(options)} {output_token} The answer is """
    elif task == 'multimodal_factual_checking':
        instructs = [
            f"Deicide if the claim can be supported by the image and the context.\n\nContext: {context}\n\nClaim: \"{text}\"{options_token} {split_token.join(options)}",
            f"Context: {context}\nCan the context support \"{text}\"? {options_token} {split_token.join(options)}",
            f"{context}\n\nRead previous text and decide if \"{text}\" is factually correct? {options_token} {split_token.join(options)}",
            f"Does the context support \"{text}\"?\n\nContext: {context} {options_token} {split_token.join(options)}",
            f"Context: {context}\n\nDoes the context support \"{text}\"? {options_token} {split_token.join(options)}"
        ]
    elif task == 'wikihow_next_step':
        context = '\n'.join(context) if len(context)>0 else '\"nothing\"'
        instructs = [
            f"For the task {meta_data['method']}, given the history steps {context} and the current step with its corresponding image, what is the next step for this task? The current step is {text}, what is the next step?",
            # f"Given the current step and its corresponding image of a task, what is the next step for this task? All previous steps are {context}\n\nThe task is {meta_data['method']} and the current step is {text}, what is the next step? ",
            f"What is the next step? You are doing {meta_data['method']} and you have finished\n\n{context}\nYou currently are at the step given by the image and the text \"{text}\". The next step is",
            # image only
            f"You are doing {meta_data['method']}. You have done\n\n{context}\nNow you are at the step described by the image. What is the next step?",
            # f"You want to achieve the task: {meta_data['method']}. You finished {context}. Now look at the image and answer what is the next step?",
            f"The goal is to \"{meta_data['method']}\". Given current step specified by the content of the image and you have finished.\n\n\All previous steps: {context}.\nWhat is the next step?",
            #text only
            f"You are doing {meta_data['method']} and you are at \"{text}\" step. The previous steps you finished are\n\n{context}\nWhat is the next step?",
        ]
    elif task == 'wikihow_text_image_step_order':
        options = ['next','previous']
        random.shuffle(options)
        instructs = [
            f"For the task \"{meta_data['method']}\", given the current step, decide if the content of the image is the next or previous step.\nThe current step is {text}.{options_token} {split_token.join(options)}",
            f"Is the image the next or previous step? You are doing \"{meta_data['method']}\" and you are currently at \"{text}\".{options_token} {split_token.join(options)}",
            f"The overall goal is to {meta_data['method']}. You are at \"{text}\" step. Is the image the next or the previous step?{options_token} {split_token.join(options)}",
            f"The goal is to \"{meta_data['method']}\". Given the current step \"{text}\", Is the picture the next or the previous step?{options_token} {split_token.join(options)}",
            f"You are doing {meta_data['method']}. Is the step specified in the picture the next or previous step to \"{text}\"?{options_token} {split_token.join(options)}",
        ]
    elif task == 'wikihow_image_text_step_order':
        options = ['next','previous']
        random.shuffle(options)
        instructs = [
            f"For the task \"{meta_data['method']}\", decide if \"{text}\" is the next or previous step to the step specified by the image.{options_token} {split_token.join(options)}",
            f"Is \"{text}\" the next or previous step? You are doing \"{meta_data['method']}\" and you are currently at the step described by the image.{options_token} {split_token.join(options)}",
            f"The overall goal is to {meta_data['method']}. You are at the step specified by the content of the image. Is \"{text}\" the next or the previous step?{options_token} {split_token.join(options)}",
            f"The goal is to \"{meta_data['method']}\". Given the current step in the picture, Is \"{text}\" the next or the previous step?{options_token} {split_token.join(options)}",
            f"You are doing {meta_data['method']}. Is the step \"{text}\" the next or previous step to the step in the image?{options_token} {split_token.join(options)}",
        ]
    elif task == 'wikihow_immediate_next_step_selection':
        instructs = [
            f"For the task \"{meta_data['method']}\", select the immediate next step to the step specified by the image.{options_token} {split_token.join(options)}",
            f"You are doing \"{meta_data['method']}\" and you are currently at the step described by the image. What is your next step?{options_token} {split_token.join(options)}",
            f"The overall goal is to {meta_data['method']}. You are at the step specified by the content of the image. Select the immediate next step from the options.{options_token} {split_token.join(options)}",
            f"The goal is to \"{meta_data['method']}\". Given the current step in the picture, what is the next step?{options_token} {split_token.join(options)}",
            f"You are doing {meta_data['method']}. What is the next step to step in the image?{options_token} {split_token.join(options)}",
        ]
    elif task == 'image_text_selection':
        instructs = [f"""Select the text from options that best describes the image. {options_token} {split_token.join(options)}""",
                     f"""Which text in the options best describes the image? {options_token} {split_token.join(options)}""",
                     f"""In this task, you are given some sentences and you need to decide which sentence best matches the image.{options_token} {split_token.join(options)}""",
                     f"""Which option in the options that is the caption of the image. {options_token} {split_token.join(options)}""",
                     f"""Select the caption of the image. {options_token} {split_token.join(options)}""",
                     ]
    elif task == 'natural_language_visual_reasoning':
        # instructs = [
        #     f"""Look at the image and decide if the text is true or not.\n\nText: {text}""",
        #     f"""Based on the image, Is \"{text}\" true?""",
        #     f"""In this task, you are given a sentence and an image and you need to decide if the image supports the sentence.\n\nSentence:{text}""",
        #     f"""\"{text}\"\n\nDecide if the above text is true or not based on the picture.""",
        #     f"""Look at the picture and is \"{text}\" true?""",
        # ]
        raise NotImplementedError
    elif task == 'visual_spatial_reasoning':
        # instructs = [
        #     f"""The text is about the spatial relationship between two objects in the image. Decide if the text is true or not.\n\nText{text}""",
        #     f"""Based on the picture, decide if the caption \"{text}\" is true.""",
        #     f"""Does the content of the image support the sentence below?\n\nSentence:{text}""",
        #     f"""Text: \"{text}\"\n\nDecide if the text is true or not based on the picture.""",
        #     f"""Is \"{text}\" true, by referring to the image?""",
        # ]
        raise NotImplementedError
    elif task == 'visual_attribute':
        instructs = [
            f"""Decide which option is the attribute of the object in the given region.\nRegion: {region_split_token.join(region)}{options_token} {split_token.join(options)}""",
            f"""Select the attribute of the object in {region_split_token.join(region)}{options_token} {split_token.join(options)}""",
            f"""Given object in {region_split_token.join(region)}, select its attribute.{options_token} {split_token.join(options)}""",
            f"""Given the region of the object, select its attribute from the options.\n\nRegion: {region_split_token.join(region)}{options_token} {split_token.join(options)}""",
            f"""Given the bounding box {region_split_token.join(region)} of the object, select its attribute.{options_token} {split_token.join(options)}""",
        ]    
    # image generation tasks
    elif task == 'infilling':
        instructs = [
            f"Fill in the missing part of the image.",
            f"Generate the missing part of the image.",
            f"Generate masked part of the image.",
            f"Generate the part of the image covered by the black square.",
            f"Generate the part of the image covered by black.",
        ]
    elif task == 'im_region_extraction':
        instructs = [
            f"Extract part of the image specified by the given region. Region: {region}.",
            f"Extract the part of image in {region}",
            f"Generate a copy of the image in the given region {region}.",
            f"Output a new image that is identical to the part of the given image specified by {region}",
            f"Generate a new image that is a precise replica of the area {region} in the given image.",
        ]
    elif task == 'im_descriptive_infilling':
        instructs = [
            f"Fill in the missing part of the image based on the description \"{text}\".",
            f"Generate the missing part of the image. The caption of the missing part is \"{text}\".",
            f"Using the caption \"{text}\" to generate the region of the image covered by black.",
            f"Based on the description \"{text}\", generate the masked part in the current image.",
            f"Generate the image that fills in the black square in the given image. The description of the black square is \"{text}\".",
        ]
    elif task == 'image_completion': # TODO: to be removed
        instructs = [
            f"Generate a new version of the original image, including a depiction of \"{text}\" in the previously missing area.",
            f"Use the description to generate a full copy of the original image with the black area filled in. You need to fill in the black area based on the description \"{text}\".",
            f"Generate a full version of the given image using the description to fill in the black area. Description: {text}.",
            f"Create a new image based on the original, with the missing area filled in by \"{text}\".",
            f"Generate a complete version of the image with the missing area, \"{text}\", filled in.",
        ]
    elif task == 'image_completion_w_region_caption':
        instructs = [
            f"Fill in the missing part of the image based on the description \"{text}\" and output the whole image.",
            f"Base on the caption \"{text}\", fill in the missing part of the image and generate the complete image.",
            f"Generate a full version of the given image using the caption to fill in the black area. Caption: {text}.",
            f"Create a new image based on the original, with the missing area filled in by \"{text}\".",
            f"Generate a complete version of the image with the missing area filled in. The caption of the missing area is \"{text}\"",
        ]
    elif task == 'image_completion_w_image_caption':
        instructs = [
            f"Complete the image based on the description \"{text}\".",
            f"Generate an image with description \"{text}\" by filling in the black area in the given image",
            f"Use the provided caption to produce a complete image by filling in the black area. Caption: \"{text}\"",
            f"Generate a new image that is the same as the given image with the missing area filled. Caption for the new image is \"{text}\".",
            f"Use the given caption to generate a new image based on the given image with the masked part filled in. Caption: \"{text}\".",
        ]
    # Visual question type matching
    
    elif task == 'VQA_activity_recognition':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task, you will answer a question about the activity of an object in the image. The question is "{question}"{options_token} {split_token.join(options)}""",
            f"""You are asked about the activity of animals or people in the image. Look at the image and answer "{question}" You should select your answer from the given options.{options_token} {split_token.join(options)}""",
            f"""Question: {question} Answer the question by first finding the object in the image and identify its activity. The answer is in the options.{options_token} {split_token.join(options)}""",
            f"""In this task, you will be asked about the activity of some object in the image. Select the best answer from options. Question: {question}{options_token} {split_token.join(options)}"""
        ]
    elif task == 'VQA_attribute':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task, you will be asked a question about the attribute of an object in the image. The question is "{question}"{options_token} {split_token.join(options)}""",
            f"""Answer the following question about the attribute of an object, "{question}" Select your answer from the given options.{options_token} {split_token.join(options)}""",
            f"""Question: {question}\n\nAnswer above question by first finding the object in the image and select its attribute from options.{options_token} {split_token.join(options)}""",
            f"""In this task, you will be asked about the attribute of some object. Select the best answer from given options. Question: {question}{options_token} {split_token.join(options)}"""
        ]
    elif task == 'VQA_color':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task, you are asked the color of some object in the image. Question: {question}{options_token} {split_token.join(options)}""",
            f"""Question: {question}\n\nAnswer the above question by first finding the object in the image and then select its color from options,{options_token} {split_token.join(options)}""",
            f"""Answer {question} based on the image. {options_token} {split_token.join(options)}""",
            f"""Answer the question: "{question}" based on the color of an object."""
        ]
    
    elif task == 'VQA_counting':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task, you are asked a question about the number of some objects in the image. The question is: {question}{options_token} {split_token.join(options)}""",
            f"""The question is: {question} Select your answer from options.{options_token} {split_token.join(options)}""",
            f"""Question: {question}\n\nPlease answer the question by counting the object mentioned in the question.{options_token} {split_token.join(options)}""",
            f"""This task tests your ability to count number of objects. Here is the question "{question}". Select the correct answer from options.{options_token} {split_token.join(options)}"""
        ]
    
    elif task == 'VQA_object_presence':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""This task asks you to identify if an object appears in the image. {question}{options_token} {split_token.join(options)}""",
            f"""In this task, you are required to answer a question about the appearance of an object.{question}{options_token} {split_token.join(options)}""",
            f"""{question} Decide if the object mentioned in previous question appears in the image.{options_token} {split_token.join(options)}""",
            f"""Question: {question} look at the image and answer the question.{options_token} {split_token.join(options)}""",
        ]
        
    elif task == 'VQA_object_recognition':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task you are asked a question about the type of an object in the image. {question}{options_token} {split_token.join(options)}""",
            f"""In this task, you will answer a question about the subclass of an object in the image. {question}{options_token} {split_token.join(options)}""",
            f"""In this task, you will be presented with an image. Your task is to answer a question about the type of object. Question: {question}{options_token} {split_token.join(options)}
            """,
            f"""Please answer a question regarding the type of an object in the image. Question: {question}{options_token} {split_token.join(options)}"""
            
        ]
        
    elif task == 'VQA_positional_reasoning':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task, you need to analyze the position of objects in an image and answer the following question. {question}{options_token} {split_token.join(options)}""",
            f"""This task requires an understanding of object location within the presented image. Please select the correct answer to the provided question. {question}{options_token} {split_token.join(options)}""",
            f"""In this task, the goal is to understand the location of objects within the presented image and provide a answer to the question provided. {question}{options_token} {split_token.join(options)}""",
            f"""Question: {question}{options_token}\n\n Please answer the question by reasoning about the positions of objects and select an answer from options. {split_token.join(options)}."""
        ]
        
    elif task == 'VQA_scene_recognition':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task, you need to pay attention to the scene in the image and answer the following question.\n {question}{options_token} {split_token.join(options)}""",
            f"""Question: {question}{options_token}. \n Please answer the question by analyzing the scene in the provided image. Here are some possible answers. {options_token} {split_token.join(options)}""",
            f"""Look at the environment in the image and answering the question accordingly.\n {question}{options_token} {split_token.join(options)}""",
            f"""Given a picture of certain environment, answer the following question by select an answer from the options. \n {question}{options_token} {split_token.join(options)}"""
        ]
        
    elif task == 'VQA_sentiment_understanding':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""This task requires an understanding of the feeling conveyed in the image. Please select the correct answer to the provided question. {question}{options_token} {split_token.join(options)}""",
            f"""Question: {question}{options_token} {split_token.join(options)}.\n Please answer the question by interpreting the sentiment in the image.""",
            f"""Please analyze the sentiment depicted in the image and answer the question.\n {question}{options_token} {split_token.join(options)}""",
            f"""In this task, you will be asked a question regarding the emotion conveyed in the image. The question is {question}{options_token} {split_token.join(options)}"""
        ]
        
    elif task == 'VQA_sport_recognition':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task, you need to pay attention to the sports depicted in the image and answer the following question. \n {question}{options_token} {split_token.join(options)}""",
            f"""Given a picture about sports, answer the following question by select an answer from the options. \n {question}{options_token} {split_token.join(options)}""",
            f"""There are some sports taking place in the image. {question}{options_token} {split_token.join(options)}""",
            f"""Please answer the following question by analyzing the sport in the given image.\n {question}{options_token} {split_token.join(options)}"""
        ]
        
    elif task == 'VQA_utility_affordance':
        instructs = [
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""In this task, you need to pay attention to the possible actions can be taken to the objects in the image and answer the following question. {question}{options_token} {split_token.join(options)}""",
            f"""Please take a look at the picture and answer the following question by thinking about what each object in the picture can be used for. {question}{options_token} {split_token.join(options)}""",
            f"""Question: {question}{options_token}\n Please select a correct answer for the question by analyzing the affordance of the objects in the image. {split_token.join(options)}""",
            f"""This task tests your ability to understand the potential actions that you can take on the objects or the usage of the objects in the image. Here is the question "{question}". Select the correct answer from options.{options_token} {split_token.join(options)}"""
        ]
        
    elif task == 'select_overlap_most_region':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, decide which region in the options overlaps most with given region.{options_token} {split_token.join(options)}""",
            f"""Select the region that shares the most common area with {given_region}.{options_token} {split_token.join(options)}""",
            f"""Which option overlaps most with {given_region}?{options_token} {split_token.join(options)}""",
            f"""Decide the region that has the most common area with the given region. Region: {given_region}{options_token} {split_token.join(options)}""",
            f"""Region: {given_region}\n\nIdentify the region overlaps most with the above given region from options.{options_token} {split_token.join(options)}"""
        ]
    elif task == 'select_overlap_least_region':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, decide which region in the options shares the least common area with given region.{options_token} {split_token.join(options)}""",
            f"""In this task, you are given a region: {given_region}, you need to select a region from the options that has the least overlap with the given region.{options_token} {split_token.join(options)}""",
            f"""Which option has the least shared area with {given_region}?{options_token} {split_token.join(options)}""",
            f"""Select the region that has the least overlap with {given_region}.{options_token} {split_token.join(options)}""",
            f"""Given region: {given_region}, decide which option has the least common area with it.{options_token} {split_token.join(options)}""",
        ]
    elif task == 'region_area':
        instructs = [
            f"""Compute the area of {region[0]}""",
            f"""What is the area of {region[0]}?""",
            f"""Calculate the area of given region. Region: {region[0]}""",
            f"""Region: {region[0]}\n\nwhat is the area of the above region?""",
            f"""In this task, you are given a region and you need to compute the area of the region. The region is {region[0]}""",
        ]
    elif task == 'select_overlaped_region':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, select an overlapping region from options.{options_token} {split_token.join(options)}""",
            f"""Select a region from options that overlaps with {given_region}{options_token} {split_token.join(options)}""",
            f"""Which region from options that shares common area with {given_region}?{options_token} {split_token.join(options)}""",
            f"""Region: {given_region}\n\nSelect a region that has overlap with the given region.{options_token} {split_token.join(options)}""",
            f"""Which region from options that has common area with {given_region}?{options_token} {split_token.join(options)}""",
            
        ]
    elif task == 'select_nonoverlaped_region':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, select an non-overlapping region from options.{options_token} {split_token.join(options)}""",
            f"""Region: {given_region}, select an non-overlapping region with the given region from options.{options_token} {split_token.join(options)}""",
            f"""Select an option that does not overlap with {given_region}{options_token} {split_token.join(options)}""",
            f"""Which option does not share common area with {given_region}?{options_token} {split_token.join(options)}""",
            f"""Tell me which option does not have shared area with {given_region}?{options_token} {split_token.join(options)}"""
        ]
    elif task == 'if_region_overlap':
        given_region = region_split_token.join(meta_data['object_regions']['given_region'])
        instructs = [
            f"""Given the region {given_region}, decide if {region[0]} overlaps with it.{options_token} {split_token.join(options)}""",
            f"""Do the following two regions overlap? Region 1: {region[0]} and Region 2: {given_region}{options_token} {split_token.join(options)}""",
            f"""Does {given_region} share common area with {region[0]}?{options_token} {split_token.join(options)}""",
            f"""Tell me if {region[0]} and {given_region} have common area.{options_token} {split_token.join(options)}""",
            f"""Do {region[0]} and {given_region} overlap?{options_token} {split_token.join(options)}"""
        ]
    elif task == 'wikihow_prev_step': # add image only version
        raise NotImplementedError
    elif task == 'detection':
        raise NotImplementedError
    elif task == 'visual_question_generation': # remove
        raise NotImplementedError
    elif task == 'VQA_differences': # remove
        raise NotImplementedError
    elif task == 'visual_skills':
        raise NotImplementedError
    elif task == 'multimodal_argument_extraction':
        raise NotImplementedError
    elif task == 'multimodal_argument_extraction_txt_or_img':
        raise NotImplementedError
    elif task == 'multimodal_argument_extraction_txt_only':
        raise NotImplementedError
    elif task == 'visual_srl':
        raise NotImplementedError
    elif task == 'visual_argument_extraction':
        raise NotImplementedError
    elif task == 'generation':
        # instructs = [
        #     f"Generate the image based on the given text.\n\n{text}",
        #     f"Generate the image described by \"{text}\"",
        #     f"Based on the caption: \"{text}\", generate the image.",
        #     f"Based on the text \"{text}\", generate the picture.",
        #     f"Generate the picture based on the text.\n\nText: \"{text}\"",
        # ]
        raise NotImplementedError
    
    # -------------------------------- NLP tasks ------------------------------------------- #
    elif task == 'nlp_only':
        instructs = [
            f"""{text}""",
            f"""{text}""",
            f"""{text}""",
            f"""{text}""",
            f"""{text}"""
        ]
    elif task == 'paragraph_nli':
        instructs = [
        f"{context}\n\nBased on the paragraph above can we conclude that \"{hypothesis}\"?\n{options_token} {split_token.join(options)}",
        f"{context}\n\nDoes the paragraph support this sentence?\n{hypothesis}\n{options_token} {split_token.join(options)}",
        f"{context}\n\nCan we draw the following conclusion?\n{hypothesis}\n{options_token} {split_token.join(options)}",
        f"{context}\nCan we infer the following?\n{hypothesis}\n{options_token} {split_token.join(options)}",
        f"Read the following paragraph and determine if the hypothesis is true:\n\n{context}\n\nHypothesis: {hypothesis} {options_token} {split_token.join(options)}",
        f"Read the text and determine if the sentence is true:\n\n{context}\n\nSentence: {hypothesis}\n{options_token} {split_token.join(options)}",
        f"Can we draw the following hypothesis from the context? \n\nContext:\n\n{context}\n\nHypothesis: {hypothesis} {options_token} {split_token.join(options)}",
        f"Determine if the sentence: \"{hypothesis}\" is supported text below:\n\n{context}{options_token} {split_token.join(options)}",
        
        f"{context}\n\nBased on the paragraph above can we conclude that \"{hypothesis}\"?\n{options_token} {split_token.join(options)}",
        f"{context}\n\nBased on the paragraph can we conclude that this sentence is true?\n{hypothesis}\n{options_token} {split_token.join(options)}",
        f"{context}\n\nCan we draw the following conclusion?\n{hypothesis}\n{options_token} {split_token.join(options)}",
        f"{context}\nDoes this next sentence follow, given the preceding text?\n{hypothesis}\n{options_token} {split_token.join(options)}",
        f"{context}\nCan we infer the following?\n{hypothesis}\n{options_token} {split_token.join(options)}",
        f"Read the following paragraph and determine if the hypothesis is true:\n\n{context}\n\nHypothesis: {hypothesis}n{options_token} {split_token.join(options)}",
        f"Read the text and determine if the sentence is true:\n\n{context}\n\nSentence: {hypothesis}n{options_token} {split_token.join(options)}",
        f"Can we draw the following hypothesis from the context? \n\nContext:\n\n{context}\n\nHypothesis: {hypothesis}n{options_token} {split_token.join(options)}",
        f"Determine if the sentence is true based on the context below.\nSentence: {hypothesis}\n\nContext: {context}{options_token} {split_token.join(options)}",
        ]
    elif task == 'nli_explanation':
        if answer == 'yes':
            instructs = [
            f"Explain why you can draw the conclusion that \"{hypothesis}\" based on the context.\n\nContext:{context}",
            f"{context}\n\nExplain how do you infer \"{hypothesis}\" from the previous text.",
            f"Context: {context}\n\nExplain why \"{hypothesis}\" can be infered from above text."
            ]
        else:
            instructs = [
            f"Why you can't draw the conclude \"{hypothesis}\" from the context.\n\nContext: {context}",
            f"{context}\n\nExplain why you can not infer \"{hypothesis}\" from the previous text.",
            f"Explain why you can not infer \"{hypothesis}\" from the following text: {context}"
            ]
    elif task == 'nli':
        instructs = [
            f"Premise: {premise}\n\nHypothesis: {hypothesis}\n\nDoes the premise entail the hypothesis?{options_token} {split_token.join(options)}",
            f"Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis entailed by the premise? {options_token} {split_token.join(options)}",
            f"Here is a premise:\n{premise}\n\nHere is a hypothesis:\n{hypothesis}\n\nIs it possible to conclude that if the premise is true, then so is the hypothesis? {options_token} {split_token.join(options)}",
            f"Sentence 1: {premise}\n\nSentence 2: {hypothesis}\nIs this second sentence entailed by the first sentence?{options_token} {split_token.join(options)}",
            f"Sentence 1: {premise}\n\nSentence 2: {hypothesis}\n\nIf the first sentence is true, then is the second sentence true? {options_token} {split_token.join(options)}",
            f"Based on the premise \"{premise}\", can we conclude the hypothesis \"{hypothesis}\" is true?{options_token} {split_token.join(options)}",
            f"Premise: \"{premise}\" If this premise is true, what does that tell us about whether it entails the hypothesis \"{hypothesis}\"?{options_token} {split_token.join(options)}",
            f"Premise:\n\"{premise}\" Based on this premise, is the hypothesis \"{hypothesis}\" true? {options_token} {split_token.join(options)}",
            f"If {premise}, can we conclude that \"{hypothesis}\"? {options_token} {split_token.join(options)}",
            f"{premise} Does it follow that \"{hypothesis}\"? {options_token} {split_token.join(options)}",
            f"If \"{premise}\", can we conclude that \"{hypothesis}\"{options_token} {split_token.join(options)}",
            f"If \"{premise}\", does it follow that \"{hypothesis}\"{options_token} {split_token.join(options)}",
            f"If \"{premise}\", is \"{hypothesis}\" correct?\n{options_token} {split_token.join(options)}",
            f"Let's say that \"{premise}\"\n\nCan we now say that \"{hypothesis}\"?\n{options_token} {split_token.join(options)}",
            f"\"{premise}\" is a true sentence.\n\nDoes this mean that \"{hypothesis}\"?\n{options_token} {split_token.join(options)}",
            f"Does \"{hypothesis}\" appear to be an accurate statement based on \"{premise}\"?\n{options_token} {split_token.join(options)}",
            f"Can we conclude that \"{hypothesis}\" if the statement \"{premise}\" is true?\n{options_token} {split_token.join(options)}",
            f"Is it possible to draw the conclusion that \"{hypothesis}\" if \"{premise}\"?\n{options_token} {split_token.join(options)}",
            f"Is \"{hypothesis}\" true if \"{premise}\"?\n{options_token} {split_token.join(options)}",
            f"Sentence 1: \"{premise}\"\n\n Sentence 2: \"{hypothesis}\"\n\nIs sentence 2 true, based on sentence 1?\n{options_token} {split_token.join(options)}",
            f"{premise}\n\nBased on the paragraph above can we conclude that \"{hypothesis}\"?\n{options_token} {split_token.join(options)}",
            f"{premise}\n\nBased on that paragraph can we conclude that this sentence is true?\n{hypothesis}\n{options_token} {split_token.join(options)}",
            f"{premise}\n\nCan we draw the following conclusion?\n{hypothesis}\n{options_token} {split_token.join(options)}",
            f"{premise} Does this next sentence follow, given the preceding text?\n{hypothesis}\n{options_token} {split_token.join(options)}",
            f"{premise} Can we infer the following?\n{hypothesis}\n{options_token} {split_token.join(options)}",
            f"Read the following paragraph and determine if the hypothesis is true:\n\n{premise}\n\nHypothesis: {hypothesis}n{options_token} {split_token.join(options)}",
            f"Read the text and determine if the sentence is true:\n\n{premise}\n\nSentence: {hypothesis}n{options_token} {split_token.join(options)}",
            f"Can we draw the following hypothesis from the context? \n\nContext:\n\n{premise}\n\nHypothesis: {hypothesis}n{options_token} {split_token.join(options)}",
            f"Determine if the sentence is true based on the text below:\n{hypothesis}\n\n{premise}{options_token} {split_token.join(options)}"
        ]
    elif task == 'question_nli':
        instructs = [
            f"Does the sentence \"{hypothesis}\" answer the question \"{question}\"\n{options_token} {split_token.join(options)}",
            f"Does the sentence \"{hypothesis}\" provide a valid answer to the question \"{question}\"{options_token} {split_token.join(options)}",
            f"Is \"{hypothesis}\" a good answer to the question \"{question}\"{options_token} {split_token.join(options)}",
            f"Does \"{hypothesis}\" correctly answer the question of {question}{options_token} {split_token.join(options)}",
            f"Does \"{hypothesis}\" contain the correct answer to \"{question}\"{options_token} {split_token.join(options)}",
            f"Q: {question}\n A: {hypothesis}\n Does the answer correctly answer the question\n{options_token} {split_token.join(options)}",
            f"Question: {question}\nAnswer: {hypothesis}\n Is the question answered in a satisfactory fashion?\n{options_token} {split_token.join(options)}",
            f"Question: {question}\n\nIs {hypothesis} a good answer to this question?\n{options_token} {split_token.join(options)}",
            f"Question: {question}\n\nIs \"{hypothesis}\" the correct answer?\n{options_token} {split_token.join(options)}"
        ]
    elif task == 'context_qa':
        instructs = [
            f"{context}\n\n{question}?\n{options_token} {split_token.join(options)}",
            f"Context: {context}\n\n{question}?\n{options_token} {split_token.join(options)}",
            f"Text: {context}\n\nQuestion: {question}?\n{options_token} {split_token.join(options)}",
            f"{context}\nBased on the above text, {question}?\n{options_token} {split_token.join(options)}",
            f"{context}\nAnswer this question, making sure that the answer is supposed by the text: {question}?\n{options_token} {split_token.join(options)}",
            f"{question} based on the following text?\n\n{context}\n{options_token} {split_token.join(options)}"
        ]
    elif task == 'record': # modify
        instructs = [
            f"Based on the passage, complete the sentence.\n\nPassage: {context}\n\nSentence: {question}\n{options_token} {split_token.join(options)}",
            f"""{context}\n\nBased on above passage, find the right ending to this sentence \"{question}\" {options_token} {split_token.join(options)}""",
            f"{context} What's the most logical way to complete this sentence? {question}\n\n{options_token} {split_token.join(options)}"
        ]
    elif task == 'PIQA':
        instructs = [
            f"Here is a goal: {text}\n\nHow would you accomplish this goal?{options_token} {split_token.join(options)}",
            f"Here is a goal: {text}\n\nWhich way makes more sense to accomplish this goal?{options_token} {split_token.join(options)}",
            f"Goal: {text}\n\nWhich of the following methods is more reasonable for accomplishing this goal?{options_token} {split_token.join(options)}",
            f"Objective: {text}\n\nWhich of the following solutions is more sound in terms of naive physics reasoning?{options_token} {split_token.join(options)}",
            f"How do you do this: {text}{options_token} {split_token.join(options)}",
            f"What is the best way to: {text}{options_token} {split_token.join(options)}",
            f"Which of the following solutions is better for the following goal:\n{text}{options_token} {split_token.join(options)}",
            f"How would someone go about accomplishing this goal?\n{text}{options_token} {split_token.join(options)}"
        ]
    elif task == 'squad_v2':
        instructs = [
            f"{context}\n\nPlease answer a question about this article. If the question is unanswerable, say \"not sure\". {question}",
            f"Read the context and answer the question. If the question is unanswerable, say \"not sure\". Context: {context}\n\nQuestion: {question}",
            f"{context}\n{question} (If the question is unanswerable, say \"not sure\")",
            f"{context}\nAnswer this question if possible (otherwise reply \"not sure\"): {question}",
            f"{context}\nIf it is possible to answer this question, answer it (else, reply \"not sure\"): {question}",
            f"{context}\n\nAnswer this question, if possible (if impossible, reply \"not sure\"): {question}",
            f"Read this: {context}\n\n{question} (If it cannot be answered, return \"not sure\")"
        ]
    elif task == 'openbook_qa':
        instructs = [
            f"Fact:{context}\n\nBased on the fact, select the best option to finish the sentence. \n\n Sentence:{text}\n{options_token} {split_token.join(options)}",
            f"Fact:{context}\n\nBased on the fact, select the best option to finish the sentence. \n\n Sentence:{text}{options_token} {split_token.join(options)}",
        ]

    # -------------------------------- compose tasks --------------------------------------- #
    
        
    # elif task == 'object_region_match_generate':
    #     instructs = [
    #         f"""Is \"{text}\" in {region_split_token.join(region)}? If not, tell me the regions that contain this object.""",
    #         f"""In this task, you need to decide if the object is in the given region. If not, generate all the regions that contain this object in image. Object: \"{text}\"; Region: {region_split_token.join(region)}""",
    #         f"""Does {region_split_token.join(region)} contain \"{text}\"? If not, tell me where is the object in image""",
    #     ]
    # elif task == 'object_region_match':
    #     instructs = [
    #         f"""Is \"{text}\" in {region_split_token.join(region)}?""",
    #         f"""Does {region_split_token.join(region)} contain \"{text}\"?""",
    #         f"""Answer if {region_split_token.join(region)} contains \"{text}\"?""",
    #         f"""In this task, you will need to decide if the object in {region_split_token.join(region)} is \"{text}\" """,
    #         f"""Object: \"{text}\"; Region: {region_split_token.join(region)}. Do they match?""",
    #         f"""Do object: \"{text}\" and {region_split_token.join(region)} match?""",
    #     ]
    
    # --------------------------- unseen tasks ---------------------------------- #
    elif task == 'commonsense_VQA':
        k_in_q = {}
        for k, v in meta_data['object_regions'].items():
            if k in question:
                question = question.replace(k, f"the {k[1:-1]} in {' '.join(v)}")
                k_in_q[k] = True
        for k, v in meta_data['object_regions'].items():
            if k in question:
                question = question.replace(k, f"the {k[1:-1]} in {' '.join(v)}")
            for i in range(len(options)):
                if k in options[i]:
                    if not k in k_in_q:
                        if target == options[i]:
                            target = options[i].replace(k, f"the {k[1:-1]} in {' '.join(v)}")
                        options[i] = options[i].replace(k, f"the {k[1:-1]} in {' '.join(v)}")
                    else:
                        if target == options[i]:
                            target = options[i].replace(k, f"the {k[1:-1]}")
                        options[i] = options[i].replace(k, f"the {k[1:-1]}")
        
        instructs = [
            # f"""Based on the image, \"{question}\" {region_info}{options_token} {split_token.join(options)}""",
            # f"""\"{question}\"\n{region_info}{options_token} {split_token.join(options)}""",
            # f"""The region information: {region_info}\nBased on the region information and the image, \"{question}\"{options_token} {split_token.join(options)}""",
            # f"""\"{question}\" {region_info} {options_token} {split_token.join(options)}"""
            # f"""Given {region_info}, answer \"{question}\"{options_token} {split_token.join(options)}"""
            f"""{question}{options_token} {split_token.join(options)}""",
            f"""Look at the image and {question}{options_token} {split_token.join(options)}""",
            f"""Look at the image and the regions in the question, {question}{options_token} {split_token.join(options)}""",
            f"""Based on the image, answer {question}{options_token} {split_token.join(options)}""",
            f"""Based on the image and the regions in the question, answer: {question}{options_token} {split_token.join(options)}""",
        ]
        # pdb.set_trace()
    else:
        raise NotImplementedError(f'Task {task} is not implemented')
    
    # add sciQA
    # Multimodal Open-Domain Dialogue
    # Conversation Style Classification
    
    assert len(instructs) >= 5, f"task {task} has only {len(instructs)} instructions"
    
    # NLP tasks
    if instruction_id != -1:
        return instructs[instruction_id], target
    else:
        return random.choice(instructs), target
