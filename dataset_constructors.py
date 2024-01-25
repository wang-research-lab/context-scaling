import string
import random
from datasets import load_dataset
from utils import Instance

def gsm_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."
    
    dataset = load_dataset("openai/gsm8k", name="main", split=split)
    
    return [
        Instance(
            instance_id=i,
            input_text=data['question'],
            reference_output=data['answer'],
        ) for i, data in enumerate(dataset)
    ] 

def openbookqa_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."
    
    dataset = load_dataset("allenai/openbookqa", name="main", split=split)

    ret = []
    for i,data in enumerate(dataset):
        question = data["question_stem"]

        for label, answer in zip(list(string.ascii_uppercase), data["choices"]["text"]):
            question += f"\n{label}. {answer}"

        answer = data["answerKey"] if data["answerKey"] in list(string.ascii_uppercase) else list(string.ascii_uppercase)[int(data["answerKey"])-1]

        ret.append(Instance(
            instance_id=i,
            input_text=question,
            reference_output=answer,
        ))

    return ret

def arc_challenge_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either test or train"

    dataset = load_dataset("allenai/ai2_arc", name="ARC-Challenge", split=split)

    ret = []
    for i,data in enumerate(dataset):
        question = data["question"]

        for label, answer in zip(list(string.ascii_uppercase), data["choices"]["text"]):
            question += f"\n{label}. {answer}"

        answer = data["answerKey"] if data["answerKey"] in list(string.ascii_uppercase) else list(string.ascii_uppercase)[int(data["answerKey"])-1]
        
        ret.append(Instance(
            instance_id=i,
            input_text=question,
            reference_output=answer,
        ))

    return ret

def arc_easy_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either test or train"

    dataset = load_dataset("allenai/ai2_arc", name="ARC-Easy", split=split)

    ret = []
    for i,data in enumerate(dataset):
        question = data["question"]

        for label, answer in zip(list(string.ascii_uppercase), data["choices"]["text"]):
            question += f"\n{label}. {answer}"

        answer = data["answerKey"] if data["answerKey"] in list(string.ascii_uppercase) else list(string.ascii_uppercase)[int(data["answerKey"])-1]
        
        ret.append(Instance(
            instance_id=i,
            input_text=question,
            reference_output=answer,
        ))

    return ret

def wmt14_cs_en_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    dataset = load_dataset("wmt/wmt14", name="cs-en", split=split)
    if split == 'train' and dataset.num_rows > 300000:
        dataset = dataset.select(random.sample(range(dataset.num_rows), 300000))

    return [
        Instance(
            instance_id=i,
            input_text=f"Czech: {data['translation']['cs']}\nEnglish:",
            reference_output=data["translation"]["en"],
        ) for i, data in enumerate(dataset)
    ]

def wmt14_de_en_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    dataset = load_dataset("wmt/wmt14", name="de-en", split=split)
    if split == 'train' and dataset.num_rows > 300000:
        dataset = dataset.select(random.sample(range(dataset.num_rows), 300000))

    return [
        Instance(
            instance_id=i,
            input_text=f"German: {data['translation']['de']}\nEnglish:",
            reference_output=data["translation"]["en"],
        ) for i, data in enumerate(dataset)
    ]

def wmt14_fr_en_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    dataset = load_dataset("wmt/wmt14", name="fr-en", split=split)
    if split == 'train' and dataset.num_rows > 300000:
        dataset = dataset.select(random.sample(range(dataset.num_rows), 300000))

    return [
        Instance(
            instance_id=i,
            input_text=f"French: {data['translation']['fr']}\nEnglish:",
            reference_output=data["translation"]["en"],
        ) for i, data in enumerate(dataset)
    ]

def wmt14_hi_en_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    dataset = load_dataset("wmt/wmt14", name="hi-en", split=split)
    if split == 'train' and dataset.num_rows > 300000:
        dataset = dataset.select(random.sample(range(dataset.num_rows), 300000))

    return [
        Instance(
            instance_id=i,
            input_text=f"Hindi: {data['translation']['hi']}\nEnglish:",
            reference_output=data["translation"]["en"],
        ) for i, data in enumerate(dataset)
    ]

def wmt14_ru_en_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    dataset = load_dataset("wmt/wmt14", name="ru-en", split=split)
    if split == 'train' and dataset.num_rows > 300000:
        dataset = dataset.select(random.sample(range(dataset.num_rows), 300000))

    return [
        Instance(
            instance_id=i,
            input_text=f"Russian: {data['translation']['ru']}\nEnglish:",
            reference_output=data["translation"]["en"],
        ) for i, data in enumerate(dataset)
    ]

def mbpp_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    dataset = load_dataset("google-research-datasets/mbpp", "full", split=split)

    return [
        Instance(
            instance_id = i,
            input_text = f"Please reply with a Python 3 solution to the below problem. Make sure to wrap your code in '```python' and '```' Markdown delimiters.\n{data['text']}",
            reference_output=data['code'],
        ) for i, data in enumerate(dataset)
    ]

def math_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."
    
    dataset = load_dataset("lighteval/MATH", "all", split=split)

    return [
        Instance(
            instance_id= i,
            input_text=data['problem'],
            reference_output=data['solution']
        ) for i, data in enumerate(dataset)
    ]

def commonsenseqa_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    if split == 'test':
        split = 'validation'

    dataset = load_dataset("tau/commonsense_qa", split=split)

    ret = []
    for i,data in enumerate(dataset):
        question = data["question"]

        for i in range(len(data["choices"]["label"])):
            question += f"\n{data['choices']['label'][i]}. {data['choices']['text'][i]}"

        ret.append(Instance(
            instance_id=i,
            input_text=question,
            reference_output=data["answerKey"]
        ))

    return ret

def deepmind_math_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    ret = []

    random.seed(kwargs['seed'])
    dataset = load_dataset("deepmind/math_dataset", kwargs['config'], split=split)
    if dataset.num_rows > 1000000:
        dataset = dataset.select(random.sample(range(dataset.num_rows), 300000))

    for i,data in enumerate(dataset):
        ret.append(Instance(
            instance_id=i,
            input_text=data["question"],
            reference_output=data["answer"]
        ))

    return ret

def aqua_rat_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    dataset = load_dataset("deepmind/aqua_rat", "raw", split=split)

    ret = []
    for i,data in enumerate(dataset):
        question = data["question"]

        for i in range(len(data["options"])):
            question += f"\n{data['options'][i]}"
        
        ret.append(Instance(
            instance_id=i,
            input_text=question,
            reference_output=data['correct'],
        ))

    return ret

def piqa_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    if split == 'test':
        split = 'validation'

    dataset = load_dataset("ybisk/piqa", split=split, trust_remote_code=True)

    return [
        Instance(
            instance_id=i,
            input_text=
                f"{data['goal']}\nPick one of the two options:\nA. {data['sol1']}\nB. {data['sol2']}",
            reference_output='A' if data['label'] == 0 else 'B'
        ) for i,data in enumerate(dataset)
    ]

def siqa_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    if split == "test":
        split = "validation"

    dataset = load_dataset("lighteval/siqa", split=split)

    return [
        Instance(
            instance_id=i,
            input_text=f"{data['context']} {data['question']}\nA. {data['answerA']}\nB. {data['answerB']}\nC. {data['answerC']}",
            reference_output='A' if data['label']=='1' else 'B' if data['label']=='2' else 'C'
        ) for i,data in enumerate(dataset)
    ]

def hellaswag_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    if split == 'test':
        split = 'validation'

    dataset = load_dataset("Rowan/hellaswag", split=split)

    ret = []
    for i,data in enumerate(dataset):
        question = f"Read the sentence and select the best ending:\n{data['ctx']}"

        for label, answer in zip(list(string.ascii_uppercase), data["endings"]):
            question += f"\n{label}. {answer}"

        answer = chr(ord('@')+(int(data["label"])+1))

        ret.append(Instance(
            instance_id=i,
            input_text=question,
            reference_output=answer,
        ))

    return ret

def winogrande_constructor(split, **kwargs):
    assert split in ['train', 'test'], "Split should be either train or test."

    if split == 'test':
        split = 'validation'

    dataset = load_dataset("allenai/winogrande", "winogrande_xl", split=split, trust_remote_code=True)

    return[
        Instance(
            instance_id=i,
            input_text=f"{data['sentence']}\nPick A or B to fill in the blank:\nA. {data['option1']}\nB. {data['option2']}",
            reference_output='A' if int(data['answer']) == 1 else 'B'
        ) for i,data in enumerate(dataset)
    ]

dataset_name_to_constructor = {
    "gsm": gsm_constructor,
    "openbookqa": openbookqa_constructor,
    "arc-challenge": arc_challenge_constructor,
    "arc-easy": arc_easy_constructor,
    "wmt14-cs-en": wmt14_cs_en_constructor,
    "wmt14-de-en": wmt14_de_en_constructor,
    "wmt14-fr-en": wmt14_fr_en_constructor,
    "wmt14-hi-en": wmt14_hi_en_constructor,
    "wmt14-ru-en": wmt14_ru_en_constructor,
    "mbpp": mbpp_constructor,
    "math": math_constructor,
    "commonsenseqa": commonsenseqa_constructor,
    "deepmind-math": deepmind_math_constructor,
    "aqua-rat": aqua_rat_constructor,
    "piqa": piqa_constructor,
    "siqa": siqa_constructor,
    "hellaswag": hellaswag_constructor,
    "winogrande": winogrande_constructor,
}