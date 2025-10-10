import json
import pandas as pd
import numpy as np


def collapse_into_visits(utterances):
    last_speaker = None
    visits = []
    visit = None
    for utterance in utterances:
        speaker, content = utterance 
        if speaker == 'action':
            continue
        elif speaker == last_speaker:
            visit += f" {content}"
        elif speaker != last_speaker:
            if visit:
                visits.append(visit)
            visit = f"Speaker: {speaker}, Content: {content}"
            last_speaker = speaker
    visits.append(visit)
    return visits

def collapse_into_roles(utterances):
    roles = ['Customer: ', 'Agent: ']
    for utterance in utterances:
        speaker, content = utterance 
        if speaker == 'action':
            continue
        elif speaker == 'customer':
            roles[0] += f" {content}"
        elif speaker == 'agent':
            roles[1] += f" {content}"
    return roles

def json2df(json_item):
    df = pd.DataFrame(columns=['visits', 'label'])
    df['visits'] = [collapse_into_visits(json_item['original'])] # collapse dataset into separate consecutive utterances or by roles
    df['label'] = [json_item['scenario']['flow']]
    return df

def load_data(path, label_names): # currently filters storewide_query and purchase_dispute
    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(columns=['visits', 'split', 'label'])

    for i, split in enumerate(['train', 'test', 'dev']):
        for json_item in data[split]:
            temp_df = json2df(json_item)
            temp_df['split'] = split
            df = pd.concat([df, temp_df], ignore_index=True)
        print(f"finished split {split}")


    df = df[df['label'].isin(label_names)]

    card_map = {label: int(label==label_names[0]) for label in label_names}
    dig_map = {label: int(label==label_names[1]) for label in label_names}
    df['card'] = df['label'].map(card_map)
    df['dig'] = df['label'].map(dig_map)
    
    return df[df['split']=='train'], df[df['split']=='test'], df[df['split']=='dev']