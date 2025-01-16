import argparse
import json
from copy import deepcopy

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def func(parsed_name):
    with open("../data/mwz2_1/ontology.json", encoding="utf-8") as f:
        ontology = json.load(f)
    slots = get_slot_information(ontology)
    cnt_turns = 0
    cnt_acc_m21 = 0
    cnt_acc_m24 = 0
    cnt_acc_gpt = 0
    coherence_m21 = 0
    coherence_m24 = 0
    with open(f'{parsed_name}.json', encoding='utf-8') as f:
        data = json.load(f)
    lst_incoherence_m21 = []
    lst_incoherence_m24 = []
    for x in data:
        last_ds_m21 = {}
        last_ds_m24 = {}
        for sample in x['turn_details']:
            cnt_turns += 1
            turn_label_m21 = get_turn_label(last_ds_m21, sample['gt_m21'])
            turn_label_m24 = get_turn_label(last_ds_m24, sample['gt_m24'])
            # m21 = set(k + '==' + v for k, v in turn_label_m21.items())
            # m24 = set(k + '==' + v for k, v in turn_label_m24.items())
            # predict = set(k + '==' + v for k, v in sample['predict_turn_label'].items())
            m21 = set(k + '==' + v.replace(' ', '') for k, v in turn_label_m21.items())
            m24 = set(k + '==' + v.replace(' ', '') for k, v in turn_label_m24.items())
            predict = set()
            for k, v in sample['predict_turn_label'].items():
                if v == 'none' and k not in sample['predict']:
                    continue
                predict.add(k + '==' + v.replace(' ', ''))
            flag_m21 = 0
            if predict == m21:
                flag_m21 = 1
                cnt_acc_m21 += 1
            flag_m24 = 0
            if predict == m24:
                flag_m24 = 1
                cnt_acc_m24 += 1
            flag_gpt = sample['gpt_turn_judge']
            if flag_gpt == 1:
                cnt_acc_gpt += 1
            if flag_m21 == flag_gpt:
                coherence_m21 += 1
            else:
                lst_incoherence_m21.append({
                    'flag': x['dialogue_idx'] + '-' + str(sample['turn_idx']),
                    'system': sample['system'],
                    'user': sample['user'],
                    'gpt_out': sample['gpt_out'],
                    "predicted": sample['predict'],
                    'turn_label_m21': turn_label_m21,
                    'turn_label_m24': turn_label_m24,
                    'predict_turn_label': sample['predict_turn_label'],
                    'judgement_m21': flag_m21,
                    'judgement_m24': flag_m24,
                    'judgement_gpt': flag_gpt,
                })
            if flag_m24 == flag_gpt:
                coherence_m24 += 1
            else:
                lst_incoherence_m24.append({
                    'flag': x['dialogue_idx'] + '-' + str(sample['turn_idx']),
                    'system': sample['system'],
                    'user': sample['user'],
                    'gpt_out': sample['gpt_out'],
                    "predicted": sample['predict'],
                    'turn_label_m21': turn_label_m21,
                    'turn_label_m24': turn_label_m24,
                    'predict_turn_label': sample['predict_turn_label'],
                    'judgement_m21': flag_m21,
                    'judgement_m24': flag_m24,
                    'judgement_gpt': flag_gpt,
                })
            last_ds_m21 = sample['gt_m21']
            last_ds_m24 = sample['gt_m24']
    print('Number of turns: {}'.format(cnt_turns))
    print('Turn Level Acc Based on MultiWOZ 2.1: {:.2f} %'.format(cnt_acc_m21 / cnt_turns * 100))
    print('Turn Level Acc Based on MultiWOZ 2.4: {:.2f} %'.format(cnt_acc_m24 / cnt_turns * 100))
    print('Turn Level Acc Based on GPT-4 Turbo: {:.2f} %'.format(cnt_acc_gpt / cnt_turns * 100))
    print('Coherence with MultiWOZ 2.1: {:.2f} %'.format(coherence_m21 / cnt_turns * 100))
    print('Coherence with MultiWOZ 2.4: {:.2f} %'.format(coherence_m24 / cnt_turns * 100))
    # with open(f'./analyze_m21_{output_name}.json', 'w', encoding='utf-8') as f:
    #     json.dump(lst_incoherence_m21, f, ensure_ascii=False, indent=2)
    # with open(f'./analyze_m24_{output_name}.json', 'w', encoding='utf-8') as f:
    #     json.dump(lst_incoherence_m24, f, ensure_ascii=False, indent=2)


def concat(a, b):
    dic = deepcopy(a)
    for k, v in b.items():
        if k not in dic:
            if v is None:
                v = ''
            dic[k] = v
    return dic


def get_turn_label(former, current):
    dic = {}
    for k, v in current.items():
        if k not in former:
            dic[k] = v
        else:
            if former[k] != v:
                dic[k] = v
    return dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed_name', default='output_parsed_original', type=str)
    args = parser.parse_args()
    func(args.parsed_name)
