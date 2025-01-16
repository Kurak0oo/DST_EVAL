import json
import re
from copy import deepcopy
import argparse


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def get_numerical_slot_values():
    dic = {
        'book people': [str(i) for i in range(1, 20)],
        'stars': [str(i) for i in range(1, 6)],
        'book stay': [str(i) for i in range(1, 20)],
    }
    return dic


def extract_json_from_string(input_string):
    # 使用正则表达式匹配 ``` 包围的 JSON 内容
    pattern = r'```json([\s\S]*?)```'
    matches = re.findall(pattern, input_string)

    if matches:
        # 提取匹配到的 JSON 内容
        json_content = matches[-1]

        try:
            # 解析 JSON
            parsed_json = json.loads(json_content)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            print(json_content)
            return None
    else:
        print("未找到匹配的 JSON 内容")
        return None


def func(comp_filename, acc_filename, output_name):
    with open("../data/mwz2_1/ontology.json", encoding="utf-8") as f:
        ontology = json.load(f)
    slots = get_slot_information(ontology)
    numerical_slot_values = get_numerical_slot_values()
    with open(f'../output/{comp_filename}.json', encoding='utf-8') as f:
        comp = json.load(f)
    with open(f'../parsed/{acc_filename}.json', encoding='utf-8') as f:
        acc = json.load(f)
    dic = {}
    for each in comp:
        dialogue_idx, turn_idx = each[0]['flag'].split('-')
        if dialogue_idx not in dic:
            dic[dialogue_idx] = {}
        dic[dialogue_idx][turn_idx] = each[0]
        dic[dialogue_idx][turn_idx]['gpt'] = each[1]
    turns = 0
    ans = []
    jga_m21 = 1
    jga_m24 = 1
    for flag, turn_data in dic.items():
        missed = set()
        incorrect = set()
        correct = {}
        tmp = {
            'dialogue_idx': flag,
            'turn_details': [],
        }
        for turn_idx in range(len(turn_data)):
            already_missed = list(deepcopy(missed))
            already_incorrect = list(deepcopy(incorrect))
            already_correct = list(deepcopy(correct))
            turns += 1
            x = extract_json_from_string(turn_data[str(turn_idx)]['gpt'].lower())
            if x is None:
                print('parsed error')
                continue
            flag_incorrect = True
            if len(acc[f'{flag}-{turn_idx}']['incorrect']) > 0:
                flag_incorrect = False
                for ea in acc[f'{flag}-{turn_idx}']['incorrect']:
                    incorrect.add(ea)
            to_be_added = []
            for incomplete_ds, v in x['missed_domain_slot'].items():
                if incomplete_ds not in slots:
                    continue
                # if incomplete_ds in correct or incomplete_ds in missed:
                if incomplete_ds in correct or incomplete_ds in turn_data[str(turn_idx)]['predict_turn_label'] or incomplete_ds in already_correct:
                    continue
                to_be_added.append(incomplete_ds)
            for miss in to_be_added:
                missed.add(miss)
            gpt_turn_judge = 0
            if flag_incorrect:
                if len(to_be_added) == 0:
                    gpt_turn_judge = 1
            for ds, v in turn_data[str(turn_idx)]['predict_turn_label'].items():
                if ds not in slots:
                    gpt_turn_judge = 0
                    incorrect.add(ds)
                    continue
                # 多余了
                if ds in correct and correct[ds] == v:
                    gpt_turn_judge = 0
                    continue
                d, s = ds.split('-')
                if s in numerical_slot_values and v not in numerical_slot_values[s]:
                    gpt_turn_judge = 0
                    continue
                if ds not in incorrect:
                    if ds in missed:
                        missed.remove(ds)
                    correct[ds] = v
            gt_m21 = set([s + '==' + v for s, v in turn_data[str(turn_idx)]["ground_truth"].items()])
            gt_m24 = set([s + '==' + v for s, v in turn_data[str(turn_idx)]["ground_truth_m24"].items()])
            predict = set([s + '==' + v for s, v in turn_data[str(turn_idx)]["predict"].items()])
            if predict == gt_m21:
                jga_m21 += 1
            if predict == gt_m24:
                jga_m24 += 1
            tmp['turn_details'].append({
                'turn_idx': turn_idx,
                'system': turn_data[str(turn_idx)]['system'],
                'user': turn_data[str(turn_idx)]['user'],
                'gt_m21': turn_data[str(turn_idx)]['ground_truth'],
                'gt_m24': turn_data[str(turn_idx)]['ground_truth_m24'],
                'predict': turn_data[str(turn_idx)]['predict'],
                'predict_turn_label': turn_data[str(turn_idx)]['predict_turn_label'],
                'gpt_turn_judge': gpt_turn_judge,
                'gpt_out': {
                    'incorrect_ds': acc[f'{flag}-{turn_idx}'],
                    'missed_ds': to_be_added,
                    # 'explanation': x['explanation'],
                },
                'already': {
                    'missed': already_missed,
                    'incorrect': already_incorrect,
                    'correct': already_correct,
                }
            })
        ans.append(tmp)
    print(turns)
    print("JGA-MWZ2.1: {:.2f}".format(jga_m21 / turns * 100))
    print("JGA-MWZ2.4: {:.2f}".format(jga_m24 / turns * 100))
    with open(f'{output_name}.json', 'w', encoding='utf-8') as f:
        json.dump(ans, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comp_filename', default='onebyone_comp_output_original_sampled', type=str)
    parser.add_argument('--acc_filename', default='parsed_acc_onebyone', type=str)
    parser.add_argument('--output_name', default='output_parsed_original', type=str)
    args = parser.parse_args()
    func(args.comp_filename, args.acc_filename, args.output_name)
