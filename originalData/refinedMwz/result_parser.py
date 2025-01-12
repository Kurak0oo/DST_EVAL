import json
from copy import deepcopy
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.fix_label import fix_general_label_error
import argparse


EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def get_turn_label(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dic = {}
    for each in data:
        if each["flag"] not in dic:
            dic[each["flag"]] = {}
        dic[each["flag"]][each["generate"]] = each["value"]
    return dic


def convert(dic, output_name):
    with open("../mwz/mwz2_1/ontology.json", encoding="utf-8") as f:
        ontology = json.load(f)
    slots = get_slot_information(ontology)
    with open("../mwz/mwz2_1/test_dials.json", encoding="utf-8") as f:
        data = json.load(f)
    lst = []
    for each in data:
        dialogue_idx = each["dialogue_idx"]
        last_belief_state = {}  # Initialization
        last_predict = {}
        last_ground_truth_m24 = {}
        last_ground_truth_m21 = {}
        history = []
        for turn in each["dialogue"]:
            k = dialogue_idx + '-' + str(turn["turn_idx"])
            if k in dic:
                turn_label = fix_general_label_error([{"slots": [[k, fix_time_label(v)]]} for k, v in dic[k].items()],
                                                        False, slots)
                last_belief_state = update_state(last_belief_state, turn_label)
            else:
                turn_label = {}
            last_belief_state = fix_general_label_error([{"slots": [[k, v]]} for k, v in last_belief_state.items()],
                                                        False, slots)
            # Track slot changes between turns without including new slots
            predict_slot_changes = get_detailed_slot_changes(last_predict, last_belief_state)
            ground_truth_m24_slot_changes = get_detailed_slot_changes(last_ground_truth_m24, turn["belief_state"])
            ground_truth_m21_slot_changes = get_detailed_slot_changes(last_ground_truth_m21, turn["belief_state"])

            # Update the last predict and ground truth to the current state for the next iteration
            last_predict = deepcopy(last_belief_state)
            last_ground_truth_m24 = deepcopy(turn["belief_state"])
            last_ground_truth_m21 = deepcopy(turn["belief_state"])
            lst.append({
                "flag": k,
                "system": turn["system_transcript"],
                "user": turn["transcript"],
                "predict_history": deepcopy(last_belief_state),
                "predict_turn_label": deepcopy(turn_label),
                "ground_truth_m21": turn["belief_state"],  # Add m21 ground truth
                "ground_truth_m24": turn["belief_state"],  # Add m24 ground truth
                "predict_slot_changes": predict_slot_changes,  # Slot changes for prediction
                "ground_truth_m21_slot_changes": ground_truth_m21_slot_changes,  # Slot changes for m21
                "ground_truth_m24_slot_changes": ground_truth_m24_slot_changes,  # Slot changes for m24
                "history": history[:],
            })
            history.append(turn["system_transcript"])
            history.append(turn["transcript"])

    with open("../mwz/mwz2_4/test_dials.json", encoding="utf-8") as f:
        m24 = json.load(f)
    cnt = 0
    for each in m24:
        dialogue_idx = each['dialogue_idx']
        for turn in each["dialogue"]:
            flag = dialogue_idx + '-' + str(turn['turn_idx'])
            assert flag == lst[cnt]['flag']
            lst[cnt]['ground_truth_m24'] = {x['slots'][0][0]: x['slots'][0][1] for x in turn['belief_state']}
            cnt += 1

    #output
    with open(f"parsed/{output_name}.json", 'w', encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)

def get_detailed_slot_changes(prev_state, current_state):
    changes = {}

    # Check for updated slots
    for slot, current_value in current_state.items():
        if slot in prev_state and prev_state[slot] != current_value:
            changes[slot] = f'"{slot}": "{prev_state[slot]}" -> "{slot}": "{current_value}"'

    # Check for removed slots
    for slot in prev_state:
        if slot not in current_state:
            # Removed slot
            changes[slot] = f'"{slot}": "{prev_state[slot]}" -> None'

    return changes


def update_state(state, turn_label):
    for s, v in turn_label.items():
        slot, value = s, fix_time_label(v)
        if value == "none":
            try:
                del state[slot]
            except:
                continue
        else:
            if value == "parking" or value == "wifi":
                value = "yes"
            state[slot] = value
    return state


def fix_time_label(value):
    x = re.search(r"\d\d\D\d\d", value)
    if x is not None:
        x = x.group()
        biaodian = re.search(r"\D", x).group()
        if biaodian != ":":
            return value.replace(biaodian, ':')
    if value == 'parking' or value == 'wifi':
        value = 'yes'
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', default='original', type=str)
    parser.add_argument('--result_path', default=r'./result-seed-10-ratio-1-original.json', type=str)
    args = parser.parse_args()
    convert(get_turn_label(args.result_path), args.output_name)
