import json
from gpt_setup import parallel_gpt_generate
from tqdm import tqdm
from argparse import ArgumentParser
from copy import deepcopy

def construct_history(history):
    dic = {}
    cnt = 1
    for i in range(0, len(history), 2):
        dic[f'Turn{cnt}'] = {
            "System": history[i],
            "user": history[i + 1]
        }
        cnt += 1
    return dic

def func(result_name):
    with open("promptTemplate/Taxonomy.txt", encoding='utf-8') as f:
        template = f.read()

    with open(f'originalData/refinedMwz/sampled/{result_name}', encoding='utf-8') as f:
        results = json.load(f)

    cnt = 0
    inputs = []
    last_bs = {}
    lst = []

    for each in tqdm(results):
        dialogue_idx, turn_idx = each['flag'].split('-')
        if turn_idx == '0':
            last_bs = {}

        # Assuming the ground truth belief states are available from each['ground_truth_m21'] and each['ground_truth_m24']
        ground_truth_m21 = each.get('ground_truth_m21', {})
        ground_truth_m24 = each.get('ground_truth_m24', {})
        ground_truth_turn_label_m21 = each.get('ground_truth_turn_label_m21',{})
        ground_truth_turn_label_m24 = each.get('ground_truth_turn_label_m24',{})
        # Get slot changes between prediction and ground truth for both versions
        predict_slot_changes = each.get('predict_slot_changes',{})
        ground_truth_m21_slot_changes = each.get('ground_truth_m21_slot_changes',{})  # comparing m21 to m21
        ground_truth_m24_slot_changes = each.get('ground_truth_m24_slot_changes',{}) # comparing m24 to m24

        # Construct the history as a string (or use your existing structure)][i + 1] for i in range(0, len(each['history']), 2)])
        turn_label = str(each['predict_turn_label'])

        # Prepare the dictionary with all necessary data
        dic = {
            'history': str(construct_history(each['history'])),
            'turn_label': turn_label,
            'user': each['user'],
            'system': each['system'],
            'last_state': last_bs,
            'predict_history': deepcopy(last_bs),
            'ground_truth_m21': ground_truth_m21,  # Add ground truth belief state for m21
            'ground_truth_m24': ground_truth_m24,  # Add ground truth belief state for m24
            'predict_slot_changes': predict_slot_changes,  # Track slot changes for predictions
            'ground_truth_turn_label_m21': ground_truth_turn_label_m24,
            'ground_truth_turn_label_m24': ground_truth_turn_label_m21,
            'ground_truth_m21_slot_changes': ground_truth_m21_slot_changes,  # Track slot changes for m21
            'ground_truth_m24_slot_changes': ground_truth_m24_slot_changes,  # Track slot changes for m24
        }

        # Generate the GPT prompt using the template
        prompt = template.format(**dic)

        inputs.append({'gpt_input': prompt, 'align_data': each})

        last_bs = each['predict_turn_label']  # Update the last belief state with the predicted state

    # Run the GPT evaluation
    res = parallel_gpt_generate(inputs)

    # Store the results# Store the results
    with open(f'./pipeline/error_classification/{result_name}', 'w', encoding='utf-8') as file:
        lst = []  # Clear list to store output

        for align_data, gpt_out, gpt_input in res:
            print(f"Align Data: {align_data}")
            print(f"GPT Output: {gpt_out}")  # Log GPT output to debug

            # Ensure that GPT output is not empty
            if not gpt_out:
                gpt_out = "No output from GPT"  # Set a default message if GPT output is empty

            # Append results to the list
            lst.append({
                'align_data': align_data,
                'gpt_out': gpt_out,
                'gpt_input': gpt_input
            })

        # Write the results to the JSON file
        json.dump(lst, file, ensure_ascii=False, indent=2)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_name', default='original_sampled.json', type=str)
    args = parser.parse_args()
    func(args.result_name)
