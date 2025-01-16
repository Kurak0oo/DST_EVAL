import argparse
import json
from copy import deepcopy

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def func(output_name):
    cnt_turns = 0
    cnt_acc_gpt = 0
    with open(f'{output_name}.json', encoding='utf-8') as f:
        data = json.load(f)
    for x in data:
        for sample in x['turn_details']:
            cnt_turns += 1
            if len(sample['already']['incorrect']) == 0 and len(sample['already']['missed']) == 0:
                cnt_acc_gpt += 1
    print('Number of turns: {}'.format(cnt_turns))
    print('GPT-JGA: {:.2f} %'.format(cnt_acc_gpt / cnt_turns * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', default='output_parsed_original', type=str)
    args = parser.parse_args()
    func(args.output_name)
