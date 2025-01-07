import json
import random
random.seed(42)


def get_sampled_dialogue_idx():
    with open('./parsed/original.json', encoding='utf-8') as f:
        data = json.load(f)
    d_idxs = set()
    for each in data:
        dialogue_idx, turn_idx = each['flag'].split('-')
        d_idxs.add(dialogue_idx)
    lst = list(d_idxs)
    random.shuffle(lst)
    with open('./parsed/sampled_dialogue_idx.txt', 'w', encoding='utf-8') as f:
        for i in range(100):
            f.write(lst[i] + '\n')


def construct_data():
    with open('./parsed/sampled_dialogue_idx.txt', encoding='utf-8') as f:
        d_idxs = set([each.strip() for each in f])
    file_names = ['original', 'da', 'da-p']
    for file_name in file_names:
        with open(f'./parsed/{file_name}.json', encoding='utf-8') as f:
            data = json.load(f)
        ans = []
        for each in data:
            dialogue_idx, turn_idx = each['flag'].split('-')
            if dialogue_idx in d_idxs:
                ans.append(each)
        print(len(ans))
        with open(f'./parsed/{file_name}_sampled.json', 'w', encoding='utf-8') as f:
            json.dump(ans, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    get_sampled_dialogue_idx()
    construct_data()
