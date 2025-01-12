import json
from gpt_setup import parallel_gpt_generate
from tqdm import tqdm
from argparse import ArgumentParser


def construct_history(history):
    dic = {}
    cnt = 1
    for i in range(0, len(history), 2):
        dic[f'Turn{cnt}'] = {
            "Agent": history[i],
            "user": history[i + 1]
        }
        cnt += 1
    return dic


def func(result_name):
    with open("promptTemplate/Taxonomy.txt", encoding='utf-8') as f:
        template = f.read()
    with open(f'./results/parsed/{result_name}', encoding='utf-8') as f:
        results = json.load(f)
    cnt = 0
    inputs = []
    last_bs = {}
    lst = []
    for each in tqdm(results):
        if len(each["predict_turn_label"]) == 0:
            lst.append([each, str({"explanation": "", "incorrect_domain_slot": {}}), ""])
            continue
        dialogue_idx, turn_idx = each['flag'].split('-')
        if turn_idx == '0':
            last_bs = {}
        history = '\n'.join(['Agent: ' + each['history'][i] + '\n' + 'User: ' + each['history'][i + 1] for i in range(0, len(each['history']), 2)])
        turn_label = str(each['predict_turn_label'])
        dic = {
            'history': str(construct_history(each['history'])),
            'turn_label': turn_label,
            'user': each['user'],
            'system': each['system'],
            'last_state': last_bs,
        }
        prompt = template.format(**dic)
        # print(prompt)
        # print('-' * 150)
        # cnt += 1
        # if cnt > 100:
        #     break
        inputs.append({'gpt_input': prompt, 'align_data': each})
    #     # cnt += 1
    #     # if cnt > 10:
    #     #     break
    #     last_bs = each['predict']
    res = parallel_gpt_generate(inputs)
    with open(f'./pipeline/error_classification/{result_name}', 'w', encoding='utf-8') as file:

        for align_data, gpt_out, gpt_input in res:
            lst.append([align_data, gpt_out, gpt_input])
        json.dump(lst, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_name', default='original.json', type=str)
    args = parser.parse_args()
    func(args.result_name)
