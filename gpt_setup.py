from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI
from func_timeout import func_set_timeout
client = OpenAI(api_key="sk-zk2530d19d3258da9048c56420da65b48e06787cb93557f1",base_url="https://api.zhizengzeng.com/")


@func_set_timeout(100)
def call_openai_api(input_text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": input_text}],
        temperature=0.0,
    )
    return my_parse(response)


def process_text(inputs):
    for _ in range(3):
        try:
            result = call_openai_api(inputs['gpt_input'])
            # print(f"Output for input:\n{result}\n{'='*30}")
            return inputs['align_data'], result, inputs['gpt_input']
        except Exception as e:
            print(f"Error calling API: {e}")
            continue
    return inputs['align_data'], '', inputs['gpt_input']


def parallel_gpt_generate(inputs):
    max_workers = 20

    with Pool(max_workers) as pool, tqdm(total=len(inputs)) as pbar:
        results = []
        for result in pool.imap_unordered(process_text, inputs):
            results.append(result)
            pbar.update(1)
    return results

    # with open('output.txt', 'w', encoding='utf-8') as file:
    #     for i, (input_text, generated_text) in enumerate(results):
    #         file.write(f"Input {i + 1}:\n{input_text}\nOutput {i + 1}:\n{generated_text}\n{'=' * 30}\n")


def my_parse(res):
    if res == '':
        return res
    else:
        return res.choices[0].message.content
