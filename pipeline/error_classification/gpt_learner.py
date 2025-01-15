import json
import sys
import os
import re
# Get the root directory path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the root directory to the system path
sys.path.insert(0, root_dir)

# Now you should be able to import gpt_setup
from gpt_setup import parallel_gpt_generate
from tqdm import tqdm
from argparse import ArgumentParser


def preprocess_gpt_output(gpt_out):
    try:
        # First, check if the output is wrapped in ```json ... ``` block using regex
        pattern = r'```json([\s\S]*?)```'
        match = re.search(pattern, gpt_out)

        if match:
            # Extract the matched JSON string inside the block
            gpt_json = match.group(1).strip()  # Remove any extra spaces or newlines
        else:
            # If no match is found, assume the output is plain JSON and use it directly
            gpt_json = gpt_out.strip()

        # Now try parsing the extracted JSON
        return json.loads(gpt_json)

    except json.JSONDecodeError:
        return None  # Return None if there's a JSON parsing error

def func(result_name):
    """
    Uses GPT to process error data for categorization and refinement, ensuring new error types are integrated into the taxonomy.
    
    :param result_name: The name of the input JSON file containing errors.
    """
    # Load the prompt template
    with open("promptTemplate/Learn.txt", encoding='utf-8') as f:
        template = f.read()

    # Load the existing error data
    with open(f'pipeline/error_classification/{result_name}', encoding='utf-8') as f:
        errors = json.load(f)

    # Load or initialize the taxonomy
    taxonomy_file = 'pipeline/error_classification/taxonomy_collection.json'
    try:
        with open(taxonomy_file, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
    except FileNotFoundError:
        taxonomy = {}

    inputs = []
    for each in tqdm(errors):
        # Prepare the taxonomy string
        existing_taxonomy = "\n".join(
            [f"- {key}: {value['description']}" for key, value in taxonomy.items()]
        )

        # Construct the dictionary to replace placeholders in the prompt
        dic = {
            "error_description": each["error_description"],
            "error_category": each["error_category"],
            "existing_taxonomy": existing_taxonomy
        }

        # Generate the prompt
        prompt = template.format(**dic)

        # Prepare the input for GPT
        inputs.append({
            'gpt_input': prompt,
            'align_data': each
        })

    # Run GPT in parallel for the inputs
    res = parallel_gpt_generate(inputs)

    # Store the results
    output_file = f'pipeline/error_classification/refined_{result_name}'
    with open(output_file, 'w', encoding='utf-8') as file:
        lst = []

        for align_data, gpt_out, gpt_input in res:
            print(f"Align Data: {align_data}")  # Debug log for align data
            print(f"GPT Output: {gpt_out}")     # Debug log for GPT output

            # Ensure GPT output is not empty
            parsed_output = preprocess_gpt_output(gpt_out)
            if parsed_output:
                error_category = parsed_output.get("error_category", "Uncategorized")
                error_description = parsed_output.get("error_description", "No description provided.")

                # Update taxonomy if new category is added
                if error_category not in taxonomy:
                    taxonomy[error_category] = {
                        "description": error_description,
                        "count": 1
                    }
                else:
                    taxonomy[error_category]["count"] += 1

                # Append the result
                lst.append({
                    'align_data': align_data,
                    'gpt_out': gpt_out,
                    'gpt_input': gpt_input,
                    'error_category': error_category,
                    'error_description': error_description
                })
            else:
                lst.append({
                    'align_data': align_data,
                    'gpt_out': gpt_out,
                    'gpt_input': gpt_input,
                    'error_category': "Parsing Error",
                    'error_description': "Failed to parse GPT output."
                })

        # Write the refined error categorizations to the output file
        json.dump(lst, file, ensure_ascii=False, indent=2)

    # Save the updated taxonomy
    with open(taxonomy_file, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    print(f"Refined errors saved to {output_file}")
    print(f"Updated taxonomy saved to {taxonomy_file}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_name', default='wrong_cases.json', type=str)
    args = parser.parse_args()
    func(args.result_name)
