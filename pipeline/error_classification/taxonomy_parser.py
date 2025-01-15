import json
from collections import Counter

def extract_error_details(input_file, output_file):
    """
    Extracts error_category and error_description from cases where the GPT judgement is "Wrong".
    
    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output JSON file for extracted errors.
    """
    def is_wrong_judgment(gpt_out):
        """
        Checks if the gpt_out indicates a wrong judgment.
        Handles various formats of gpt_out.
        
        :param gpt_out: The gpt_out string from the JSON data.
        :return: True if the judgement is "Wrong", otherwise False.
        """
        try:
            # Parse gpt_out if it's a JSON string
            if gpt_out.startswith("{"):
                parsed_output = json.loads(gpt_out)
                judgement = parsed_output.get("judgement", "").strip().lower()
                # Normalize judgement to handle cases like "Wrong" and "[Wrong]"
                return judgement == "wrong" or judgement == "[wrong]"
            # Handle cases where gpt_out is just "Wrong" or "[Wrong]"
            return gpt_out.strip().lower() == "wrong" or gpt_out.strip().lower() == "[wrong]"
        except (json.JSONDecodeError, AttributeError):
            # If gpt_out is not a valid JSON or not a string, it's invalid
            return False

    # Load the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter wrong cases and extract error details
    error_details = []
    for item in data:
        gpt_out = item.get('gpt_out', "")
        if is_wrong_judgment(gpt_out):
            try:
                parsed_output = json.loads(gpt_out) if gpt_out.startswith("{") else {}
                error_details.append({
                    "error_category": parsed_output.get("error_category", "Unknown Category"),
                    "error_description": parsed_output.get("error_description", "No description available.")
                })
            except json.JSONDecodeError:
                error_details.append({
                    "error_category": "Unknown Category",
                    "error_description": "No description available."
                })

    # Save the error details into the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(error_details, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(error_details)} error details to {output_file}")

extract_error_details(
     input_file="pipeline/error_classification/original_sampled.json",
     output_file="pipeline/error_classification/wrong_cases.json"
)
