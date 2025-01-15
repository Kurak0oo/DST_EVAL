import json

output = "{\n  \"judgement\": \"Wrong\",\n  \"error_category\": \"Incomplete State Tracking Resulting in Partial Belief State Update\",\n  \"error_description\": \"The model's prediction only captures a part of the conversation's belief state related to booking hotel rooms (specifically the number of people), while omitting other relevant slots such as 'hotel-area', 'hotel-parking', 'hotel-pricerange', 'hotel-stars', and 'hotel-type'. This suggests a failure in tracking and compiling the full context and user preferences throughout the dialogue. The model may be focusing on the most recent user utterance and missing information provided in earlier dialogue turns. It also indicates a weakness in integrating past dialogue context into the present conversation state. Such an error can result from limitations in the model's ability to manage long-term dialogue context or its failure to aggregate and update all relevant information correctly in its belief state.\"\n}"

# Parse the string into a valid JSON object
parsed_output = json.loads(output)

# Print as a properly formatted JSON object
print(json.dumps(parsed_output, indent=2))
