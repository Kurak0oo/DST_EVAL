import os
import requests

# Load environment variables from a .env file (optional but recommended)

# Base URL for the API
BASE_URL = "https://api.zhizengzeng.com/v1"

def credit_grants(query=None):
    """
    Fetches credit grants from the Zhizengzeng API.

    Args:
        query (dict, optional): A dictionary of query parameters to send in the body of the POST request.

    Returns:
        dict: The JSON response from the API if the request is successful.
        None: If the request fails.
    """
    # Retrieve the API secret key from environment variables for security
    api_secret_key = "sk-zk2530d19d3258da9048c56420da65b48e06787cb93557f1"


    # Construct the full URL for the credit grants endpoint
    url = f"{BASE_URL}/dashboard/billing/credit_grants"

    # Set up the headers, including the authorization token
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f"Bearer {api_secret_key}"
    }

    try:
        # Make the POST request to the API
        response = requests.post(url, headers=headers, json=query)

        # Raise an exception for HTTP error codes
        response.raise_for_status()

        # Parse and return the JSON response
        data = response.json()
        print("Credit Grants Response:", data)
        return data

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response Content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except ValueError:
        print("Error decoding JSON response.")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

    return None

def main():
    """
    Main function to test the credit_grants function.
    """
    # Example query parameters (modify as needed)
    test_query = {
        # 'param1': 'value1',
        # 'param2': 'value2',
        # Add any necessary query parameters here
    }

    # Call the credit_grants function with the test_query
    result = credit_grants(query=test_query)

    if result:
        # Process the result as needed
        print("Successfully retrieved credit grants.")
    else:
        print("Failed to retrieve credit grants.")

if __name__ == "__main__":
    main()
