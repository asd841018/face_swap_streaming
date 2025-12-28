import urllib.request
import json
import sys

url = "http://localhost:9997/v3/paths/list"

print(f"Checking URL: {url}")

try:
    with urllib.request.urlopen(url) as response:
        print(f"Status Code: {response.getcode()}")
        data = response.read()
        try:
            json_data = json.loads(data)
            print("Response JSON:")
            print(json.dumps(json_data, indent=2))
        except json.JSONDecodeError:
            print("Response Text:")
            print(data.decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
