import urllib.request
import json
import urllib.error

url = "http://127.0.0.1:8001/api/query"
data = json.dumps({"query": "What is this study about?"}).encode("utf-8")
headers = {"Content-Type": "application/json"}
req = urllib.request.Request(url, data=data, headers=headers)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print("Success:", result)
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code}")
    print(e.read().decode())
except Exception as e:
    print(f"Error: {e}")
