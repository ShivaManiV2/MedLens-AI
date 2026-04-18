import requests

url = "http://localhost:8000/api/query"
data = {"query": "Tell me a long medical story."}

with requests.post(url, json=data, stream=True) as r:
    for line in r.iter_lines():
        if line:
            print(line.decode('utf-8'))
