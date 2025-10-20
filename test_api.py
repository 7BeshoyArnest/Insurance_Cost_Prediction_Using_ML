import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "age": 40,
    "sex": "male",      
    "bmi": 25.0,
    "children": 2,
    "smoker": "no",
    "region": "southwest"
}

response = requests.post(url, json=data)
print(response.json())

