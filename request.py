import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Name', 'Purpose','Completion Year','Type','Length (m)' 'Max Height above Foundation (m)'})

print(r.json())
