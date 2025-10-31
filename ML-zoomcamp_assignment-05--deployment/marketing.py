import requests
url='http://0.0.0.0:9696/predict'
records={
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}


response=requests.post(url,json=records)
predict=response.json()

if predict['outcome']:
    print(f"Customer will convert with a score of {predict['score']}, Therefore send subscription offer")
else:
    print(f"Customer will not convert with a score of {predict['score']}, Therefore send free trial offer")