import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI(title= 'Score-Dashboard')
with open('pipeline_v1.bin','rb') as f_in:
    pipeline=pickle.load(f_in)

class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

class PredictResponse(BaseModel):
    score: float
    outcome: bool

def predict_single(records: dict)-> float:
    results=pipeline.predict_proba(records)[0,1]
    return float(results)

@app.post('/predict') 
def predict(records: LeadData)->PredictResponse:
    prob=predict_single(records.model_dump())
    return PredictResponse(score=prob,
            outcome=bool(prob>=0.5))
 
if __name__== '__main__' :
    uvicorn.run(app, host='0.0.0.0', port=9696)  