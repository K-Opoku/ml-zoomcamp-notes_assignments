# Module 5 — Deployment with FastAPI, Docker & uv

This note explains how to deploy a machine learning model as a web service.  
It follows **Module 5 (Deployment)** from the *Machine Learning Zoomcamp*, updated for the 2025 workshop.

---

## 🎯 Goal
Turn a trained ML pipeline into an API that anyone can run on their computer or inside a container.  
The goal is to make the model easy to use, repeatable, and ready for production.

---

## 🧱 Step 1: Environment Setup with uv

We use **uv**, a modern Python package manager, to make sure everyone gets the same versions of the tools.

```bash
uv init
uv add fastapi uvicorn scikit-learn==1.6.1
uv lock
The uv.lock file saves exact versions of every package.
This makes the setup reproducible for anyone who clones the repo.

📦 Step 2: Model Preparation
The file pipeline_v1.bin is a saved model pipeline. It contains:

a DictVectorizer for turning features into numbers

a Logistic Regression model trained on the lead scoring dataset

We check the file to be sure it’s not corrupted:

bash
Copy code
md5sum pipeline_v1.bin
🌐 Step 3: Serve the Model with FastAPI
We create a small web app using FastAPI. It accepts JSON input and returns a prediction.

python
Copy code
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

@app.post("/predict")
def predict(lead: Lead):
    X = dv.transform([lead.dict()])
    prob = model.predict_proba(X)[0, 1]
    return {"conversion_probability": float(prob)}
Example input:

json
Copy code
{
  "lead_source": "organic_search",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0
}
Example output:

json
Copy code
{ "conversion_probability": 0.59 }
Run locally with:

bash
Copy code
uvicorn score:app --host 0.0.0.0 --port 9696 --reload
🐳 Step 4: Containerize with Docker
We package the app into a Docker image so it can run anywhere.

Dockerfile:

dockerfile
Copy code
FROM python:3.13-slim

WORKDIR /app

COPY requirements/pyproject.toml requirements/uv.lock ./
RUN pip install uv && uv pip install --system -r pyproject.toml

COPY app/ .

EXPOSE 9696
CMD ["uvicorn", "score:app", "--host", "0.0.0.0", "--port", "9696"]
Build and run:

bash
Copy code
docker build -t zoomcamp-score:local .
docker run --rm -p 9696:9696 zoomcamp-score:local
🧪 Step 5: Test the API
Use a simple Python script (test.py) to send data to the container and check the response.

bash
Copy code
python test.py
You should get a probability close to 0.59, showing the service works correctly inside Docker.

🌍 Step 6: (Optional) Share the Image
You can upload the image to Docker Hub so others can use it:

bash
Copy code
docker tag zoomcamp-score:local your-dockerhub-username/zoomcamp-score:1.0
docker push your-dockerhub-username/zoomcamp-score:1.0
Then anyone can run it with:

bash
Copy code
docker run --rm -p 9696:9696 your-dockerhub-username/zoomcamp-score:1.0
🧠 Key Points
Reproducible setup: uv.lock keeps dependency versions fixed.

Easy API: FastAPI makes it simple to send data and get predictions.

Portable: Docker makes sure it runs the same everywhere.

Professional: A clean structure and documentation make this project ready for clients or recruiters.

📁 Folder Layout
pgsql
Copy code
ML-zoomcamp_assignment-05-deployment/
├── app/
│   ├── score.py
│   ├── pipeline_v1.bin
│   └── marketing.py
├── requirements/
│   ├── pyproject.toml
│   ├── uv.lock
│   └── .python-version
├── Dockerfile
├── test.py
└── README.md
