🧠 Machine Learning Zoomcamp – Conceptual Notes & Assignments

This repository contains my personal notes, summaries, and assignments from the Machine Learning Zoomcamp course by DataTalksClub.

I’m documenting my learning journey with a focus on:

📘 Conceptual clarity for beginners  
🛠️ Practical assignments  
🧩 Clean Markdown formatting for recruiter visibility  

---

## 📌 Learning Progress – Machine Learning Zoomcamp

### ✔️ Intro Section Completed  
Key learnings:
- What Machine Learning is (and isn’t)
- Differences between ML, RL, rule-based systems, and traditional programming
- The CRISP-DM framework for ML projects
- Model selection and data splitting (train/validation/test)
- Setting up the ML environment

### ✔️ Module 2 – Regression Completed  
Key learnings:
- Preparing and cleaning data with Pandas & NumPy
- Exploratory Data Analysis (EDA) with Seaborn/Matplotlib
- Building proper train/validation/test splits for fair evaluation
- Linear regression theory and vectorized implementation
- Feature engineering (numerical + categorical)
- Evaluating models with RMSE
- Regularization and tuning for better performance
- Using the trained model for predictions

### ✔️ Module 3 – Classification Completed  
Key learnings:
- Binary classification using logistic regression
- Encoding categorical features and scaling numerical ones
- Using `.predict_proba()` vs `.predict()` for decision-making
- Applying decision thresholds to convert probabilities into binary predictions
- Understanding class imbalance and its impact on accuracy
- Building reusable training and prediction pipelines
- Saving and loading models for deployment
- Preparing for evaluation with proper data splits and preprocessing discipline

### ✔️ Module 4 – Evaluation Completed  
Key learnings:
- Why accuracy alone is misleading in imbalanced datasets
- Using `DummyClassifier` to establish a baseline
- Understanding the confusion matrix: TP, FP, TN, FN
- Computing precision, recall, and F1 score from confusion matrix
- Tuning decision thresholds to balance precision and recall
- Plotting precision-recall curves to visualize tradeoffs
- Plotting ROC curves and computing AUC to measure ranking quality
- Evaluating individual features using ROC AUC
- Applying 5-fold cross-validation to assess model stability
- Interpreting mean and standard deviation of AUC across folds
- Documenting evaluation results with metrics, plots, and threshold decisions

### ✔️ Module 5 – Deployment Completed  
📁 Folder: [`ML-zoomcamp_assignment-05--deployment`](./ML-zoomcamp_assignment-05--deployment)

Key deliverables:
- Containerized FastAPI app serving a churn prediction model
- Dockerfile with uv integration for reproducible deployment
- Locked dependencies using `uv.lock` and `pyproject.toml`
- API test script for local validation
- Model file and scoring logic packaged for production

How to run:
```bash
docker build -t zoomcamp-score:local ML-zoomcamp_assignment-05--deployment
docker run --rm -p 9696:9696 zoomcamp-score:local
python ML-zoomcamp_assignment-05--deployment/test.py
```
### ✔️ Module 8 – Deep Learning Completed
Key learnings:
- **CNN Fundamentals:** Understanding why Dense networks fail on images and how Convolutional layers (filters/kernels) preserve spatial data.
- **Architecture Mechanics:** How Pooling layers (Max/Average) provide translation invariance and reduce computation.
- **Transfer Learning:** Leveraging pre-trained models (Xception, ResNet50) trained on ImageNet to solve custom problems with less data.
- **Implementation Frameworks:** Building and training models using both the Keras (Functional API) and PyTorch (Class-based) approaches.
- **Data Preprocessing:** Managing image loading pipelines, resizing, and normalization for neural networks.
- **Fighting Overfitting:**
    - **Dropout:** Randomly deactivating neurons to force robust feature learning.
    - **Data Augmentation:** Generating new training samples via rotation, flips, and shifts.
- **Tuning Strategy:** Systematically tuning Learning Rate (LR), inner layer sizes, and using Checkpointing to save only the best model.

### ✔️ Module 9 – Serverless Deep Learning Completed
📁 Folder: [`ML-zoomcamp_assignment-09--serverless`](./ML-zoomcamp_assignment-09--serverless)

Key learnings:
- **Serverless Architecture:** Understanding AWS Lambda, cold starts, and the cost/scaling benefits over traditional servers.
- **Model Optimization:** Converting heavy Keras/TensorFlow models into lightweight **TFLite** or **ONNX** formats for faster inference.
- **Lambda-Ready Containers:** Building Docker images based on Amazon Linux to match the AWS production environment.
- **Cross-Platform Debugging:** Solving critical OS-level dependency conflicts (e.g., `glibc` versions for `onnxruntime` and `numpy`) between local dev (Windows/Python 3.12) and production (Linux/Python 3.10).
- **Build Context Hygiene:** Optimizing Docker builds using `.dockerignore` to exclude large datasets and virtual environments.

Key deliverables:
- AWS Lambda function (`lambda_function.py`) capable of downloading and classifying images via URL.
- Production-grade `Dockerfile` using multi-stage builds and `uv` for fast dependency installation.
- `pyproject.toml` with pinned versions to ensure stability on Amazon Linux.

How to run:
```bash
# Build the Lambda-compatible image
docker build -t hair-classifier:v1 ML-zoomcamp_assignment-09--serverless

# Run the container locally (emulating Lambda)
docker run -it --rm -p 8080:8080 hair-classifier:v1

# Test the endpoint
python ML-zoomcamp_assignment-09--serverless/test.py
