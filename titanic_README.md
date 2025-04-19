# Titanic Survival Prediction - Streamlit App

This project is an end-to-end machine learning application built to predict survival on the Titanic using a Random Forest Classifier. The model is trained and evaluated using the Titanic dataset from Kaggle and deployed as an interactive Streamlit web application.

---
## Live app
[Click here to launch the app](https://titanicpredictionapp-tebbzapph5futzidcjz257c.streamlit.app/)

## Project Structure
titanic-survival-prediction/
├── app.py                          # Streamlit web app script
├── End-to-End Titanic Survival Prediction project.ipynb  # Jupyter notebook for EDA, training, and model saving
├── train.csv                       # Dataset used for training and evaluation
├── random_forest_model.pkl         # Saved RandomForest model using pickle
├── scaler.pkl                      # Saved StandardScaler used for input normalization
├── requirements.txt                # List of required Python packages
|── README.md                       # Project documentation
|__test.csv                         # Dataset used for test and evaluation


---

## Features

- Data preprocessing
- Random Forest model training
- Feature engineering (`FamilySize`, `IsAlone`)
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score
- Confusion matrix heatmap
- Interactive user input form to predict survival

---

## Installation & Setup

```bash
# Clone the repository
https://github.com/your-username/titanic-streamlit-app.git
cd titanic-streamlit-app

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Model Overview

- **Model:** Random Forest Classifier
- **Preprocessing:**
  - `Sex` encoded as 0 (male) and 1 (female)
  - `Embarked` one-hot encoded (`Embarked_Q`, `Embarked_S`)
  - `FamilySize` and `IsAlone` engineered as new features
  - Features scaled using `StandardScaler`
---
## Technologies Used:

Python:	Core programming language
Pandas:	Data manipulation and preprocessing
NumPy:	Numerical operations
Matplotlib & Seaborn:	Data visualization
Scikit-learn:	Machine learning models and metrics
Streamlit:	Web application framework for deploying ML models
Jupyter Notebook:	Exploratory data analysis and prototyping
Pickle:	Saving and loading the trained model and scale

---
## Conclusion:

In this project, we successfully built and evaluated a machine learning model to predict the survival of passengers aboard the Titanic. The process involved:

- Thorough data cleaning and preprocessing to handle missing values and encode categorical features.

- Exploratory Data Analysis (EDA) to understand data distribution and uncover patterns.

- Engineering meaningful features like FamilySize and IsAlone, which helped improve model performance.

- Comparing multiple models including Logistic Regression, Random Forest, and XGBoost.

- Selecting the Random Forest Classifier as the best-performing model based on accuracy and F1 score.

Finally, the trained model was deployed as an interactive web application using Streamlit, allowing users to input passenger details and receive real-time survival predictions.
This end-to-end workflow demonstrates the power of combining data science techniques with interactive tools to create practical machine learning solutions.

---

## Contact

Made with by [Harsh Bala]. Feel free to reach out!

- GitHub: [@HarshGit-tech](https://github.com/HarshGit-tech)

---

## License

This project is licensed under the MIT License.

