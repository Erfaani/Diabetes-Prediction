# Diabetes Prediction Project

This project is a **ML and DL-based classification task** aimed at predicting diabetes in patients based on their medical and demographic information. It involves exploratory data analysis (EDA), data preprocessing, training machine learning models, and evaluating their performance.

---

## 📁 Dataset

The dataset contains **medical and demographic data** from patients, including:
- **Features**:
  - `age`: Age of the patient.
  - `gender`: Gender of the patient (Male/Female).
  - `hypertension`: Whether the patient has hypertension (1: Yes, 0: No).
  - `heart_disease`: Whether the patient has heart disease (1: Yes, 0: No).
  - `smoking_history`: Smoking status of the patient (e.g., never, current, former).
  - `bmi`: Body Mass Index.
  - `HbA1c_level`: Hemoglobin A1c level (a diabetes-related measure).
  - `blood_glucose_level`: Random blood glucose level.
- **Target**:
  - `diabetes`: Indicates whether the patient has diabetes (1: Yes, 0: No).

Dataset Size: **100,000 rows**  
Source:([kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data))

---

## 🛠️ Tools and Libraries

The following tools and libraries were used in this project:
- **Python** (Programming language)
- **Jupyter Notebook** (Development environment)
- **Pandas** (Data manipulation)
- **Matplotlib & Seaborn** (Data visualization)
- **Scikit-learn** (Machine learning models)
- **TensorFlow/Keras** (Deep learning model)

---

## 🕊️ Steps in the Project

### 1. **Exploratory Data Analysis (EDA)**
- Visualized feature distributions (e.g., age, BMI, HbA1c level).
- Analyzed relationships between features and the target variable (`diabetes`).
- Handled outliers and checked for missing values.

### 2. **Data Preprocessing**
- Encoded categorical features like `gender` and `smoking_history` into numerical values.
- Scaled numerical features for consistency across models.
- Split data into **training** (80%) and **test** (20%) sets.

### 3. **Model Training**
Trained and compared the following models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Neural Network (Deep Learning)**

### 4. **Model Evaluation**
- Evaluated models using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
- Visualized confusion matrices for each model.

---

## 📈 Results

### Model Comparison (Accuracy):
- **Neural Network**: 0.9699
- **Random Forest**: 0.9680
- **SVM**: 0.9627
- **KNN**: 0.9597
- **Logistic Regression**: 0.9579
- **Decision Tree**: 0.9503

---

## 📝 How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/Erfaani/diabetes-prediction.git
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open the notebook and follow the steps for EDA, preprocessing, and model training.

---

## 🚀 Future Work
- Improve feature engineering to enhance model accuracy.
- Explore advanced deep learning architectures.
- Deploy the model as a web application using Flask or Streamlit.

---

## 🙌 Acknowledgements
Special thanks to the creators of the dataset and the open-source tools used in this project.

---

## 📧 Contact
For any questions or feedback, feel free to reach out:

**Email:** [Erfanjouybar@gmail.com]  
**GitHub:** [[Github Link](https://github.com/Erfaani/)]  
**LinkedIn:** [[Linkedin link](https://www.linkedin.com/in/erfanjouybar)]  


