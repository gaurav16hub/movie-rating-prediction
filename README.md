Movie Rating Prediction with Python

Project Overview

This project uses Machine Learning to predict movie ratings based on features such as genre, director, actors, duration, year, and votes.

Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

Machine Learning Model

Random Forest Regressor

Model Evaluation

Initial evaluation results:

- MAE: 0.819
- RMSE: 1.094
- R² Score: 0.356

These results are from the current train-test split and may vary if the dataset or model settings change.

Project Files

- "main.py" — data preprocessing, model training, evaluation, and plotting
- "IMDb Movies India.csv" — input dataset
- "prediction_results.csv" — actual and predicted movie ratings
- "actual_vs_predicted.png" — prediction comparison graph
- "requirements.txt" — Python dependencies

How to Run

1. Install Python.
2. Install dependencies using "pip install -r requirements.txt".
3. Place the dataset in the project folder.
4. Run "python main.py".

Future Improvements

- Improve feature engineering.
- Compare different regression algorithms.
- Tune model hyperparameters.
- Evaluate performance using cross-validation.

Author

Gaurav Kumar