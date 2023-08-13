# ForestFireClassifier
# Forest Fire Prediction

This repository contains a Python script for predicting the severity of forest fires using machine learning models. The goal of this project is to demonstrate how different classification algorithms can be used to predict the severity of forest fires based on various environmental factors.

## Table of Contents
- [Dataset Credits](#dataset-credits)
- [Description](#description)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)
## Dataset Credits

**Credits:** Dataset provided by ([NishmaNayana](https://github.com/NishmaNayana)).

## Description

Forest fires can cause significant damage to natural habitats and ecosystems. Predicting the severity of forest fires can help authorities allocate resources effectively and mitigate the impact. This project aims to explore and compare the performance of different machine learning models in predicting forest fire severity.

## Dataset

The project utilizes a dataset containing information about forest fires, including environmental variables such as temperature, humidity, wind speed, and more. The dataset is preprocessed to categorize the severity of burned areas as "low," "medium," or "high."

## Preprocessing

- Categorical features are encoded using one-hot encoding.
- The class imbalance is addressed using the Synthetic Minority Over-sampling Technique (SMOTE).

## Models

The following classification algorithms are used to predict forest fire severity:

- Decision Trees
- Random Forests
- Support Vector Machines

## Evaluation

Each model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Area Under the Receiver Operating Characteristic Curve (AUC-ROC)

## Usage

1. Clone this repository: `git clone https://github.com/your-username/forest-fire-prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the script: `python forest_fire_prediction.py`

Feel free to experiment with different parameters, models, and preprocessing techniques to further improve the performance.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code in accordance with the terms of the license.

