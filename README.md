# StackOverflow Tag Predictor

This project, developed as part of a school project at EPF, my engineering school, focuses on creating and deploying a machine learning model to predict tags for StackOverflow posts. It serves as a practical exploration into managing machine learning models in production, highlighting the requirements and rigor needed in Production environment, the significance of unit testing, and the setup of the development environment. The project is complemented by a `requirements.txt` file, ensuring easy setup and reproducibility.

## Project Overview

The project integrates several aspects of machine learning and web development, including data preprocessing, model training, prediction, and a web interface for interaction. It's structured into four main components, each playing a crucial role in the model's workflow from data preparation to user interaction.

### Detailed Description

- **Data Preprocessing**: Utilizes Keras for data preparation, ensuring input data is in the correct format for model training.
- **Model Training**: Involves building and training a deep learning model using TensorFlow and scikit-learn.
- **Prediction**: The model predicts tags based on new input text, making it practical for real-world applications.
- **Flask Application**: Acts as an interface between the model and the user, handling HTTP requests and responses.

Comprehensive testing is included to ensure functionality and reliability.

### Project Goals

- Understanding how to manage ML models in production.
- Learning about the requirements for deploying ML models.
- Emphasizing the importance of unit testing in software development.
- Establishing a robust development environment.

### Project Structure

| Folder        | Description                                           |
|---------------|-------------------------------------------------------|
| App           | The application                                       |
| Preprocessing | Prepare and divide the dataset into different batches |
| Train         | Train our deep learning model                         |
| Predict       | Make predictions based on the previously trained model|

## Usage

**To train the model:**

```bash
python train/train/run.py "train/data/training-data/stackoverflow_posts.csv" "train/conf/train-conf.yml" "train/data/artefacts/"
```

**To launch the Flask application:**
```bash
python predict/predict/app.py
```

To test the API with curl:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model_path": "artefacts", "text": ["How to create a list in Python?", "How to convert a pandas dataframe to a numpy array?"]}' http://localhost:5000/predict
```

