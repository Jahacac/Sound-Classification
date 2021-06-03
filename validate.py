import data_processing as dp
import os
import pandas as pd
import train as tr
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import load_model

# function for converting predictions to labels
def prep_submissions(preds_array):
    preds_df = pd.DataFrame(preds_array)
    predicted_labels = preds_df.idxmax(axis=1)  # convert back one hot encoding to categorical variabless
    return predicted_labels


# function to draw confusion matrix
def draw_confusion_matrix(true, preds, labels):
    title = f"{tr.model_name} - Confusion Matrix"
    plt.figure(figsize=(20, 20))
    conf_matx = confusion_matrix(true, preds)
    sns.heatmap(conf_matx, annot=True, annot_kws={"size": 12}, fmt='g', cbar=False, cmap="viridis", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title(title, fontdict={"fontsize": 36})
    plt.savefig(os.path.join(dp.image_path, title))
    plt.show()


def validate_model(model_filename):
    # load validation dataset
    validation = dp.load_dataset_from_json(dp.validation_dataset_path_json, tr.model_feature)
    validation_data = []
    validation_labels = []

    # prepare data for prediction
    for data in validation:
        features = data[tr.feature_key]
        features = np.array(features)
        validation_data.append(features)
        validation_labels.append(data["label"])

    # read labels from file
    labels = dp.get_labels()

    # shape data for model
    validation_data = np.array(validation_data)
    validation_data = validation_data.reshape(validation_data.shape[0], validation_data.shape[1], validation_data.shape[2], 1)
    validation_labels = tr.shape_labels(validation_labels, labels)
    validation_labels = tr.to_categorical(validation_labels, len(labels))

    # open trained model
    model = load_model(os.path.join(dp.trained_model_path, f'{model_filename}.h5'))

    # Preparing data for Classification report / Confusion matrix
    validation_preds = model.predict(validation_data)
    validation_preds_labels = prep_submissions(validation_preds)
    validation_labels = prep_submissions(validation_labels)

    # Classification report
    print(classification_report(validation_labels, validation_preds_labels))

    # Confusion matrix
    draw_confusion_matrix(validation_labels, validation_preds_labels, labels)


validate_model(tr.model_name)
