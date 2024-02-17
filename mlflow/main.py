import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from utils import calculate_results
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

max_tokens = 68000
custom_path = "../model"

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.4)
# parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
args = parser.parse_args()


def get_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def preprocess_text_with_line(filename):
    input_lines = get_lines(filename)
    abstract_lines = ""
    abstract_samples = []

    for line in input_lines:
        if line.startswith("###"):
            abstract_id = line
            abstract_lines = ""
        elif line.isspace():
            abstract_line_split = abstract_lines.splitlines()
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t")
                line_data['target'] = target_text_split[0]
                line_data['text'] = target_text_split[1].lower()
                line_data['line_number'] = abstract_line_number
                line_data['total_lines'] = len(abstract_line_split)-1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line
    return abstract_samples


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    train_data = preprocess_text_with_line("../dataset/train.txt")
    val_data = preprocess_text_with_line("../dataset/val.txt")
    test_data = preprocess_text_with_line("../dataset/test.txt")

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    exp = mlflow.set_experiment(experiment_name="experiment_skilite_tf1")
    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    train_sentences = train_df['text'].tolist()
    val_sentences = val_df['text'].tolist()
    test_sentences = test_df['text'].tolist()

    # create text vectorizer
    text_vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=55)

    text_vectorizer.adapt(train_sentences)

    rct_20k_text_vocab = text_vectorizer.get_vocabulary()

    # create token embedding layer
    token_embed = layers.Embedding(
        input_dim=len(rct_20k_text_vocab), output_dim=128, mask_zero=True, name="token_embedding")

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    train_labels_one_hot = one_hot_encoder.fit_transform(train_df['target'].to_numpy().reshape(-1, 1))
    val_labels_one_hot = one_hot_encoder.transform(val_df['target'].to_numpy().reshape(-1, 1))
    test_labels_one_hot = one_hot_encoder.transform(test_df['target'].to_numpy().reshape(-1, 1))

    # turn our data into tensorflow datasets
    train_datasets = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
    valid_datasets = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))
    test_datasets = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df['target'].to_numpy())
    val_labels_encoded = label_encoder.transform(val_df['target'].to_numpy())
    test_labels_encoded = label_encoder.transform(test_df['target'].to_numpy())

    # take the Tensorslice datasets and turn them into prefetch datasets
    train_datasets = train_datasets.batch(32).prefetch(tf.data.AUTOTUNE)
    valid_datasets = valid_datasets.batch(32).prefetch(tf.data.AUTOTUNE)
    test_datasets = test_datasets.batch(32).prefetch(tf.data.AUTOTUNE)

    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_

    with mlflow.start_run(experiment_id=exp.experiment_id, run_name="12"):
        inputs = layers.Input(shape=(1,), dtype=tf.string)
        text_vectors = text_vectorizer(inputs)
        token_embeddings = token_embed(text_vectors)
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model_1 = tf.keras.Model(inputs, outputs)

        # compile
        model_1.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(),
                        metrics=['accuracy'])

        # fit the model
        history_model_1 = model_1.fit(train_datasets, epochs=3, validation_data=valid_datasets)
        evaluation_val = model_1.evaluate(valid_datasets)
        model_1_pred_probs = model_1.predict(valid_datasets)
        model_1_preds = tf.argmax(model_1_pred_probs, axis=1)
        model_1_results = calculate_results(y_true=val_labels_encoded, y_pred=model_1_preds)

        # mlflow.tensorflow.save_model(model_1, "model")
        mlflow.log_metric("precision", model_1_results['precision'])
        mlflow.log_metric("precision", model_1_results['precision'])
        mlflow.log_metric("recall", model_1_results['recall'])
        mlflow.log_metric("f1", model_1_results['f1'])
        mlflow.sklearn.log_model(model_1, "my_model_1")

