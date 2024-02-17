import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = Flask(__name__)

loaded_model = joblib.load('../mlflow/model/data/MLmodel.pkl')
labels = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']


def get_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()


def preprocess_text_with_line_numbers(filename):
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
                line_data['total_lines'] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line
    return abstract_samples


def transform_and_predict(text_to_predict: str) -> str:
    text_to_predict = [text_to_predict]
    loaded_predict_probs = loaded_model.predict(text_to_predict, verbose=1)
    predicted_output = tf.argmax(loaded_predict_probs, axis=1)
    return labels[predicted_output.numpy()[0]]


@app.route('/skilite')
def predict():
    # Get the text data from the request
    logger.info(f'*******************************')
    try:
        logger.info(f'here1: {request.json}')
    except Exception as e:
        logger.info(f'error: {e}')
    data = request.json

    text = data['text']
    logger.info('here2')
    prediction_out = transform_and_predict(text)
    logger.info('here3')

    # Use your NLP model to make predictions
    # Example:
    # prediction = nlp_model.predict(text)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction_out})


if __name__ == '__main__':
    logger.info('here0')
    app.run(debug=True)
