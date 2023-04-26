from flask import Flask, render_template, request
from shift_predictor import ShiftPredictor

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['post'])
def predict():
    # Get the data from the POST request.
    data = request.form.to_dict().values()

    # make prediction

    predictor = ShiftPredictor.get_predictor()

    print(data)
    result = predictor.predict(data)

    shift = result[0] == 'yes'
    result_percent = result[1]
    message = 'You\'re good!' if not shift else 'You suck!'

    return render_template(
        'predict.html',
        shift=shift,
        result_percent="{:.2f}%".format(result_percent * 100),
        message=message,
    )
