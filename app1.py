from flask import Flask, request, render_template,url_for,redirect
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    data = [int(request.form['Name']), int(request.form['Purpose']), int(request.form['Completion Year']), int(request.form['Type']), int(request.form['Length (m)']), int(request.form['Max Height above Foundation (m)'])]
    final_features = [np.array(data)]
    print(data)
    print(final_features )
    prediction = model.predict_proba(final_features)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if float(output) > 0.5:
        return render_template('home.html', prediction_text='Your dam is at risk.\nProbability of risk occurrence: {}'.format(output), action_needed="Immediate actions are required.")
    else:
        return render_template('home.html', prediction_text='Your dam is safe.\nProbability of risk occurrence: {}'.format(output), action_needed="Your dam is currently safe.")

if __name__ == '__main__':
    app.run(debug=True)