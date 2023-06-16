from flask import Flask, render_template, request
import numpy as np
import pickle

filename = './model/model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def hello_world():
    x = float(request.form["years"])
    x = np.array([[x]])
    p = model.predict(x)[0][0].round()
    return render_template('index.html', y=p)

if __name__ == "__main__":
    app.run(debug = False)