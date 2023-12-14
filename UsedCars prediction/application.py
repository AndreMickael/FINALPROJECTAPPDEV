from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
cars = pd.read_csv("Used_car_price_cleaned.csv")

@app.route('/')
def index():
    manufactured= sorted(cars['manufactured'].unique(), reverse=True)
    companies = sorted(cars['brand'].unique())
    car_models = sorted(cars['variant'].unique())
    owners = sorted(cars['owner'].unique())
    fuels = sorted(cars['fuel'].unique())
    types= sorted(cars['type'].unique())
    cities = sorted(cars['city'].unique())
    companies.insert(0, 'Select Brand')
    return render_template('index.html', manufactured=manufactured, companies=companies, car_models=car_models, owners=owners,
                           fuels=fuels, types=types, cities=cities)


@app.route('/predict', methods=['POST'])
def predict():
    manufacture = int(request.form.get('manufacture'))
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    owner = request.form.get('owner')
    fuel = request.form.get('fuel')
    type = request.form.get('type')
    city = request.form.get('city')
    kms = int(request.form.get('kms'))

    prediction = model.predict(pd.DataFrame([[manufacture, company, car_model, owner, fuel, type, kms, city]],
                        columns = ['manufactured', 'brand', 'variant', 'owner', 'fuel', 'type', 'kms', 'city']))

    return str(np.round(prediction[0], 2))

if __name__=="__main__":
    app.run(debug=True)