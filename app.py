from flask import Flask, request, render_template
from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pickle
import json

app = Flask(__name__)
app.secret_key = "secret_key"

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = 'static/files'

model = joblib.load("HouseRentLstm.pkl")

model1 = joblib.load('loan.pkl')

__locations = None
__data_columns = None
model3 = pickle.load(open('banglore_home_prices_model.pickle','rb'))

f = open('columns.json')
__data_columns = json.loads(f.read())['data_columns']
__locations = __data_columns[3:]

def get_estimated_price(input_json):
    try:
        loc_index = __data_columns.index(input_json['location'].lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = input_json['sqft']
    x[1] = input_json['bath']
    x[2] = input_json['bhk']
    if loc_index >= 0:
        x[loc_index] = 1
    result = round(model3.predict([x])[0],2)
    return result

@app.route('/',methods=['GET', 'POST'])
def login():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        
                # If account exists in accounts table in out database
        if username=="admin" and password=="admin":
            # Create session data, we can access this data in other routes
            # Redirect to home page
            return render_template('index.html')
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home1')
def home1():
    return render_template('service.html')

@app.route('/home2')
def home2():
    return render_template('loan.html')

@app.route('/home3')
def home3():
    return render_template('priceindex.html', locations=__locations)


@app.route('/prediction',methods=['GET', 'POST'])
def prediction():
     if request.method == 'POST':
        bhk = int(request.form['bhk'])
        size = int(request.form['size'])
        area_type = int(request.form['area_type'])
        city = int(request.form['city'])
        furnishing_status = int(request.form['furnishing_status'])
        tenant_preferred = int(request.form['tenant_preferred'])
        bathroom = int(request.form['bathroom'])

        # Prepare the feature vector for prediction
        input_features = np.array([[bhk, size, area_type, city, furnishing_status, tenant_preferred, bathroom]])
        input_features = input_features.reshape(input_features.shape[0], input_features.shape[1], 1)

        # Predict the rent
        prediction = model.predict(input_features)
        predicted_rent = prediction[0][0]

        return render_template('service.html', prediction_text='Predicted Rent: ₹ {:.2f}'.format(predicted_rent))



@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Extract features from the form data
    no_of_dependents = int(data['no_of_dependents'])
    education = data['education']
    self_employed = data['self_employed']
    income_annum = float(data['income_annum'])
    loan_amount = float(data['loan_amount'])
    loan_term = int(data['loan_term'])
    cibil_score = int(data['cibil_score'])
    residential_assets_value = float(data['residential_assets_value'])
    commercial_assets_value = float(data['commercial_assets_value'])
    luxury_assets_value = float(data['luxury_assets_value'])
    bank_asset_value = float(data['bank_asset_value'])

    # Encode categorical variables
    education_encoded = 1 if education == 'Graduate' else 0
    self_employed_encoded = 1 if self_employed == 'Yes' else 0

    # Create a feature array
    features = np.array([[no_of_dependents, education_encoded, self_employed_encoded,
                          income_annum, loan_amount, loan_term, cibil_score,
                          residential_assets_value, commercial_assets_value,
                          luxury_assets_value, bank_asset_value]])

    # Make prediction
    prediction = model1.predict(features)

    # Map the prediction to a human-readable form
    loan_status = 'Approved' if prediction[0] == 1 else 'Rejected'

    # Render the result template
    return render_template('loan.html', loan_status=loan_status)


@app.route('/predict2',methods=['POST'])
def predict2():

    if request.method == 'POST':
        input_json = {
            "location": request.form['sLocation'],
            "sqft": request.form['Squareft'],
            "bhk": request.form['uiBHK'],
            "bath": request.form['uiBathrooms']
        }
        result = get_estimated_price(input_json)

        if result > 100:
            result = round(result/100, 2)
            result = str(result) + ' Crore'
        else:
            result = str(result) + ' Lakhs'

    return render_template('priceresult.html',result=result)




@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
