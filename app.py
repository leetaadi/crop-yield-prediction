from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

# Load data to get unique values
df = pd.read_csv(r"C:\Users\Aadi\crop-yield-prediction\data\Crop_production.csv")

# Clean and standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9]', '_', regex=True)

# Extract unique values
states = sorted(df['state_name'].dropna().str.title().unique())
crop_types = sorted(df['crop_type'].dropna().str.title().unique())
crops = sorted(df['crop'].dropna().str.title().unique())

@app.route('/')
def home():
    return render_template('index.html',
                         states=states,
                         crop_types=crop_types,
                         crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map form inputs to EXACT column names used in model training
        input_data = {
            'State_Name': request.form['state_name'].strip().lower(),
            'Crop_Type': request.form['crop_type'].strip().lower(),
            'Crop': request.form['crop'].strip().lower(),
            'N': float(request.form['nitrogen']),
            'P': float(request.form['phosphorus']),
            'K': float(request.form['potassium']),
            'pH': float(request.form['ph']),
            'rainfall': float(request.form['rainfall']),
            'temperature': float(request.form['temperature']),
            'Area_in_hectares': float(request.form['area'])
        }

        # Ensure column order matches training data EXACTLY
        input_df = pd.DataFrame([input_data], columns=[
            'State_Name',
            'Crop_Type', 
            'Crop',
            'N',
            'P',
            'K',
            'pH',
            'rainfall',
            'temperature',
            'Area_in_hectares'
        ])
        
        prediction = model.predict(input_df)[0]
        return render_template('result.html',
                             prediction=round(prediction, 2),
                             crop=request.form['crop'].title())

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)