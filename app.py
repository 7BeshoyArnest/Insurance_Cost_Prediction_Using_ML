from flask import Flask, request, jsonify
import joblib
import numpy as np
from flasgger import Swagger

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)

# Load the trained model
model = joblib.load('model_joblib_gbr')

@app.route('/')
def home():
    return " Insurance Cost Prediction API is running! Visit /apidocs for Swagger UI."


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict insurance cost based on customer data
    ---
    tags:
      - Insurance Cost Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            -age
            -sex
            -bmi
            -children
            -smoker
            -region
          properties:
            age:
              type: integer
              example: 29
            sex:
              type: string
              example: "female"
            bmi:  
              type: number
              example: 26.2
            children:
              type: integer
              example: 2            
            smoker:
              type: string
              example: "yes"
            region:
              type: string
              example: "northeast"
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            Insurance_cost:
              type: integer
              example: 0
    """

    data = request.get_json()

    # List of required fields
    required_fields = [
        'age', 'sex', 'bmi', 'children', 'smoker', 'region'
    ]

    # Check for missing fields
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

        # Check for empty or null values
        if data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            return jsonify({"error": f"Field '{field}' cannot be empty"}), 400

    # Type validation
    numeric_fields = [
        'age', 'bmi', 'children'
    ]

    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            return jsonify({"error": f"Field '{field}' must be a number"}), 400

    if not isinstance(data['sex'], str):
        return jsonify({"error": "Field 'Sex' must be a string"}), 400
    if not isinstance(data['smoker'], str):
        return jsonify({"error": "Field 'Smoker' must be a string"}), 400
    if not isinstance(data['region'], str):
        return jsonify({"error": "Field 'Region' must be a string"}), 400

    # Logical range validation
    if data['age'] <= 0 or data['age'] > 120:
        return jsonify({"error": "Age must be between 1 and 120"}), 400
    if data['bmi'] <= 0:
        return jsonify({"error": "Bmi must be a positive number"}), 400
    if data['children'] < 0:
        return jsonify({"error": "Children cannot be negative"}), 400
    
     

    # Validate categorical values
    valid_genders = ['male', 'female']
    smoker_status = ['yes', 'no']
    region_map = {'southwest':1,'southeast':2,'northwest':3,'northeast':4}

    
    
    if data['sex'].lower() not in valid_genders:
        return jsonify({"error": f"Invalid Sex. Must be one of {valid_genders}"}), 400
    if data['smoker'].lower() not in smoker_status:
        return jsonify({"error": f"Invalid Smoker status. Must be one of {smoker_status}"}), 400
    if data['region'].lower() not in region_map:
        return jsonify({"error": f"Invalid Region. Must be one of {region_map}"}), 400
    

    # Encode Gender
    gender = 1 if data['sex'].lower() == 'male' else 0
    # Encode Smoker
    smoker_status = 1 if data['smoker'].lower() == 'yes' else 0

     # Get the region value from the input
    region = data.get('region', '').lower()

    # Apply the same mapping (with error handling)
    region_encoded = region_map[region]
    
    

    # Arrange features exactly as the model was trained
    features = np.array([[
        data['age'],
        gender,    
        data['bmi'],
        data['children'],
        smoker_status,   
        region_encoded
    ]])

    # Make prediction safely
    try:
        prediction = model.predict(features)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    return jsonify({'Insurance Cost is': int(prediction[0])})


if __name__ == "__main__":
    app.run(port=5000, debug=True)







# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model
# model = joblib.load('churn_predict_model')

# @app.route('/')
# def home():
#     return " Bank Customer Churn Prediction API is running!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get JSON data from request
#     data = request.get_json()

#     # One-hot encode Geography (France is baseline → both 0)
#     geo_germany = 1 if data['Geography'].lower() == 'germany' else 0
#     geo_spain = 1 if data['Geography'].lower() == 'spain' else 0

#     # Encode Gender
#     gender_male = 1 if data['Gender'].lower() == 'male' else 0

#     # Arrange features in the same order the model was trained
#     features = np.array([[
#         data['CreditScore'],
#         data['Age'],
#         data['Tenure'],
#         data['Balance'],
#         data['NumOfProducts'],
#         data['HasCrCard'],
#         data['IsActiveMember'],
#         data['EstimatedSalary'],
#         geo_germany,
#         geo_spain,
#         gender_male
#     ]])

#     # Make prediction
#     prediction = model.predict(features)

#     # Return prediction as JSON
#     return jsonify({'Exited': int(prediction[0])})


# if __name__ == "__main__":
#     app.run(port=5000, debug=True)
