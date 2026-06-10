from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the logistic regression model safely
MODEL_PATH = 'logitic_pkl.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Could not find model file at {MODEL_PATH}. Ensure it is placed in the root folder.")

# Single-file HTML layout utilizing Tailwind CSS & FontAwesome
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Failure Risk Prediction Tool</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-50 font-sans min-h-screen flex flex-col justify-between">

    <header class="bg-gradient-to-r from-rose-600 to-red-700 text-white shadow-md py-4 px-6">
        <div class="max-w-6xl mx-auto flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <i class="fa-solid fa-heart-pulse text-3xl animate-pulse"></i>
                <h1 class="text-2xl font-bold tracking-tight">CardioRisk AI Portal</h1>
            </div>
            <p class="text-xs bg-red-800/40 px-3 py-1 rounded-full border border-red-400/30">Logistic Regression Engine</p>
        </div>
    </header>

    <main class="flex-grow max-w-4xl w-full mx-auto p-4 md:p-8">
        
        <div class="bg-white rounded-2xl shadow-xs border border-slate-100 p-6 mb-8">
            <h2 class="text-xl font-semibold text-slate-800 mb-2">Clinical Decision Support Tool</h2>
            <p class="text-slate-600 text-sm leading-relaxed">
                Provide the patient's continuous physical stats and specific lab parameters to calculate survival indicators. This deployment features categorical conversion from raw probability scores.
            </p>
        </div>

        {% if prediction_text %}
        <div class="mb-8 rounded-2xl p-6 shadow-xs border transition-all duration-300
            {% if result_class == 'success' %} bg-emerald-50 border-emerald-200 text-emerald-900 {% elif result_class == 'danger' %} bg-rose-50 border-rose-200 text-rose-900 {% else %} bg-amber-50 border-amber-200 text-amber-900 {% endif %}">
            <div class="flex items-start space-x-4">
                <div class="p-3 rounded-xl 
                    {% if result_class == 'success' %} bg-emerald-500 text-white {% elif result_class == 'danger' %} bg-rose-500 text-white {% else %} bg-amber-500 text-white {% endif %}">
                    {% if result_class == 'success' %}
                        <i class="fa-solid fa-circle-check text-xl"></i>
                    {% elif result_class == 'danger' %}
                        <i class="fa-solid fa-circle-exclamation text-xl"></i>
                    {% else %}
                        <i class="fa-solid fa-triangle-exclamation text-xl"></i>
                    {% endif %}
                </div>
                <div>
                    <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Model Assessment</h3>
                    <p class="text-2xl font-bold mb-1">{{ prediction_text }}</p>
                    {% if probability_text %}
                        <p class="text-sm opacity-90 font-medium"><i class="fa-solid fa-chart-pie mr-1"></i> {{ probability_text }}</p>
                    {% endif %}
                </div>
            </div>
        </div>
        {% endif %}

        <form method="POST" action="/" class="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 md:p-8 space-y-8">
            
            <div>
                <h3 class="text-md font-semibold text-slate-700 border-b border-slate-100 pb-2 mb-4 flex items-center">
                    <i class="fa-solid fa-user-md text-rose-500 mr-2"></i> Patient Baseline Info
                </h3>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Age (Years)</label>
                        <input type="number" step="any" name="age" value="{{ inputs.get('age', '60') }}" required
                            class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Biological Sex</label>
                        <select name="sex" required class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                            <option value="1" {% if inputs.get('sex') == '1' %}selected{% endif %}>Male</option>
                            <option value="0" {% if inputs.get('sex') == '0' %}selected{% endif %}>Female</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Follow-up Period (Days)</label>
                        <input type="number" step="any" name="time" value="{{ inputs.get('time', '100') }}" required
                            class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                    </div>
                </div>
            </div>

            <div>
                <h3 class="text-md font-semibold text-slate-700 border-b border-slate-100 pb-2 mb-4 flex items-center">
                    <i class="fa-solid fa-notes-medical text-rose-500 mr-2"></i> Clinical Comorbidities
                </h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Anaemia</label>
                        <select name="anaemia" required class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                            <option value="0" {% if inputs.get('anaemia') == '0' %}selected{% endif %}>No</option>
                            <option value="1" {% if inputs.get('anaemia') == '1' %}selected{% endif %}>Yes</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Diabetes</label>
                        <select name="diabetes" required class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                            <option value="0" {% if inputs.get('diabetes') == '0' %}selected{% endif %}>No</option>
                            <option value="1" {% if inputs.get('diabetes') == '1' %}selected{% endif %}>Yes</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">High Blood Pressure</label>
                        <select name="high_blood_pressure" required class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                            <option value="0" {% if inputs.get('high_blood_pressure') == '0' %}selected{% endif %}>No</option>
                            <option value="1" {% if inputs.get('high_blood_pressure') == '1' %}selected{% endif %}>Yes</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Smoking History</label>
                        <select name="smoking" required class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                            <option value="0" {% if inputs.get('smoking') == '0' %}selected{% endif %}>No</option>
                            <option value="1" {% if inputs.get('smoking') == '1' %}selected{% endif %}>Yes</option>
                        </select>
                    </div>
                </div>
            </div>

            <div>
                <h3 class="text-md font-semibold text-slate-700 border-b border-slate-100 pb-2 mb-4 flex items-center">
                    <i class="fa-solid fa-flask text-rose-500 mr-2"></i> Biochemical Metrics
                </h3>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">CPK Enzyme (mcg/L)</label>
                        <input type="number" step="any" name="creatinine_phosphokinase" value="{{ inputs.get('creatinine_phosphokinase', '250') }}" required
                            class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Ejection Fraction (%)</label>
                        <input type="number" step="any" name="ejection_fraction" value="{{ inputs.get('ejection_fraction', '38') }}" required
                            class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Platelets (cells/mL)</label>
                        <input type="number" step="any" name="platelets" value="{{ inputs.get('platelets', '250000') }}" required
                            class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Serum Creatinine (mg/dL)</label>
                        <input type="number" step="any" name="serum_creatinine" value="{{ inputs.get('serum_creatinine', '1.1') }}" required
                            class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Serum Sodium (mEq/L)</label>
                        <input type="number" step="any" name="serum_sodium" value="{{ inputs.get('serum_sodium', '137') }}" required
                            class="w-full rounded-lg border border-slate-200 p-2.5 text-sm focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition">
                    </div>
                </div>
            </div>

            <div class="pt-4 flex items-center justify-end space-x-4 border-t border-slate-100">
                <a href="/" class="px-5 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg font-medium text-sm transition">Reset</a>
                <button type="submit" class="px-6 py-2.5 bg-rose-600 hover:bg-rose-700 text-white rounded-lg font-medium text-sm shadow-md shadow-rose-200 transition flex items-center cursor-pointer">
                    <i class="fa-solid fa-calculator mr-2"></i> Evaluate Risk Profiles
                </button>
            </div>
        </form>
    </main>

    <footer class="bg-slate-900 text-slate-400 py-4 text-center text-xs border-t border-slate-800 mt-12">
        <div class="max-w-6xl mx-auto px-4">
            <p>&copy; 2026 CardioRisk AI Engine. Built securely using Scikit-Learn 1.6.1 model architecture.</p>
        </div>
    </footer>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    probability_text = None
    result_class = None
    inputs = {}

    if request.method == 'POST':
        try:
            # Explicit feature ordering to feed the model
            feature_names = [
                'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                'ejection_fraction', 'high_blood_pressure', 'platelets', 
                'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
            ]
            
            # Map form input into dictionary data structure
            form_data = {}
            for feature in feature_names:
                form_data[feature] = float(request.form[feature])
                inputs[feature] = request.form[feature] # Persists context state in the UI

            # Standardized DataFrame object matching exact training layout
            input_df = pd.DataFrame([form_data])
            
            # Execute Model Class Estimations
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            # Categorical prediction transformations
            if pred == 1:
                prediction_text = "High Risk / Mortality Predicted"
                result_class = "danger"
            else:
                prediction_text = "Low Risk / Survival Predicted"
                result_class = "success"
                
            probability_text = f"{prob * 100:.2f}% Risk of Mortality (vs. {(1-prob)*100:.2f}% Chance of Survival)"
            
        except Exception as e:
            prediction_text = f"Error evaluating inputs: {str(e)}"
            result_class = "error"

    # Serve the page layout straight from the string object variable
    return render_template_string(HTML_LAYOUT, 
                                  prediction_text=prediction_text, 
                                  probability_text=probability_text, 
                                  result_class=result_class,
                                  inputs=inputs)

if __name__ == '__main__':
    app.run(debug=True)
