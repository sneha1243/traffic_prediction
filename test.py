from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from flask import Flask, render_template, request, jsonify
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import warnings
import re

app = Flask(__name__, static_url_path='')

n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist', 'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist', 'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds', 'haze', 'proximity thunderstorm', 'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 'fog', 'heavy intensity drizzle', 'shower snow', 'snow', 'thunderstorm with rain', 'thunderstorm with heavy rain', 'thunderstorm with light rain', 'proximity thunderstorm with rain', 'thunderstorm with drizzle', 'smoke', 'thunderstorm', 'proximity shower rain', 'very heavy rain', 'proximity thunderstorm with drizzle', 'light rain and snow', 'light intensity shower rain', 'SQUALLS', 'shower drizzle', 'thunderstorm with light drizzle']

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
regr = MLPRegressor(random_state=1, max_iter=500)

TOMTOM_API_KEY = "QRnzIDmdFd2QtcXKAyTOofUL0xJB4E6z"
TOGETHER_API_KEY = "19d372c5d48189c1e14310bb193e21081a66696ef77f2d042e11e2d3d791e2e4"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/train')
def train():
    data = pd.read_csv('static/Train.csv')
    data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)

    last_n_hours = [1, 2, 3, 4, 5, 6]
    for n in last_n_hours:
        data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)

    data = data.dropna(subset=[f'last_{n}_hour_traffic' for n in last_n_hours]).reset_index(drop=True)
    data['is_holiday'] = data['is_holiday'].apply(lambda x: 0 if x == 'None' or pd.isna(x) else 1)
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['month_day'] = data['date_time'].dt.day
    data['weekday'] = data['date_time'].dt.weekday + 1
    data['month'] = data['date_time'].dt.month
    data['year'] = data['date_time'].dt.year
    data.to_csv("traffic_volume_data.csv", index=None)

    sns.set()
    warnings.filterwarnings('ignore')
    data = pd.read_csv("traffic_volume_data.csv")
    data = data.sample(10000).reset_index(drop=True)
    label_columns = ['weather_type', 'weather_description']
    numeric_columns = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month']

    n1 = data['weather_type']
    n2 = data['weather_description']

    n11 = [(n1features.index(x) + 1) if x in n1features else 0 for x in n1]
    n22 = [(n2features.index(x) + 1) if x in n2features else 0 for x in n2]

    data['weather_type'] = n11
    data['weather_description'] = n22

    features = numeric_columns + label_columns
    target = ['traffic_volume']
    X = data[features]
    y = data[target]

    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y).flatten()

    regr.fit(X, y)

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
    y_pred = regr.predict(testX)

    print('Mean Absolute Error:', mean_absolute_error(testY, y_pred))
    return render_template('index.html', n1features=n1features, n2features=n2features)


@app.route('/predict', methods=['POST'])
def predict():
    ip = []
    ip.append(1 if request.form['isholiday'] == 'yes' else 0)
    ip.append(int(request.form['temperature']))
    ip.append(int(request.form['day']))
    ip.append(int(request.form['time'][:2]))
    D = request.form['date']
    ip.append(int(D[8:]))
    ip.append(int(D[:4]))
    ip.append(int(D[5:7]))

    s1 = request.form.get('x0')
    s2 = request.form.get('x1')
    ip.append((n1features.index(s1)+1) if s1 in n1features else 0)
    ip.append((n2features.index(s2)+1) if s2 in n2features else 0)

    ip = x_scaler.transform([ip])
    out = regr.predict(ip)
    y_pred = y_scaler.inverse_transform([out])

    val = y_pred[0][0]
    if val <= 1000:
        s = "🟢 No Traffic\n\n" \
        "Smooth travel conditions. Proceed with your trip.\n\n" \
        "Best time for delivery services or appointments."
    elif val <= 3000:
        s = "🟠 Normal Traffic\n\n" \
        "Traffic flow is manageable, but expect occasional slowdowns.\n\n" \
        "Stay alert and follow traffic rules.\n\n" \
        "AI chatbot can assist with real-time route tips."
    elif val <= 5500:
        s = "🟡 Medium Traffic\n\n" \
        "Expect some delays. Plan extra time for your journey.\n\n" \
        "Check real-time updates for faster routes using AI chatbot assistance."
    else:
        s = "🔴 Heavy Traffic\n\n" \
        "Avoid travel unless necessary.\n\n" \
        "Use alternate routes: ask AI chatbot.\n\n" \
        "Consider public transport or delaying your trip."
    return render_template('output.html', data1=ip, op=y_pred, statement=s)
def extract_locations(text):
    pattern = re.search(r'from ([\w\s]+) to ([\w\s]+)', text, re.IGNORECASE)
    return pattern.groups() if pattern else (None, None)


@app.route('/tomtom-traffic', methods=['POST'])
def tomtom_traffic():
    src = request.form.get("source")
    dest = request.form.get("destination")
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{src}:{dest}/json?key={TOMTOM_API_KEY}&traffic=true"
    response = requests.get(url)
    data = response.json()
    try:
        summary = data['routes'][0]['summary']
        travel_time = summary['travelTimeInSeconds'] // 60
        traffic_delay = summary['trafficDelayInSeconds'] // 60
        return jsonify({
            "route": f"{src} to {dest}",
            "travel_time_mins": travel_time,
            "delay_mins": traffic_delay
        })
    except:
        return jsonify({"error": "Invalid route or API issue"}), 400
def clean_response(response: str) -> str:
    # Remove <think> ... </think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return response.strip()

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_msg = request.form.get("message", "")
    src, dest = extract_locations(user_msg)
    
    if src and dest:
        tomtom_url = "http://127.0.0.1:5000/tomtom-traffic"
        response = requests.post(tomtom_url, data={"source": src.strip(), "destination": dest.strip()})
        if response.status_code == 200:
            tomtom_data = response.json()
            return jsonify({"reply": f"Live route info:\nRoute: {tomtom_data['route']}\nEstimated Time: {tomtom_data['travel_time_mins']} mins\nDelay due to traffic: {tomtom_data['delay_mins']} mins"})
    if not user_msg.strip():
        return jsonify({"reply": "Please ask something."})

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",  # ✅ Works without server
        "messages": [{"role": "user", "content": user_msg}],
        "temperature": 0.7
    }

    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        data = response.json()
        if "choices" in data and data["choices"]:
            reply = data["choices"][0]["message"]["content"]
        else:
            reply = "No valid reply received from AI."
        clean_responses = clean_response(reply)
        return jsonify({"reply": clean_responses})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)
