from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import re
import csv
import os
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# =====================================================
# LOAD + CLEAN RTF DATASET
# =====================================================

with open("ai_vs_human_1200_dataset.csv", "r", encoding="utf-8", errors="ignore") as file:
    raw_data = file.read()

# Remove RTF formatting
cleaned = re.sub(r'{\\.*?}|\\\\[a-z]+[0-9]* ?', '', raw_data)

# Extract lines
lines = cleaned.splitlines()

dataset_lines = []

for line in lines:

    line = line.strip()

    if "," in line and len(line) > 10:

        # remove ending slash
        line = line.replace("\\", "")

        dataset_lines.append(line)

# Convert into dataframe
csv_data = "\n".join(dataset_lines)

df = pd.read_csv(StringIO(csv_data))

# Rename columns safely
df.columns = ["text", "label"]

# Clean
df = df.dropna()

# Convert label
df["label"] = df["label"].astype(int)

print(df.head())

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =====================================================
# TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=20000,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================================================
# ADVANCED MODEL
# =====================================================

model = LogisticRegression(
    max_iter=1000,
    C=2,
    solver='liblinear'
)

model.fit(X_train_vec, y_train)

# =====================================================
# ACCURACY
# =====================================================

pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, pred)

print("===================================")
print("MODEL TRAINED SUCCESSFULLY")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("===================================")

# =====================================================
# HTML PAGE
# =====================================================

HTML_PAGE = """

<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>AI Detector</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:Arial;
}

body{
background:#050816;
color:white;
overflow-x:hidden;
}

.bg{
position:fixed;
width:100%;
height:100%;
background:linear-gradient(45deg,#0f172a,#111827,#1e293b);
background-size:400% 400%;
animation:bg 12s infinite alternate;
z-index:-1;
}

@keyframes bg{
0%{background-position:left;}
100%{background-position:right;}
}

.container{
width:90%;
max-width:1200px;
margin:auto;
padding-top:40px;
text-align:center;
}

h1{
font-size:55px;
margin-bottom:20px;
background:linear-gradient(to right,#00d4ff,#7c3aed);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.accuracy{
display:inline-block;
padding:15px 30px;
border-radius:15px;
background:rgba(255,255,255,0.1);
margin-bottom:30px;
}

.main{
background:rgba(255,255,255,0.08);
padding:40px;
border-radius:25px;
backdrop-filter:blur(10px);
}

textarea{
width:100%;
height:220px;
border:none;
outline:none;
padding:20px;
border-radius:20px;
background:#0f172a;
color:white;
font-size:18px;
margin-bottom:20px;
resize:none;
}

button{
padding:18px 45px;
border:none;
border-radius:15px;
font-size:18px;
cursor:pointer;
background:linear-gradient(to right,#06b6d4,#7c3aed);
color:white;
transition:0.4s;
}

button:hover{
transform:scale(1.05);
}

.result{
margin-top:40px;
padding:30px;
border-radius:20px;
background:rgba(255,255,255,0.08);
}

.cards{
display:flex;
justify-content:center;
gap:25px;
flex-wrap:wrap;
margin-top:25px;
}

.card{
background:#111827;
padding:30px;
border-radius:20px;
width:220px;
}

.card p{
font-size:35px;
margin-top:10px;
font-weight:bold;
}

.chart{
margin-top:40px;
background:rgba(255,255,255,0.08);
padding:30px;
border-radius:20px;
}

</style>

</head>

<body>

<div class="bg"></div>

<div class="container">

<h1>AI Powered Human Text Detector</h1>

<div class="accuracy">
Model Accuracy : {{accuracy}}%
</div>

<div class="main">

<textarea id="textInput"
placeholder="Paste AI or Human text here..."></textarea>

<button onclick="analyzeText()">
Analyze Text
</button>

</div>

<div class="result">

<h2 id="prediction">
Waiting For Analysis...
</h2>

<div class="cards">

<div class="card">
<h3>Human Score</h3>
<p id="humanScore">0%</p>
</div>

<div class="card">
<h3>AI Score</h3>
<p id="aiScore">0%</p>
</div>

</div>

</div>

<div class="chart">
<canvas id="chart"></canvas>
</div>

</div>

<script>

let chart;

async function analyzeText(){

const text =
document.getElementById("textInput").value;

const formData = new FormData();

formData.append("text", text);

const response = await fetch('/predict',{

method:'POST',
body:formData

});

const data = await response.json();

document.getElementById("prediction").innerText =
data.prediction;

document.getElementById("humanScore").innerText =
data.human_score + "%";

document.getElementById("aiScore").innerText =
data.ai_score + "%";

updateChart(data.human_score, data.ai_score);

}

function updateChart(human, ai){

const ctx =
document.getElementById("chart").getContext("2d");

if(chart){
chart.destroy();
}

chart = new Chart(ctx,{

type:'doughnut',

data:{
labels:['Human','AI'],

datasets:[{
data:[human, ai],

backgroundColor:[
'#06b6d4',
'#7c3aed'
],

borderWidth:0
}]
},

options:{
responsive:true
}

});

}

</script>

</body>
</html>

"""

# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    return render_template_string(
        HTML_PAGE,
        accuracy=round(accuracy * 100, 2)
    )

# =====================================================
# PREDICT
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form.get("text")

    if text.strip() == "":
        return jsonify({
            "prediction": "Please Enter Text",
            "human_score": 0,
            "ai_score": 0
        })

    vectorized_text = vectorizer.transform([text])

    prediction = model.predict(vectorized_text)[0]

    probability = model.predict_proba(vectorized_text)[0]

    human_score = round(probability[0] * 100, 2)
    ai_score = round(probability[1] * 100, 2)

    if prediction == 0:
        result = "Human Written Text"
    else:
        result = "AI Generated Text"

    return jsonify({
        "prediction": result,
        "human_score": human_score,
        "ai_score": ai_score
    })

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )