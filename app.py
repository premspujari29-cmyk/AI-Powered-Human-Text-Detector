from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# LOAD DATASET
# =====================================================

try:

    data = pd.read_csv("ai_vs_human_1200_dataset.csv")

    # Ensure correct columns
    data.columns = ["text", "label"]

    # Remove null values
    data = data.dropna()

    # Convert label to integer
    data["label"] = data["label"].astype(int)

    print("Dataset Loaded Successfully")

except Exception as e:

    print("DATASET ERROR:", e)

    # Empty fallback dataframe
    data = pd.DataFrame({
        "text": ["sample text"],
        "label": [0]
    })

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42
)

# =====================================================
# TF-IDF VECTORIZER
# =====================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================================================
# MACHINE LEARNING MODEL
# =====================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_vec, y_train)

# =====================================================
# ACCURACY
# =====================================================

prediction = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, prediction)

print("MODEL ACCURACY:", round(accuracy * 100, 2), "%")

# =====================================================
# HTML PAGE
# =====================================================

HTML_PAGE = """

<!DOCTYPE html>
<html>

<head>

<title>AI Text Detector</title>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

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
animation:bgMove 10s infinite alternate;
z-index:-1;
}

@keyframes bgMove{
0%{background-position:left;}
100%{background-position:right;}
}

.container{
width:90%;
max-width:1100px;
margin:auto;
padding-top:40px;
text-align:center;
}

h1{
font-size:50px;
margin-bottom:20px;
background:linear-gradient(to right,#00d4ff,#7c3aed);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.accuracy{
display:inline-block;
padding:15px 30px;
background:rgba(255,255,255,0.1);
border-radius:15px;
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
resize:none;
margin-bottom:20px;
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
background:rgba(255,255,255,0.08);
padding:30px;
border-radius:20px;
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

.card h3{
margin-bottom:10px;
}

.card p{
font-size:35px;
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

if(text.trim() === ""){
alert("Please enter text");
return;
}

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
# HOME ROUTE
# =====================================================

@app.route("/")
def home():

    return render_template_string(
        HTML_PAGE,
        accuracy=round(accuracy * 100, 2)
    )

# =====================================================
# PREDICT ROUTE
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form.get("text")

    if not text or text.strip() == "":

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
# RUN APP
# =====================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
