from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import io
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# GLOBAL VARIABLES
# =====================================================

model1 = None
model2 = None
vectorizer = None

accuracy = 0

human_count = 0
ai_count = 0
dataset_size = 0

# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)

    text = re.sub(r"\\s+", " ", text).strip()

    return text

# =====================================================
# HTML PAGE
# =====================================================

HTML_PAGE = """

<!DOCTYPE html>
<html>

<head>

<title>AI Powered Human Text Detector</title>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script src="https://cdn.jsdelivr.net/npm/tsparticles@2/tsparticles.bundle.min.js"></script>

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

#tsparticles{
position:fixed;
width:100%;
height:100%;
z-index:-2;
top:0;
left:0;
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
0%{
background-position:left;
}
100%{
background-position:right;
}
}

.container{
width:90%;
max-width:1300px;
margin:auto;
padding-top:40px;
padding-bottom:50px;
text-align:center;
}

h1{
font-size:58px;
margin-bottom:20px;
background:linear-gradient(to right,#00d4ff,#7c3aed);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.upload-box{
background:rgba(255,255,255,0.08);
padding:40px;
border-radius:25px;
margin-bottom:30px;
backdrop-filter:blur(10px);
box-shadow:0 0 30px rgba(0,0,0,0.3);
transition:0.4s;
}

.upload-box:hover{
transform:translateY(-5px);
}

input[type=file]{
width:100%;
padding:18px;
border-radius:15px;
background:#111827;
color:white;
border:none;
margin-bottom:20px;
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
margin-top:20px;
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
font-weight:bold;
}

button:hover{
transform:scale(1.05);
box-shadow:0 0 25px #7c3aed;
}

.status{
margin-top:20px;
font-size:20px;
font-weight:bold;
}

.result{
margin-top:30px;
background:rgba(255,255,255,0.08);
padding:30px;
border-radius:20px;
backdrop-filter:blur(10px);
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
transition:0.4s;
}

.card:hover{
transform:translateY(-10px);
}

.card p{
font-size:35px;
font-weight:bold;
margin-top:10px;
}

.stats{
display:flex;
justify-content:center;
gap:20px;
margin-top:30px;
flex-wrap:wrap;
}

.stat-card{
background:#111827;
padding:25px;
border-radius:18px;
width:180px;
transition:0.4s;
}

.stat-card:hover{
transform:translateY(-8px);
}

.stat-card p{
font-size:28px;
margin-top:10px;
font-weight:bold;
}

.progress-container{
margin-top:30px;
}

.progress-title{
margin-bottom:12px;
font-size:20px;
}

.progress-bar{
width:100%;
height:25px;
background:#1e293b;
border-radius:20px;
overflow:hidden;
}

.progress-fill{
height:100%;
width:0%;
background:linear-gradient(to right,#06b6d4,#7c3aed);
transition:1s;
}

.loader{
width:60px;
height:60px;
border:6px solid rgba(255,255,255,0.2);
border-top:6px solid #7c3aed;
border-radius:50%;
margin:20px auto;
display:none;
animation:spin 1s linear infinite;
}

@keyframes spin{
100%{
transform:rotate(360deg);
}
}

.chart-container{
margin-top:40px;
display:grid;
grid-template-columns:1fr 1fr;
gap:25px;
align-items:center;
}

.chart-card{
background:rgba(255,255,255,0.08);
padding:30px;
border-radius:25px;
backdrop-filter:blur(10px);
}

.small-chart{
width:300px;
height:300px;
margin:auto;
}

.dataset-stats{
display:flex;
flex-direction:column;
gap:20px;
}

.dataset-card{
background:rgba(255,255,255,0.08);
padding:30px;
border-radius:20px;
backdrop-filter:blur(10px);
transition:0.4s;
}

.dataset-card:hover{
transform:translateY(-8px);
}

.dataset-card p{
font-size:38px;
font-weight:bold;
margin-top:12px;
background:linear-gradient(to right,#06b6d4,#7c3aed);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.history-box{
margin-top:40px;
background:rgba(255,255,255,0.08);
padding:30px;
border-radius:20px;
text-align:left;
}

.history-box ul{
list-style:none;
margin-top:20px;
}

.history-box li{
padding:15px;
margin-bottom:12px;
background:#111827;
border-radius:12px;
}

@media(max-width:900px){

.chart-container{
grid-template-columns:1fr;
}

.small-chart{
width:240px;
height:240px;
}

}

</style>

</head>

<body>

<div id="tsparticles"></div>

<div class="bg"></div>

<div class="container">

<h1>AI Powered Human Text Detector</h1>

<div class="upload-box">

<h2>Upload Dataset CSV</h2>

<input type="file" id="datasetFile">

<button onclick="uploadDataset()">
Train Advanced Model
</button>

<div class="status" id="trainStatus">
No Dataset Uploaded
</div>

</div>

<div class="upload-box">

<h2>Paste Text For Detection</h2>

<textarea id="textInput"
placeholder="Paste AI or Human text here..."></textarea>

<button onclick="analyzeText()">
Analyze Text
</button>

<div id="loader" class="loader"></div>

</div>

<div class="result">

<h2 id="prediction">
Waiting For Analysis...
</h2>

<div class="progress-container">

<div class="progress-title">
AI Purity Level
</div>

<div class="progress-bar">

<div class="progress-fill" id="purityBar">
</div>

</div>

</div>

<div class="cards">

<div class="card">
<h3>Human Score</h3>
<p id="humanScore">0%</p>
</div>

<div class="card">
<h3>AI Score</h3>
<p id="aiScore">0%</p>
</div>

<div class="card">
<h3>Confidence</h3>
<p id="confidence">0%</p>
</div>

</div>

<div class="stats">

<div class="stat-card">
<h3>Words</h3>
<p id="wordCount">0</p>
</div>

<div class="stat-card">
<h3>Characters</h3>
<p id="charCount">0</p>
</div>

<div class="stat-card">
<h3>Diversity</h3>
<p id="diversity">0</p>
</div>

<div class="stat-card">
<h3>Avg Word</h3>
<p id="avgWord">0</p>
</div>

</div>

</div>

<div class="chart-container">

<div class="chart-card">

<h2>AI vs Human Comparison</h2>

<div class="small-chart">
<canvas id="chart"></canvas>
</div>

</div>

<div class="dataset-stats">

<div class="dataset-card">
<h3>Total Human Text</h3>
<p id="datasetHuman">0</p>
</div>

<div class="dataset-card">
<h3>Total AI Text</h3>
<p id="datasetAI">0</p>
</div>

<div class="dataset-card">
<h3>Dataset Size</h3>
<p id="datasetSize">0</p>
</div>

</div>

</div>

<div class="history-box">

<h2>Recent Analysis</h2>

<ul id="historyList"></ul>

</div>

</div>

<script>

let chart;

tsParticles.load("tsparticles",{

particles:{

number:{
value:60
},

color:{
value:"#7c3aed"
},

links:{
enable:true,
distance:120,
color:"#06b6d4",
opacity:0.2
},

move:{
enable:true,
speed:1
},

size:{
value:2
},

opacity:{
value:0.5
}

}

});

async function uploadDataset(){

const fileInput =
document.getElementById("datasetFile");

const file = fileInput.files[0];

if(!file){
alert("Please upload CSV dataset");
return;
}

const formData = new FormData();

formData.append("file", file);

document.getElementById("trainStatus").innerText =
"Training Advanced AI Model...";

const response = await fetch('/train',{

method:'POST',
body:formData

});

const data = await response.json();

document.getElementById("trainStatus").innerText =
data.message;

document.getElementById("datasetHuman").innerText =
data.human_count;

document.getElementById("datasetAI").innerText =
data.ai_count;

document.getElementById("datasetSize").innerText =
data.dataset_size;

}

async function analyzeText(){

const text =
document.getElementById("textInput").value;

if(text.trim() === ""){
alert("Please enter text");
return;
}

document.getElementById("loader").style.display =
"block";

const formData = new FormData();

formData.append("text", text);

const response = await fetch('/predict',{

method:'POST',
body:formData

});

const data = await response.json();

document.getElementById("loader").style.display =
"none";

if(data.error){
alert(data.error);
return;
}

document.getElementById("prediction").innerText =
data.prediction;

document.getElementById("humanScore").innerText =
data.human_score + "%";

document.getElementById("aiScore").innerText =
data.ai_score + "%";

document.getElementById("confidence").innerText =
data.confidence + "%";

document.getElementById("diversity").innerText =
data.lexical_diversity;

document.getElementById("avgWord").innerText =
data.avg_word_length;

document.getElementById("purityBar").style.width =
data.ai_score + "%";

updateChart(data.human_score, data.ai_score);

const words = text.trim().split(/\\s+/).length;
const chars = text.length;

document.getElementById("wordCount").innerText =
words;

document.getElementById("charCount").innerText =
chars;

const history =
document.getElementById("historyList");

const item =
document.createElement("li");

item.innerText =
data.prediction +
" | Confidence: " +
data.confidence + "%";

history.prepend(item);

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

hoverOffset:15,

borderWidth:2,

borderColor:'#050816'

}]

},

options:{

responsive:true,

cutout:'75%',

plugins:{

legend:{
position:'bottom',

labels:{
color:'white'
}
}

},

animation:{
animateRotate:true,
duration:2000
}

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
    return render_template_string(HTML_PAGE)

# =====================================================
# TRAIN MODEL
# =====================================================

@app.route("/train", methods=["POST"])
def train():

    global model1
    global model2
    global vectorizer
    global accuracy
    global human_count
    global ai_count
    global dataset_size

    if "file" not in request.files:

        return jsonify({
            "message": "Please Upload CSV File"
        })

    file = request.files["file"]

    try:

        data = pd.read_csv(
            io.StringIO(
                file.stream.read().decode(
                    "utf-8",
                    errors="ignore"
                )
            ),
            sep=",",
            engine="python",
            on_bad_lines="skip"
        )

        data = data.iloc[:, :2]

        data.columns = ["text", "label"]

        data = data.dropna()

        data["label"] = pd.to_numeric(
            data["label"],
            errors="coerce"
        )

        data = data.dropna()

        data["label"] = data["label"].astype(int)

        data["text"] = data["text"].apply(clean_text)

        human_count = len(data[data["label"] == 0])
        ai_count = len(data[data["label"] == 1])
        dataset_size = len(data)

        X_train, X_test, y_train, y_test = train_test_split(

            data["text"],
            data["label"],

            test_size=0.2,

            random_state=42,

            stratify=data["label"]

        )

        vectorizer = TfidfVectorizer(

            stop_words="english",

            ngram_range=(1,3),

            max_features=30000,

            sublinear_tf=True,

            min_df=2,

            max_df=0.9

        )

        X_train_vec = vectorizer.fit_transform(X_train)

        X_test_vec = vectorizer.transform(X_test)

        model1 = PassiveAggressiveClassifier(

            max_iter=2000,

            C=0.5,

            random_state=42

        )

        model2 = MultinomialNB()

        model1.fit(X_train_vec, y_train)

        model2.fit(X_train_vec, y_train)

        pred1 = model1.predict(X_test_vec)

        pred2 = model2.predict(X_test_vec)

        final_pred = []

        for p1, p2 in zip(pred1, pred2):

            if p1 == p2:
                final_pred.append(p1)
            else:
                final_pred.append(p1)

        accuracy = round(

            accuracy_score(y_test, final_pred) * 100,

            2

        )

        return jsonify({

            "message":
            f"Advanced AI Model Trained | Accuracy: {accuracy}%",

            "accuracy":accuracy,

            "human_count":human_count,

            "ai_count":ai_count,

            "dataset_size":dataset_size

        })

    except Exception as e:

        return jsonify({
            "message": f"Error: {str(e)}"
        })

# =====================================================
# PREDICT
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    global model1
    global model2
    global vectorizer

    if model1 is None:

        return jsonify({
            "error":
            "Please Upload And Train Dataset First"
        })

    text = request.form.get("text")

    cleaned_text = clean_text(text)

    vectorized_text = vectorizer.transform([cleaned_text])

    pred1 = model1.predict(vectorized_text)[0]

    pred2 = model2.predict(vectorized_text)[0]

    if pred1 == pred2:
        prediction = pred1
    else:
        prediction = pred1

    score1 = model1.decision_function(vectorized_text)[0]

    confidence = round(
        min(abs(score1) * 15, 99),
        2
    )

    if prediction == 1:

        ai_score = confidence

        human_score = round(100 - ai_score, 2)

        result = "AI Generated Text"

    else:

        human_score = confidence

        ai_score = round(100 - human_score, 2)

        result = "Human Written Text"

    words = len(text.split())

    chars = len(text)

    avg_word_length = round(

        np.mean([len(word) for word in text.split()]),

        2

    )

    unique_words = len(set(text.split()))

    lexical_diversity = round(

        unique_words / max(words, 1),

        2

    )

    return jsonify({

        "prediction": result,

        "human_score": human_score,

        "ai_score": ai_score,

        "confidence": confidence,

        "words": words,

        "characters": chars,

        "avg_word_length": avg_word_length,

        "lexical_diversity": lexical_diversity

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
