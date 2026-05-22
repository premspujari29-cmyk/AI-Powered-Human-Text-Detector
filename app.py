from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import io
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
# GLOBAL VARIABLES
# =====================================================

model = None
vectorizer = None
accuracy = 0

human_count = 0
ai_count = 0
dataset_size = 0

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
animation:floatCard 4s ease-in-out infinite;
}

.small-chart{
width:320px;
height:320px;
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
position:relative;
overflow:hidden;
}

.dataset-card::before{
content:'';
position:absolute;
width:100%;
height:5px;
top:0;
left:0;
background:linear-gradient(to right,#06b6d4,#7c3aed);
}

.dataset-card:hover{
transform:translateY(-8px) scale(1.02);
}

.dataset-card p{
font-size:38px;
font-weight:bold;
margin-top:12px;
background:linear-gradient(to right,#06b6d4,#7c3aed);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

@keyframes floatCard{
0%{
transform:translateY(0px);
}
50%{
transform:translateY(-10px);
}
100%{
transform:translateY(0px);
}
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
width:260px;
height:260px;
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
Train Model
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
<h3>Sentences</h3>
<p id="sentenceCount">0</p>
</div>

<div class="stat-card">
<h3>Read Time</h3>
<p id="readTime">0 Min</p>
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

document.getElementById("purityBar").style.width =
data.ai_score + "%";

updateChart(data.human_score, data.ai_score);

const words = text.trim().split(/\\s+/).length;
const chars = text.length;
const sentences = text.split(/[.!?]+/).length - 1;
const reading = Math.ceil(words / 200);

document.getElementById("wordCount").innerText =
words;

document.getElementById("charCount").innerText =
chars;

document.getElementById("sentenceCount").innerText =
sentences;

document.getElementById("readTime").innerText =
reading + " Min";

const history =
document.getElementById("historyList");

const item =
document.createElement("li");

item.innerText =
data.prediction +
" | AI Score: " +
data.ai_score + "%";

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
color:'white',
padding:20,
font:{
size:15
}
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

document.getElementById("textInput")
.addEventListener("input", function(){

const text = this.value;

const words = text.trim().split(/\\s+/).length;

document.getElementById("wordCount").innerText =
words;

});

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

    global model
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

        human_count = len(data[data["label"] == 0])
        ai_count = len(data[data["label"] == 1])
        dataset_size = len(data)

        X_train, X_test, y_train, y_test = train_test_split(
            data["text"],
            data["label"],
            test_size=0.2,
            random_state=42
        )

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(X_train_vec, y_train)

        prediction = model.predict(X_test_vec)

        accuracy = round(
            accuracy_score(y_test, prediction) * 100,
            2
        )

        return jsonify({

            "message":
            f"Model Trained Successfully | Accuracy: {accuracy}%",

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

    global model
    global vectorizer

    if model is None:

        return jsonify({
            "error":
            "Please Upload And Train Dataset First"
        })

    text = request.form.get("text")

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
