from flask import Flask, request, render_template
import os
from model_utils import preprocess_image, predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        image = request.files['file']
        img_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(img_path)
        img_array = preprocess_image(img_path)
        label, confidence = predict(img_array)
        result = f"Prediction: {label}, Confidence: {confidence}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

