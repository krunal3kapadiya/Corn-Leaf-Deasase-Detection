# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:45:44 2023

@author: Krunal
"""

import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, datasets
from PIL import Image


UPLOAD_FOLDER = 'static/uploaded_images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def upload_form():
	return render_template('upload.html')


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploaded_images/' + filename), code=301)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('upload_file',
             #                       filename=filename))
            data_transform = transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.ToTensor(),
            ])
            
            data = datasets.ImageFolder(root="data", transform=data_transform)
            labels_for_viz = {v: k for k, v in data.class_to_idx.items()}
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.load('corn_model')
            new_image= Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            transformed_image = data_transform(new_image)
            new_prediction = model.forward((transformed_image).to(device).unsqueeze(0))
            new_prediction = int(torch.max(new_prediction, 1)[1])
            print("Predicted Class:", labels_for_viz[new_prediction])

            return render_template('upload.html', filename=filename)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
   app.run(debug=True)
