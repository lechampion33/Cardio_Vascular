import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ecg import ECG

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the ECG image
            ecg = ECG()
            ecg_user_image_read = ecg.getImage(filepath)
            ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
            dividing_leads = ecg.DividingLeads(ecg_user_image_read)
            ecg.PreprocessingLeads(dividing_leads)
            ecg.SignalExtraction_Scaling(dividing_leads)
            ecg_1dsignal = ecg.CombineConvert1Dsignal()
            ecg_final = ecg.DimensionalReduction(ecg_1dsignal)
            prediction = ecg.ModelLoad_predict(ecg_final)
            
            return render_template('index.html', 
                                   uploaded_image=filename,
                                   gray_image='Preprossed_Leads_1-12_figure.png',
                                   divided_leads='Leads_1-12_figure.png',
                                   long_lead='Long_Lead_13_figure.png',
                                   preprocessed_leads='Preprossed_Leads_1-12_figure.png',
                                   preprocessed_lead_13='Preprossed_Leads_13_figure.png',
                                   contour_leads='Contour_Leads_1-12_figure.png',
                                   prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

