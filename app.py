from flask import Flask, render_template, request,url_for,redirect
import numpy as np
import pickle
import os
import glob
import re
import sys

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from keras.initializers import glorot_uniform

##############################################################

###############################################################

filename1 = 'cancer-model.pkl'
classifier1 = pickle.load(open(filename1,'rb'))

filename2 = 'diabetes-model.pkl'
classifier2 = pickle.load(open(filename2,'rb'))

filename3 = 'liver-model.pkl'
classifier3 = pickle.load(open(filename3,'rb'))

filename4 = 'heart-model.pkl'
classifier4 = pickle.load(open(filename4,'rb'))

filename5 = 'kidney-model.pkl'
classifier5 = pickle.load(open(filename5,'rb'))

#load the model architecture

model2_path = 'Pnemonia_vgg16.h5'
model1 = load_model(model2_path)
model2 = load_model(model2_path)
model3 = load_model(model2_path)

UPLOAD_FOLDER = 'uploads'
###############################################################

app = Flask(__name__)

###############################################################
app.static_folder = 'static'

@app.route('/')
def home_view():
    return render_template('index.html')


@app.route('/thank_you')
def thank_you():
    return render_template('thanks.html')

@app.route('/cancer_view')
def cancer_view():
    return render_template('breast.html')

@app.route('/cancer_result',methods = ['POST'])
def cancer_result():
    temp_value = list()
    if request.method == 'POST':
        mean_radius = float(request.form['mean_radius']);
        mean_texture = float(request.form['mean_texture']);
        mean_perimeter = float(request.form['mean_perimeter']);
        mean_area = float(request.form['mean_area']);
        mean_smoothness = float(request.form['mean_smoothness']);
        mean_compactness = float(request.form['mean_compactness']);
        mean_concativity = float(request.form['mean_concativity']);
        mean_concave_points = float(request.form['mean_concave_points']);
        mean_symmetry = float(request.form['mean_symmetry']);
        mean_fractal_dimension = float(request.form['mean_fractal_dimension']);
        radius_error = float(request.form['radius_error']);
        texture_error = float(request.form['texture_error']);
        perimeter_error = float(request.form['perimeter_error']);
        area_error = float(request.form['area_error']);
        smoothness_error = float(request.form['smoothness_error']);
        compactness_error = float(request.form['compactness_error']);
        concavity_error  = float(request.form['concavity_error']);
        concave_points_error = float(request.form['concave_points_error']);
        symmetry_error = float(request.form['symmetry_error']);
        fractal_dimension_error = float(request.form['fractal_dimension_error']);
        worst_radius = float(request.form['worst_radius']);
        worst_texture = float(request.form['worst_texture']);
        worst_perimeter = float(request.form['worst_perimeter']);
        worst_area = float(request.form['worst_area']);
        worst_smoothness = float(request.form['worst_smoothness']);
        worst_compactness = float(request.form['worst_compactness']);
        worst_concavity = float(request.form['worst_concavity']);
        worst_concave_points = float(request.form['worst_concave_points']);
        worst_symmetry = float(request.form['worst_symmetry']);
        worst_fractal_dimension = float(request.form['worst_fractal_dimension']);

        temp_value = [mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concativity,mean_concave_points,
                        mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,
                        concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,
                        worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]

        data = np.array([temp_value])

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        data = sc.fit_transform(data)

        my_prediction = int(classifier1.predict(data))

    if my_prediction == 0:
        return render_template('result_page.html',result_text = 'Your report suggests that you are not suffering from breast cancer')
    elif my_prediction == 1:
        return render_template('result_page.html',result_text = 'Unfortunately, your report suggest that you are suffering from breast cancer')

@app.route('/diabetes_view')
def diabetes_view():
    return render_template('diabetes.html')

@app.route('/diabetes_result',methods = ['POST'])
def diabetes_result():
    temp_value = list()
    if request.method == 'POST':
        num_preg = int(request.form['pregnancy'])
        glucose_conc = int(request.form['glucose'])
        blood_pressure = int(request.form['blood_pressure'])
        thickness = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diab_pred = float(request.form['diabetes_pred'])
        age = int(request.form['age'])
        skin = float(request.form['skin'])

        temp_value = [num_preg,glucose_conc,blood_pressure,thickness,insulin,bmi,diab_pred,age,skin]

        data = np.array([temp_value])
        my_prediction = int(classifier2.predict(data))

    if my_prediction == 0:
        return render_template('result_page.html',result_text = 'Your report suggests that you are not suffering diabetes')
    elif my_prediction == 1:
        return render_template('result_page.html',result_text = 'Unfortunately, your report suggest that you are suffering from diabetes')

@app.route('/liver_view')
def liver_view():
    return render_template('liver.html')

@app.route('/liver_result',methods = ['POST'])
def liver_result():
    temp_value = list()
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        total_bilrubin = float(request.form['tbilrubin'])
        direct_bilrubin = float(request.form['dbilrubin'])
        alk_phosphate = int(request.form['alphosphate'])
        alaminotrans = int(request.form['alaminotrans'])
        asaminotrans = int(request.form['asaminotrans'])
        tproteins = float(request.form['tproteins'])
        albumins = float(request.form['albumins'])
        albu_and_globu = float(request.form['albu_and_globu'])

        temp_value = [age,gender,total_bilrubin,direct_bilrubin,alk_phosphate,alaminotrans,asaminotrans,tproteins,albumins,albu_and_globu]

        data = np.array([temp_value])
        my_prediction = int(classifier3.predict(data))

    if my_prediction == 2:
        return render_template('result_page.html',result_text = 'Your report suggests that you are not suffering Liver Cancer')
    elif my_prediction == 1:
        return render_template('result_page.html',result_text = 'Unfortunately, your report suggest that you are suffering from Liver Cancer')

@app.route('/heart_view')
def heart_view():
    return render_template('heart.html')

@app.route('/heart_result',methods = ['POST'])
def heart_result():
    temp_value = list()
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chest_pain = int(request.form['chest_pain'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        temp_value = [age,sex,chest_pain,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

        data = np.array([temp_value])

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        data = sc.fit_transform(data)

        my_prediction = int(classifier4.predict(data))

    if my_prediction == 0:
        return render_template('result_page.html',result_text = 'Your report suggests that you are not suffering from Heart Disease')
    elif my_prediction == 1:
        return render_template('result_page.html',result_text = 'Unfortunately, your report suggest that you are suffering from Heart Disease')

@app.route('/kidney_view')
def kidney_view():
    return render_template('kidney.html')

@app.route('/kidney_result',methods = ['POST'])
def kidney_result():
    temp_value = list()
    if request.method == 'POST':

        age = float(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        albu = float(request.form['albu'])
        sugar = float(request.form['sugar'])
        rbc = float(request.form['rbc'])
        pus = float(request.form['pus'])
        pcc = int(request.form['pcc'])
        bact = int(request.form['bact'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        na = float(request.form['na'])
        ka = float(request.form['ka'])
        haem = float(request.form['haem'])
        pcv = request.form['pcv']
        wbcc =  request.form['wbcc']
        rbcc = request.form['rbcc']
        hyper = int(request.form['hyper'])
        dm = int(request.form['dm'])
        cad = int(request.form['cad'])
        appet = int(request.form['appet'])
        pedal = int(request.form['pedal'])
        anemia = int(request.form['anemia'])
        temp_value = [age,bp,sg,albu,sugar,rbc,pus,pcc,bact,bgr,bu,sc,na,ka,haem,pcv,wbcc,rbcc,hyper,dm,cad,appet,pedal,anemia]
        data = np.array([temp_value])

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        data = sc.fit_transform(data)

        my_prediction = int(classifier5.predict(data))

    if my_prediction == 0:
        return render_template('result_page.html',result_text = 'Your report suggests that you are not suffering from Kidney Disease')
    elif my_prediction == 1:
        return render_template('result_page.html',result_text = 'Unfortunately, your report suggest that you are suffering from Kidney Disease')



def model1_predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))

    x = image.img_to_array(img)
    x = x/(1./255)
    x = np.expand_dims(x,axis = 0)

    x = preprocess_input(x)
    pred = model1.predict(x)
    pred = np.argmax(pred,axis = 1)
    return pred

def model2_predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))

    x = image.img_to_array(img)
    x = x/(1./255)
    x = np.expand_dims(x,axis = 0)

    x = preprocess_input(x)
    pred = model2.predict(x)
    pred = np.argmax(pred,axis = 1)
    return pred

def model3_predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))

    x = image.img_to_array(img)
    x = x/(1./255)
    x = np.expand_dims(x,axis = 0)

    x = preprocess_input(x)
    pred = model3.predict(x)
    pred = np.argmax(pred,axis = 1)
    return pred


@app.route('/malaria_view')
def malaria_view():
    return render_template('malaria.html')


@app.route('/malaria_predict',methods=['POST'])
def malaria_predict():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER,f.filename)
        f.save(file_path)

        pred = model1_predict(file_path)

    if pred == 1:
        return render_template('result_page.html',result_text = 'Your report suggests that you are not suffering from Malaria')
    elif pred == 0:
        return render_template('result_page.html',result_text = 'Unfortunately, your report suggest that you are suffering from Malaria')


@app.route('/pnemonia_view')
def pnemonia_view():
    return render_template('pneumonia.html')



@app.route('/pnemonia_result',methods=['GET','POST'])
def pnemonia_result():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER,f.filename)
        f.save(file_path)

        pred = model2_predict(file_path)

    if pred == 1:
        return render_template('result_page.html',result_text = 'Unfortunately, your report suggest that you are suffering from Pnemonia')
    elif pred == 0:
        return render_template('result_page.html',result_text = 'Your report suggests that you are not suffering from Pnemonia')


@app.route('/brain_view')
def brain_view():
    return render_template('brain.html')


@app.route('/brain_result',methods=['GET','POST'])
def brain_result():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER,f.filename)
        f.save(file_path)

        pred = model3_predict(file_path)

    if pred == 1:
        return render_template('result_page.html',result_text = 'Unfortunately, your report suggest that you are suffering from Brain Tumor')
    elif pred == 0:
        return render_template('result_page.html',result_text = 'Your report suggests that you are not suffering from Brain Tumor')


################################################################

if __name__ == "__main__":
    app.run(debug=True)

################################################################
