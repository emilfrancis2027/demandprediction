import pickle
from flask import Flask, request
##creating a flask app and naming it "app"
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
import io, base64, os, json, re, glob
import datetime
from datetime import timedelta
import pandas as pd
#import pydata_google_auth
import numpy as np
from fbprophet import Prophet
#import statsmodels.api as sm
import cv2
from PIL import Image
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure





app = Flask('app')
@app.route('/', methods=['POST'])
def predict():
    csvfile=request.files['file']
    dst = pd.read_csv(csvfile,low_memory=False,parse_dates=['date'],index_col=['date'])
    dataset=dst[dst['item'] == 20]
    dataset.reset_index(level=0, inplace=True)
    dataset = dataset[['date', 'sales']]
    dataset.columns = ["ds", "y"]
    prophet_basic = Prophet()
    prophet_basic.fit(dataset)
    fut= prophet_basic.make_future_dataframe(periods=48,freq='M')
    forecast = prophet_basic.predict(fut)

    #plt.savefig('plo.png')
    #im=Image.open('plo.png')
    #rgb_im = im.convert('RGB')
    #rgb_im.save('plo.jpg','JPEG')
    #img = cv2.imread('plo.jpg')
    #string_img = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

    fig = prophet_basic.plot(forecast)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)