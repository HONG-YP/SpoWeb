# file name : index.py
# pwd : /project_name/app/main/index.py
from flask import Blueprint, request, render_template, flash, redirect, url_for
from flask import current_app as current_app
from matplotlib import pyplot as plt
import re
import math
import pandas as pd
import numpy as np
from . import Predict

main = Blueprint('main', __name__, url_prefix='/')

@main.route('/main', methods = ['GET'])
def index():
    return render_template('/main/project.html')


@main.route('/main/success',  methods=['POST'])
def success():
   if request.method == 'POST':

       Review = request.form['review']
       Genre = request.form['genre']
       Final_Review = Genre + '  ' + Review
       pred  = Predict.isitspo(Genre, Review)
       spo = format(pred[0][0] * 100, ".3f")
       nonspo = 1 - pred[0][0]
       nonspo = format(nonspo * 100, ".3f")

       if float(spo) >= float(nonspo):
           result = '스포 리뷰'
       else:
           result = '논스포 리뷰'

       return render_template('/main/success.html', Final_Review =Final_Review, Review = Review, Genre = Genre, spo = spo, nonspo = nonspo, result = result)
   else:
       pass

