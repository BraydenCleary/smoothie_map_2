
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask.ext.wtf import Form
from wtforms import IntegerField, StringField, SubmitField, SelectField, DecimalField
from wtforms.validators import Required, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from datetime import datetime
import pickle

#Initialize Flask App
app = Flask(__name__)

my_model = pickle.load(open('my_model.pkl'))

@app.route('/', methods=['GET', 'POST'])
def model():
  SAN_FRANCISCO_ZIP_CODES = [94102, 94109, 94123, 94117, 94134, 94112, 94124, 94121, 94133, 94116, 94115, 94110, 94127, 94114, 94107, 94132, 94122, 94103, 94105, 94104, 94108, 94118, 94158, 94111, 94131, 94130, 94014, 94129, 94015]

  WEEKEND_DAYS = ['Saturday', 'Sunday']

  EARLY_MORNING   = [5,6,7]
  LATE_MORNING    = [8,9,10]
  EARLY_AFTERNOON = [11,12,13]
  LATE_AFTERNOON  = [14,15,16]
  EARLY_EVENING   = [17,18,19]
  LATE_EVENING    = [20,21,22]
  EARLY_NIGHT     = [23,0,1]
  LATE_NIGHT      = [2,3,4]

  def determine_time_of_day_bucket(datetime):
    hour = datetime.hour
    if hour in EARLY_MORNING:
        return 1
    elif hour in LATE_MORNING:
        return 2
    elif hour in EARLY_AFTERNOON:
        return 3
    elif hour in LATE_AFTERNOON:
        return 4
    elif hour in EARLY_EVENING:
        return 5
    elif hour in LATE_EVENING:
        return 6
    elif hour in EARLY_NIGHT:
        return 7
    elif hour in LATE_NIGHT:
        return 8


  def parse_for_map(datetime):
    zip_predictions = []
    for zip in SAN_FRANCISCO_ZIP_CODES:
      day_of_month        = datetime.day
      time_of_day_bucket  = determine_time_of_day_bucket(datetime)
      month_of_year       = datetime.month
      year                = datetime.year
      DayOfWeek_Friday    = datetime.isoweekday() == 5
      DayOfWeek_Wednesday = datetime.isoweekday() == 3
      DayOfWeek_Tuesday   = datetime.isoweekday() == 2
      DayOfWeek_Thursday  = datetime.isoweekday() == 4
      DayOfWeek_Monday    = datetime.isoweekday() == 1
      DayOfWeek_Saturday  = datetime.isoweekday() == 6
      DayOfWeek_Sunday    = datetime.isoweekday() == 7
      is_weekend          = datetime.isoweekday() in [6,7]
      zip_predictions.append({'zip': zip, 'result': my_model.predict([day_of_month, time_of_day_bucket, month_of_year, year, DayOfWeek_Friday, DayOfWeek_Wednesday, DayOfWeek_Tuesday, DayOfWeek_Thursday, DayOfWeek_Monday, DayOfWeek_Saturday, DayOfWeek_Sunday, is_weekend, zip])[0], 'result_proba': my_model.predict_proba([day_of_month, time_of_day_bucket, month_of_year, year, DayOfWeek_Friday, DayOfWeek_Wednesday, DayOfWeek_Tuesday, DayOfWeek_Thursday, DayOfWeek_Monday, DayOfWeek_Saturday, DayOfWeek_Sunday, is_weekend, zip])[0][0]})
    return zip_predictions

  zip_preds = parse_for_map(datetime.now())

  return render_template('model.html', data=zip_preds)

if __name__ == '__main__':
    app.run(debug=True)
