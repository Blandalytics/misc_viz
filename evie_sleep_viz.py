import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import date, timedelta
import xgboost as xgb
import plotly.graph_objects as go
import time

sleep_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1tYSDr-8iMYPb9PpXWA48-NjNJuQpWIJeEO8CmI83Pmg/export?format=csv&gid=0').astype({
    'Start Time':'int',
    'Overnight':'int'
})
sleep_df[['Start Date','End Date']] = sleep_df[['Start Date','End Date']].apply(pd.to_datetime)

# Only use days that have an overnight recording
# Otherwise backfilling will return crazy long sleep duration
sleep_df = sleep_df.groupby('Start Date').filter(lambda x: x['Overnight'].max()>0)[['Start Date','Start Time','End Date','End Time','Overnight']]

# Create a df of all possible minutes across the date ranges
time_df = pd.DataFrame()
for date in sleep_df['Start Date'].unique():
  temp_df = pd.DataFrame(range(0,1440), columns=['time'])
  temp_df['date'] = date
  time_df = pd.concat([time_df,
                       temp_df])

# Merge in the start and end times for sleep
time_df = time_df.reset_index(drop=True).merge(sleep_df[['Start Date','Start Time']],how='left', left_on=['date','time'],right_on=['Start Date','Start Time'])
time_df = time_df.merge(sleep_df[['End Date','End Time']],how='left', left_on=['date','time'],right_on=['End Date','End Time']).sort_values(['date','time'],ascending=False)

# Combine sleep start and end
time_df['asleep'] = time_df['Start Time']
time_df.loc[time_df['asleep']>1,'asleep'] = 1
time_df['asleep'] = time_df['asleep'].fillna(time_df['End Time'])
time_df.loc[time_df['asleep']>1,'asleep'] = 0

# Backfill sleep start/end
time_df['asleep'] = time_df['asleep'].fillna(method='bfill').fillna(1).astype('int')

# Weigh the df by how recent the date is (how many weeks ago)
time_df['weight'] = (time_df['date'].max() - time_df['date']) / np.timedelta64(1, 'D') / 7
time_df['weight'] = time_df['weight'].astype('int').max() - time_df['weight'].astype('int') + 1
time_df = time_df.loc[time_df.index.repeat(time_df.weight)][['time','asleep']].reset_index(drop=True)

# Streamlit needs a workaround for xgboost model training
@st.cache_resource()
def train_and_predict_class(xtrain, ytrain):
    try:
        model = xgb.XGBClassifier(objective='binary:logistic')
        model.fit(xtrain,ytrain)
        return model
    except ValueError as er:
        st.error(er)

train_and_predict_class(time_df['time'],time_df['asleep'])
        
pred_df = pd.DataFrame(range(0,1440), columns=['time'])
pred_df[['pred_awake','pred_asleep']] = model.predict_proba(pred_df['time'])
pred_df = pred_df.sort_values('time')
pred_df = pd.concat([pred_df.drop(columns=['time']),
                     pred_df])
pred_df['pred_asleep'] = pred_df['pred_asleep'].rolling(60,min_periods=1).mean()
pred_df['pred_awake'] = 1 - pred_df['pred_asleep']
pred_df = pred_df.dropna(subset='time')
pred_df['time_label'] = pred_df['time'].apply(lambda x: time.strftime("%M:%S", time.gmtime(x)))

fig = go.Figure([go.Bar(y=pred_df['time_label'], x=pred_df.assign(filler = 1)['filler'],
                        hovertemplate = '<b>%{y}</b><br>Likelihood: %{customdata:.1f}%<br><extra></extra>',
                        marker=dict(color=pred_df['pred_asleep'],
                                    colorscale=['#a9373b', '#faf5f5', '#2369bd'],
                                    cmax=0.8,
                                    cmin=0.2,
                                    line=dict(width=0)),
                        width=2,
                        customdata=pred_df['pred_asleep'].mul(100),
                        orientation='h'
             )])
fig.update_yaxes(tickvals=[x*60 for x in range(0,24)])
fig.update_xaxes(range=[0,1],
                 showticklabels=False)

fig.update_layout(
    title={
        'text': "Likelihood of Evie Being Asleep, by Hour",
        'y':0.975,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font=dict(
        size=18
    ),
    autosize=False,
    width=550,
    height=800,
    margin=dict(
        l=75,
        r=25,
        b=10,
        t=50,
        pad=0
    ),
)

st.plotly_chart(fig)
