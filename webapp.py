# streamlit webapp

import streamlit as st

import os
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

#from minizinc import Instance, Model, Solver
import pymzn
import pydeck as pdk

# Define geographic locations

location_data = {'name':['Mannheim', 'Karlsruhe', 'Baden_Baden', 'Buehl', 'Offenburg', 'Lahr_Schwarzwald', 'Loerrach','Heidelberg','Freiburg_im_Breisgau'],
'latitude':[49.50004032,49.0158491,48.75732995,48.69282575,48.4747585,48.33432035,48.33432035,49.4057284,47.98731115],
'longitude':[8.50207514,8.40953385,8.21440805,8.14527,7.94506255,7.88272955,7.88272955,8.68361415,7.79642005]}

location = pd.DataFrame(location_data)

# list of paths

path_data = {'name': ['Mannheim to Karlsruhe'],
            'color': ['faa61a'],
            'path': [[[49.50004032, 8.50207514], [49.0158491, 8.40953385]]]}

path_df = pd.DataFrame(path_data)


@st.cache(persist=True)
def load_model(name):

    return name


@st.cache(persist=True)
def load_data(name):

    return name


st.set_page_config(
    page_title="ISR Collection Management App",
    page_icon="🛃",
    layout="wide",
    initial_sidebar_state="collapsed", #expanded, collapsed
    )

st.title('Automation of ISR Collection Management')
st.markdown('Exploration Dashboard')

# build the sidebar
mpath = './models'
mfiles = os.listdir(mpath)
models = []

for f in mfiles:
    name, ext = os.path.splitext(f)
    if ext == '.mzn':
        models.append(f)

dpath = './data'
dfiles = os.listdir(dpath)
data_files = []

for f in dfiles:
    name, ext = os.path.splitext(f)
    if ext == '.dzn':
        data_files.append(f)

# add selectors for models and data  in the sidebar

current_model = st.sidebar.selectbox(
    "Available Minizinc Models",
    models)

current_data = st.sidebar.selectbox(
    "Available Minizinc Data",
    data_files)

current_solver = st.sidebar.selectbox(
    "Available Minizinc Solvers",
    ('gecode', 'cplex', 'gurobi', 'scip', 'xpress'))

# create the minizinc Model
# create model
# 1) minizinc model does not work
#my_solver = Solver.lookup(current_solver)
#instance = Instance(my_solver, Model(load_model(os.path.join(mpath, current_model))))
#instance.add_file(load_data(os.path.join(dpath, current_data)))
#results = instance.solve()
#st.write(results['ctl'])

current_model_full = os.path.join(mpath, current_model)
current_data_full = os.path.join(dpath, current_data)
results = pymzn.minizinc(current_model_full, current_data_full)

ctl = results[0]['ctl']
allocated_asset = results[0]['ctl']
coll_start = results[0]['allocated_collection_start']
coll_duration = results[0]['allocated_collection_duration']

# prepare the data to display

no_crs = len(coll_start)

ctl_df = pd.DataFrame()

date1 = datetime(2021, 3, 10, 7, 0, 0)

for i in range(1,no_crs):
    if allocated_asset[i].name == 'NO_ASSET':
        cr = dict(Task='CR'+str(i),Start=date1,Finish=date1+timedelta(hours = 1),Resource=allocated_asset[i].name)
        ctl_df = ctl_df.append(cr,ignore_index=True)
    else:
        cr = dict(Task='CR'+str(i),Start= date1 + timedelta(hours = coll_start[i]),Finish=date1 + timedelta(hours = coll_start[i]+coll_duration[i])
        , Resource=allocated_asset[i].name)
        ctl_df = ctl_df.append(cr,ignore_index=True)


st.header('CTL - Grid View')
st.write(ctl_df)
# streamlit elements


st.header('Map View')
st.map(location[['latitude', 'longitude']])


st.header('Synch Matrix View')


fig = px.timeline(ctl_df, x_start="Start", x_end="Finish", y="Task", color="Resource")

fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
fig.update_layout(autosize=True)

st.write(fig)

with st.expander('Collection Task List', expanded=False):
    st.markdown('Tabular view')

    midpoint = (np.average(location['latitude']), np.average(location['longitude']))

    st.pydeck_chart(pdk.Deck(
        tooltip={"text":"NAI: {name}"},
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
        latitude=midpoint[0],
        longitude=midpoint[1],
        zoom=8,
        pitch=0,
        ),
        layers=[
            pdk.Layer(
                type='ScatterplotLayer',
                data=location,
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=1000,
                ),
            pdk.Layer(
                type="PathLayer",
                data=path_df,
                pickable=True,
                get_color="color",
                width_scale=20,
                width_min_pixels=2,
                get_path="path",
                get_width=5,
            ),
        ],
    ))

with st.expander('Collection Requirements List'):
    st.write('add CRs')

with st.expander('ISR assets'):
    st.write('add assests')

with st.expander('Minzinc Model and Data'):
    col1, col2 = st.columns(2)
    with col1:
        st.header(current_model)
        with open(current_model_full) as f:
            contents = f.readlines()
        st.write(contents)
    with col2:
        st.header(current_data)
        with open(current_data_full) as f:
            contents = f.readlines()
        st.write(contents)
