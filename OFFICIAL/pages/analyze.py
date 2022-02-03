import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd
import sqlite3

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json

# from copy import deepcopy
# import base64

import matplotlib as mpl
mpl.use("agg")
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


def app():
    
    dirname = st.sidebar.text_input("Enter your data directory path:", os.path.dirname(__file__))
    # dirname = os.path.join(os.path.dirname(__file__), '../data')
    local_files_list = [file for file in os.listdir(dirname) if file.endswith('.csv') and not file.startswith('.')] # Ignore hidden files
    
    with st.sidebar.expander("See data in current directory"):
        st.write(local_files_list)

    uploaded_file = st.sidebar.file_uploader("Upload data", type = ['csv', 'xlsx', 'sqlite', 'db'])
    st.sidebar.markdown("## Select Visualization Parameters")

    num_graphs = st.sidebar.slider('Number of graphs', 1, 10, 1)  # min, max, default

    st.title('DATA ANALYSIS')

    def graph_data(graph_key):
        
        selected_dataset = st.selectbox('Select Datasets', local_files_list + [uploaded_file.name] if uploaded_file is not None else local_files_list, index=0, key=graph_key)
        if uploaded_file is not None and selected_dataset == uploaded_file.name:
            dataframe = pd.read_csv(uploaded_file)
        else:
            dataframe = pd.read_csv(dirname + "/" + selected_dataset)
        
        # dataframe = pd.read_csv(dirname + "/" + selected_dataset)
        st.subheader("Raw Data")
        with st.expander("See raw data"):
            st.write(dataframe)
        parameter_list = dataframe.columns.tolist()
        
        with st.expander("See graph parameters"):
            graph_title = st.text_input("Enter your graph title:", "Graph Title", key=graph_key)
            col1, col2 = st.columns(2)

            graph_xdata = col1.selectbox('X Data', parameter_list, key=graph_key)
            graph_ydata = col2.multiselect('Y Data', parameter_list, key=graph_key)

            # st.subheader("Datatype: ", dataframe[graph1_ydata].dtype)
            regression = st.checkbox('Regression', value=True, key=graph_key)
            if regression == True:
                regression_type = st.selectbox('Regression Type', 
                    ['Ordinary Least Squares', 
                    'Locally WEighted Scatterplot Smoothing',
                    'Moving Averages: Rolling',
                    'Moving Averages: Exponential',
                    'Moving Averages: Expanding'], key=graph_key)
        
        # Types of regressions: https://plotly.com/python/linear-fits/   
        regression_options = {
            'Ordinary Least Squares': ['ols', None], 
            'Locally WEighted Scatterplot Smoothing': ['lowess', dict(frac=0.1)], 
            'Moving Averages: Rolling': ['rolling', dict(window=5)],
            'Moving Averages: Exponential': ['ewm', dict(halflife=2)],
            'Moving Averages: Expanding': ['expanding', None],
            }
        
        graph_fig = px.scatter(
            dataframe,
            x=graph_xdata,
            y=graph_ydata,
            trendline= None if regression != True else regression_options[regression_type][0],
            trendline_options= None if regression != True else regression_options[regression_type][1],
        )
        
        st.subheader(graph_title)
        st.write(graph_fig)

    for i in range(num_graphs):
        graph_data(i)

    with st.expander("See notes"):

        st.markdown("""
    About the example data:                

    This work examines the effects of different agent types on prediction market outcomes and effects of prediction market environments on agent utilities.

    Agent types:
    - Random
    - Biased
    - Analytic
    - Myopic

    Key questions:
    * How do different agent types affect prediction market outcomes?
    * How do different prediction market environments affect payoffs for different agent types?
    
    """)


    st.subheader("About this app")
    st.markdown("""
    This app allows users to easily visualize and analyze statistics for their data.
    """)