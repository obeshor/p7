import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import numpy as np

plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')
#Title display
html_temp = """
<div style="background-color: black; padding:10px; border-radius:10px">
<h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
</div>
<p style="font-size: 20px; font-weight: bold; text-align:center">Prédictions de scoring client et comparaison à l'ensemble des clients </p>
"""
st.markdown(html_temp, unsafe_allow_html=True)