Project: Hooplas Mini Basketball Chatbot
Authors:
Joshua Kearstan and Cole Stevenosky

Semester: Fall 2024

Course: CHEG 472

Project Layout:
The project by Josh and Cole involved analyzing bowling data to understand the factors influencing performance. We preprocessed the data, handled missing values and outliers, and identified key correlations between features. Machine learning models, including Random Forest, Gradient Boosting, and Linear Regression, were trained and evaluated using K-Fold cross-validation. The best-performing model was selected based on R-squared scores. Finally, we developed a prototype AI assistant using OpenAI to provide personalized bowling tips.

Files In This Repository:
CHEG472_Hooplas.csv
script.py

Requirements:
import pandas as pd
import numpy as np
import numpy as np
import joblib
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import shap
import pickle

