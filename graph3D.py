import plotly.express as px
import numpy as np
import pandas as pd

# Load dataset
path = r".\manualDataTrials_10_10_91.csv"
df = pd.read_csv(path)

# df = df[df["Feature Noised"].isin(['Dirty Important Feature Reversed', 'Noise', 'Clean Important Feature w/ Higher Variance'])]
#df = df[df["Importance Ratio"] > 1]

df = df[df["Batch Size"] == 8]
# df = df[df["Initial Perm Importance Mean"] > 0]
df = df[df["Noise Std Dev"] == 0.5]
# df = df[(df["Importance Ratio"] < 2)]
# df = df[~df["Feature Noised"].isin(["0.25 Std. Dev. Noise", "0.50 Std. Dev. Noise", "1.00 Std. Dev. Noise", "2.00 Std. Dev. Noise"])]

fig = px.scatter_3d(df, x='RF Feature Importance', y='Importance Ratio', z='Noise Std Dev',
                    color = 'Feature Noised')
fig.show()