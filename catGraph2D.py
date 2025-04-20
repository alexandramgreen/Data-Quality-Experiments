import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "./forestCover_trials.csv"
df = pd.read_csv(file_path)
df = df[df["Feature Noised"].isin(['Slope', 'constant', 'noise, std dev 0.5', 'Elevation', 'Hillshade_Noon'])]
df = df[df['Noise Std Dev'] == 1.0]

# Extract relevant columns
x = df['Feature Importance']
y = -df['Importance Difference'] + df['Feature Importance']
categories = df['Feature Noised']

# Assign a color to each category
unique_categories = categories.unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))  # Or use 'tab20', 'Set1', etc.
category_to_color = dict(zip(unique_categories, colors))
color_vals = categories.map(category_to_color)

# Plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(x, y, c=color_vals, edgecolor='k', alpha=0.7)

# Create a custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                      markerfacecolor=category_to_color[cat], markersize=10)
           for cat in unique_categories]
plt.legend(handles=handles, title='Feature Noised')

# Labels and title
plt.xlabel("Initial Importance")
plt.ylabel("New Importance")
plt.title("Change in Importance according to Feature")

plt.show()
