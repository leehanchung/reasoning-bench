import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset from CSV file
df = pd.read_csv('results.csv')  # Replace 'your_dataset.csv' with the actual file path

# Extract latency and reasoning tokens columns
reasoning_tokens = df['reasoning_tokens']
latency = df['latency']

# Scatter plot of latency vs. reasoning tokens
plt.scatter(reasoning_tokens, latency, color='blue', label='Data points')

# Fit a best-fit straight line
coefficients = np.polyfit(reasoning_tokens, latency, 1)  # Degree 1 for linear fit
best_fit_line = np.poly1d(coefficients)

# Plot the best-fit line
plt.plot(reasoning_tokens, best_fit_line(reasoning_tokens), color='red', label='Best-fit line')

# Labels and legend
plt.xlabel('Reasoning Tokens')
plt.ylabel('Latency')
plt.title('Latency vs Reasoning Tokens')
plt.legend()

# Show plot
plt.show()
