import matplotlib.pyplot as plt

# Define the data and positions
x_positions_label = [i * 2 for i in range(4)] + [9, 11]  # Increasing the gap before SPA 48
x_positions_prediction = [i * 2 + 0.8 for i in range(4)] + [9.8, 11.8]  # Matching gap for prediction bars
label_errors = [26.56, 24.71, 24.71, 25.08, 14.87, 18.22]  # Labeling errors
prediction_errors = [48.71, 39.48, 46.06, 43.72, 23.97, 29.84]  # Updated Majority prediction error

# Colors for the bars
colors_label = ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']
colors_prediction = ['lightcoral', 'lightcoral', 'lightcoral', 'lightcoral', 'lightcoral', 'lightcoral']

# Create the plot
fig, ax = plt.subplots()

# Plot the bars for label errors
bars_label = ax.bar(x_positions_label, label_errors, color=colors_label, width=0.6)

# Plot the bars for prediction errors (right bars in each pair, with gaps for None values)
bars_prediction = ax.bar([x_positions_prediction[i] for i in range(len(prediction_errors)) if prediction_errors[i] is not None],
                         [e for e in prediction_errors if e is not None], color=colors_prediction, width=0.6)

# Set the labels for x-ticks with larger fonts
ax.set_xticks([i * 2 + 0.4 for i in range(4)] + [9.4, 11.4])
ax.set_xticklabels(['Majority', 'Borda', 'MC', 'Copeland', 'SPA 48', 'SPA 49'], fontsize=14)

# Set the y-axis label with larger font size
ax.set_ylabel('Error (%)', fontsize=14)

# Increase tick size for the y-axis
ax.tick_params(axis='y', labelsize=12)

# Display the plot
plt.show()
