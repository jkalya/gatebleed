import matplotlib.pyplot as plt

# Initialize an empty list to store the first column values
first_column_data = []

# Read the file
with open('/build/readings.txt', 'r') as file:
    for line in file:
        # Skip empty lines
        if line.strip():
            # Split the line by whitespace and convert the first column to an integer
            first_column_data.append(int(line.split()[0]))

# Plot the first column data as a histogram
plt.hist(first_column_data, bins=30, edgecolor='black')

# Set the labels
plt.xlabel('time(cycle)')
plt.ylabel('frequency')

# Save the plot as an image
plt.savefig('/build/histogram.png', format='png')