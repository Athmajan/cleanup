import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
dfTime_Random = pd.read_csv("finite_results/time_history_master.csv")
dfTime_Base = pd.read_csv("finite_results/BasePolicyResults.csv")

# Group by 'Experiment' and calculate mean and std for dfTime_Random
grouped_random = dfTime_Random.groupby('Experiment')['TimeTaken'].agg(['mean', 'std']).reset_index()
grouped_random['upper'] = grouped_random['mean'] + grouped_random['std']
grouped_random['lower'] = grouped_random['mean'] - grouped_random['std']

# Group by 'Experiment' and calculate mean and std for dfTime_Base
grouped_base = dfTime_Base.groupby('Experiment')['TimeTaken'].agg(['mean', 'std']).reset_index()
grouped_base['upper'] = grouped_base['mean'] + grouped_base['std']
grouped_base['lower'] = grouped_base['mean'] - grouped_base['std']

# Plotting the data points for dfTime_Random with lighter 'x' signs
plt.scatter(dfTime_Random['Experiment'], dfTime_Random['TimeTaken'], color='gray', marker='x', label='Random Data Points')

# Plotting the data points for dfTime_Base with lighter 'o' signs
plt.scatter(dfTime_Base['Experiment'], dfTime_Base['TimeTaken'], color='lightblue', marker='x', label='Base Data Points', alpha=0.5)

# Plotting the mean with error bars representing std for dfTime_Random
plt.errorbar(grouped_random['Experiment'], grouped_random['mean'], yerr=grouped_random['std'], fmt='o', color='red', ecolor='lightcoral', elinewidth=2, capsize=5, label='Random Mean ± Std Dev')

# Plotting the mean with error bars representing std for dfTime_Base
plt.errorbar(grouped_base['Experiment'], grouped_base['mean'], yerr=grouped_base['std'], fmt='o', color='blue', ecolor='lightblue', elinewidth=2, capsize=5, label='Base Mean ± Std Dev')

# Connecting the mean points with a line for dfTime_Random
plt.plot(grouped_random['Experiment'], grouped_random['mean'], color='red', linestyle='-', linewidth=2)

# Connecting the mean points with a line for dfTime_Base
plt.plot(grouped_base['Experiment'], grouped_base['mean'], color='blue', linestyle='-', linewidth=2)

# Connecting the upper and lower limits with dotted lines for dfTime_Random (without connecting points of each experiment)
plt.plot(grouped_random['Experiment'], grouped_random['upper'], color='red', linestyle='--', linewidth=0.5)
plt.plot(grouped_random['Experiment'], grouped_random['lower'], color='red', linestyle='--', linewidth=0.5)

# Connecting the upper and lower limits with dotted lines for dfTime_Base (without connecting points of each experiment)
plt.plot(grouped_base['Experiment'], grouped_base['upper'], color='blue', linestyle='--', linewidth=0.5)
plt.plot(grouped_base['Experiment'], grouped_base['lower'], color='blue', linestyle='--', linewidth=0.5)

# Adding labels and title
plt.xlabel('Experiment')
plt.ylabel('Time Taken')
plt.title('Time Taken per Experiment')
plt.legend()

# Show the plot
plt.show()
