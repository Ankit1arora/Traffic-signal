import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import os

# Load the dataset
data_path = r"C:\Users\ankit\OneDrive\Desktop\adpmp\newdata.xlsx"
traffic_data = pd.read_excel(data_path)

# Verify correct column names
expected_columns = ['Date', 'Time', 'Traffic Signal', 'Number of Vehicles', 'Green Light Duration (sec)']
if list(traffic_data.columns) != expected_columns:
    raise ValueError(f"The Excel file must have columns named: {expected_columns}. "
                     f"Current columns are: {list(traffic_data.columns)}")

# Ensure 'Green Light Duration (sec)' is numeric
traffic_data['Green Light Duration (sec)'] = pd.to_numeric(traffic_data['Green Light Duration (sec)'], errors='coerce')
traffic_data = traffic_data.dropna(subset=['Green Light Duration (sec)'])

# 🔹 Q-Learning Parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Exploration-exploitation balance

# Initialize Q-table
q_table = {}

# 🚀 Define Fuzzy Logic System
traffic_volume = ctrl.Antecedent(np.arange(0, 100, 1), 'traffic_volume')
green_light_duration = ctrl.Consequent(np.arange(10, 80, 1), 'green_light_duration')

# Membership functions
traffic_volume['low'] = fuzz.trimf(traffic_volume.universe, [0, 0, 40])
traffic_volume['medium'] = fuzz.trimf(traffic_volume.universe, [20, 50, 80])
traffic_volume['high'] = fuzz.trimf(traffic_volume.universe, [60, 100, 100])

green_light_duration['short'] = fuzz.trimf(green_light_duration.universe, [10, 20, 40])
green_light_duration['medium'] = fuzz.trimf(green_light_duration.universe, [30, 50, 70])
green_light_duration['long'] = fuzz.trimf(green_light_duration.universe, [50, 70, 80])

# Define Fuzzy Rules
rule1 = ctrl.Rule(traffic_volume['low'], green_light_duration['short'])
rule2 = ctrl.Rule(traffic_volume['medium'], green_light_duration['medium'])
rule3 = ctrl.Rule(traffic_volume['high'], green_light_duration['long'])

# Control System
traffic_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
traffic_simulation = ctrl.ControlSystemSimulation(traffic_ctrl)

# Process Each Traffic Signal
results = []
for index, row in traffic_data.iterrows():
    time, signal, vehicles, green_light = row['Time'], row['Traffic Signal'], row['Number of Vehicles'], row['Green Light Duration (sec)']
    
    # 🔹 Get average traffic for this signal
    avg_traffic = round(traffic_data[traffic_data['Traffic Signal'] == signal]['Number of Vehicles'].mean())


    # 🔹 Apply Fuzzy Logic for initial decision
    traffic_simulation.input['traffic_volume'] = vehicles
    traffic_simulation.compute()
    fuzzy_green_light = traffic_simulation.output['green_light_duration']

    # 🔹 Q-Learning Implementation
    state = (signal, vehicles)
    
    # Initialize Q-table for unseen states
    if state not in q_table:
        q_table[state] = np.random.uniform(low=10, high=80, size=(3,))
    
    # Choose action using epsilon-greedy strategy
    if np.random.rand() < epsilon:
        action = np.random.choice([0, 1, 2])  # Explore: 0 (decrease), 1 (no change), 2 (increase)
    else:
        action = np.argmax(q_table[state])  # Exploit best action
    
    # Map action to green light duration change
    action_effect = [-5, 0, 5]  # Adjust green light by -5, 0, or +5 sec
    new_green_light = fuzzy_green_light + action_effect[action]
    new_green_light = max(10, min(new_green_light, 80))  # Keep within limits
    
    # 🔹 Define reward function
    deviation = abs(avg_traffic - vehicles)
    reward = -deviation if deviation > 20 else 10 - deviation * 0.1

    # 🔹 Q-value Update (Bellman Equation)
    next_state = (signal, avg_traffic)
    if next_state not in q_table:
        q_table[next_state] = np.random.uniform(low=10, high=80, size=(3,))

    max_next_q = np.max(q_table[next_state])
    q_table[state][action] = (1 - learning_rate) * q_table[state][action] + \
                             learning_rate * (reward + discount_factor * max_next_q)
    
    # Store results
    results.append([time, signal, vehicles, green_light, new_green_light, fuzzy_green_light, avg_traffic])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=[
    'Time', 'Traffic Signal', 'Number of Vehicles', 
    'Given Green Light (sec)', 'RL Optimized Green Light (sec)', 'Fuzzy Green Light (sec)', 'Average Vehicles'
])

# Display the updated table
print("\n🚦 Updated Traffic Signal Timing Adjustments 🚦")
print(results_df)

# Save results to CSV
output_path = r"C:\Users\ankit\OneDrive\Desktop\adpmp\optimized_results11.csv"
results_df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")

# Create the plot for ISBT 43
isbt43_data = results_df[results_df['Traffic Signal'] == 'ISBT 43'].copy()
isbt43_data.loc[:, 'Time'] = pd.to_datetime(isbt43_data['Time'], format='%I:%M %p').dt.time
isbt43_data = isbt43_data.sort_values(by='Time')

plt.figure(figsize=(12, 6))
plt.plot(isbt43_data['Time'].astype(str), isbt43_data['Given Green Light (sec)'], label='Given Green Light', marker='o')
plt.plot(isbt43_data['Time'].astype(str), isbt43_data['RL Optimized Green Light (sec)'], label='RL Optimized Green Light', marker='x')
plt.plot(isbt43_data['Time'].astype(str), isbt43_data['Fuzzy Green Light (sec)'], label='Fuzzy Green Light', linestyle='--')

plt.xlabel('Time')
plt.ylabel('Green Light Duration (sec)')
plt.title('ISBT 43: Given vs. RL Optimized vs. Fuzzy Green Light Duration')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
graph_output_path = r"C:\Users\ankit\OneDrive\Desktop\adpmp\ISBT43_green_light_optimization.png"
plt.savefig(graph_output_path)
plt.close()

print(f"\n✅ Graph saved to: {graph_output_path}")
