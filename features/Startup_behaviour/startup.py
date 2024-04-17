import json
import numpy as np
import matplotlib.pyplot as plt

def load_json_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def calculate_derivative(data):
    return np.diff(data)

def find_stabilization_point(derivatives, threshold=0.01, consecutive_points=5):
    abs_derivatives = np.abs(derivatives)
    count = 0  # Counter for consecutive points under threshold
    for i in range(len(abs_derivatives)):
        if abs_derivatives[i] < threshold:
            count += 1
            if count >= consecutive_points:
                return i - consecutive_points + 2  # Adjust for the window of points
        else:
            count = 0  # Reset counter if the point is above the threshold
    return -1  # Returns -1 if no stabilization point is found

def main():
    filename = 'data.json'
    reward_data = load_json_data(filename)
    
    episode = input("Enter the episode number to analyze: ")
    try:
        rewards = reward_data[episode]
        smoothed_rewards = moving_average(rewards)
        derivatives = calculate_derivative(smoothed_rewards)
        
        stabilization_point = find_stabilization_point(derivatives)
        
        print(f"Stabilization occurs at timestep: {stabilization_point}")

        # Plotting the rewards and their smoothed version
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label='Original Rewards', alpha=0.5)
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label='Smoothed Rewards', color='red')
        if stabilization_point != -1:
            plt.axvline(x=stabilization_point, color='green', label='Stabilization Point')
        plt.title('Reward Stabilization Analysis')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('reward_plot.png')  # Saves the plot as a PNG file
        plt.close()  # Close the plot to free up memory
        print("Plot saved as 'reward_plot.png'.")
        
    except KeyError:
        print("Episode number not found in the data. Please enter a valid episode number.")

if __name__ == "__main__":
    main()
