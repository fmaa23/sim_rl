import json
import numpy as np
import matplotlib.pyplot as plt
import os

class StartupBehavior:
    def __init__(self, window_size, threshold, consecutive_points, episode):
        """Initialize parameters for the startup behavior analysis."""
        self.window_size = window_size
        self.threshold = threshold
        self.consecutive_points = consecutive_points
        self.episode = episode



    def load_json_data(self):
        # Current directory
        current_dir = os.path.dirname(os.path.dirname(os.getcwd()))

        # Construct the relative path to the target JSON file
        relative_path = os.path.join(current_dir, 'foundations', 'output_csv', 'reward_dict.json')

        # Normalize the path to avoid any cross-platform issues
        normalized_path = os.path.normpath(relative_path)

        # Load the JSON file using a context manager
        with open(normalized_path, 'r') as file:
            data = json.load(file)
        
        return data

    def moving_average(self, data):
        """Calculate moving average of the data using the defined window size."""
        return np.convolve(data, np.ones(self.window_size) / self.window_size, mode='valid')

    def calculate_derivative(self, data):
        """Calculate the first derivative of the data."""
        return np.diff(data)

    def find_stabilization_point(self, derivatives):


        abs_derivatives = np.abs(derivatives)
        count = 0  # Counter for consecutive points under threshold
        for i in range(len(abs_derivatives)):
            if abs_derivatives[i] < self.threshold:
                count += 1
                if count >= self.consecutive_points:
                    return i - self.consecutive_points + 2  # Adjust for the window of points
            else:
                count = 0  # Reset counter if the point is above the threshold
        return -1  # Returns -1 if no stabilization point is found

    def plot_rewards(self, rewards, smoothed_rewards, stabilization_point):
        """Plot the original and smoothed rewards with stabilization point."""
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label='Original Rewards', alpha=0.5)
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label='Smoothed Rewards', color='red')
        if stabilization_point != -1:
            plt.axvline(x=stabilization_point, color='green', label='Stabilization Point')
        plt.title('Reward Stabilization Analysis')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('reward_plot.png')
        plt.close()
        print("Plot saved as 'reward_plot.png'.")

if __name__ == "__main__":
    
    window_size = 5
    threshold = 0.01
    consecutive_points = 5
    episode = 0

    startup_engine = StartupBehavior(window_size, threshold, consecutive_points, episode)
    reward_data = startup_engine.load_json_data()   
    rewards = reward_data[str(episode)]
    smoothed_rewards = startup_engine.moving_average(rewards)
    derivatives = startup_engine.calculate_derivative(smoothed_rewards)
    
    stabilization_point = startup_engine.find_stabilization_point(derivatives)
    
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
    plt.savefig('reward_plot.png') 
    plt.close()  
    print("Plot saved as 'reward_plot.png'.")