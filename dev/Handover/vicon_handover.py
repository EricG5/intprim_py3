from dev.util import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_date = "2026_04_27"
    data_dir = Path(__file__).parent / data_date
    data_ = {}
    for file in data_dir.glob("*.csv"):
        # print(f"Processing {file.name}...")
        prefix = "traj_vicon_"
        name = file.stem.removeprefix(prefix)
        data_[name] = np.loadtxt(file, delimiter=",", skiprows=1)

    # visualize_pose_trajectory(data_['Baton'], title="Baton trajectory")
    time_prev = data_['Baton'][0, 0]
    # print(f"Starting time: {time_prev:.2f} seconds")
    for i in range(0, len(data_['Baton'])):
        if data_['Baton'][i, 0] - time_prev > 0.5:
            print(f"Time gap detected at index {i}: {data_['Baton'][i, 0] - time_prev:.2f} seconds")
        time_prev = data_['Baton'][i, 0]

    for name in data_:
        plt.figure()
        plt.plot(data_[name][:, 0], data_[name][:, 1], label="x")
        plt.plot(data_[name][:, 0], data_[name][:, 2], label="y")
        plt.plot(data_[name][:, 0], data_[name][:, 3], label="z")
        plt.title(f"{name} trajectory")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid()
    plt.show()