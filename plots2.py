import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel('./DistanceCalibration.xlsx',
                   skiprows=29).iloc[:, :3]


for drone, sf in df.groupby("DroneID"):
    plt.scatter(sf.Time, sf.Distance, label=drone)
    fit = np.polyfit(sf.Time, sf.Distance, 1)
    plt.plot(sf.Time,
             sf.Time * fit[0] + fit[1],
             label=f"{drone} fit, {fit[0]:.2f}")

plt.title("Actual Drone Speed at 50 Drone-cm/sec")
plt.xlabel("time (s)")
plt.ylabel("distance (cm)")

plt.legend()
plt.savefig("actual_dronecm.png")
plt.show()
