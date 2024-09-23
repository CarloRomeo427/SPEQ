import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametri dell'array
num_states = 296  # 296 stati
num_time_steps = 300000  # 300,000 istanti di tempo

# Carichiamo i dati
data = np.load(
    "/home/ganjiro/PycharmProjects/dropRL/DropQ/OriginalREDQCodebase/runs/lomo/Humanoid/td_error_Humanoid-v2/td_error_Humanoid-v2_s0/result_td_eval.npy")

downsample_factor = 1500  # Factor for mean aggregation
data_clipped = np.clip(data, None, 1000)  # Clip values
data_clipped = np.sqrt(data_clipped)  # Apply square root transformation

# Compute the mean in chunks of size 'downsample_factor'
num_new_time_steps = data_clipped.shape[1] // downsample_factor
data_mean = data_clipped[:, :num_new_time_steps * downsample_factor].reshape(num_states, num_new_time_steps, downsample_factor).mean(axis=2)
data_mean = data_mean[:,:300]

# Create a 3D plot
x = np.arange(data_mean.shape[-1])  # New x-axis (time after mean aggregation)
y = np.arange(num_states)  # y-axis (states)
X, Y = np.meshgrid(x, y)

# Creiamo la figura 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Creiamo una "heatmap" 3D
ax.plot_surface(X, Y, data_mean, cmap='binary')

# Aggiungiamo piani verticali ai vari stati (5, 15, 25, ...)
plane_states = np.arange(3, num_states, 10)  # Stati: 5, 15, 25, ...

for state in plane_states:
    # Creiamo una matrice per il piano verticale
    x_plane, z_plane = np.meshgrid(x, [0, 1])  # Due righe di X (tempo)
    y_plane = np.full_like(x_plane, state)  # Stato fisso

    # Aggiungiamo il piano verticale (stato fisso e valori lungo il tempo)
    z_values = np.tile(data_mean[state, :], (2, 1))  # Duplicate values for plane height

    ax.plot_surface(x_plane, y_plane, z_values, color='red', alpha=1)

# Label degli assi
ax.set_xlabel('States')
ax.set_ylabel('Time steps')
ax.set_zlabel('Valori')

# Set the azimuth and elevation angles
ax.view_init(azim=0, elev=0)

plt.tight_layout()
plt.show()
