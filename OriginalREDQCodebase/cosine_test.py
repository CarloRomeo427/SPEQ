def cosine_annealing(start_value, final_value, current_step, max_steps):
    """
    Computes the cosine annealed value given the start and final values, 
    the current step, and the total steps.
    
    Parameters:
        start_value (float): The starting value of the parameter.
        final_value (float): The final value of the parameter.
        current_step (int): The current step in the process.
        max_steps (int): The total number of steps.
    
    Returns:
        float: The cosine annealed value at the current step.
    """
    return final_value + 0.5 * (start_value - final_value) * (1 + np.cos(np.pi * current_step / max_steps))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')
T = 300_000
steps = np.arange(0, T + 1)
threshold = cosine_annealing(start_value=0.75, final_value=0.25, current_step=steps, max_steps=T)
improvement = cosine_annealing(start_value=0.01, final_value=0.1, current_step=steps, max_steps=T)
heldout = cosine_annealing(start_value=0.25, final_value=0.1, current_step=steps, max_steps=T)

plt.figure(figsize=(12, 7))
plt.plot(steps, threshold, label="threshold", linewidth=2)
plt.plot(steps, improvement, label="improvement", linewidth=2)
plt.plot(steps, heldout, label="heldout", linewidth=2)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Parameter Value", fontsize=12)
plt.title("Cosine Annealing Schedule", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.show()