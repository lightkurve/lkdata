"""
Your Example Title
==================

This is a description of the example and what it demonstrates.
"""

# Code to create a plot or figure
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Simple plot")
plt.show()
