#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
fig = plt.figure(1)
fig.suptitle("All in One", fontsize="x-small")
spec = plt.GridSpec(ncols=2, nrows=3)

ax1 = fig.add_subplot(spec[0])
ax1.plot(y0, "r-")
ax2 = fig.add_subplot(spec[1])
ax2.scatter(x1, y1, color="m", marker="o")
ax2.set_title("Men's Height vs Weight", fontsize="x-small")
ax2.set_xlabel("Height (in)", fontsize="x-small")
ax2.set_ylabel("Weight (lbs)", fontsize="x-small")
ax3 = fig.add_subplot(spec[2])
ax3.plot(x2, y2)
ax3.set_title("Exponential Decay of C-14", fontsize="x-small")
ax3.set_xlabel("Time (years)", fontsize="x-small")
ax3.set_ylabel("Fraction Remaining", fontsize="x-small")
ax3.set_yscale("log")
ax4 = fig.add_subplot(spec[3])
ax4.plot(x3, y31, "r--", label="C-14")
ax4.plot(x3, y32, "g-", label="Ra-226")
ax4.legend(loc="upper right")
ax4.set_title("Exponential Decay of Radioactive Elements", fontsize="x-small")
ax4.set_xlim(0, 20000)
ax4.set_ylim(0, 1)
ax4.set_xlabel("Time (years)", fontsize="x-small")
ax4.set_ylabel("Fraction Remaining", fontsize="x-small")
ax5 = fig.add_subplot(spec[4:6])
ax5.hist(student_grades, bins=range(40, 110, 10), edgecolor="black")
ax5.set_xlabel("Grades", fontsize="x-small")
ax5.set_ylabel("Number of Students", fontsize="x-small")
ax5.set_xlim(0, 100)
ax5.set_ylim(0, 30)
ax5.set_title("Project A", fontsize="x-small")
plt.tight_layout()
plt.show()
