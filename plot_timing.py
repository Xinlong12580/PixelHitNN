import matplotlib.pyplot as plt
import numpy as np
batch_size = 200
#with open("output_y.txt", "r") as f: 
#    lines_NN = [float(line)/batch_size * 2 / 1000 for line in f.readlines()] #Need to do both x and y
with open("output_merged.txt", "r") as f: 
    lines_NN = [float(line)/batch_size / 1000 for line in f.readlines()] #Need to do both x and y
with open("../../CMSSW_14_0_1/src/output.log", "r") as f:
    #lines_Template = [float(line.split()[-2]) / 1000 for line in f.readlines() if "Template Execution time:" in line ]
    lines = f.readlines() 
    lines_Template_before = [float(line.split()[-2]) / 1000 for line in lines if "before_main" in line ]
    lines_Template_after = [float(line.split()[-2]) / 1000 for line in lines if "after_main" in line ]
    print(len(lines_Template_before), len( lines_Template_after))
    lines_Template = [y - x for x,y in zip(lines_Template_before, lines_Template_after) ]
fig = plt.figure()
ax = fig.add_subplot(111)
bins = np.linspace(0, 40, 301) 
mean_NN = np.mean(lines_NN)
mean_Template = np.mean(lines_Template)
print("mean_NN:", mean_NN, "mean_Template", mean_Template)
ax.hist(lines_NN, density = True, bins=bins, facecolor = "none", edgecolor="blue", label = f"NN, batch size={batch_size}, mean = {mean_NN:.2f} ms")
ax.hist(lines_Template, density=True, bins=bins, facecolor = "none", edgecolor="red", label = f"Template, mean = {mean_Template:.2f} ms")
ax.legend()
ax.set_xlabel("ns")
ax.set_ylabel("count")
ax.set_title("Average cost of time per cluster, NN and Template")
fig.savefig(f"CPE_timing_{batch_size}.png")
