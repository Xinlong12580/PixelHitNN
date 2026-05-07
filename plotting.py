import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

def microns_to_pixel(position, n_pixels, pixel_pitch):
    center_pixel = n_pixels / 2 + 1
    pixel_pos = position / pixel_pitch + center_pixel
    return pixel_pos

def plot_clusters(data_sets, file_name):
    #Create and save a figure showing cluster charges, prediction and true position of a hit for multiple clusters

    def make_1d_pixels_plot(cluster, prediction, error, true_position, ax, cot_alpha, cot_beta):
        #Create a plot for a single cluster
        ax.imshow([cluster], cmap="Blues", extent=(0, len(cluster), 0, 1), vmin=0, vmax=max(cluster) * 1.5)
        ax.set_yticks([])

        for i, value in enumerate(cluster):
            ax.text(i + 0.5, 0.5, f'{value:.2f}', color='black', ha='center', va='center', fontsize=8)

        ax.errorbar(prediction, 0.75, xerr=error, fmt='o', color='red', markersize=4)
        ax.plot(true_position, 0.75, color="orange", marker='X', markersize=8, linewidth=3, zorder=999)

        # Set x-axis ticks to show only integers with a step of 3 (adjust as needed)
        step = 3
        ax.set_xticks(np.arange(0, len(cluster), step))
        ax.set_xticklabels(map(int, ax.get_xticks()))

        # Add cotangent values and angles above the plot
        ax.text(0.4, 1.15, f'cot $\\alpha$: {cot_alpha:.4f}', color='black', ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.text(0.6, 1.15, f'cot $\\beta$: {cot_beta:.4f}', color='black', ha='center', va='center', fontsize=10, transform=ax.transAxes)  
 
    # Create subplots for each figure
    n_clusters = len(data_sets)
    fig, axs = plt.subplots(nrows=n_clusters, figsize=(10, n_clusters))

    for ax, data_set in zip(axs, data_sets):
        cluster = data_set['cluster']
        prediction_uncertainty = data_set['prediction_uncertainty']
        position = data_set['position']
        pixel_pitch = data_set['pixel_pitch']
        angles = data_set['angles']

        prediction = microns_to_pixel(prediction_uncertainty[0], len(cluster), pixel_pitch)
        error = prediction_uncertainty[1] / pixel_pitch
        true_position = microns_to_pixel(position, len(cluster), pixel_pitch)

        make_1d_pixels_plot(cluster, prediction, error, true_position, ax, cot_alpha=angles[0], cot_beta=angles[1])

    # Add resolution and bias on top of the entire figure
    resolution = data_sets[0]['resolution']  # assuming all datasets have the same resolution
    bias = data_sets[0]['bias']  # assuming all datasets have the same bias
    fig.text(0.5, 1.02, f'Resolution: {resolution:.0f} microns\nBias: {bias:.1f} microns', color='black', ha='center', va='center', fontsize=10)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot to a file
    print(f"Saving {file_name}")
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(file_name.replace(".pdf", ".png"), bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_dnn_loss(history,output_file):
    #print(history)
    plt.plot(history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['%s-train'%(label), '%s-validation'%(label)], loc='upper right')
    #plt.show()
    plt.savefig(output_file)
    plt.close()

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def fit_gaussian(data, bins_centers,initial_params = [1.,0.,1.]):
    params, covariance = curve_fit(gaussian, bins_centers, data, p0=initial_params)
    amplitude, mean, stddev = params
    return amplitude, mean, stddev

def plot_residuals(residuals, output_file, plot_type="Residuals",name=""):
    if plot_type == "Residuals":
        bins = np.linspace(-300, 300, 100)  # microns
        initial_params = [1., 0., 100.]  # Amplitude, mean, width
    elif plot_type == "Pulls":
        bins = np.linspace(-5, 5, 100)
        initial_params = [1., 0., 1.]  # Amplitude, mean, width
    else:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Allowed choices are 'Residuals' or 'Pulls' ")

    legend_label = f"{name} {plot_type}"
    n, bins, patches = plt.hist(residuals, bins=bins, density=True, alpha=0.7, color='lightblue', edgecolor='black',label=legend_label)

    threshold = 200  # microns or units matching your data
    residuals = np.array(residuals)
    fraction_above_threshold = np.mean(np.abs(residuals) > threshold)
    print(f"Fraction of residuals with |value| > {threshold} um: {fraction_above_threshold:.4f}")


    bins_centers = 0.5 * (bins[:-1] + bins[1:])
    # Plot Gaussian fit
    try:
        amplitude, mean, stddev = fit_gaussian(n, bins_centers, initial_params=initial_params)  # Use n instead of residuals
        fit_curve = gaussian(bins_centers, amplitude, mean, stddev)
        plt.plot(bins_centers, fit_curve, 'r--', label='Gaussian Fit')
            # Print fit parameters
        fit_params_str = f'Mean: {mean:.2f}\nStd Deviation: {stddev:.2f}'
        plt.text(0.2, 0.9, fit_params_str, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5,edgecolor="none"), horizontalalignment='center', verticalalignment='top')
    except Exception as e:
        print(f"Error plotting Gaussian fit: {e}")

    plt.legend(loc="upper right")
    plt.xlabel(plot_type)
    plt.ylabel("Cluster density / bin")

    plt.tick_params(axis='both', direction='in')

    print(f"Saving {output_file}")
    plt.savefig(output_file)
    if ".pdf" in output_file:
        plt.savefig(output_file.replace(".pdf", ".png"))
    plt.close()


def plot_uncertainties(uncertainties, file_name):
    plt.figure(figsize=(8, 6))
    plt.hist(uncertainties, bins=26, color='skyblue', edgecolor='black', alpha=0.7, range=(0, 130))
    
    plt.xlabel('Uncertainty (microns)', fontsize=16)
    plt.ylabel('Frequency [a.u.]', fontsize=16)

    mean_uncertainty = np.mean(uncertainties)
    std_uncertainty = np.std(uncertainties)
    max_uncertainty = np.max(uncertainties)
    print(f"MAX UNCERTAINTY: {max_uncertainty:.1f}")
    
    plt.text(0.95, 0.95, f'Mean: {mean_uncertainty:.2f} microns\nStd: {std_uncertainty:.2f} microns',
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    uncertainty_5 = np.sum(uncertainties <= 5) / len(uncertainties) * 100
    uncertainty_120 = np.sum(uncertainties >= 120) / len(uncertainties) * 100

    print(f"Percentage of entries with uncertainty <= 5 microns: {uncertainty_5:.2f}%")
    print(f"Percentage of entries with uncertainty >= 120 microns: {uncertainty_120:.2f}%")

    print(f"Saving {file_name}")
    plt.tight_layout()
    plt.savefig(file_name)
    if ".pdf" in file_name:
        plt.savefig(file_name.replace(".pdf", ".png"))
    plt.close()


def plot_nll_and_mse(history, output_file_prefix):
    keys = history.keys()
    
    # Try to detect the loss and mse keys (adjust if needed)
    nll_train_key = None
    nll_val_key = None
    mse_train_key = None
    mse_val_key = None
    
    for key in keys:
        if 'val' not in key and 'nll' in key:
            nll_train_key = key
        elif 'val' in key and 'nll' in key:
            nll_val_key = key
        if 'val' not in key and 'mse_position' in key:
            mse_train_key = key
        elif 'val' in key and 'mse_position' in key:
            mse_val_key = key


    # Plot NLL loss train vs val
    plt.figure(figsize=(8,5))
    if nll_train_key and nll_val_key:
        if(len(history[nll_train_key])<3):
            print("WARNING: Need at least three epochs to plot NLL and MSE history")
            return
        #We skip the first two epochs because we can have a very large loss/mse at the start of training, messing up the scale
        plt.plot(history[nll_train_key][2:], label='NLL Train')
        plt.plot(history[nll_val_key][2:], label='NLL Validation')
        plt.xlabel('Epoch')
        plt.ylabel('NLL Loss')
        plt.title('NLL Loss: Train vs Validation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_file_prefix}_nll_loss.png")
        plt.close()
    else:
        print("Could not find NLL keys for train and validation in history")

    # Plot MSE position train vs val
    plt.figure(figsize=(8,5))
    if mse_train_key and mse_val_key:
        plt.plot(history[mse_train_key][2:], label='MSE Position Train')
        plt.plot(history[mse_val_key][2:], label='MSE Position Validation')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Position')
        plt.title('MSE Position: Train vs Validation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_file_prefix}_mse_position.png")
        plt.close()
    else:
        print("Could not find MSE Position keys for train and validation in history")