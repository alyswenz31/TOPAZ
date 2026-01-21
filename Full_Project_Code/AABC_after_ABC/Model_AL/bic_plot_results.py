import numpy as np
import matplotlib.pyplot as plt
import os

def process_folder(suffix):
    folder = f"Simulated_Grid/ODE_Align/BIC_Values_{suffix}"
    
    # Will need to manually save the BIC results to the folder above as "bic_results_#.npy"
    # update to include only desired sample sizes
    sample_sizes = [0, 25000, 50000, 75000, 125000] #, 150000]

    x_vals = []
    y_vals = []

    print(f"\nProcessing folder: {folder}")

    for n in sample_sizes:
        filename = os.path.join(folder, f"bic_results_{n}.npy")

        try:
            data = np.load(filename)
            second_val = data[1]

            x_vals.append(n)
            y_vals.append(second_val)

            print(f"  Loaded {filename}: second value = {second_val}")

        except Exception as e:
            print(f"  Could not load {filename}: {e}")

    # Make plot
    plt.figure(figsize=(7, 5))
    plt.scatter(x_vals, y_vals)
    plt.plot(x_vals, y_vals)

    plt.xlabel("Number of AABC Samples")
    plt.ylabel("RSS Error")
    plt.title(f"RSS Error vs AABC Samples ({suffix})")
    plt.grid(True)
    plt.tight_layout()

    # Save to PDF with suffix
    save_name = f"Simulated_Grid/ODE_Align/bic_results_plot_{suffix}.pdf"
    plt.savefig(save_name)
    print(f"  Saved figure: {save_name}")

    plt.close()


def main():
    for suffix in ["00", "05"]:
        process_folder(suffix)


if __name__ == "__main__":
    main()
