import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pandas as pd
import io
import math
import numpy as np
from scipy.optimize import curve_fit
import os


# Global variables
datasets = []
latest_fig = None
# def make_errorbars(ax):
#     """
#     make error bars
#     :param ax: the plot axes
#     :return:
#     """
#     ax.errorbar(
#         x_val, y_val,
#         xerr=x_err, yerr=y_err,
#         fmt='x', ecolor='red', capsize=3, markersize=0, label='Data'
#     )

def name_data(ax):
    x_label = xname_entry.get().strip()
    y_label = yname_entry.get().strip()
    graph_title = name_entry.get().strip()

    if not x_label:
        x_label = "X Values"
    if not y_label:
        y_label = "Y Values"
    if not graph_title:
        graph_title = "Scatter Plot with Error Bars"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(graph_title)

def change_graph(fig):
    """
    adds the new graph to the figure
    :param fig:
    :return:
    """
    # Save plot to memory buffer
    plt.tight_layout()
    fig.canvas.draw()  # Force draw before saving
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')  # ensure full plot is captured
    buf.seek(0)
    plt.close(fig)

    # Load plot into Tkinter-compatible image
    image = Image.open(buf)
    global latest_fig
    latest_fig = image
    # image = image.resize((500, 250))  # High-quality resampling
    photo = ImageTk.PhotoImage(image)

    # Update the image label
    img_label.configure(image=photo, text="", font=None, width=0, height=0)
    img_label.image = photo  # Keep reference alive
    buf.close()


# Function to update the image slot with a plot
def update_plot_image():
    global img_label

    if not datasets:
        print("No data to plot.")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust size as needed

    #make_errorbars(ax)

    name_data(ax)

    if show_origin.get():
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # === Attempt to fit the curve to the data ===
    fit_expr = fit_entry.get().strip() # the expression
    for idx, (x_val, x_err, y_val, y_err) in enumerate(datasets):
        # plot error bars
        ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, fmt='x', label=f'Data {idx + 1}', capsize=3)
        print((x_val, x_err, y_val, y_err))
        # perform fit and plot it
        if fit_expr:
            fit_func_raw = create_fit_function(fit_expr)

            def fit_wrapper(x, *args):
                param_names = ['a', 'b', 'c', 'd', 'e']
                params_dict = {name: val for name, val in zip(param_names, args)}
                return np.array([fit_func_raw(val, **params_dict) for val in x])

            try:
                guess_str = init_guess_entry.get().strip()

                try:
                    # Convert comma-separated string to list of floats
                    initial_guess = [float(val) for val in guess_str.split(',')]
                except ValueError:
                    print("Invalid initial guess format. Please enter comma-separated numbers like: 1, 0.5, 2")
                    return

                popt, pcov = curve_fit(fit_wrapper, x_val, y_val, p0=initial_guess)
                perr = np.sqrt(np.diag(pcov))

                x_start = float(start_fit_entry.get()) if start_fit_entry.get() else min(x_val)
                x_end = float(end_fit_entry.get()) if end_fit_entry.get() else max(x_val)
                x_fit = np.linspace(x_start, x_end, 300)
                y_fit = fit_wrapper(x_fit, *popt)

                y_fit_filtered = fit_wrapper(x_val, *popt)
                ss_res = np.sum((y_val - y_fit_filtered) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                fit_label = "Fit: "
                for i in range(len(initial_guess)):
                    fit_label += f"{chr(97 + i)} = {popt[i]:.4f} ± {perr[i]:.4f}\n"
                fit_label += f"$R^2$ = {r_squared:.3f}"

                ax.plot(x_fit, y_fit, label=fit_label)
                # Show legend
                ax.legend()

            except Exception as e:
                print(f"Fit failed for dataset {idx + 1}: {e}")

    ax.grid(True)

    change_graph(fig)

# Function to open and read CSV
def open_excel():
    global datasets
    try:
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Excel Files", "*.xlsx *.xls")],
            title="Select One or More Excel Files"
        )
        print(file_paths)
        # Reset existing data
        datasets.clear()

        for file_path in file_paths:
            try:
                df = pd.read_excel(file_path)
                if df.shape[1] < 4:
                    print(f"{file_path}: not enough columns.")
                    continue

                x_val, x_err, y_val, y_err = [], [], [], []
                for _, row in df.iterrows():
                    try:
                        x_val.append(float(row.iloc[0]))
                        x_err.append(float(row.iloc[1]))
                        y_val.append(float(row.iloc[2]))
                        y_err.append(float(row.iloc[3]))
                    except:
                        continue

                datasets.append((x_val, x_err, y_val, y_err))
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

            for dataset in datasets:
                print(f"Loaded {len(dataset[0])} data points.")
            update_plot_image()

    except Exception as e:
        print("Error loading Excel:", e)

# Save button action
def save_action():
    global latest_fig
    if latest_fig is None:
        print("No image to save.")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png")],
        title="Save Graph As"
    )

    if file_path:
        try:
            latest_fig.save(file_path, format="PNG")
            print(f"Image saved to {file_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

def create_fit_function(expr: str):
    """
    Converts a string like 'a*sin(x) + b*cos(x)' into a Python function.
    """
    # Safe math functions allowed in the expression
    safe_funcs = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "pow": math.pow,
        "pi": math.pi,
        "e": math.e,
    }

    def fit_func(x, **params):
        try:
            return eval(expr, {"x": x, **params, **safe_funcs, "__builtins__": {}})
        except Exception as e:
            print(f"Error evaluating fit expression: {e}")
            return None

    return fit_func


# === Build UI ===
root = tk.Tk()
root.title("Graphicator")
root.geometry("1000x500")

# === Top Frame (image + open CSV) ===
top_frame = tk.Frame(root)
top_frame.grid(row=0, column=0, padx=20, pady=20, sticky='w')

# Create a label to display the image
img_label = tk.Label(top_frame, text="[No Graph Yet]", font=("Arial", 16), width=50, height=10)
img_label.pack(side='left', padx=10)

# "Open CSV" button (right)
open_button = tk.Button(top_frame, text="Open Excel", command=open_excel, font=("Arial", 12))
open_button.pack(side='right', padx=10, pady=10)
open_button.pack(side='right', padx=10, pady=10)

# === Bottom Frame for text inputs and save ===
bottom_frame = tk.Frame(root)
bottom_frame.grid(row=1, column=0, padx=20, pady=20, sticky='w')

# --- Row 0: Name, X name, Y name ---
name_label = tk.Label(bottom_frame, text="Name", font=("Arial", 12))
name_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

name_entry = tk.Entry(bottom_frame, width=20, font=("Arial", 12))
name_entry.grid(row=0, column=1, padx=5, pady=5)

xname_label = tk.Label(bottom_frame, text="X Value name", font=("Arial", 12))
xname_label.grid(row=0, column=2, padx=5, pady=5)

xname_entry = tk.Entry(bottom_frame, width=20, font=("Arial", 12))
xname_entry.grid(row=0, column=3, padx=5, pady=5)

yname_label = tk.Label(bottom_frame, text="Y Value name", font=("Arial", 12))
yname_label.grid(row=0, column=4, padx=5, pady=5)

yname_entry = tk.Entry(bottom_frame, width=20, font=("Arial", 12))
yname_entry.grid(row=0, column=5, padx=5, pady=5)

# --- Row 1: Fit and Save ---
fit_label = tk.Label(bottom_frame, text="Fit", font=("Arial", 12))
fit_label.grid(row=1, column=0, padx=(0, 10), pady=5, sticky='w')

fit_entry = tk.Entry(bottom_frame, width=30, font=("Arial", 12))
fit_entry.grid(row=1, column=1, padx=5, pady=5, columnspan=1)

# --- Row 2: Start Fit and End Fit ---
start_fit_label = tk.Label(bottom_frame, text="Start Fit", font=("Arial", 12))
start_fit_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')

start_fit_entry = tk.Entry(bottom_frame, width=15, font=("Arial", 12))
start_fit_entry.grid(row=2, column=1, padx=5, pady=5)

end_fit_label = tk.Label(bottom_frame, text="End Fit", font=("Arial", 12))
end_fit_label.grid(row=2, column=2, padx=5, pady=5)

end_fit_entry = tk.Entry(bottom_frame, width=15, font=("Arial", 12))
end_fit_entry.grid(row=2, column=3, padx=5, pady=5)

init_guess_label = tk.Label(bottom_frame, text="Initial Guess", font=("Arial", 12))
init_guess_label.grid(row=1, column=2, padx=5, pady=5)

init_guess_entry = tk.Entry(bottom_frame, width=20, font=("Arial", 12))
init_guess_entry.grid(row=1, column=3, padx=5, pady=5)
init_guess_entry.insert(0, "1, 1, 1")

save_button = tk.Button(bottom_frame, text="Save", command=save_action, font=("Arial", 12))
save_button.grid(row=1, column=5, padx=10, pady=5, sticky='e')

show_origin = tk.IntVar(value=0)
origin_checkbox = tk.Checkbutton(
    bottom_frame,
    text="Show (0, 0)",
    variable=show_origin,
    font=("Arial", 11)
)
origin_checkbox.grid(row=1, column=4, padx=5, pady=5)


# Run app
root.mainloop()

