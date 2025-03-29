import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import csv
import io
import math
import numpy as np
from scipy.optimize import curve_fit
import os


# Global variables
x_val, x_err, y_val, y_err = [], [], [], []
latest_fig = None
def make_errorbars(ax):
    """
    make error bars
    :param ax: the plot axes
    :return:
    """
    ax.errorbar(
        x_val, y_val,
        xerr=x_err, yerr=y_err,
        fmt='o', ecolor='red', capsize=3, markersize=5, label='Data'
    )

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

    if not x_val or not y_val:
        print("No data to plot.")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 4))  # Adjust size as needed

    make_errorbars(ax)

    name_data(ax)

    if show_origin.get():
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # === Attempt to fit the curve to the data ===
    fit_expr = fit_entry.get().strip() # the expression

    if fit_expr:
        try:
            # Create dynamic fit function
            fit_func_raw = create_fit_function(fit_expr)

            # === Define a wrapper to work with curve_fit ===
            def fit_wrapper(x, *args):
                param_names = ['a', 'b', 'c', 'd', 'e']
                params_dict = {name: val for name, val in zip(param_names, args)}
                return np.array([fit_func_raw(val, **params_dict) for val in x])

            # === Perform the curve fitting ===
            initial_guess = [0] * (param_number.get()) # initial values for params
            popt, pcov = curve_fit(fit_wrapper, x_val, y_val, p0=initial_guess)
            # calculate error
            perr = np.sqrt(np.diag(pcov))  # standard errors

            # Predict y using the optimal parameters
            y_fit_filtered = fit_wrapper(x_val, *popt)

            # Calculate R-squared
            ss_res = np.sum((y_val - y_fit_filtered) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
#dfdf
            #set the start and end of the fit
            try:
                x_start = float(start_fit_entry.get())
                x_end = float(end_fit_entry.get())
            except ValueError:
                x_start, x_end = min(x_val), max(x_val)

            # === Generate the fit line using optimal parameters ===
            x_fit = np.linspace(x_start, x_end, 300)
            y_fit = fit_wrapper(x_fit, *popt)
            # create fit label
            fit_label = "Fit: "
            for i in range(len(initial_guess)):
                fit_label += f"{chr(97+i)} = {popt[i]:.4f} ± {perr[i]:.2f}\n"
            fit_label += f"$R^2$ = {r_squared:.3f}"
            # add the fit
            ax.plot(x_fit, y_fit, color="green", label=fit_label)
            # Show legend
            ax.legend()

        except Exception as e:
            print(f"Could not generate fitted curve: {e}")

    ax.grid(True)

    change_graph(fig)

# Function to open and read CSV
def open_csv():
    global x_val, x_err, y_val, y_err

    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    x_val.clear()
    x_err.clear()
    y_val.clear()
    y_err.clear()

    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                # Skip header row
                if i == 0:
                    continue
                if len(row) >= 4:
                    try:
                        x_val.append(float(row[0]))
                        x_err.append(float(row[1]))
                        y_val.append(float(row[2]))
                        y_err.append(float(row[3]))
                    except ValueError:
                        continue  # Skip invalid rows

        print(f"Loaded {len(x_val)} data points.")
        update_plot_image()

    except Exception as e:
        print("Error loading CSV:", e)

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
open_button = tk.Button(top_frame, text="Open CSV", command=open_csv, font=("Arial", 12))
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

save_button = tk.Button(bottom_frame, text="Save", command=save_action, font=("Arial", 12))
save_button.grid(row=1, column=5, padx=10, pady=5, sticky='e')

param_label = tk.Label(bottom_frame, text="Param Number", font=("Arial", 12))
param_label.grid(row=2, column=4, padx=10, pady=5)

param_number = tk.IntVar(value=3)
param_slider = tk.Scale(
    bottom_frame,
    from_=1,
    to=5,
    orient=tk.HORIZONTAL,
    variable=param_number
)
param_slider.grid(row=2, column=5, padx=10, pady=5)

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

