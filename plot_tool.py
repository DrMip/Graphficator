from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import csv
import io
import math
import numpy as np
from scipy.optimize import curve_fit


def update_plot_image(x_val, x_err, y_val, y_err):
    global img_label

    if not x_val or not y_val:
        print("No data to plot.")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 4))  # Adjust size as needed

    ax.errorbar(
        x_val, y_val,
        xerr=x_err, yerr=y_err,
        fmt='o', ecolor='red', capsize=3, markersize=5, label='Data'
    )
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
    if show_origin.get():
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # === Attempt to fit the curve to the data ===
    fit_expr = fit_entry.get().strip()

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

            # calculate R_squared
            # Predict y using the optimal parameters
            y_fit_filtered = fit_wrapper(x_val, *popt)

            # Calculate R-squared
            ss_res = np.sum((y_val - y_fit_filtered) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            try:
                x_start = float(start_fit_entry.get())
                x_end = float(end_fit_entry.get())
            except ValueError:
                x_start, x_end = min(x_val), max(x_val)

            # === Generate the fit line using optimal parameters ===
            x_fit = np.linspace(x_start, x_end, 300)
            y_fit = fit_wrapper(x_fit, *popt)
            fit_label = "Fit: "
            for i in range(len(initial_guess)):
                fit_label += f"a{i} = {popt[i]:.4f} ± {perr[i]:.2f}\n"
            fit_label += f"$R^2$ = {r_squared:.3f}"
            ax.plot(x_fit, y_fit, color="green", label=fit_label)
            # Show legend
            ax.legend()

        except Exception as e:
            print(f"Could not generate fitted curve: {e}")

    ax.grid(True)

    # Save plot to memory buffer
    plt.tight_layout()
    fig.canvas.draw()   # Force draw before saving
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')  # ensure full plot is captured
    buf.seek(0)
    plt.close(fig)

    # Load plot into Tkinter-compatible image
    image = Image.open(buf)
    global latest_fig
    latest_fig = image
    #image = image.resize((500, 250))  # High-quality resampling
    photo = ImageTk.PhotoImage(image)

    # Update the image label
    img_label.configure(image=photo, text="", font=None, width=0, height=0)
    img_label.image = photo  # Keep reference alive
    buf.close()

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