
from plotting_config import set_plt_settings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

set_plt_settings()



# Define square root function
# def powerlaw_func(x, a, b):
#     return a * x ** b
    # return a * (1 - np.exp(-b * x))

def asymptotic_func(x, a, b):
    return a * x / (x + b)

def plot_fit_rmse_lead_time(rmse_dict, max_lead_time_dict, n_lags, zone):

    plt.figure(figsize=(6, 4))

    error_colors = {
        "Wind Onshore": "tab:green",  
        "load": "darkred",  
    }

    for error_type, rmse_per_step in rmse_dict.items():
        horizons = np.arange(1, len(rmse_per_step) + 1)
        error_type_nice_name = "Wind" if error_type == "Wind Onshore" else "Load"
        horizons = np.arange(1, max_lead_time_dict[error_type] + 1)
        # Ensure rmse_per_step is a numpy array
        rmse_per_step = np.array(rmse_per_step)
        plotting_linspace = np.linspace(0, 12, 100)
        # Fit the function to the data

        asymptotic_params, _ = curve_fit(asymptotic_func, horizons, rmse_per_step)
        asymptotic_fitted_rmse = asymptotic_func(plotting_linspace, *asymptotic_params)

        plt.scatter(horizons, 100 * rmse_per_step, color=error_colors[error_type], marker='x', label=f'{error_type_nice_name}', s=100)
        plt.plot(plotting_linspace, 100 * asymptotic_fitted_rmse, linestyle='--', color=error_colors[error_type], label=rf'Fit {error_type}')

    plt.xlabel("Lead time [h]")
    plt.xticks([0, 3, 6, 9, 12])

    plt.ylabel("NRMSE [%]")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0., 1.02), loc='lower left', borderaxespad=0., ncols=2)
    plt.tight_layout()
    plt.xlim((0, 12))
    plt.savefig(f'plots/nrmse_lag_{n_lags}_{zone}_top_legend.pdf', bbox_inches='tight')
    plt.show()


def plot_rmse(rmse_dict, zone):
    """
    Plot RMSE for different error types. No fit. 
    """

    set_plt_settings()
    pretty_labels = {
        'Wind Onshore': 'Wind Onshore',
        'Solar': 'Solar',
        'load': 'Load',
    }
    type_colors = {
        'Wind Onshore': '#009879',
        'Solar': '#f4a259',
        'load': "#b62c41",
    }

    for error_type, rmse_zone_dict in rmse_dict.items():
        rmse_ser = rmse_zone_dict[zone]
        plt.plot(rmse_ser.index, rmse_ser, label=pretty_labels[error_type], color=type_colors[error_type])
    plt.ylim((0, 0.08))
    plt.xlim((1, 8))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel('Lead time [h]')
    plt.ylabel('RMSE [p.u.]')

    plt.savefig(f'plots/rmse_plot_{zone}.pdf', bbox_inches='tight')
    plt.show()
    return 