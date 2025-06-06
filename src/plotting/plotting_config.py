# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:27:03 2024

@author: wouterko
"""
import matplotlib.pyplot as plt


PLOTTING_PARAMS = {
    'dpi': 800,
    'label_size': 20,
    'title_size': 20,
    'tick_size': 18,
    'font_style': 'Times New Roman',
    'legend_title_fontsize': 20,
    'legend_fontsize': 18,
}

def set_plt_settings():
    plt.rcParams["font.sans-serif"] = PLOTTING_PARAMS['font_style']
    plt.rcParams['font.family'] = PLOTTING_PARAMS['font_style']
    plt.rcParams['figure.dpi'] = PLOTTING_PARAMS['dpi']
    plt.rcParams['axes.labelsize'] = PLOTTING_PARAMS['label_size']
    plt.rcParams['axes.titlesize'] = PLOTTING_PARAMS['title_size']
    plt.rcParams['xtick.labelsize'] = PLOTTING_PARAMS['tick_size']
    plt.rcParams['ytick.labelsize'] = PLOTTING_PARAMS['tick_size']
    plt.rcParams['legend.fontsize'] = PLOTTING_PARAMS['legend_fontsize']
    plt.rcParams['legend.title_fontsize'] = PLOTTING_PARAMS['legend_title_fontsize']
    # plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    return 

