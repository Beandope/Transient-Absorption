import matplotlib.pyplot as plt

def fig_fmt():
    font = {'family': 'Arial',
            #'color':  'darkred',
    #         'weight': 'bold',
            'size': 12,
            }
    plt.rc('font', **font)
    plt.rc('axes', linewidth = '1')
    plt.rc('xtick', labelsize = 12)
    plt.rc('ytick', labelsize = 12)
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.size'] = 6
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.minor.size'] = 6
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)