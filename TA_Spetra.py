import numpy as np
import matplotlib.pyplot as plt
from Figure_Size import set_size, fig_fmt
import matplotlib.pylab as pl
import os
from matplotlib.ticker import MultipleLocator

# load data ta ------------------------------------------

filepath_in= os.getcwd()
filename_in= ['/10K_0mT']
ext='_proc.dat'
conc_naming = '5gL'
cl = '5 g/L'

unitf = '$\mu$J/cm$^2$'
temp = ['5', '300']
unitt = 'K'

sz_time = np.zeros(len(filename_in)) #create a 1*n array with all-zero elements
sz_lambda = np.zeros(len(filename_in))

data_list = []

for i in range(len(filename_in)):
    data = np.loadtxt(filepath_in + filename_in[i] + ext)
    data = data
    data_list.append(data)
    sz_time[i] = data.shape[0] - 1
    sz_lambda[i] = data.shape[1] - 1

sz_t_max = np.max(sz_time)
sz_l_max = np.max(sz_lambda)


def unit_conv(V):
    wvl = 1240 / V
    return ['%1.0f' % z for z in wvl]

markers = ['o', 's', 'v', 'p', 'D', '1', 'x']
colors= pl.cm.RdBu(np.linspace(0.2, 1, len(filename_in)))
#plt.rcParams["figure.figsize"] = (7, 6)
unit = 'ps'
## Plot Dynamics--------------------------------

time_ranges = [0.15, 1]
bkg_limits = [-0.2, -0.0] # for bkg subtr, in time
norm_lambda = 840

osc_lambda = [810, 1000]
M_spe = np.zeros([int(sz_t_max), 2*len(filename_in), len(time_ranges)])

# x-axis: size of time delay, y: twice the number of files for the wavelengths and absorbance, z: times inspected
#map = np.zeros(len(filename_in), sz_t_max, sz_l_max)
wvl_set = np.array([800, 850, 900, 950, 1000])
new_tick_loc = np.array(1240/wvl_set)
mws = np.array([850, 950, 1050])
new_tick_loc_m = np.array(1240/mws)


fig_fmt()
fig1 = plt.figure(figsize=(5, 5), dpi = 200)
ax1 = fig1.add_subplot(111)
ax1.axhline(y=0, color='k', linestyle='-')
ax2 = plt.twiny(ax1)

M_spe = np.zeros([int(sz_l_max), 2 * len(filename_in)])
M_dev = np.zeros([int(sz_l_max), len(filename_in)])

x_iso_list = []
x_iso_range_list = []  ## range of the isosbestic points within time_ranges
x_iso_dev_list = []
ol_ratio_list = []
uol_ratio_list = []

for j in range(len(filename_in)):
    print(j)
    map1 = data_list[j]
    map1[1:, 1:] = (10 ** (-map1[1:, 1:] * 0.001) - 1) * 100
    # if j in [ 3,4]:
    #     map1[1:, 1:] = -1 * map1[1:, 1:]

    T_lower = np.argmin(np.abs(map1[1:, 0] - time_ranges[0]))
    T_upper = np.argmin(np.abs(map1[1:, 0] - time_ranges[1]))
    M_spe[0: int(sz_lambda[j]), 2 * j] = map1[0, 1:]

    for m in range(int(sz_lambda[j])):
        M_spe[m, 2 * j + 1] = np.average(map1[T_lower:T_upper, m])
        M_dev[m, j] = np.std(map1[T_lower:T_upper, m])

    #fig1.canvas.set_window_title(str(time_ranges[j]) +'_spec_temp')
    ## zero cross point
    x_iso = np.argmin(np.abs(M_spe[10:int(sz_lambda[j]) - 5, 2 * j + 1]))
    #     print(x_iso)
    x_iso_list.append(M_spe[x_iso, 2 * j])
    x_iso_dev_list.append(M_dev[x_iso, j])

    # Plot curves
    M_spe[0:int(sz_lambda[j]), 2 * j] = 1240 / M_spe[0:int(sz_lambda[j]), 2 * j]
    e_cut = np.argmin(np.abs(M_spe[2:-2, 2 * j] - 1.7))
    ax1.plot(M_spe[2:-2, 2 * j],
                  M_spe[2:-2, 2 * j + 1]/np.max(np.abs(M_spe[2:-2, 2 * j + 1])) ,
                  linestyle='-', marker = markers[j], markevery=7, markerfacecolor='white',
                  linewidth=1, color=colors[j], label = filename_in[j])

ax1.set_xlabel('Photon Energy (eV)')
# ax1.xaxis.set_label_coords(0.5, -0.1)
ax1.set_ylabel('$\Delta$T/T (%)')
# ax1.yaxis.set_label_coords(-0.13, 0.5)
# ax1.legend(frameon=False, loc='upper left', fontsize=8, bbox_to_anchor = (0.96, 1))
ax1.legend(frameon=False, loc='upper left', fontsize=8)
# ax1.set_title(filename_in[j], fontdict = font)
#     ax1.annotate(filename_in[j],xy=(0.87, 0.93), xycoords ='axes fraction', fontsize = 12)
# ax1.title(filename_in[j], fontsize =12)
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_loc)
ax2.set_xticklabels(unit_conv(new_tick_loc))
ax2.set_xlabel('Wavelength (nm)')
# ax1.set_yticks(np.arange(-0.2, 1.2, 0.2))
# ax1.axvline(x=1.639, ymin=0.5, linewidth=68, color='w')
# ax1.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.set_xticks(new_tick_loc_m, minor=True)
fig1.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85, wspace=None, hspace=None)
ax2.annotate(cl, xy=(0.86, 0.9), xycoords='axes fraction')

set_size(4,3)

plt.show()