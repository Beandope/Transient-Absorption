import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mplc
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib import colors
import os
from Figure_Size import set_size, fig_fmt


# load data ta ------------------------------------------
filepath_in= os.getcwd()
filename_in= ['10K_0mT']
ext='_proc.dat'
conc_naming = '5gL'
cl = '5 g/L'

sz_time = np.zeros(len(filename_in))
sz_lambda = np.zeros(len(filename_in))

data_list = []

for i in range(len(filename_in)):
    data = np.loadtxt(filepath_in +'/'+ filename_in[i] + ext)
    data_list.append(data)
    sz_time[i] = data.shape[0] - 1
    sz_lambda[i] = data.shape[1] - 1

sz_t_max = np.max(sz_time) #find the most delay times, which can be used to create matrix later
sz_l_max = np.max(sz_lambda)
markers = ['o', 's', 'v', 'p', 'D', '1', 'x']

fig_fmt()

# plot dynamics -------------------------------------------
spe_ranges = [825, 930, 1030]
bkg_limits = [-.2, -0.3]
t0_limit_m = -0.1  # to normalize at t0
t0_limit_M = 0.3  # to normalize at t0
t_lin_min = -1
t_lin_max = 1

# x, y = np.zeros([len(filename_in)], [len(sz_time)]), np.zeros([len(filename_in)],[len(sz_lambda)])

# plot all lambdas for a single map - Normalized - bkg subtracted
M_kin = np.zeros([int(sz_t_max), 2 * len(spe_ranges), len(filename_in)])
# parula_map = par.parula_map
c_list = ['k', 'darkred', 'g', 'm']


def myfmt(x, pos):  # colorbar ticks decimal places
    return '{0:.1f}'.format(x)


for j in range(len(filename_in)):
    data_list[j][1:, 1:] = np.abs(data_list[j][1:, 1:])
    map1 = data_list[j]
    fig_fmt()
    fig1 = plt.figure(figsize=(5, 5), dpi=200)
    # fig1.canvas.set_window_title(str(filename_in[j]))

    # noinspection PyInterpreter
    for i in range(len(spe_ranges)):
        delta = np.min(np.abs(map1[0, 1:] - spe_ranges[i]))
        L_1 = np.argmin(np.abs(map1[0, 1:] - spe_ranges[i]))  # min index of the wavelength inspected
        deltat0 = np.min(np.abs(map1[1:, 0] - t0_limit_M))  # norm at t0
        t0_M = np.argmin(np.abs(map1[1:, 0] - t0_limit_M))
        deltat0 = np.min(np.abs(map1[1:, 0] - t0_limit_m))  # norm at t0
        t0_m = np.argmin(np.abs(map1[1:, 0] - t0_limit_m))
        delta_bkg_m = np.min(np.abs(map1[1:, 0] - bkg_limits[0]))  # bkg subtr
        t_bkg_m = np.argmin(np.abs(map1[1:, 0] - bkg_limits[0]))
        delta_bkg_M = np.min(np.abs(map1[1:, 0] - bkg_limits[1]))  # bkg subtr
        t_bkg_M = np.argmin(np.abs(map1[1:, 0] - bkg_limits[1]))
        M_kin[0:int(sz_time[j]), 2 * i, j] = map1[1:, 0]
        M_kin[0:int(sz_time[j]), 2 * i + 1, j] = map1[1:, L_1] - \
                                                 np.nanmean(map1[t_bkg_m: t_bkg_M, L_1])  # bkg subtr
        M_kin[0:int(sz_time[j]), 2 * i + 1, j] = map1[1:, L_1]  # no bkg subtr
        M_kin[0:int(sz_time[j]), 2 * i + 1, j] = M_kin[0:int(sz_time[j]), 2 * i + 1, j] / \
                                                 np.max(np.abs(
                                                     M_kin[0:int(sz_time[j]), 2 * i + 1, j]))  # norm at max, preserving
        # M_kin[0:int(sz_time[j]), 2 * i + 1, j] = M_kin[0: int(sz_time[j]), 2 * i + 1, j] / \
        #                                np.max(np.abs(M_kin[t0_m : t0_M, 2 * i + 1, j])) #norm at t0, preserving sign
        t_max = np.max(np.abs(M_kin[t0_m:t0_M, 2 * i + 1, j]))  # norm at +1
        x = np.argmax(np.abs(M_kin[t0_m:t0_M, 2 * i + 1, j]))
        # if M_kin[x+t0_m][2 * i + 1][j] < 0:
        #    M_kin[0: int(sz_time[j]), 2 * i + 1, j] = -1 * M_kin[0:int(sz_time[j]), 2 * i + 1, j]

        ax2 = plt.subplot(111)
        #         ax2.set_xlim(1.2, 1000)

        #         ax2.set_yticklabels([])
        plot2 = plt.plot(M_kin[0:int(sz_time[j]), 2 * i, j], M_kin[0:int(sz_time[j]), 2 * i + 1, j],
                         label=str(spe_ranges[i]) + ' nm',
                         linestyle='', linewidth=2, color=c_list[i],
                         marker=markers[i], markevery=2, markerfacecolor='white')
        ax2.set_xlabel('Delay Time (ps)')
        ax2.set_ylabel('Norm. $\Delta$T/T')

        #         plt.xscale('symlog')
        plt.subplots_adjust(wspace=0)
        # ax2.set_title(str(Fluence[j]) + '$\mu$J/cm$^2$', fontdict = font, loc = 'left')
        ax2.legend(loc='upper right', fontsize=18, frameon=False)

        ##plot tail
        # plt.plot(M_kin[200:int(sz_time[j]), 2*i, j], M_kin[200:int(sz_time[j]), 2 * i + 1, j],
        #                  label = str(round(1240/spe_ranges[i],2)) + ' eV', linestyle = 'dotted', linewidth =2)
        # plt.xlabel('Delay Time (ps)', fontdict=font)
        # plt.ylabel('Norm. $\Delta$A', fontdict = font)
        # plt.legend(loc='upper right', fontsize=18, frameon=False)
        # plt.tight_layout()
    # plt.savefig(filepath_out + filename_in[j] +'_' + str(spe_ranges[0]) + 'nm_4gL.png', transparent=True)
    # np.savetxt(str(filename_out[j]), M_kin[:, :, j], delimiter= ' ')
    # plt.show()


## 2D Map
fig_list = []
c = 0
fig_fmt()
for n in range(len(filename_in)):
    f, ax = plt.subplots(figsize = (4,3), dpi = 200)
    data = np.loadtxt(filepath_in +'/'+ filename_in[n] + ext)
#     if n in [1, 2, 3, 5]:
#         pref = -1
#     else:
    pref = -1
    data[1:, 1:] = pref * data[1:, 1:]  # sometimes the sign of the map is completely reversed
    for j in range(len(data[0, 1:])):
        data[0, j + 1] = 1240 / data[0, j + 1]
    data_list.append(data)
    X = data[0, 1:]
    Y = data[1:, 0]
    data[1:, 1:] = data[1:, 1:] / np.max(data[1:, 1:])
    Z = data[1:, 1:]
    lower_bound = np.min(data[1:, 1:])
    upper_bound = np.max(data[1:, 1:])
    print(lower_bound)
    print(upper_bound)
    levels = np.linspace(lower_bound, upper_bound, 40, endpoint=True)
    divnorm = mplc.TwoSlopeNorm(vmin=lower_bound, vcenter=0, vmax=upper_bound)
    fig = ax.contourf(X, Y, Z, cmap='RdBu_r', levels=levels, norm=divnorm)
    fig_list.append(fig)
    # fig2.canvas.set_window_title(str(filename_in[j]))

#     ax.set_yscale('symlog', linthreshy = 1e1, subsy = np.linspace(1, 10, 10))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.annotate(filename_in[n], xy=(0.70,  0.90), xycoords='axes fraction')
    c += 1
    ax.set_xlabel('Photon Energy (eV)')
    ax.set_ylabel('Delay Time (ps)')
    ## Color bars
                    ## get_position.([left, bottom, width, height ]
    cax = f.add_axes([ax.get_position().x1+0.02, ax.get_position().y0,
                      0.03, ax.get_position().y1 - ax.get_position().y0])
    norm = mplc.TwoSlopeNorm(vmin = -0.2, vcenter = 0, vmax = 1)
    cbar = plt.colorbar(fig, cax=cax, ticks = [],
                        orientation = 'vertical')#format=ticker.FuncFormatter(myfmt))
    bx = cbar.ax.twinx()
    bx.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    bx.tick_params(size = 0,)
    cbar.ax.set_title('$\Delta$T/T')
    # f.tight_layout()
    set_size(4, 3)
# cbar.ax.set_xticklabels(cbar.ax.get_xticklabels() , direction = 'out')
# plt.savefig('/Users/yulong/Dropbox (GaTech)/JPCL2020/figs/2D_FourPanel.pdf')
plt.show()
