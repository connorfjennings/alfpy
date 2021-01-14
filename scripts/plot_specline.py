import astropy, os, sys, scipy, math, copy, numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec, rc, gridspec, rcParams, ticker
from matplotlib import gridspec
from astropy.io import ascii as astro_ascii


# ---------------------------------------------------------------- #
def plot_specindex(ax, wave1, wave2, textin='nan', l1=0.1, l2=0.06, fontsize=14):   
    teml = ax.get_ylim()[1]-ax.get_ylim()[0]
    ax.plot([wave1, wave1], 
            [ax.get_ylim()[1]-l1*teml, ax.get_ylim()[1]-l2*teml], 'k-', lw=0.75)
    ax.plot([wave1, wave2], 
            [ax.get_ylim()[1]-l2*teml, ax.get_ylim()[1]-l2*teml], 'k-', lw=0.75)
    ax.text(wave1, ax.get_ylim()[1]-0.4*l1*teml, textin, fontsize=fontsize)
    
# ---------------------------------------------------------------- #    
def plot_singleline(ax, wave1, textin='nan', l1=0.1, l2=0.06, fontsize=14):
    teml = ax.get_ylim()[1]-ax.get_ylim()[0]
    ax.plot([wave1, wave1], 
            [ax.get_ylim()[1]-l1*teml, ax.get_ylim()[1]-l2*teml], 'k-', lw=0.75)
    ax.text(wave1-10, ax.get_ylim()[1]-0.8*l2*teml, textin, fontsize=fontsize)

# ---------------------------------------------------------------- #
def plot_doubleline(ax, wave1, wave2, textin='nan', l1=0.1, l2=0.06, fontsize=14):
    teml = ax.get_ylim()[1]-ax.get_ylim()[0]
    ax.plot([wave1, wave1], 
            [ax.get_ylim()[1]-l1*teml, ax.get_ylim()[1]-l2*teml], 'k-', lw=0.75)
    ax.plot([wave2, wave2], 
            [ax.get_ylim()[1]-l1*teml, ax.get_ylim()[1]-l2*teml], 'k-', lw=0.75)
    ax.plot([wave1, wave2], 
            [ax.get_ylim()[1]-l2*teml, ax.get_ylim()[1]-l2*teml], 'k-', lw=0.75)
    ax.text(wave1, ax.get_ylim()[1]-0.8*l2*teml, textin, fontsize=fontsize)
    
    
# ---------------------------------------------------------------- #    
def plot_tripleline(ax, wave1, wave2, wave3, textin='nan'):
    teml = ax.get_ylim()[1]-ax.get_ylim()[0]
    ax.plot([wave1, wave1], 
            [ax.get_ylim()[1]-0.2*teml, ax.get_ylim()[1]-0.1*teml], 'k-', lw=0.75)
    ax.plot([wave2, wave2], 
            [ax.get_ylim()[1]-0.2*teml, ax.get_ylim()[1]-0.1*teml], 'k-', lw=0.75)
    ax.plot([wave3, wave3], 
            [ax.get_ylim()[1]-0.2*teml, ax.get_ylim()[1]-0.1*teml], 'k-', lw=0.75)
    ax.plot([wave1, wave2], 
            [ax.get_ylim()[1]-0.1*teml, ax.get_ylim()[1]-0.1*teml], 'k-', lw=0.75)
    ax.plot([wave2, wave3], 
            [ax.get_ylim()[1]-0.1*teml, ax.get_ylim()[1]-0.1*teml], 'k-', lw=0.75)
    ax.text(wave2, ax.get_ylim()[1]-0.08*teml, textin, fontsize=14)

    
# --------------------------------
def add_reflines(ax, line_zone = [1,]):
    xmin, xmax = ax.get_xlim()[0], ax.get_xlim()[1]
    ref = '/Users/menggu/WORK/Manga/observation/redux154/ref/'
    ref_line = astro_ascii.read(ref+'udgpaper_index_air.lis')
    Hlines = [6564.61, 4862.68, 4341.68, 4102.89]
    Hlines_name = [r'$\rm{H}$$\alpha$', r'$\rm{H}$$\beta$', r'$\rm{H}$$\gamma$', r'$\rm{H}$$\delta$']
    CaTline = [8498, 8542, 8662]; CaTline_name = [r'$\rm{CaT}$', r'$\rm{CaT}$', r'$\rm{CaT}$']
    for il, line in enumerate(Hlines):
        if line < ax.get_xlim()[0] or line > ax.get_xlim()[1]: 
            continue
        else:
            teml = ax.get_ylim()[1] - ax.get_ylim()[0]
            if line>xmin and line<xmax:
                ax.plot([line,line], [ax.get_ylim()[1]-0.2*teml, ax.get_ylim()[1]-0.1*teml] , 'k-', lw=0.75)
                ax.text(line-10, ax.get_ylim()[1]-0.10*teml, (Hlines_name+CaTline_name)[il], fontsize=15)
                    
    if 1 in line_zone:
        plot_doubleline(ax, 3934.777, 3969.591, textin=r'$\mathrm{CaII}$')
        plot_specindex(ax, 4222.25, 4234.75, textin=r'$\mathrm{Ca}$')
        plot_specindex(ax, 4282.6, 4317.6, textin=r'$\mathrm{CH}$')
        plot_specindex(ax, 4369.125, 4420.375, textin=r'$\mathrm{Fe4384}$')
    if 2 in line_zone:
        plot_singleline(ax, 5168.761, textin=r'$\mathrm{Mgb}$')
        plot_singleline(ax, 5267.2, textin=r'$\mathrm{FeI}$')
    if 3 in line_zone:
        plot_singleline(ax, (5878.5+5911.0)/2., textin=r'$\mathrm{NaD}$')
        plot_singleline(ax, (6189.625+6272.125)/2., textin=r'$\mathrm{TiO}$')
    if 4 in line_zone:
        plot_singleline(ax, (8177.+8205.)/2., textin=r'$\mathrm{NaI}$')
        plot_tripleline(ax, (8484.+8513.)/2, (8522.+8562.)/2, (8642.+8682.)/2, textin=r'$\mathrm{CaII}$')
    if 5 in line_zone:
        plot_singleline(ax, 9910., textin=r'$\mathrm{FeH}$') 