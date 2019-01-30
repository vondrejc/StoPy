

def set_pars(mpl):
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}"]
    params={'text.usetex': True,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'lines.markeredgewidth': 2.0,
            'font.size': 20,
            'legend.fontsize': 15,
            }
    mpl.rcParams.update(params)
    fig_par={'dpi': 1000,
             'facecolor': 'w',
             'edgecolor': 'k',
             'figsize': (8, 6),
             'figsize_square': (6, 6),
             'pad_inches': 0.05,
             }

    return fig_par

def set_labels():
    lines=['bx-', 'ro--', 'rs-', 'go:', 'gs:', 'b+--']
    labels={}
    return lines, labels

def copy_files(src, dest, files='all'):
    import os
    from shutil import copy
    src_files=os.listdir(src)
    for file_name in src_files:
        if files=='all' or file_name in files:
            full_file_name=os.path.join(src, file_name)
            if (os.path.isfile(full_file_name)):
                copy(full_file_name, dest)
        else:
            continue
    print('copy of files is finished')
    return

if __name__=='__main__':
    pass
