import py3Dmol
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def islistoflists(x):
    if not isinstance(x, list):
        return False
    if all(isinstance(i, list) for i in x):
        return True
    return False

def view_molecule(file, view_size=(600,200), color_regions=False, regions=None, colormap='tab10',interactive=True):
    view = py3Dmol.view(width=view_size[0],height=view_size[1])
    view.addModel(open(f'{file}','r').read(),'pdb')
    
    cmap1 = matplotlib.colormaps[colormap] # matplotlib colormaps selection
    if color_regions:
        view.setStyle({"cartoon": {'color': f'silver'}})
        for idx, region_idx in enumerate(regions):
            if islistoflists(region_idx):
                region_range = [ res for region in region_idx for res in range(region[0],region[1]+1)]
                view.addStyle({'resi':region_range}, {"cartoon": {'color':f'{to_hex(cmap1(idx))}'}})
            else:
                view.addStyle({'resi':list(range(region_idx[0],region_idx[1]+1))}, {"cartoon": {'color': f'{to_hex(cmap1(idx))}'}})
        
    if interactive :
        view.setClickable({}, True, """
                        function(atom, viewer, event, container) {
                        if (!atom) return;
                        if (atom.label) {
                        viewer.removeLabel(atom.label);
                        atom.label = null;
                        } else {
                        atom.label = viewer.addLabel(
                        atom.resn + " " + atom.resi ,
                        {position: atom, backgroundColor: "white", fontColor: "black", fontSize: 20}
                        );
                        }
                        }
                        """)
    view.setProjection("orthographic")
    view.zoomTo()
    view.zoom()
    view.show()

def make_ridge_plots(
    probab_cars,
    hinge_order=None,
    figsize=(8, 12),
    hspace=-0.5,
    cmap='viridis',
    fill_alpha=0.55,
    line_width=1,
    xlabel='z$_{e2e}$, nm',
    fontsize=10,
    dpi=300,
    show_plot=True,
    text_offset=0.05,
    ticksize=10,
    yscale=None,
):
    """
    Ridge-plot utility for CAR KDE dictionaries.

    probab_cars format:
        {
            hinge_length: {
                'kdes': array-like of shape (n_kdes, n_points) or (n_points,),
                'xval': array-like of shape (n_points,) or (n_points, 1)
            },
            ...
        }
    """
    if not isinstance(probab_cars, dict) or len(probab_cars) == 0:
        raise ValueError("probab_cars must be a non-empty dictionary.")

    if hinge_order is None:
        hinge_order = sorted(probab_cars.keys())

    n_hinges = len(hinge_order)
    fig, axes = plt.subplots(n_hinges, 1, figsize=figsize, dpi=dpi, sharex=True)
    if n_hinges == 1:
        axes = [axes]
    if cmap in list(matplotlib.colormaps):
        cmap_obj = matplotlib.colormaps[cmap]
    else:
        cmap_obj = None

    for idx, hinge_len in enumerate(hinge_order):
        if hinge_len not in probab_cars:
            raise ValueError(f"Hinge length `{hinge_len}` is missing in probab_cars.")

        entry = probab_cars[hinge_len]
        if 'kdes' not in entry or 'xval' not in entry:
            raise ValueError(f"probab_cars[{hinge_len}] must contain `kdes` and `xval`.")

        xvals = np.asarray(entry['xval']).reshape(-1)
        kdes = np.asarray(entry['kdes'], dtype=float)
        if kdes.ndim == 1:
            yvals = kdes
        elif kdes.ndim == 2:
            yvals = np.mean(kdes, axis=0)
        else:
            raise ValueError(f"`kdes` for hinge `{hinge_len}` must be 1D or 2D.")

        if xvals.shape[0] != yvals.shape[0]:
            raise ValueError(
                f"xval and KDE length mismatch for hinge `{hinge_len}`."
            )

        ax = axes[idx]

        if cmap_obj is None:
            color = cmap
        else:
            color = cmap_obj(idx / max(n_hinges - 1, 1))

        ax.fill_between(xvals, yvals, 0, color=color, alpha=fill_alpha)
        ax.plot(xvals, yvals, color=color, lw=line_width)
        if yscale is None:
            ax.set_ylim([0,max(yvals)+0.02])
            ax.set_yticks([])
            ax.tick_params(axis='y', which='both', length=0)
        else:
            ax.set_yscale("log")
            ax.set_ylim([1e-3,max(yvals)+0.02])
            ax.set_yticks([])
            ax.tick_params(axis='y', which='both', length=0)
            
        ax.set_xlim([0,max(xvals)+0.02])

        ax.patch.set_alpha(0)
        ax.spines['top'].set_visible(False)

        if idx != n_hinges - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        if idx == n_hinges:
            ax.spines['bottom'].set_visible(True)
            ax.set_xticks(fontsize=ticksize)
        if idx == 0:
            ax.spines['top'].set_visible(True)

        ax.text(-text_offset,0.,f"{hinge_len}",transform=ax.transAxes,fontsize=fontsize)

    axes[-1].set_xlabel(xlabel,fontsize=fontsize)

    fig.subplots_adjust(hspace=hspace)

    if show_plot:
        plt.show()

    return fig, axes