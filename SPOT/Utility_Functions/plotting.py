# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 23:54:37 2023

@author: S205272
"""

import pylab as plt 
import numpy as np


def get_colors(inp, colormap, vmin=None, vmax=None, bg_label=None):

	r""" Maps a given numpy input array with the specified Matplotlib colormap with optional specified minimum and maximum values. For an array that is integer such as that from multi-label segmentation, bg_label helps specify the background class which will automatically be mapped to a background color of [0,0,0,0]

    Parameters
    ----------
    inp : numpy array
        input n-d array to color
  	colormap :  matplotlib.cm colormap object
        colorscheme to apply e.g. cm.Spectral, cm.Reds, cm.coolwarm_r
    vmin : int/float
        specify the optional value to map as the minimum boundary of the colormap
    vmax : int/float
    	specify the optional value to map as the maximum boundary of the colormap
    bg_label: int
    	for an input array that is integer such as a segmentation mask, specify which integer label to mark as background. These values will all map to [0,0,0,0]

    Returns
    -------
    colored : numpy array
        the colored version of input as RGBA, the 4th being the alpha. colors are specified as floats in range 0.-1.
    """
	import pylab as plt 
	norm = plt.Normalize(vmin, vmax)

	colored = colormap(norm(inp))
	if bg_label is not None:
		colored[inp==bg_label] = 0 # make these all black!

	return colored

def draw_affinity_similarity_graph_networkx(matrix, ax, 
                                            node_positions=None, 
                                            node_names=None,
                                            edge_color=(0,0,0,1),
                                            font_size=18):
    
    r"""
    Parameters
    ----------
    matrix : numpy array
        an affinity, distance or connectivity matrix 
    ax : matplotlib axes object
        matplotlib axes to plot figure onto
    node_positions : (len(matrix), 2) array
        if not None, the (x,y) coordinates to place node in the plotting. 
    node_names : dict       
        if not None, a dictionary of the mapping between node id and the name of the nodes
    edge_color : rgba tuple
        color of the drawn nodes as an rgba value 
    font_size : int 
        size of the label text 

    Returns
    -------
    G : networkx graph object
        input matrix converted to networkx using nx.from_numpy_array
    
    """
    import networkx as nx 
    import numpy as np 
    
    # convert matrix to networkx object. 
    G = nx.from_numpy_array(matrix)
    
    # nx.draw(G)
    # fig, ax = plt.subplots(figsize=(5,5))
    if node_names is None:
        node_names = {ii: ii for ii in np.arange(len(matrix))}
    # if node_positions is None:
    #     pos = nx.kamada_kawai_layout(G, pos=mean_uu_all)
    
    nx.draw(G, 
            labels=node_names, 
            pos = node_positions,
            with_labels=True, 
            font_weight='bold',
            edge_color=edge_color,
            ax=ax,
            font_size=font_size)
    
    return G

# pos = nx.kamada_kawai_layout(G, pos=mean_uu_all)
    
#     fig, ax = plt.subplots(figsize=(15,15))
#     nx.draw(G, labels={ii: all_labels_names[ii] for ii in np.arange(len(all_labels_names))}, 
#             pos = pos,
#             with_labels=True,
#             node_size=5, 
#             # node_color='k',
#             edge_color=(0,0,0,0.1),
#             # node_color='r',
#             # font_weight='bold',
#             font_family='Arial',
#             font_size=18, 
#             clip_on=False,
#             ax=ax)
#     # plt.savefig(os.path.join(saveplotsfolder_cells, 
#     #                          'UMAP-actor-action_coord_relationships_spread.pdf'), dpi=300, bbox_inches='tight')
#     plt.savefig(os.path.join(saveplotsfolder_cells, 
#                              '%s-actor-action_coord_relationships_spread.png' %(method_name)), dpi=600, bbox_inches='tight')
#     plt.show()


def plot_image_patches(positions,
                       patches,
                       ax,
                       subsample=10,
                       zoom=0.25):
    
    r""" visualize image patches onto the first two coordinates of positions 
    
    Parameters 
    ----------
    positions: (N_objects, N_dim) array 
        The first two dimension are taken as the x-, y- coordinate positions to place patches
    patches: list or array
        list or array of 2D image patches, the same number as the number of positions. If grayscale, patches will be replicated to produce a gray image
    ax: matplotlib axes object
        matplotlib axes object, which we will plot on 
    subsample: int
        this is to avoid plotting every patch for large datasets. It will instead plot every subsample patch.
    zoom : float
        this tunes the size of the plotted patches, increase this to make patches larger to fill empty space, or decrease to decrease overlap with neighboring positioned patches    
    
    Returns 
    -------
    inp_colored : inp.shape + (4,) array 
        RGBA colored input array
    
    """
    
    import numpy as np 
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        
    ax.scatter(positions[:,0], 
               positions[:,1], c='lightgrey')

    for jj in np.arange(0,len(positions),subsample):
        patch = patches[jj]
        if len(patch.shape) == 2:
            patch = np.dstack([patch, patch, patch])
        im = OffsetImage(patch, zoom=zoom)
        x, y = positions[jj]
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)
            
    # ax.axis("square")
    ax.axis("off")
    
    return []

# plots a simple barplot of cluster proportions
def barplot_cluster_proportions(cluster_labels, 
                                all_conditions, 
                                unique_conditions, 
                                clust_labels_colors=None,
                                normalize_counts=True,
                                figsize=(7,5),
                                saveplotsfolder=None):
    
    r""" Plot a barplot of the proportions per cluster for each specified unique condition
    
    Parameters
    ----------
    cluster_labels : 
        
    all_conditions : (N_objects,) array
        the experimental condition label of each object instance 
    unique_conditions : (N_unique_objects,) array 
        the unique conditions in the experiment to plot the barplots for
        
        
    Returns
    -------
    None
    
    """
    import numpy as np 
    import seaborn as sns 
    import os 
    import pylab as plt
    
    uniq_clust_labels = np.unique(cluster_labels)
    
    if clust_labels_colors is None:
        clust_labels_colors = sns.color_palette('rainbow', len(uniq_clust_labels))
    
    all_hist_lab = []
    
    for cond_ii, cond in enumerate(unique_conditions):
        
        select = np.hstack(all_conditions)==cond
        lab_select = cluster_labels[select].copy()
        
        hist_lab = np.hstack([np.sum(lab_select==lab) for lab in uniq_clust_labels])
        
        if normalize_counts:
            hist_lab = hist_lab/float(np.sum(hist_lab)) # normalize across clusters. 
        
        all_hist_lab.append(hist_lab)
        
        fig, ax = plt.subplots(figsize=(7,5))
        plt.title(cond)
        for jjjj in np.arange(len(hist_lab)):
            ax.bar(np.arange(len(hist_lab))[jjjj], 
                (hist_lab[jjjj]),
                color=clust_labels_colors[jjjj],
                width=1., linewidth=3, edgecolor='k')
        
        plt.xlim([-0.5,len(uniq_clust_labels)-0.5])
        plt.ylim([0,0.3])
        plt.xticks(fontsize=14, fontname='Arial')
        plt.yticks(fontsize=14, fontname='Arial')
        plt.tick_params(axis='both', length=10, right=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if saveplotsfolder is not None: 
            plt.savefig(os.path.join(saveplotsfolder,
                                     cond+'_per_cluster_proportion.svg'), bbox_inches='tight')
        plt.show()
        
    all_hist_lab = np.array(all_hist_lab)
        
    return uniq_clust_labels, all_hist_lab
    

def stacked_barplot_temporal_cluster_proportions(cluster_proportions_time, 
                                                 unique_conditions, 
                                                 time_intervals = None, 
                                                 clust_labels_colors=None,
                                                 figsize=(7,5),
                                                 saveplotsfolder=None):
    
    r""" Plot a stacked barplot of the proportions per cluster for each specified unique condition over time 
    
    Parameters
    ----------
    cluster_proportions_time : (n_unique_condition, n_unique_clusters, n_time_intervals) array 
        the fractional proportion of objects in each time interval in each phenotype cluster
    unique_conditions : list or array
        the unique conditions in the experiment. If not specified, the unique conditions will be obtained by np.unique(all_conditions)     
    time_intervals : lenght (n_time_intervals+1,) list or array 
        the global time interval sampling, the density of objects mapping to the phenomic landscape within each interval will be used to construct the 'majority' phenomic coordinate to represent the phenotypic diversity in that interval.
    figsize : 2-tuple
        size of figure canvas
    saveplotsfolder : str
        if specified, will save the generated heatmap figures to this folder 
    
    Returns
    -------
    None
    
    """
    
    import numpy as np 
    import pylab as plt 
    import os 
    import seaborn as sns 
    
    if clust_labels_colors is None:
        clust_labels_colors = sns.color_palette('Spectral', cluster_proportions_time.shape[-1])
    
    
    # iterate over each condition 
    for jjjj in np.arange(len(cluster_proportions_time))[:]:
        
        cluster_order = np.arange(cluster_proportions_time.shape[-1])[::-1]
        data = cluster_proportions_time[jjjj].copy()
        data = data[:, cluster_order].copy() # put in the correct order for plotting. 
        # create the cumulative totals for bar graph plotting. 
        data_cumsums = np.cumsum(data, axis=1)

        fig, ax = plt.subplots(figsize=(5,5))
        plt.title(unique_conditions[jjjj])
        # this requires us to build and program the composition up. in order of the histogram 
        for ord_ii in np.arange(data.shape[1]):
            if ord_ii>0:
                ax.bar(np.arange(cluster_proportions_time.shape[1]), 
                        data[:,ord_ii], 
                        bottom = data_cumsums[:,ord_ii-1], 
                        # color=clust_labels_colors[order_hist[::-1][ord_ii]],
                        color=clust_labels_colors[::-1][ord_ii],
                        width=1,
                        edgecolor='k')
            else:
                ax.bar(np.arange(cluster_proportions_time.shape[1]), 
                        data[:,ord_ii], 
                        # color=clust_labels_colors[order_hist[::-1][ord_ii]],
                        color=clust_labels_colors[::-1][ord_ii],
                        width=1,
                        edgecolor='k')
        
        if time_intervals is not None:
            plt.xticks(np.arange(len(time_intervals))-0.5, 
                       time_intervals, rotation=90)    
        plt.xlim([0-0.5,data.shape[0]-0.5])
        plt.ylim([0,1])

        if saveplotsfolder is not None:
            plt.savefig(os.path.join(saveplotsfolder,
                                     str(unique_conditions[jjjj])+'_temporal_cluster_composition_stackedbarplots.svg'), bbox_inches='tight',dpi=300)
        plt.show()
        
    return []


def barplot_cluster_statistics(cluster_scores,
                               featnames=None,
                               colormap=plt.cm.coolwarm,
                               style='hbar',
                               shared_canvas = False,
                               figsize=(1,4),
                               vmin=-2, 
                               vmax=2,
                               nticks=5,
                               save_dpi=300, 
                               saveplotsfolder=None):
    
    r""" Convenience function to produce barplot of the mean scores per cluster 
    
    Parameters 
    ----------
    cluster_scores: (N_clusters, N_feats) array 
        The mean feature value per unique cluster
    featnames : (N_feats,) array
        The names of the features, if specified will be added to annotate the bars. 
    colormap : matplotlib.cm colormap object
        The colormap to use to color the bar according to value. default = matplotlib.cm.coolwarm
    style: 'hbar' or 'vbar'
        if 'hbar' plot horizontal bar or 'vbar' which plots vertical bars
    shared_canvas : bool
        if True, result will be plotted onto a single figure object
    figsize : 2-tuple
        the size and aspect ratio of the figure axis 
    vmin : float
        minimum x-axis value to plot 
    vmax : float
        maximum x-axis value to plot 
    nticks : int
        the number of ticks to use on x-axis
    save_dpi : int 
        if saveplotsfolder variable is specified, this parameter determines the dpi of the saved figure
    saveplotsfolder : folderpath
        if not None, the plots will be saved to this folder 
    
    Returns 
    -------
    plot_objs : matplotlib figure object
        list of (fig, ax) object tuples per cluster. fig, ax is the same as when calling fig, ax = plt.subplots()
    """
    import pylab as plt 
    import os 
    
    plot_objs = []
    # iterate over the phenotype cluster.
    
    if shared_canvas: 
        
        if style =='hbar':
            fig, ax = plt.subplots(nrows=1, ncols=cluster_scores.shape[0], figsize=figsize)
        
            for ii in np.arange(cluster_scores.shape[0]):
                data = cluster_scores[ii,::-1]
                
                bar_colors = get_colors(data, colormap=colormap, 
                                    vmin=vmin, vmax=vmax)
            
                for jj in np.arange(len(data)):
                    ax[ii].barh(jj, 
                            data[jj], edgecolor='k', color=bar_colors[jj])
                    
                ax[ii].vlines(0, -0.5, cluster_scores.shape[1]-0.5, color='k')
                
                ax[ii].set_yticks(np.arange(cluster_scores.shape[1]))
                if ii == 0: 
                    ax[ii].set_yticklabels(featnames[::-1])
                else:
                    ax[ii].set_yticklabels([])
                ax[ii].set_xlim([vmin,vmax])
                ax[ii].set_ylim([-0.5, cluster_scores.shape[1]-0.5])
                
                # make the right and top spines away. 
                ax[ii].spines['right'].set_visible(False)
                ax[ii].spines['top'].set_visible(False)
                ax[ii].set_xticks(np.linspace(vmin,vmax,nticks))

            if saveplotsfolder is not None:
                plt.savefig(os.path.join(saveplotsfolder,
                                          'Mean_SAM_features-Cluster_all_hbar.svg'), 
                            dpi=save_dpi, 
                            bbox_inches='tight')
            plt.show()
            
            plot_objs.append((fig, ax))
            
        elif style == 'vbar':
            
            fig, ax = plt.subplots(nrows=cluster_scores.shape[0], ncols=1, figsize=figsize)
            
            for ii in np.arange(cluster_scores.shape[0]):   
                data = cluster_scores[ii]

                bar_colors = get_colors(data, colormap=colormap, 
                                    vmin=vmin, vmax=vmax)
            
                for jj in np.arange(len(data)):
                    ax[ii].bar(jj, 
                            data[jj], edgecolor='k', color=bar_colors[jj])
                    
                ax[ii].hlines(0, -0.5, cluster_scores.shape[1]-0.5, color='k')
                ax[ii].set_xticks(np.arange(cluster_scores.shape[1]))
                if ii == cluster_scores.shape[0]-1: 
                    ax[ii].set_xticklabels(featnames[:], rotation=90)
                else:
                    ax[ii].set_xticklabels([])
                ax[ii].set_ylim([vmin,vmax])
                ax[ii].set_xlim([-0.5, cluster_scores.shape[1]-0.5])
                
                ax[ii].spines['right'].set_visible(False)
                ax[ii].spines['top'].set_visible(False)
                ax[ii].set_yticks(np.linspace(vmin,vmax,nticks))

                
            if saveplotsfolder is not None:
                plt.savefig(os.path.join(saveplotsfolder,
                                          'Mean_SAM_features-Cluster_all_vbar.svg'), 
                            dpi=save_dpi, 
                            bbox_inches='tight')
            plt.show()
            
            plot_objs.append((fig, ax))
            
        else:
            
            fig, ax = plt.subplots(nrows=1, ncols=cluster_scores.shape[0], figsize=figsize)
        
            for ii in np.arange(cluster_scores.shape[0]):
                data = cluster_scores[ii,::-1]
                
                bar_colors = get_colors(data, colormap=colormap, 
                                    vmin=vmin, vmax=vmax)
            
                for jj in np.arange(len(data)):
                    ax[ii].barh(jj, 
                            data[jj], edgecolor='k', color=bar_colors[jj])
                    
                ax[ii].vlines(0, -0.5, cluster_scores.shape[1]-0.5, color='k')
                
                ax[ii].set_yticks(np.arange(cluster_scores.shape[1]))
                if ii == 0: 
                    ax[ii].set_yticklabels(featnames[::-1])
                else:
                    ax[ii].set_yticklabels([])
                ax[ii].set_xlim([vmin,vmax])
                ax[ii].set_ylim([-0.5, cluster_scores.shape[1]-0.5])
                
                # make the right and top spines away. 
                ax[ii].spines['right'].set_visible(False)
                ax[ii].spines['top'].set_visible(False)
                ax[ii].set_xticks(np.linspace(vmin,vmax,nticks))

            if saveplotsfolder is not None:
                plt.savefig(os.path.join(saveplotsfolder,
                                          'Mean_SAM_features-Cluster_all_hbar.svg'), 
                            dpi=save_dpi, 
                            bbox_inches='tight')
            plt.show()
            
            plot_objs.append((fig, ax))
    else:
        
        
        for ii in np.arange(cluster_scores.shape[0]):
    
            # map the values to a colormap namely blue and red.         
    
            fig, ax = plt.subplots(figsize=figsize)
            
            if style =='hbar':
                data = cluster_scores[ii,::-1]
            elif style == 'vbar':
                data = cluster_scores[ii]
            else:
                data = cluster_scores[ii,::-1] # default. 
            
            bar_colors = get_colors(data, colormap=colormap, 
                                    vmin=vmin, vmax=vmax)
            
            for jj in np.arange(len(data)):
                
                if style =='hbar':
                    ax.barh(jj, 
                            data[jj], edgecolor='k', color=bar_colors[jj])
                    plt.vlines(0, -0.5, cluster_scores.shape[1]-0.5, color='k')
                    plt.yticks(np.arange(cluster_scores.shape[1]), 
                                featnames[::-1])
                    plt.xlim([vmin,vmax])
                    plt.ylim([-0.5, cluster_scores.shape[1]-0.5])
                elif style == 'vbar':
                    offset = 2 # not sure why we need this. 
                    ax.bar(jj-offset, 
                           data[jj], edgecolor='k', color=bar_colors[jj], align='center')
                    plt.hlines(0, -0.5-offset, cluster_scores.shape[1]-0.5-offset, color='k')
                    plt.xticks(np.arange(cluster_scores.shape[1]), 
                                featnames, rotation=90)
                    plt.ylim([vmin,vmax])
                    plt.xlim([-0.5-offset, cluster_scores.shape[1]-0.5-offset])
                else:
                    ax.barh(jj, 
                            data[jj], edgecolor='k', color=bar_colors[jj])
                    plt.vlines(0, -0.5, cluster_scores.shape[1]-0.5, color='k')
                    plt.yticks(np.arange(cluster_scores.shape[1]), 
                                featnames[::-1])
                    plt.xlim([vmin,vmax])
                    plt.ylim([-0.5, cluster_scores.shape[1]-0.5])
                    
                
            # make the right and top spines away. 
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticks(np.linspace(vmin,vmax,nticks))
            
            if saveplotsfolder is not None:
                plt.savefig(os.path.join(saveplotsfolder,
                                          'Mean_SAM_features-Cluster_%s.svg' %(str(ii).zfill(3))), 
                            dpi=save_dpi, 
                            bbox_inches='tight')
            plt.show()
            
            plot_objs.append((fig, ax))
        
    return plot_objs


