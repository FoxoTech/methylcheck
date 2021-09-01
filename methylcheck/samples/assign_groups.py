import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import methylcheck

# dummy function for pytest to test assign()
def get_input(text):
    return input(text)

def assign(df, two_pass=False):
    """Manually and interactively assign each sample to a group, based on beta-value distribution shape.
    This function is not documented or supported anymore.

how:
    sorts samples by the position of their peak intensity in beta dist, then lets the human assign a number to
    each.
    assumes that peaks that differ by > 0.1 of the 0 to 1 range are different clusters. fills in expected cluster #.

options:
    if two_pass==True: it lets user go through every cluster a second time and split larger clusters further.
    sometimes a cluster isn't obviously two groups until it is separated from the larger dataset.
    """
    # calculate the highest bin of the sample beta histogram for each sample
    def calc_max_bin_of_hist(nparray):
        hist_vals, bin_edges = np.histogram(nparray, bins=16)
        # want the bin_edge corresponding to the max hist_val
        bin_of_max_val = np.where(hist_vals == np.amax(hist_vals))
        return bin_of_max_val[0][0] # it could occur in two bins; if so, return the first bin only
    colors = {-1:"xkcd:scarlet", 2:'xkcd:brick red', 1:"xkcd:blue", 2:"xkcd:green", 3:'xkcd:gold', 4:'xkcd:purple', 5:'xkcd:grey blue',
        6:'xkcd:brown', 7:'xkcd:bubblegum', 8:'xkcd:tangerine', 9:'xkcd:poop', 10:"xkcd:baby blue", 11:"xkcd:violet", 12:"xkcd:apple green",
        13:"xkcd:neon pink", 14:"xkcd:deep green", 15:"xkcd:blush", 16:"xkcd:lemon"}
    def sample_plot_color_clusters(df, colormap=None, verbose=False, save=False, silent=False, reduce=0.01, ymax=5, figsize=(14,8)):
        """ colormap: dict of sample_ids --> group number (some int). Sample_ids must match df columns"""
        #qualitative_colors = sns.color_palette("Set3", 10)
        #sns.set_color_codes()
        # ensure sample ids are unique
        if df.shape[0] > df.shape[1]:
            df = df.transpose()
            # must have samples in rows/index to reindex. you can't "recolumn".
            if list(df.index) != list(set(df.index)):
                LOGGER.info("Your sample ids contain duplicates.")
                # rename these for this plot only
                df['ids'] = df.index.map(lambda x: x + '_' + str(int(1000000*np.random.rand())))
                df = df.set_index('ids', drop=True)
                df = df.transpose()

        if df.shape[0] < df.shape[1]:
            ## ensure probes in rows and samples in cols
            df = df.transpose()
        if reduce != None and reduce < 1.0:
            if not isinstance(reduce, (int, float)):
                try:
                    reduce = float(reduce)
                except Exception as e:
                    raise ValueError(f"{reduce} must be a floating point number between 0 and 1.0")
            probes = np.random.choice(df.shape[0], int(reduce*df.shape[0]))
        fig, ax = plt.subplots(figsize=figsize) #was (12, 9)
        for idx, sample_id in enumerate(df.columns): # samples
            if sample_id != 'Name': # probe name
                if reduce:
                    values = df[sample_id].values[probes]
                else:
                    values = df[sample_id].values
                if len(values.shape) > 1:
                    raise ValueError("Your df probaby contains duplicate sample names.")

                this_color = colors[-1] if colormap is None else colors.get(colormap.get(sample_id,-1),'k')
                if idx+1 == len(df.columns):
                    kde_kws = {'linestyle':'--', 'linewidth':2}
                else:
                    kde_kws = {'linewidth':1}

                #sns.distplot(
                #    values, hist=False, rug=False, kde=True,
                #    ax=ax, axlabel='beta', color=this_color, kde_kws=kde_kws)
                sns.kdeplot(values, ax=ax, label='beta', color=this_color, **kde_kws)
        (data_ymin,data_ymax) = ax.get_ylim()
        if data_ymax > ymax:
            ax.set_ylim(0,ymax)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    # orient so samples in columns
    if len(df.columns) > df.shape[0]:
        df = df.transpose() # now more rows(probes) than cols(samples)

    def color_guide(user_defined_groups):
        color_names = {k: v.replace('xkcd:','') for k,v in colors.items()}
        color_count = Counter(user_defined_groups.values())
        guide = []
        for group,count in color_count.most_common():
            guide.append(f"{group} {color_names[group]}")
        return ' | '.join(guide)

    sample_predicted_groups = {} # sample_name --> bin where peak occurs | as a 0-10 number for clustering
    for sample_id in df.columns: # columns
        #betas[sample] = vectorized_func(meth, unmeth)
        sample_predicted_groups[sample_id] = calc_max_bin_of_hist(df[sample_id].values)
    print('Predicted sample groups, based on beta-peak position')
    prediction_note = " | ".join([f"{k}: {v}" for k,v in Counter(sample_predicted_groups.values()).most_common()])
    print(prediction_note)
    # sort by cluster frequency
    sample_predicted_groups_counts = Counter(sample_predicted_groups.values())
    sample_predicted_groups = dict(sorted(sample_predicted_groups.items(), key=lambda kv: sample_predicted_groups_counts[kv[1]], reverse=True))

    # go through each sample, plotting it, and letting user assign to one of the groups
    user_defined_groups = {}
    colormap = {}
    past_samples = []
    for sample_id,predicted_group in sample_predicted_groups.items():
        past_samples.append(sample_id)
        sub_sample_df = pd.DataFrame(data = df[past_samples], index = df.index).transpose()
        sample_plot_color_clusters(sub_sample_df, colormap=colormap, reduce=0.01)
        print(color_guide(user_defined_groups))
        while True:
            g = get_input(f"Group ({predicted_group}): ")
            if g in (""," "):
                g = predicted_group
            try:
                g = int(g)
            except:
                print('Only use numbers between -1 and 16!')
                g = get_input(f"Group ({predicted_group}): ")
            if g not in colors:
                print('Only use numbers between -1 and 16!')
                g = get_input(f"Group ({predicted_group}): ")
            else:
                break
        user_defined_groups[sample_id] = g
        if g == -1:
            g = 10
        colormap[sample_id] = g

    # finally, produce a table and plot of each cluster:
    clusters = list(set(user_defined_groups.values()))
    cluster_samples = {}
    for cluster in clusters:
        samples = [sample_id for sample_id in df.columns if user_defined_groups.get(sample_id) == cluster]
        sub_sample_df = pd.DataFrame(data = df[samples], index = df.index)
        print(f"Group ({cluster}) {len(samples)} samples: {samples}")
        methylcheck.sample_plot(sub_sample_df)
        if two_pass:
            yesno = get_input(f"Rework Group ({cluster}) (y/n)?")
            if yesno in ('y','Y','yes'):
                past_samples = []
                for sample_id in samples:
                    past_samples.append(sample_id)
                    sub_sample_df = pd.DataFrame(data = df[past_samples], index = df.index).transpose()
                    sample_plot_color_clusters(sub_sample_df, colormap=colormap, reduce=0.01)
                    print(color_guide(user_defined_groups))
                    while True:
                        g = get_input(f"Group ({cluster}): ")
                        if g in (""," "):
                            g = cluster
                        try:
                            g = int(g)
                        except:
                            print('Only use numbers between -1 and 16!')
                            g = get_input(f"Group ({cluster}): ")
                        if g not in colors:
                            print('Only use numbers between -1 and 16!')
                            g = get_input(f"Group ({cluster}): ")
                        else:
                            break
                    user_defined_groups[sample_id] = g
                    if g == -1:
                        g = 10
                samples = [sample_id for sample_id in df.columns if user_defined_groups.get(sample_id) == cluster]
                sub_sample_df = pd.DataFrame(data = df[samples], index = df.index)
                print(f"Group ({cluster}) {len(samples)} samples: {samples} REVISED")
                methylcheck.sample_plot(sub_sample_df)
        cluster_samples[cluster] = samples
    return user_defined_groups

def plot_assigned_groups(df, user_defined_groups):
    """ takes the 'sample: group' dictionary and plots each sub-set.
    returns a lookup dict of each cluster --> [list of samples]."""
    # finally, produce a table and plot of each cluster:
    clusters = list(set(user_defined_groups.values()))
    cluster_samples = {}
    for cluster in clusters:
        samples = [sample_id for sample_id in df.columns if user_defined_groups.get(sample_id) == cluster]
        sub_sample_df = pd.DataFrame(data = df[samples], index = df.index)
        plot_title = f"Group ({cluster}) {len(samples)} samples"
        methylcheck.beta_density_plot(sub_sample_df, plot_title=plot_title)
        cluster_samples[cluster] = samples
    return cluster_samples
