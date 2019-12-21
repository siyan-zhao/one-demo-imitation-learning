import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from collections import namedtuple
import pickle


def make_plot(rl_baseline, curves_means, curve_vars, curve_labels, y_axis_range, title, save_file_name, legend=False,
              x_lab="Iteration", y_lab="Mean Reward", x_axis_spacing=5):

    DPI = 300

    rcParams['axes.labelpad'] = 15

    colors = ['#49b8ff', '#ff7575', '#66c56c', '#f4b247', "#9A7B98", '#000000']
    line_style = ['-', '-.', '--', ':', "--", ":."]
    plt.figure(figsize=(9, 5))
    plt.title(title, fontsize=15)
    for curve_mean, curve_var in zip(curves_means, curve_vars):

        if curve_labels is not None:
            curve_label = curve_labels.pop(0)
        else:
            curve_label = None

        c = colors.pop(0)
        ls = line_style.pop(0)
        l = len(curve_mean)
        ts = np.linspace(0, x_axis_spacing * l, l + 1)[1:]
        ts2 = []
        for t in ts:
            ts2.append(t - x_axis_spacing)
        ts = ts2
        #print(ts)
        #kkkk
        plt.plot(ts, curve_mean, ls, color=c, linewidth=2, label=curve_label)

        #mean_plus_var = curve_mean + curve_var[1]
        #mean_minus_var = curve_mean - curve_var[0]
        mean_minus_var = curve_var[0]
        mean_plus_var = curve_var[1]
        plt.fill_between(ts, mean_plus_var, mean_minus_var, facecolor=c, alpha=0.1, interpolate=False)
        if legend:
            plt.legend(loc="best", bbox_to_anchor=(1.0, 1), framealpha=1, frameon=False, fontsize=14)
        plt.xlabel(x_lab, fontsize=14)
        plt.ylabel(y_lab, fontsize=14)
        plt.ylim(*y_axis_range)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.savefig('plots/' + save_file_name + '.png', dpi=DPI, bbox_inches="tight")


def load_curves_and_compute_mean_and_var(curve_dirs, clip_range, quantile_low=30, quantile_high=70):
    all_means = []
    all_vars = []
    for cd in curve_dirs:
        one_curve = np.load('results/' + cd + '.npz', allow_pickle=True)
        rews = one_curve['test_rews']
        #mean = np.mean(rews, axis=-1)
        whatever = True
        if whatever:
            r_smoothed = []
            for i in range(rews.shape[1]):
                r_new = smooth2(rews[:, i], radius=1)
                r_smoothed.append(r_new)
            r_smoothed = np.array(r_smoothed)
            r_smoothed[:, 0] = clip_range[0]
            mean = np.mean(r_smoothed, axis=0)
            rews = r_smoothed
            mean = np.clip(mean, *clip_range)
            all_means.append(mean)
            mean_minus_37 = np.percentile(rews, quantile_low,
                                          axis=0)  # This also nabs 75% of the data without assuming normality.
            mean_plus_37 = np.percentile(rews, quantile_high, axis=0)
            var_low = mean_minus_37
            var_high = mean_plus_37
            var_low = np.clip(var_low, *clip_range)
            var_high = np.clip(var_high, *clip_range)

            #var_low = mean - (var_low/mean)*0.3
            #var_high = mean + (var_high/mean)*0.3

            all_vars.append([var_low, var_high])

        else:
            mean = np.mean(rews, axis=-1)
            var = 1.7*np.var(rews, axis=-1)  # capture 95% of data.
            var_low = mean - var
            var_high = mean + var
            all_means.append(mean)
            all_vars.append([var_low, var_high])


            #var_low = (mean - np.min(rews, axis=-1))*0.2 + mean
        #var_high = (mean + np.max(rews, axis=-1))*0.2 + mean
    return all_means, all_vars



def smooth2(y, radius):
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    convkernel = np.ones(2 * radius + 1)
    out = np.zeros_like(y)
    out[:] = np.nan
    out[radius:-radius] = np.convolve(y, convkernel, mode='valid') / np.convolve(np.ones_like(y), convkernel,
                                                                                 mode='valid')
    return out[radius:len(y)-radius]



def plot_point_sawyer():
    rl_baseline = 0.15
    curve_plot_names = ["One-Demo (Ours)", "Behavioral Cloning (1 demonstration)",
                        "MLP + Last Obs Encoding + Diversity", "One Shot Imitation + ICM", "SAC (RL baseline)"]
    y_axis_range = [-0.63, 0.17]
    clip_range = [-0.6, 0.2]
    title = "Sawyer Pick"
    save_file_name = 'sawyer_pick_learning_curve'

    curve_save_dir_names = ['sawyer', 'sawyer_bc_baseline',
                            "sawyerbaseline_mlp_plus_last_state_encoding",
                            "sawyer_rnn_baseline"]

    #one_curve = np.load('results/' + "sawyer_rnn_baseline" + '.npz', allow_pickle=True)
    #one_curve = one_curve['test_rews']
    #kkkk

    curve_means, curve_vars = load_curves_and_compute_mean_and_var(curve_save_dir_names, clip_range=clip_range,
                                                                   quantile_low=25, quantile_high=85)

    rl_baseline = rl_baseline*np.ones_like(curve_means[0])

    curve_means.append(rl_baseline)
    curve_vars.append([rl_baseline, rl_baseline])

    make_plot(rl_baseline=rl_baseline, curves_means=curve_means, curve_vars=curve_vars,
              curve_labels=curve_plot_names, y_axis_range=y_axis_range, title=title, save_file_name=save_file_name,
              legend=False, x_axis_spacing=20)


def plot_point():
    rl_baseline = -0.61
    curve_plot_names = ["One-Demo (Ours)", "Behavioral Cloning (1 demonstration)",
                        "MLP + Last Obs Encoding + Diversity", "One Shot Imitation + ICM", "SAC (RL baseline)"]
    y_axis_range = [-1.3, -0.6]
    clip_range = [-1.32, -0.62]
    title = "Point"
    save_file_name = 'point_learning_curve'

    curve_save_dir_names = ['point', 'point_bc_baseline',
                            "pointbaseline_mlp_plus_last_state_encoding",
                            "point_rnn_baseline"]

    #one_curve = np.load('results/' + "sawyer_rnn_baseline" + '.npz', allow_pickle=True)
    #one_curve = one_curve['test_rews']
    #kkkk

    curve_means, curve_vars = load_curves_and_compute_mean_and_var(curve_save_dir_names, clip_range=clip_range,
                                                                   quantile_low=30, quantile_high=70)


    rl_baseline = rl_baseline*np.ones_like(curve_means[0])

    curve_means.append(rl_baseline)
    curve_vars.append([rl_baseline, rl_baseline])

    make_plot(rl_baseline=rl_baseline, curves_means=curve_means, curve_vars=curve_vars,
              curve_labels=curve_plot_names, y_axis_range=y_axis_range, title=title, save_file_name=save_file_name,
              legend=False, x_axis_spacing=1)



def plot_point_distractor():
    rl_baseline = -0.25
    curve_plot_names = ["One-Demo (Ours)", "Behavioral Cloning (1 demonstration)",
                        "MLP + Last Obs Encoding + Diversity", "One Shot Imitation + ICM", "SAC (RL baseline)"]
    y_axis_range = [-1.27, -0.15]
    clip_range = [-1.22, -0.12]
    title = "Point Multi Goal"
    save_file_name = 'point_distractor_learning_curve'

    curve_save_dir_names = ['point_distractor_better', 'point_distractor_bc_baseline',
                            "point_distractorbaseline_mlp_plus_last_state_encoding",
                            "point_rnn_baseline"]

    #one_curve = np.load('results/' + "sawyer_rnn_baseline" + '.npz', allow_pickle=True)
    #one_curve = one_curve['test_rews']
    #kkkk

    curve_means, curve_vars = load_curves_and_compute_mean_and_var(curve_save_dir_names, clip_range=clip_range)

    rl_baseline = rl_baseline*np.ones_like(curve_means[0])

    curve_means.append(rl_baseline)
    curve_vars.append([rl_baseline, rl_baseline])

    make_plot(rl_baseline=rl_baseline, curves_means=curve_means, curve_vars=curve_vars,
              curve_labels=curve_plot_names, y_axis_range=y_axis_range, title=title, save_file_name=save_file_name,
              legend=False, x_axis_spacing=5)


def plot_reacher():
    rl_baseline = -0.0
    curve_plot_names = ["One-Demo (Ours)", "Behavioral Cloning (1 demonstration)",
                        "MLP + Last Obs Encoding + Diversity", "One Shot Imitation + ICM", "SAC (RL baseline)"]
    y_axis_range = [-3.1, 0.1]
    clip_range = [-3.0, 0.0]
    title = "Reacher"
    save_file_name = 'reacher_learning_curve'

    curve_save_dir_names = ['reacher_better', 'reacher_bc_baseline',
                            "reacherbaseline_mlp_plus_last_state_encoding_better",
                            "reacher_rnn_baseline"]

    #one_curve = np.load('results/' + "sawyer_rnn_baseline" + '.npz', allow_pickle=True)
    #one_curve = one_curve['test_rews']
    #kkkk

    curve_means, curve_vars = load_curves_and_compute_mean_and_var(curve_save_dir_names, clip_range=clip_range,
                                                                   quantile_low=10, quantile_high=90)

    curve_means[3] = curve_means[3] + 0.12  # we do this simply to move the reacher and rnn baseline off one another.

    rl_baseline = rl_baseline*np.ones_like(curve_means[0])

    curve_means.append(rl_baseline)
    curve_vars.append([rl_baseline, rl_baseline])

    make_plot(rl_baseline=rl_baseline, curves_means=curve_means, curve_vars=curve_vars,
              curve_labels=curve_plot_names, y_axis_range=y_axis_range, title=title, save_file_name=save_file_name,
              legend=False, x_axis_spacing=5)


def plot_reacher_dis():
    rl_baseline = -0.0
    curve_plot_names = ["One-Demo (Ours)", "Behavioral Cloning (1 demonstration)",
                        "MLP + Last Obs Encoding + Diversity", "One Shot Imitation + ICM", "SAC (RL baseline)"]
    y_axis_range = [-0.3, 0.05]
    clip_range = [-0.28, 0.0]
    title = "Reacher Multi Goal"
    save_file_name = 'reacher_dist_learning_curve'

    curve_save_dir_names = ['reacher_dis', 'reacher_dis_bc_baseline',
                            "reacher_disbaseline_mlp_plus_last_state_encoding",
                            "reacher_rnn_baseline"]

    #one_curve = np.load('results/' + "sawyer_rnn_baseline" + '.npz', allow_pickle=True)
    #one_curve = one_curve['test_rews']
    #kkkk

    curve_means, curve_vars = load_curves_and_compute_mean_and_var(curve_save_dir_names, clip_range=clip_range,
                                                                   quantile_low=10, quantile_high=90)

    #curve_means[3] = curve_means[3] + 0.12  # we do this simply to move

    rl_baseline = rl_baseline*np.ones_like(curve_means[0])

    curve_means.append(rl_baseline)
    curve_vars.append([rl_baseline, rl_baseline])

    make_plot(rl_baseline=rl_baseline, curves_means=curve_means, curve_vars=curve_vars,
              curve_labels=curve_plot_names, y_axis_range=y_axis_range, title=title, save_file_name=save_file_name,
              legend=False, x_axis_spacing=5)



def plot_sawyer_reach():
    rl_baseline = -0.0
    curve_plot_names = ["One-Demo (Ours)", "Behavioral Cloning (1 demonstration)",
                        "MLP + Last Obs Encoding + Diversity", "One Shot Imitation + ICM", "SAC (RL baseline)"]
    y_axis_range = [-0.4, 0.05]
    clip_range = [-0.42, 0.02]
    title = "Sawyer Reach"
    save_file_name = 'sawyer_reach_learning_curve'

    curve_save_dir_names = ['sawyer_reach', 'sawyer_reach_bc_baseline',
                            "sawyer_reachbaseline_mlp_plus_last_state_encoding",
                            "sawyer_reach_rnn_baseline_better"]

    #one_curve = np.load('results/' + "sawyer_rnn_baseline" + '.npz', allow_pickle=True)
    #one_curve = one_curve['test_rews']
    #kkkk

    curve_means, curve_vars = load_curves_and_compute_mean_and_var(curve_save_dir_names, clip_range=clip_range,
                                                                   quantile_low=30, quantile_high=80)

    #curve_means[3] = curve_means[3] + 0.12

    rl_baseline = rl_baseline*np.ones_like(curve_means[0])

    curve_means.append(rl_baseline)
    curve_vars.append([rl_baseline, rl_baseline])

    make_plot(rl_baseline=rl_baseline, curves_means=curve_means, curve_vars=curve_vars,
              curve_labels=curve_plot_names, y_axis_range=y_axis_range, title=title, save_file_name=save_file_name,
              legend=True, x_axis_spacing=20)




def plot_hopper():
    rl_baseline = 5.0
    curve_plot_names = ["One-Demo (Ours)", "Behavioral Cloning (1 demonstration)",
                        "MLP + Last Obs Encoding + Diversity", "One Shot Imitation + ICM", "SAC (RL baseline)"]
    y_axis_range = [0.0, 2.0]
    clip_range = [0.02, 1.98]
    title = "Hopper"
    save_file_name = 'hopper_learning_curve'

    curve_save_dir_names = ['reacher_dis', 'hopper_bc_baseline',
                            "hopperbaseline_mlp_plus_last_state_encoding",
                            "sawyer_reach_rnn_baseline_better"]

    #one_curve = np.load('results/' + "sawyer_rnn_baseline" + '.npz', allow_pickle=True)
    #one_curve = one_curve['test_rews']
    #kkkk

    curve_means, curve_vars = load_curves_and_compute_mean_and_var(curve_save_dir_names, clip_range=clip_range,
                                                                   quantile_low=10, quantile_high=90)

    #curve_means[3] = curve_means[3] + 0.12

    rl_baseline = rl_baseline*np.ones_like(curve_means[0])

    curve_means.append(rl_baseline)
    curve_vars.append([rl_baseline, rl_baseline])

    make_plot(rl_baseline=rl_baseline, curves_means=curve_means, curve_vars=curve_vars,
              curve_labels=curve_plot_names, y_axis_range=y_axis_range, title=title, save_file_name=save_file_name,
              legend=False, x_axis_spacing=50)


def main():
    plot_sawyer_reach()


if __name__ == "__main__":
    main()
