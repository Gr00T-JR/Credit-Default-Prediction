import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
# from scratch_3 import select_plot_data

with open('results/results.pkl', 'rb') as f:
    MODELS_PERFORMANCES = pickle.load(f)

# print(MODELS_PERFORMANCES)

model_name = 'ada'

model_name_list = MODELS_PERFORMANCES.keys()

tuning_measures_list = ['accuracy tuning', 'roc auc tuning']
tuning_methods_dict = {'train_validation_split' : {}, 'train_validation_split_randomized' : {}}
performance_name_list = ['test accuracy', 'test roc auc', 'test r2']

def select_plot_data(tuning_method_list : list, tuning_method_value : float, max_evals : int = 25, performance_score : str = 'test accuracy', tuning_measures_list : list = tuning_measures_list) -> tuple:
    score_list = []
    for tuning_method_iterator in tuning_method_list:
        for tuning_measure_name in tuning_measures_list:
            score_list.append(MODELS_PERFORMANCES[model_name][tuning_method_iterator][tuning_method_value][tuning_measure_name][max_evals][performance_score])

    accuracy_tuned_scores = score_list[::2]
    roc_auc_tuned_scores = score_list[1::2]

    return accuracy_tuned_scores, roc_auc_tuned_scores

ratio_range = range(20,30+1,5)
max_evals_range = range(25,35+1,10)
random_state = 0

debug = False  # COPY pasted / to be removed -----
if debug:
    max_evals_range = range(1,1+1,1)
    ratio_range = range(20,25+1,5)



ratio_0 = 0.25
max_evals_0 = 25

tuning_method_name = 'train_validation_split'
tuning_measure_name = tuning_measures_list[0] # = 'accuracy tuning'

performance_scores = ['test accuracy' , 'test roc auc', 'test r2']
selected_performance_score = performance_scores[0]

print(MODELS_PERFORMANCES['ada'][tuning_method_name][0.25][tuning_measure_name])

selected_data = select_plot_data(tuning_method_list=list(tuning_methods_dict), tuning_method_value=ratio_0, max_evals=max_evals_0, performance_score=selected_performance_score)
accuracy_tuned_score_list, roc_auc_tuned_score_list = selected_data

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.20)

xticks_variables = list(tuning_methods_dict)

# The x position of bars
bar_width = 0.25
r1 = np.arange(len(xticks_variables))
r2 = [x + bar_width for x in r1]
positions = [r1, r2]

bars1 = ax.bar(r1, accuracy_tuned_score_list, color ='b', width = bar_width, label='accuracy tuned')
bars2 = ax.bar(r2, roc_auc_tuned_score_list, color ='g', width = bar_width, label='roc auc tuned')

# bars2.remove()
bars_list = [bars1, bars2]

ax.legend()

plt.ylim([0.82, 0.84])
plt.xticks(r1+bar_width/2, xticks_variables)
plt.xlabel('Tuning method')
plt.ylabel(selected_performance_score)


axcolor = 'lightgoldenrodyellow'
ax_ratio = plt.axes([0.15, 0.2, 0.03, 0.675], facecolor=axcolor)
s_ratio = Slider(ax=ax_ratio, label='ratio', valmin=ratio_range[0]/100, valmax=ratio_range[-1]/100, valinit=ratio_0, valstep=ratio_range.step/100, orientation='vertical')

ax_max_evals = plt.axes([0.05, 0.2, 0.03, 0.675], facecolor=axcolor)
s_max_evals = Slider(ax=ax_max_evals, label='max_evals', valmin=max_evals_range[0], valmax=max_evals_range[-1], valinit=max_evals_0, valstep=max_evals_range.step, orientation='vertical')

def bar_plot(y_label, data1, data2):

    for bars in bars_list:
        bars.remove()

    bars_list[0] = ax.bar(r1, data1, color='b', width=bar_width, label='accuracy tuned')
    bars_list[1] = ax.bar(r2, data2, color='g', width=bar_width, label='roc auc tuned')

    plt.ylabel(y_label)


def update(val):
    ratio = round(s_ratio.val*100)/100
    max_evals = s_max_evals.val
    updated_data = select_plot_data(tuning_method_list=list(tuning_methods_dict), tuning_method_value=ratio,  max_evals=max_evals, performance_score=selected_performance_score)
    selected_accuracy_tuned_score_list, selected_roc_auc_tuned_score_list = updated_data
    bar_plot(selected_performance_score, selected_accuracy_tuned_score_list, selected_roc_auc_tuned_score_list)
    fig.canvas.draw_idle()

s_ratio.on_changed(update)
s_max_evals.on_changed(update)

plt.ioff()

pickle.dump(fig, open('../train_val.fig.pickle', 'wb'))
plt.show()