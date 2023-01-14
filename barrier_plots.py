import numpy as np
import torch
import os
from utils.plot import plot_interp_metric, plot_metric_contour
import matplotlib.pyplot as plt

def plot_width_acc(
    train_barriers,
    test_barriers,
    title,
    widths=np.array([1,2,4])

):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(
        widths,
        train_barriers,
        linestyle="solid",
        color="tab:blue",
        linewidth=2,
        label="Train",
    )
    ax.plot(
        widths,
        test_barriers,
        linestyle="solid",
        color="tab:orange",
        linewidth=2,
        label="Test",
    )
    ax.set_xlabel("width multiplier")
    ax.set_xticks([1, 2, 4])
    ax.set_ylabel(f"accuracy barrier")

    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig


def loss_barrier(line):

    lowest_acc = np.min(line)
    start_acc = line[0]
    end_acc = line[-1]
    mean_end_acc = (start_acc + end_acc) / 2
    
    return mean_end_acc - lowest_acc

output_files = os.listdir('notebooks/outputs/')

txt_files = [f for f in output_files if '.txt' in f]

### first do loss barrier plots
n_points = 20
lambdas = torch.linspace(0, 1, steps=n_points)

train_acc_barriers = []
test_acc_barriers = []

for txt_file in sorted(txt_files):
    print(txt_file)
    with open(f'notebooks/outputs/{txt_file}', 'r') as f:
        lines = f.readlines()
        
        good_lines = lines[-4:]

        output_lines = []

        for line in good_lines:
            line = line.replace('[','').replace(']','').split(',')
            line = [float(item) for item in line]
            output_lines += [line]

        if not 'mlp' in txt_file:
            train_acc_barriers.append(-loss_barrier(output_lines[-2]))
            test_acc_barriers.append(-loss_barrier(output_lines[-1]))

        fig = plot_interp_metric("accuracy", lambdas, *output_lines)
        name = txt_file.replace('.txt', '.png')
        fig.savefig(name)

resnet_train_barriers, vgg_train_barriers = train_acc_barriers[:3], train_acc_barriers[3:]
resnet_test_barriers, vgg_test_barriers = test_acc_barriers[:3], test_acc_barriers[3:]

fig = plot_width_acc(np.array(resnet_train_barriers), np.array(resnet_test_barriers), 'resnet cifar10')
fig.savefig('resnet_width_barrier.png')

fig = plot_width_acc(np.array(vgg_train_barriers), np.array(vgg_test_barriers), 'vgg cifar10')
fig.savefig('vgg_width_barrier.png')

#npy_files = [f for f in output_files if '.npy' in f]
#
#### then do contour plots
#for npy_file in sorted(npy_files):
#    grid = np.load(f'notebooks/outputs/{npy_file}')
#
#    plot_metric_contour(
#        "accuracy",
#        t1s,
#        t2s,
#        test_acc_grid,
#        model_vectors_dict={
#            "A": utils.projection(v1, contour_plane),
#            "B": utils.projection(v2, contour_plane),
#            "B permuted": utils.projection(v3, contour_plane),
#        },
#    )

