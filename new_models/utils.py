import time

import torch
import torch.nn as nn


class Timer:
    def __init__(self, config):
        self.start_t = 0
        self.device = config.device

    def start(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.start_t = time.time()

    def report(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        return time.time() - self.start_t


def split_parameters(net):

    params_non_bias = []
    params_bias = []

    for name, p in net.named_parameters():

        if p.requires_grad:
            if "bias" in name:
                params_bias.append(p)
            else:
                params_non_bias.append(p)

    return params_non_bias, params_bias


LOGGING_COLUMNS_LIST = [
    "epoch",
    "train_loss",
    "train_acc",
    "eval_loss",
    "eval_acc",
    "epoch_time",
]


format_for_table = (
    lambda x, locals: (f"{locals[x]}".rjust(len(x)))
    if type(locals[x]) == int
    else "{:0.4f}".format(locals[x]).rjust(len(x))
    if locals[x] is not None
    else " " * len(x)
)


# define the printing function and print the column heads
def print_training_details(
    columns_list,
    separator_left="|  ",
    separator_right="  ",
    final="|",
    column_heads_only=False,
    is_final_entry=False,
):
    print_string = ""
    if column_heads_only:
        for column_head_name in columns_list:
            print_string += separator_left + column_head_name + separator_right
        print_string += final
        print("-" * (len(print_string)))  # print the top bar
        print(print_string)
        print("-" * (len(print_string)))  # print the bottom bar
    else:
        for column_value in columns_list:
            print_string += separator_left + column_value + separator_right
        print_string += final
        print(print_string)
    if is_final_entry:
        print("-" * (len(print_string)))  # print the final output bar
