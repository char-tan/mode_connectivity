import matplotlib.pyplot as plt


def plot_interp_metric(
    metric_name: str,
    lambdas,
    train_metric_interp_naive,
    test_metric_interp_naive,
    train_metric_interp_clever,
    test_metric_interp_clever,
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        lambdas,
        train_metric_interp_naive,
        linestyle="dashed",
        color="tab:blue",
        alpha=0.5,
        linewidth=2,
        label="Train, naïve interp.",
    )
    ax.plot(
        lambdas,
        test_metric_interp_naive,
        linestyle="dashed",
        color="tab:orange",
        alpha=0.5,
        linewidth=2,
        label="Test, naïve interp.",
    )
    ax.plot(
        lambdas,
        train_metric_interp_clever,
        linestyle="solid",
        color="tab:blue",
        linewidth=2,
        label="Train, permuted interp.",
    )
    ax.plot(
        lambdas,
        test_metric_interp_clever,
        linestyle="solid",
        color="tab:orange",
        linewidth=2,
        label="Test, permuted interp.",
    )
    ax.set_xlabel("$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel(f"{metric_name.title()}")
    # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
    ax.set_title(f"{metric_name.title()} between the two models")
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig


def plot_interp_acc(
    lambdas,
    train_acc_interp_naive,
    test_acc_interp_naive,
    train_acc_interp_clever,
    test_acc_interp_clever,
):
    return plot_interp_metric(
        "accuracy",
        lambdas,
        train_acc_interp_naive,
        test_acc_interp_naive,
        train_acc_interp_clever,
        test_acc_interp_clever,
    )
