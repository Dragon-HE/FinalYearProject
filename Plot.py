import matplotlib.pyplot as plt


def plot_histogram(data_set):
    plt.figure(1)
    plt.xlim(0.0, data_set.var15.max())
    # learn how to set the width of X axis
    # https://stackoverflow.com/questions/17734587/why-is-set-xlim-not-setting-the-x-limits-in-my-figure
    plt.hist(data_set['var15'], bins=100)
    plt.xlabel('var15')
    plt.ylabel('Frequency')
    plt.title('distribution of feature var15')
    plt.show()
