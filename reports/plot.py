#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np

YELLOW = '#F8A32A'
RED = "#F33C68"
BLUE = "#1C93D6"
BLACK = "#555555"


def main(args):
    data = np.loadtxt(args.scores[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(data, color=BLUE)
    ax.set_xlabel("# Episodes", color=BLACK)
    ax.set_ylabel("Avg Score", color=BLACK)

    ax.set_ylim([0, np.max(data) * 1.01])
    ax.set_xlim([0, len(data) * 1.01])

    ax.spines['bottom'].set_color(BLACK)
    ax.spines['top'].set_color(BLACK)
    ax.spines['left'].set_color(BLACK)
    ax.spines['right'].set_color(BLACK)
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    ax.tick_params(axis='x', colors=BLACK)
    ax.tick_params(axis='y', colors=BLACK)

    plt.savefig(args.out[0])
    plt.show()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scores', required=True, nargs=1, help='Scores to plot in tsv.')
    parser.add_argument('-o', '--out', required=True, nargs=1, help='Output file')
    args = parser.parse_args()

    main(args)
