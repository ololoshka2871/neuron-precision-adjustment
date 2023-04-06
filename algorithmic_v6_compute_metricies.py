#!/usr/bin/env python

import argparse

import multiprocessing as mp
import pandas as pd

from algorithmic_v6_compute import sim_main


def run_single_simulation(_: int) -> dict:
    return sim_main()


if __name__ == '__main__':
    # parce command line arguments:
    # -n --number-of-simulations <int>
    # -o --output-file <str>
    parcer = argparse.ArgumentParser(description='Run simulation and analise.')
    parcer.add_argument('-n', '--number-of-simulations', type=int,
                        default=4, help='Number of simulations to run.')
    parcer.add_argument('-o', '--output-file', type=str,
                        default='results.csv', help='Output file name.')
    args = parcer.parse_args()

    # run simulations in separate processes:

    with mp.Pool() as pool:
        results = pool.map(run_single_simulation,
                           range(args.number_of_simulations))

    # save results to file:
    df = pd.DataFrame(data=results)
    df.to_csv(args.output_file)
