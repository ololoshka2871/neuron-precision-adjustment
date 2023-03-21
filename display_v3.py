#!/usr/bin/env python

import matplotlib.pyplot as plt

from parameters_v3 import *
from display.progress_display_v3 import ProgressDisplay
from misc.Rezonator import Rezonator
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates
from misc.f_s_transformer import FSTransformer
from graders.controller_grader_v3 import ControllerGrager
from models.rezonator_model import RezonatorModel
from models.sim_stop_detector_v3 import SimStopDetector
from simulators.simulator_v3 import Simulator
from controllers.controller_v3 import NNController


if __name__ == "__main__":
    import argparse
    import pickle

    from deap import creator

    from deap_elements.fitnes_max import register_finex_max
    from deap_elements.individual import register_individual

    from old.constants_v2 import *

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', type=int, help='item (default: hof)', default='-1')
    parser.add_argument('-p', type=int, help='Population item')
    parser.add_argument('-r', action=argparse.BooleanOptionalAction)
    parser.add_argument(
        'file', type=str, help='Simulation history file', default='learn_v2.ckl')
    args = parser.parse_args()

    register_finex_max()
    register_individual(creator.FitnessMax)  # type: ignore

    # read history file
    with open(args.file, "rb") as cp_file:
        cp = pickle.load(cp_file)  # type: ignore

    gen_hof = cp["gen_hof"]
    population = cp["population"]

    f, ax = plt.subplots(1, 3)

    if args.p is not None:
        individuum = population[args.p]
    else:
        individuum = gen_hof[args.i]

    if args.r:
        while True:
            params = gen_sim_parameters()
            rezonator = RezonatorModel(
                power_threshold=POWER_THRESHOLD, layer_thikness=params['ag_thikness'])
            adjustment_freq = rezonator.possible_freq_adjust * \
                params['initial_freq_diff']  # [Hz]
            if adjustment_freq > FREQ_PRECISION:
                break
    else:
        # Смещение и угол поворота из файла
        params = dict(
            offset=individuum.rezonator_offset,
            angle=individuum.rezonator_angle,
            initial_freq_diff=individuum.initial_freq_diff,
            ag_thikness=individuum.ag_layer_thikness,
        )
        rezonator = RezonatorModel(
            power_threshold=POWER_THRESHOLD, layer_thikness=params['ag_thikness'])
        adjustment_freq = rezonator.possible_freq_adjust * \
            params['initial_freq_diff']  # [Hz]

    print('Offset: {}, angle: {}, i_fd: {}, Ag: {} mm'.format(
        params['offset'], params['angle'], params['initial_freq_diff'], params['ag_thikness']))
    print(
        f"Adjustment freq: +{adjustment_freq:.2f} Hz ({rezonator.possible_freq_adjust:.2f} Hz * {params['initial_freq_diff']:.2f})")

    initial_pos = WorkzoneRelativeCoordinates(-1.0, 1.0)  # Верхний левый угол
    rez = Rezonator.load()
    coord_transformer = CoordinateTransformer(
        rez, (0, 0), params['offset'], params['angle'])

    NNController.init_model(F_HISTORY_SIZE, MOVE_HISTORY_SIZE, mean_layers=NN_MEAN_LAYERS)
    weights = individuum
    controller = NNController(weights, save_history=True)

    sim = Simulator(rezonator_model=rezonator,
                    controller_v3=controller,
                    coord_transformer=coord_transformer,
                    fs_transformer=FSTransformer(255.0, MAX_F),
                    laser_power=LASER_POWER,
                    initial_freq_diff=params['initial_freq_diff'],
                    freqmeter_period=FREQMETER_PERIOD,
                    modeling_period=SIM_CYCLE_TIME,
                    freq_history_size=F_HISTORY_SIZE,
                    initial_wz_pos=initial_pos)

    stop_detector = SimStopDetector(timeout=SIM_TIMEOUT,
                                    history_len_s=SIM_TIMEOUT,
                                    max_temperature=MAX_T,
                                    self_grade_epsilon=0.01,
                                    start_timestamp=0.0)

    grader = ControllerGrager(dest_freq_ch=adjustment_freq,
                              f_penalty=f_penalty,
                              max_temperature=MAX_T)

    model = rezonator.get_model_view(params['offset'], params['angle'])
    input_display = ProgressDisplay(
        *ax, rez, model, coord_transformer,  # type: ignore
        move_history_size=MOVE_HISTORY_SIZE, initial_pos=initial_pos,
        possible_freq_adjust=rezonator.possible_freq_adjust,
    )

    plt.show(block=False)

    stop_condition = sim.perform_modeling(stop_detector, input_display)

    rm = rezonator.get_metrics()
    total, g = grader.get_grade(rm, stop_detector.summary(), stop_condition)
    precision = (1.0 - (adjustment_freq - rm['static_freq_change']) / adjustment_freq) * 100.0
    print(
        f"""Done {stop_condition} Score = {total}:
- Adjust rgade: {g[0]:.2f} @ {rm['static_freq_change']:.2f} Hz/{adjustment_freq:.2f} Hz: {precision:.2f}%,
- Penalty: {g[1]:.6f} @ {rm['penalty_energy']},
- dissbalance: {g[2] * 100:.2f} %,
- Self grade: {g[3]:.2f},
- Tmax grade: {g[4]:.2f},
- Avarage speed: {g[5]:.2f},
- Time spent: {SIM_TIMEOUT * g[6]:.2f} s, ({g[6] * 100:.2f} %),
- Total path: {g[7]:.2f},
- Stop condition grade: {g[8]:.2f}"""
    )

    sf, ax = plt.subplots(1, 2)
    stop_detector.plot_summary(ax[0])
    ax[1].imshow(controller.history().T, interpolation='none', cmap='gray', origin='lower')  # type: ignore
    
    plt.show(block=True)