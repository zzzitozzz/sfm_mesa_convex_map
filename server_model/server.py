import sys
from model import MoveAgent

import numpy as np


def make_new_model_instance(human_var, forceful_human_var, wall_arr, pop_num, for_pop, target_arr, tmp_seed, len_sq, f_r, max_f_r, f_tau, csv_plot):
    ex_num = 1 # force_tau
    # if csv_plot:
    #     file_name_array = [
    #         f"/local_home/keito/simple_convex_map/agst_dir/goal_up_forceful_tau/ex{ex_num}_for_{for_pop}_len_{int(len_sq)}_csv/tau_{int(f_tau*100)}/"]
    # else:
    #     file_name_array = [
    #         f"/local_home/keito/simple_convex_map/agst_dir/goal_up_forceful_tau/ex{ex_num}_for_{for_pop}_len_{int(len_sq)}/tau_{int(f_tau*100)}/"]
    if csv_plot:
        file_name_array = [f"./tmp_data/tau_{int(f_tau*100)}/"]
    else:
        file_name_array = [f"./tmp_data/tau_{int(f_tau*100)}/"]

    m = MoveAgent(
        population=pop_num,
        for_population=for_pop,
        target_arr=target_arr,
        v_arg=[1., 1.],
        wall_arr=wall_arr,
        seed=tmp_seed,  # 乱数生成用
        r=0.5,  # 避難者の大きさ
        wall_r=1.0,  # うそ壁の大きさß
        human_var=human_var,
        forceful_human_var=forceful_human_var,
        width=60,  # 見かけの大きさ(マップ)
        height=60,  # 見かけの大きさ(マップ)
        dt=0.3,
        in_target_d=3,
        vision=1.5,  # 10
        time_step=0,
        add_file_name="",
        add_file_name_arr=file_name_array,
        len_sq=len_sq,
        f_r=f_r,
        max_f_r=max_f_r,
        csv_plot=csv_plot)
    return m


if __name__ == '__main__':
    pop_num = int(sys.argv[1])  # 通常の人数
    f_tau = float(sys.argv[2])  # 変更する変数の値
    tmp_seed = int(sys.argv[3])  # seed値
    f_r = 0.5
    for_pop = 5  # 強引な避難者の人数 #tmp
    csv_plot = False  # csvファイル(各エージェントの動きの軌跡)を出力するかどうか
    len_sq = 3  # 長方形の一辺の長さはlen_sq*2
    # max_f_r = 1.01
    max_f_r = 0.5
    human_var = {"m": 80., "tau": 0.5, "k": 120000., "kappa": 240000.,
                 "repul_h": [2000., 0.08], "repul_m": [2000., 0.08]}
    forceful_human_var = {"f_m": 80., "f_tau": f_tau, "f_k": 120000.,
                          "f_kappa": 240000., "f_repul_h": [2000., 0.08], "f_repul_m": [2000., 0.08]}

    # wall_arr = [[4., 40., 1], [54., 40., 1],
    #             [4., 26., 3], [16., 26., 3],
    #             [22., 26., 3], [54., 26., 3],
    #             [16., 4., 2], [16., 26., 2],
    #             [22., 4., 0], [22., 26., 0]]

    wall_arr = np.array([[[4., 40.], [54., 40.]],
                [[4., 26.], [16., 26.]],
                [[22., 26.], [54., 26.]],
                [[16., 4.], [16., 26.]],
                [[22., 4.], [22., 26.]]])

    target_arr = [[0, [19., 32.5], 2.], [1, [54., 33.], 2],
                  [2, [19., 32.5], 6.], [3, [19., 14.], 2.]]
    while 1:
        m = make_new_model_instance(
            human_var, forceful_human_var, wall_arr, pop_num, for_pop, target_arr, tmp_seed, len_sq, f_r, max_f_r,  f_tau, csv_plot)
        m.running = True
        while m.running:
            m.step()
        sys.exit()

