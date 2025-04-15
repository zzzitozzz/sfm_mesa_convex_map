import sys
from s_model import MoveAgent


def make_new_model_instance(human_var, forceful_human_var, wall_arr, pop_num, for_pop, target_arr, tmp_seed, len_sq, f_r, max_f_r, f_m, csv_plot):
    # ex_num = 11 #len_sq ex11
    # ex_num = 2 #f_r, f_repul
    ex_num = 1
    # if for_pop == 1:
    #     if csv_plot:
    #         file_name_array = [
    #             f"/local_home/keito/ex{ex_num}_convex_for_{for_pop}_csv/", "_force_m_"]
    #     else:
    #         file_name_array = [
    #             f"/local_home/keito/ex{ex_num}_convex_for_{for_pop}/", "_force_m_"]
    # else:
    #     if csv_plot:
    #         file_name_array = [
    #             f"/local_home/keito/ex{ex_num}_convex_for_{for_pop}_len_{int(len_sq)}_csv/", "_force_m_"]
    #     else:
    #         file_name_array = [
    # #             f"/local_home/keito/ex{ex_num}_convex_for_{for_pop}_len_{int(len_sq)}/", "_force_m_"]

    # if csv_plot:
    #     file_name_array = [
    #         f"/local_home/keito/ex{ex_num}_convex_for_{for_pop}_len_{int(len_sq)}_f_r_{int(f_r*10)}_f_m_{int(f_m)}_csv/", "_force_m_"]
    # else:
    #     file_name_array = [
    #         f"/local_home/keito/ex{ex_num}_convex_for_{for_pop}_len_{int(len_sq)}_f_r_{int(f_r*10)}_f_m_{int(f_m)}/", "_force_m_"]

    # file_name_array = [  # Debug
    #     f"./data/test/ex{ex_num}_convex_for_{for_pop}_len_{len_sq}_f_r_{int(f_r*10)}_f_m_{int(f_m)}/", "_force_m_"]
    # file_name_array = [  # Debug
    #     f"./data/test_{for_pop}_len_{int(len_sq)}_f_r_{int(f_r*10)}/", "_force_m_"]_
    file_name_array = [  # Debug
        f"./refact_data4/", "_force_m_"]

    # file_name_array = [
    #     f"/local_home/keito/test_{for_pop}/", "_force_m_"]
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
        width=200,  # 見かけの大きさ(マップ)
        height=200,  # 見かけの大きさ(マップ)
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
    # f_m = float(sys.argv[2])  # 強引な避難者の質量
    # f_r = float(sys.argv[2])  # 強引な避難者の半径(接触範囲の増加)
    f_r = 0.5
    f_m = float(sys.argv[2]) # 強引な避難者の質量
    tmp_seed = int(sys.argv[3])  # seed値
    for_pop = 5  # 強引な避難者の人数 #tmp
    csv_plot = True  # csvファイル(各エージェントの動きの軌跡)を出力するかどうか
    len_sq = 3  # 長方形の一辺の長さはlen_sq*2
    max_f_r = 1.01
    human_var = {"m": 80., "tau": 0.5, "k": 120000., "kappa": 240000.,
                 "repul_h": [2000., 0.08], "repul_m": [2000., 0.08]}
    forceful_human_var = {"f_m": f_m, "f_tau": 0.5, "f_k": 120000.,
                          "f_kappa": 240000., "f_repul_h": [2000., 0.08], "f_repul_m": [2000., 0.08]}

    wall_arr = [[4., 3., 3], [154., 3., 3],
                [4., 15., 1], [154., 15., 1],
                ]
    target_arr = [[0, [154., 9], 2.]] #チェックポイントとゴール
    while 1:
        m = make_new_model_instance(
            human_var, forceful_human_var, wall_arr, pop_num, for_pop, target_arr, tmp_seed, len_sq, f_r, max_f_r, f_m, csv_plot)
        m.running = True
        while m.running:
            m.step()
        sys.exit()
