import mesa
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

from s_agent import Human, HumanSpecs, ForcefulHuman, ForcefulHumanSpecs, Wall
warnings.simplefilter('ignore', UserWarning)


class MoveAgent(mesa.Model):

    def __init__(
            self, population=100, for_population=1, target_arr=[], v_arg=[], wall_arr=[[]], seed=1, r=0.5,
            wall_r=0.5, human_var={}, forceful_human_var={},
            width=100, height=100, dt=0.1,
            in_target_d=3, vision=3, time_step=0,
            add_file_name="", add_file_name_arr=[],
            len_sq=3., f_r=0., max_f_r=1.,
            csv_plot=False):
        super().__init__()
        self.population = population
        self.for_population = for_population
        self.target_arr = target_arr
        self.v_arg = v_arg
        self.wall_arr = wall_arr
        self.seed = seed
        self.r = r
        self.wall_r = wall_r
        self.human_var = human_var
        self.forceful_human_var = forceful_human_var
        self.width = width
        self.height = height
        self.dt = dt
        self.in_target_d = in_target_d
        self.vision = vision
        self.time_step = time_step
        self.add_file_name_arr = add_file_name_arr
        self.len_sq = len_sq
        self.f_r = f_r
        self.max_f_r = max_f_r
        self.csv_plot = csv_plot
        human_var_inst, forceful_human_var_inst = self.assign_ini_human_and_forceful_human_var()
        self.dir_parts()
        # self.schedule = mesa.time.RandomActivation(self) #すべてのエージェントをランダムに呼び出し、各エージェントでstep()を一回呼ぶ。step()だけで変更を適用する
        # すべてのエージェントを順番に呼び出し、すべてのエージェントで順番にstep()を一回読んだ後、すべてのエージェントで順番にadvance()を一回呼ぶ。step()で変更を準備し、advance()で変更を適用する
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, True)
        np.random.seed(self.seed)
        self.make_agents(human_var_inst, human_var_inst)
        self.running = True
        print(f"change para: {self.check_f_parameter()}")
        self.make_basic_dir()
        self.save_specs_to_file(human_var_inst, forceful_human_var_inst)

    def dir_parts(self):
        basic_file_name = f"{self.add_file_name_arr[0]}{self.population}{self.add_file_name_arr[1]}"
        self.add_file_name = f"{basic_file_name}{int(self.f_m)}/"
        self.add_file_name = self.add_file_name + "seed_" + str(self.seed)
        return None

    def assign_ini_human_and_forceful_human_var(self):
        self.m = self.human_var["m"]
        self.tau = self.human_var["tau"]
        self.k = self.human_var["k"]
        self.kappa = self.human_var["kappa"]
        self.repul_h = self.human_var["repul_h"]
        self.repul_m = self.human_var["repul_m"]
        self.alpha = self.human_var["alpha"]
        self.f_m = self.forceful_human_var["f_m"]
        self.f_tau = self.forceful_human_var["f_tau"]
        self.f_k = self.forceful_human_var["f_k"]
        self.f_kappa = self.forceful_human_var["f_kappa"]
        self.f_repul_h = self.forceful_human_var["f_repul_h"]
        self.f_repul_m = self.forceful_human_var["f_repul_m"]
        self.f_alpha = self.forceful_human_var["f_alpha"]
        human_var_inst = HumanSpecs(self.r, self.m, self.tau, self.k, self.kappa, self.repul_h, self.repul_m, self.alpha, self.dt, self.in_target_d, self.vision)
        forceful_human_var_inst = ForcefulHumanSpecs(self.f_r, self.f_m, self.f_tau, self.f_k, self.f_kappa, self.f_repul_h, self.f_repul_m, self.f_alpha)
        return human_var_inst, forceful_human_var_inst

    def make_basic_dir(self):
        path = f"{self.add_file_name}/Data/"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}nolmal.dat", "w") as f:
            f.write("evacuation_time\n")
        os.makedirs(path, exist_ok=True)
        with open(f"{path}forceful.dat", "w") as f:
            f.write("evacuation_time\n")
        print(f"{self.add_file_name=}")
        self.ini_force_dataframe()

    def save_specs_to_file(self, human_specs, forceful_human_specs):
        path = f"{self.add_file_name}/../humans_specs.yaml"
        if not os.path.exists(path):
            data = {
                "human": vars(human_specs),
                "forcefulhuman": vars(forceful_human_specs)
            }
            with open(f"{path}", "w") as f:
                yaml.dump(data, f, sort_keys=False)

    def ini_force_dataframe(self):
        tmp_path = self.add_file_name.replace(f"/seed_{self.seed}", "")
        if os.path.isfile(f"{tmp_path}/forceful_time.csv"):
            None
        else:
            df = pd.DataFrame(
                columns=["m", "nol_pop", "seed", "id", "evacuation_time"])
            df.to_csv(f"{tmp_path}/forceful_time.csv", index=False)

    def make_agents(self, human_var_inst, forceful_human_var_inst):
        tmp_id = 0
        tmp_id = self.generate_human(tmp_id, human_var_inst, forceful_human_var_inst)
        self.generate_wall(tmp_id)

    def generate_human(self, tmp_id, human_var_inst, forceful_human_var_inst):
        tmp_div = 1.
        pos_array = []
        human_array = []
        tmp_forceful_num = self.for_population
        for i in range(tmp_id, tmp_id + self.population + self.for_population):  # 1人多く作成(強引な人)
            pos = []
            velocity = []
            if tmp_forceful_num:  # 強引な人(強引な人の位置が先に決まったのち通常の避難者の位置が決まる)
                velocity = self.decide_vel()
                # pos, pos_array = self.decide_forceful_postion(pos_array)
                if self.for_population == 1:
                    pos = np.array((19., 36.))
                    pos_array.append(pos)
                else:
                    pos = self.decide_forceful_postion(human_array)
                forceful_target = [19., 14.]
                target = forceful_target
                human = ForcefulHuman(i, self, pos, velocity,
                                      target, tmp_div, human_var_inst,
                                      self.space, self.add_file_name,
                                      forceful_human_var_inst,
                                      )
                self.space.place_agent(human, pos)
                self.schedule.add(human)
                human_array.append(human)
                tmp_forceful_num -= 1
            else:  # 通常の人
                pos = self.decide_positon(human_array, i - tmp_id) #tmp
                velocity = self.decide_vel()
                nolmal_target = self.decide_nolmal_target()
                target = nolmal_target
                human = Human(i, self, pos, velocity, target, tmp_div,
                              human_var_inst, self.space,
                              self.add_file_name,)
                self.space.place_agent(human, pos)
                self.schedule.add(human)
                human_array.append(human)
        tmp_id += self.population + self.for_population
        return tmp_id

    def decide_positon(self, human_array, i):
        while 1:
            x = np.random.randint(4, 34) + np.random.rand()
            y = np.random.randint(26, 40) + np.random.rand()
            if 4. + self.r * 2 <= x <= 34. - self.r * 2 and 26. + self.r * 2 <= y <= 40. - self.r * 2:
                tmp_pos = np.array((x, y))
                if self.human_pos_check(tmp_pos, human_array):
                    pos = tmp_pos
                    break
        return pos

    def decide_forceful_postion(self, human_array):
        while 1:
            x = np.random.randint(19.-self.len_sq, 19. +
                                  self.len_sq) + np.random.rand()
            y = np.random.randint(32.5-self.len_sq, 32.5 +
                                  self.len_sq) + np.random.rand()
            if 19.-self.len_sq + self.max_f_r <= x <= 19.+self.len_sq - self.max_f_r and 32.5-self.len_sq + self.max_f_r <= y <= 32.5+self.len_sq - self.max_f_r:
                tmp_pos = np.array((x, y))
                if self.forceful_human_pos_check(tmp_pos, human_array):
                    pos = tmp_pos
                    break
        return pos

    def decide_vel(self):
        while 1:
            velocity = np.random.normal(
                loc=self.v_arg[0], scale=self.v_arg[1], size=2)
            # 初期速度(および希望速さのx,y成分)は0.5以上1以下
            if 0.5 <= np.linalg.norm(velocity, 2) <= 1.:
                break
        return velocity

    def decide_nolmal_target(self):
        while 1:
            y = np.random.randint(26, 40) + np.random.rand()
            if 26. + self.r <= y <= 40. - self.r:
                nolmal_target = [54., y]
                break
        return nolmal_target

    def human_pos_check(self, tmp_pos, human_array):
        for hu in human_array:
            dis = self.space.get_distance(tmp_pos, hu.pos)
            if type(hu) is Human:   ##強引な避難者の大きさを変更したときの条件式
                if dis < self.r + self.r:
                    return False
            elif type(hu) is ForcefulHuman:
                if dis < self.r + self.max_f_r:
                    return False
        return True

    def forceful_human_pos_check(self, tmp_pos, human_array):
        for hu in human_array:
            dis = self.space.get_distance(tmp_pos, hu.pos)
            if type(hu) is Human: ##強引な避難者の大きさを変更したときの条件式
                if dis < self.r + self.r:
                    return False
            if type(hu) is ForcefulHuman:
                if dis < self.max_f_r + self.r:
                    return False
        return True

    def generate_wall(self, id):  # 壁を作る
        i = 0
        while 1:
            if i >= len(self.wall_arr) - 1:
                break
            tmp_wall_1 = self.wall_arr[i]
            tmp_wall_2 = self.wall_arr[i + 1]
            tmp_x, tmp_y = tmp_wall_1[0], tmp_wall_1[1]
            not_skip = 1
            if tmp_wall_1[2] == 0 or tmp_wall_1[2] == 2:
                while 1:
                    if tmp_y >= tmp_wall_2[1]:
                        break
                    if i != 0:
                        neighbors = self.space.get_neighbors(
                            np.array((27., 20.)), 30, False)
                        for neighbor in neighbors:
                            if type(neighbor) is Wall:
                                if tmp_x == neighbor.pos[0] and tmp_y == neighbor.pos[1]:
                                    not_skip = 0
                                    break
                    if not_skip:
                        self.generate_block(tmp_x, tmp_y, id, tmp_wall_1[2])
                        id += 1
                        tmp_y += self.height * 0.001
                    else:
                        not_skip = 1
                        tmp_y += self.height * 0.001
            elif tmp_wall_1[2] == 1 or tmp_wall_1[2] == 3:
                while 1:
                    if tmp_x >= tmp_wall_2[0]:
                        break
                    if i != 0:
                        neighbors = self.space.get_neighbors(
                            np.array((27., 20.)), 30, False)
                        for neighbor in neighbors:
                            if type(neighbor) is Wall:
                                if tmp_x == neighbor.pos[0] and tmp_y == neighbor.pos[1]:
                                    not_skip = 0
                                    break
                    if not_skip:
                        self.generate_block(tmp_x, tmp_y, id, tmp_wall_1[2])
                        id += 1
                        tmp_x += self.width * 0.001
                    else:
                        not_skip = 1
                        tmp_x += self.width * 0.001
            else:
                print("generate_wall_error")
            i += 2
        return id  # tmp_y -= self.height * 0.001

    def generate_block(self, x, y, id, cnt):  # 1つブロックを生成する
        pos = np.array((x, y))
        dir = cnt  # 1:左 2:上　3:右　4:下　の方向に避難者に力を与える → 0:左 1:上　2:右　3:下
        wall = Wall(
            id,
            self,
            pos,
            self.wall_r,
            dir,
        )
        self.space.place_agent(wall, pos)
        self.schedule.add(wall)

    def check_f_parameter(self):
        count = 0
        if self.m != self.f_m:
            print(f"self.m change {self.m=} {self.f_m=}")
            count += 1
        if self.tau != self.f_tau:
            print("tau change")
            count += 1
        if self.repul_h[0] != self.f_repul_h[0]:
            print("repul_h[0] change")
            count += 1
        if self.repul_h[1] != self.f_repul_h[1]:
            print("repul_h[1] change")
            count += 1
        if self.repul_m[0] != self.f_repul_m[0]:
            print("repul_m[0] change")
            count += 1
        if self.repul_m[1] != self.f_repul_m[1]:
            print("repul_m[1] change")
            count += 1
        if self.k != self.f_k:
            print("k change")
            count += 1
        if self.kappa != self.f_kappa:
            print("kappa change")
            count += 1
        if self.f_r != self.r:
            print("f_r change")
            count += 1
        return count

    def step(self):
        self.schedule.step()
        self.time_step += 1
        if self.time_step % 100 == 0:
            if self.all_agent_evacuate():
                self.running = False
        if self.time_step >= 1500:
            self.timeout_check()
            self.running = False

    def all_agent_evacuate(self):
        cur_pop_num = (
            len(open(f"{self.add_file_name}/Data/nolmal.dat").readlines()))
        if cur_pop_num == self.population + 1:
            if (len(open(f"{self.add_file_name}/Data/forceful.dat").readlines())) + cur_pop_num == self.population + self.for_population + 2:
                return True
        return False

    def timeout_check(self):
        if self.csv_plot:
            for obj in self.schedule.agents:
                if type(obj) is Human or type(obj) is ForcefulHuman:
                    path = obj.add_file_name
                    obj.make_dir(path)
                    obj.write_record(path, obj.max_population_around)
        self.write_interrupt()

    def write_interrupt(self):
        nolmal_num = len(
            open(f"{self.add_file_name}/Data/nolmal.dat").readlines())
        forceful_num = len(
            open(f"{self.add_file_name}/Data/forceful.dat").readlines())
        if nolmal_num < self.population + 1:
            if forceful_num < self.for_population + 1:
                with open(f"{self.add_file_name}/Data/nolmal.dat", "a") as f:
                    f.write(f"interrupt\n")
                with open(f"{self.add_file_name}/Data/forceful.dat", "a") as f:
                    f.write(f"interrupt\n")
            else:
                with open(f"{self.add_file_name}/Data/nolmal.dat", "a") as f:
                    f.write(f"interrupt\n")
        else:
            with open(f"{self.add_file_name}/Data/forceful.dat", "a") as f:
                f.write(f"interrupt\n")

