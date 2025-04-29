import os
import copy

import mesa
import numpy as np
import pandas as pd
import math

class SharedParams:
    "_shared: Common human-related parameters shared between Human and ForcefulHuman instances."
    def __init__(self, in_target_d, vision, dt):
        self.in_target_d = in_target_d
        self.vision = vision
        self.dt = dt

class HumanSpecs:
    "_hspecs: Human-related common specs set used in Human (and ForcefulHuman)"
    def __init__(self, r, m, tau, k, kappa, repul_h, repul_m, alpha):
        self.r = r
        self.m = m
        self.tau = tau
        self.k = k
        self.kappa = kappa
        self.repul_h = repul_h
        self.repul_m = repul_m
        self.alpha = alpha


class Human(mesa.Agent):
    def __init__(self, unique_id, model,
                 pos, velocity,
                 target, tmp_div,
                 shared,
                 human_var_inst,
                 space, add_file_name,
                 target_id=1,
                 tmp_pos=(0., 0.), pos_array=[],
                 in_goal=False,elapsed_time=0.,  # 経過時間
                 ):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.velocity = velocity
        self._shared = shared
        self._hspecs = human_var_inst
        self.tmp_div = tmp_div #特定の人同士の反発力の大きさを除算もしくは乗算する値
        self.target = target #避難所の座標
        self.space = space #エージェントが動き回る空間を管理するモジュール
        self.add_file_name = add_file_name #保存するファイル名(の基礎.最終的には絶対パスまたは相対パスができる)
        self.target_id = target_id #避難所または中継地を指定するid
        self.tmp_pos = np.array((0., 0.)) #一時的に計算した結果の位置を保存する値(将来的には壁を乗り越えるなどのありえない挙動をした時に元の位置に戻すために一旦計算した位置を保存している)
        self.in_goal = in_goal #目的地に到着したか判定するboolean型の変数
        self.pos_array = [] #自分の位置をステップごとに記録する配列
        self.pos_array.append(self.pos)
        self.elapsed_time = elapsed_time #経過時間

    @property
    def hspecs(self):
        return self._hspecs
        
    def step(self):  # 次の位置を特定するための計算式を書く
        self.target_update()
        i = 5
        while i:
            self._calculate()
            target_dis = abs(self.pos[0] - 54.)
            self.goal_check(target_dis)
            self.tmp_pos[0] = self.pos[0] + \
                self.velocity[0] * self._shared.dt  # 仮の位置を計算
            self.tmp_pos[1] = self.pos[1] + self.velocity[1] * self._shared.dt
            if self.pos_check():
                break
            else:
                i -= 1
                self.reset_target()
        return None

    def advance(self):
        self.pos = copy.deepcopy(self.tmp_pos)
        self.pos_array.append(self.pos)
        self.elapsed_time += self._shared.dt
        if (self.in_goal):  # goalした場合
            path = self.add_file_name
            self.make_dir(path)
            self.write_record(path)
            self.model.space.remove_agent(self)
            self.model.schedule.remove(self)
            return None
        else:
            self.model.space.move_agent(self, self.pos)  # goalしていない場合
        return None

    def target_update(self):
        if type(self) is Human:
            if self.target_id == 1 and self.pos[1] < 26:
                self.target_id = 0
                self.target = self.model.target_arr[self.target_id][1]
            elif self.target_id == 0:
                tmp_y_dis = abs(self.pos[1]-self.target[1])
                if tmp_y_dis < self.model.target_arr[self.target_id][2]:
                    self.target_id = 1
                    self.target = self.model.target_arr[self.target_id][1]
            else:
                None
        elif type(self) is ForcefulHuman:
            if self.target_id == 3 and self.pos[0] >= 22:
                self.target_id = 2
                self.target = self.model.target_arr[self.target_id][1]
            elif self.target_id == 2:
                tmp_x_dis = abs(self.pos[0]-self.target[0])
                if tmp_x_dis < self.model.target_arr[self.target_id][2]:
                    self.target_id = 3
                    self.target = self.model.target_arr[self.target_id][1]
            else:
                None    

    def goal_check(self, target_dis):
        if target_dis < 0.5:
            self.in_goal = True
            self.velocity = [0.0, 0.0]
            return None

    def reset_target(self):
        if type(self) is Human:
            if self.target_id == 1:  # 正常
                while 1:
                    y = np.random.randint(26, 40) + np.random.rand()
                    if 26.5 <= y <= 39.5:
                        break
                self.target[1] = y
            elif self.target_id == 0:  # 例外
                while 1:
                    x = np.random.randint(16, 22) + np.random.rand()
                    if 16.5 <= x <= 21.5:
                        break
                self.target[0] = x
        elif type(self) is ForcefulHuman:
            if self.target_id == 3:  # 正常
                while 1:
                    x = np.random.randint(16, 22) + np.random.rand()
                    if 16.5 <= x <= 21.5:
                        break
                self.target[0] = x
            elif self.target_id == 2:  # 例外
                while 1:
                    y = np.random.randint(26, 40) + np.random.rand()
                    if 26.5 <= y <= 39.5:
                        break
                self.target[1] = y
        return None

    def make_dir(self, path):
        os.makedirs(f"{path}/Data", exist_ok=True)
        if self.model.csv_plot:
            os.makedirs(f"{path}/csv", exist_ok=True)
        return None

    def write_record(self, path):
        if self.model.csv_plot:
            np.savetxt(f"{path}/csv/id{self.unique_id}_nolmal"
                       f".csv", self.pos_array, delimiter=",")
        if self.in_goal:
            with open(f"{self.add_file_name}/Data/"
                      f"nolmal.dat", "a") as f:
                f.write(f"{self.elapsed_time} \n")

    def _sincos(self, x2):
        r_0 = np.sqrt((x2[0] - self.pos[0]) ** 2 + (x2[1] - self.pos[1]) ** 2)
        sin = (x2[1] - self.pos[1]) / r_0
        cos = (x2[0] - self.pos[0]) / r_0
        return cos, sin

    def _force(self, target):
        fx, fy = 0., 0.
        theta = self._sincos(target)
        neighbors = self.model.space.get_neighbors(
            self.pos, self._shared.vision, False)
        fx, fy = self.force_from_goal(theta)
        wall_dis = {"left_wall_dis": 1000., "right_wall_dis": 1000.,
                    "upper_wall_dis": 1000., "bottom_wall_dis": 1000.}
        wall_obj = {"left_wall": 1., "right_wall": 1.,
                    "upper_wall": 1., "bottom_wall": 1.}

        for neighbor in neighbors:
            if self.unique_id == neighbor.unique_id:
                continue
            if type(neighbor) is Human or type(neighbor) is ForcefulHuman:
                if type(self) == type(neighbor):
                    tmp_fx, tmp_fy = self.force_from_human(neighbor)
                else:
                    tmp_fx, tmp_fy = self.force_from_human_alpha(neighbor)
                fx += tmp_fx
                fy += tmp_fy
            elif type(neighbor) is Wall:
                wall_dis, wall_obj = self.choose_wall(neighbor,
                                                      wall_dis, wall_obj)
        tmp_fx, tmp_fy = 0., 0.
        tmp_fx, tmp_fy, wall_obj = self.force_from_wall(wall_obj)
        fx += tmp_fx
        fy += tmp_fy
        fx /= self.hspecs.m
        fy /= self.hspecs.m
        return fx, fy

    def force_from_goal(self, theta):
        fx = self.hspecs.m * (0.8 * theta[0] - self.velocity[0]) / self.hspecs.tau
        fy = self.hspecs.m * (0.8 * theta[1] - self.velocity[1]) / self.hspecs.tau
        return fx, fy

    def force_from_human(self, neighbor):
        fx, fy = 0., 0.
        n_ij = (self.pos - neighbor.pos) / \
            self.space.get_distance(self.pos, neighbor.pos)
        t_ij = [-n_ij[1], n_ij[0]]
        dis = (self.hspecs.r + neighbor.hspecs.r) - \
            self.space.get_distance(self.pos, neighbor.pos)
        if dis >= 0:
            fx += (self.hspecs.repul_h[0] * (math.e ** (dis / self.hspecs.repul_h[1])) + self.hspecs.k * dis) * \
                n_ij[0] + self.hspecs.kappa * dis * \
                np.dot(
                (neighbor.velocity - self.velocity), t_ij)*t_ij[0]
            fy += (self.hspecs.repul_h[0] * (math.e ** (dis / self.hspecs.repul_h[1])) + self.hspecs.k * dis) * \
                n_ij[1] + self.hspecs.kappa * dis * \
                np.dot(
                    (neighbor.velocity - self.velocity), t_ij)*t_ij[1]
        else:
            fx += self.hspecs.repul_h[0] * (math.e **
                                     (dis / self.hspecs.repul_h[1])) * n_ij[0]
            fy += self.hspecs.repul_h[0] * (math.e **
                                     (dis / self.hspecs.repul_h[1])) * n_ij[1]
        return fx, fy

    def force_from_human_alpha(self, neighbor):
        fx, fy = 0., 0.
        tmp_A = self.hspecs.repul_h[0] * (1. + self.hspecs.alpha)
        n_ij = (self.pos - neighbor.pos) / \
            self.space.get_distance(self.pos, neighbor.pos)
        t_ij = [-n_ij[1], n_ij[0]]
        dis = (self.hspecs.r + neighbor.hspecs.r) - \
            self.space.get_distance(self.pos, neighbor.pos)
        if dis >= 0:
            fx += (tmp_A * (math.e ** (dis / self.hspecs.repul_h[1])) + self.hspecs.k * dis) * \
                n_ij[0] + self.hspecs.kappa * dis * \
                np.dot(
                (neighbor.velocity - self.velocity), t_ij)*t_ij[0]
            fy += (tmp_A * (math.e ** (dis / self.hspecs.repul_h[1])) + self.hspecs.k * dis) * \
                n_ij[1] + self.hspecs.kappa * dis * \
                np.dot(
                    (neighbor.velocity - self.velocity), t_ij)*t_ij[1]
        else:
            fx += tmp_A * (math.e **
                                     (dis / self.hspecs.repul_h[1])) * n_ij[0]
            fy += tmp_A * (math.e **
                                     (dis / self.hspecs.repul_h[1])) * n_ij[1]
        return fx, fy

    def choose_wall(self, neighbor, wall_dis, wall_obj):
        tmp_wall_dis = self.space.get_distance(
            self.pos, neighbor.pos)
        if abs(self.pos[1] - neighbor.pos[1]) < 102 * 0.001 * 2:
            if neighbor.dir == 2:
                if tmp_wall_dis < wall_dis["left_wall_dis"]:
                    wall_dis["left_wall_dis"] = tmp_wall_dis
                    wall_obj["left_wall"] = neighbor
            if neighbor.dir == 0:
                if tmp_wall_dis < wall_dis["right_wall_dis"]:
                    wall_dis["right_wall_dis"] = tmp_wall_dis
                    wall_obj["right_wall"] = neighbor
        elif abs(self.pos[0] - neighbor.pos[0]) < 158. * 0.001 * 2:
            if neighbor.dir == 3:
                if tmp_wall_dis < wall_dis["upper_wall_dis"]:
                    wall_dis["upper_wall_dis"] = tmp_wall_dis
                    wall_obj["upper_wall"] = neighbor
            if neighbor.dir == 1:
                if tmp_wall_dis < wall_dis["bottom_wall_dis"]:
                    wall_dis["bottom_wall_dis"] = tmp_wall_dis
                    wall_obj["bottom_wall"] = neighbor
        return wall_dis, wall_obj

    def force_from_wall(self, wall_obj):
        fx, fy = 0., 0.
        tmp_wall = np.array([22., 26.])
        tmp_dis = self.space.get_distance(self.pos, tmp_wall)
        if tmp_dis <= 1.5:  # 壁に対して右斜め上に垂直なとき
            if type(wall_obj["right_wall"]) is Wall or type(wall_obj["upper_wall"]) is Wall:
                None
            else:
                n_iw = (self.pos - tmp_wall) / tmp_dis
                t_iw = [-n_iw[1], n_iw[0]]
                dis = tmp_dis + 1.
                tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
                fx += tmp_fx
                fy += tmp_fy

        tmp_wall = np.array([16., 26.])  # 壁に対して左斜め上に垂直なとき
        tmp_dis = self.space.get_distance(self.pos, tmp_wall)
        if tmp_dis <= 1.5:
            if type(wall_obj["left_wall"]) is Wall or type(wall_obj["upper_wall"]) is Wall:
                None
            else:
                n_iw = (self.pos - tmp_wall) / tmp_dis
                t_iw = [-n_iw[1], n_iw[0]]
                dis = tmp_dis + 1.
                tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
                fx += tmp_fx
                fy += tmp_fy

        if type(wall_obj["left_wall"]) is Wall:
            n_iw = [1., 0.]
            t_iw = [-n_iw[1], n_iw[0]]
            dis = self.hspecs.r - (self.pos[0] - wall_obj["left_wall"].pos[0])
            dis += wall_obj["left_wall"].wall_r
            tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
            fx += tmp_fx
            fy += tmp_fy
            wall_obj["left_wall"] = 1.
        if type(wall_obj["right_wall"]) is Wall:
            n_iw = [-1., 0.]
            t_iw = [-n_iw[1], n_iw[0]]
            dis = self.hspecs.r - (wall_obj["right_wall"].pos[0] - self.pos[0])
            dis += wall_obj["right_wall"].wall_r
            tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
            fx += tmp_fx
            fy += tmp_fy
            wall_obj["right_wall"] = 1.
        if type(wall_obj["upper_wall"]) is Wall:
            n_iw = [0., 1.]
            t_iw = [-n_iw[1], n_iw[0]]
            dis = self.hspecs.r - (self.pos[1] - wall_obj["upper_wall"].pos[1])
            dis += wall_obj["upper_wall"].wall_r
            tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
            fx += tmp_fx
            fy += tmp_fy
            wall_obj["upper_wall"] = 1.
        if type(wall_obj["bottom_wall"]) is Wall:
            n_iw = [0., -1.]
            t_iw = [-n_iw[1], n_iw[0]]
            dis = self.hspecs.r - (wall_obj["bottom_wall"].pos[1] - self.pos[1])
            dis += wall_obj["bottom_wall"].wall_r
            tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
            fx += tmp_fx
            fy += tmp_fy
            wall_obj["bottom_wall"] = 1.
        return fx, fy, wall_obj

    def wall_force_core(self, dis, n_iw, t_iw):
        fx, fy = 0., 0.
        if dis >= 0:
            fx += (self.hspecs.repul_m[0] * (math.e ** (dis / self.hspecs.repul_m[1])) + self.hspecs.k *
                    dis) * n_iw[0] - self.hspecs.kappa * dis * np.dot(self.velocity, t_iw) * t_iw[0]
            fy += (self.hspecs.repul_m[0] * (math.e ** (dis / self.hspecs.repul_m[1])) + self.hspecs.k *
                    dis) * n_iw[1] - self.hspecs.kappa * dis * np.dot(self.velocity, t_iw) * t_iw[1]
        else:
            fx += (self.hspecs.repul_m[0] * (math.e **
                    (dis / self.hspecs.repul_m[1]))) * n_iw[0]
            fy += (self.hspecs.repul_m[0] * (math.e **
                    (dis / self.hspecs.repul_m[1]))) * n_iw[1]
        return fx, fy

    def _calculate(self):
        fx, fy = self._force(self.target)
        self.velocity[0] += fx * self._shared.dt
        self.velocity[1] += fy * self._shared.dt
        if (np.linalg.norm(self.velocity, 2) > 1.):  # review
            v = copy.deepcopy(self.velocity)
            vn = np.linalg.norm(v)
            self.velocity = v / vn
        return None
    
    def pos_check(self):
        area = [[4., 26 + self.hspecs.r], [54., 40. - self.hspecs.r],
                [16. + self.hspecs.r, 4.], [22. - self.hspecs.r, 40. - self.hspecs.r]]       
        area_check = False
        i = 0
        while 1:
            if i >= len(area):
                break
            if area[i][0] <= self.tmp_pos[0] <= area[i + 1][0] and area[i][1] <= self.tmp_pos[1] <= area[i + 1][1]:
                area_check = True
                break
            else:
                i += 2
        if area_check:
            return True
        else:
            self.tmp_pos = copy.deepcopy(self.pos)
            return False

    def get_distance(pos_1, pos_2):
        x1, y1 = pos_1
        x2, y2 = pos_2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return math.sqrt(dx * dx + dy * dy)


class ForcefulHumanSpecs:
    "fhspecs:ForcefulHuman-related specs set used in ForcefulHuman"
    def __init__(self, f_r, f_m, f_tau, f_k, f_kappa, f_repul_h, f_repul_m, alpha):
        self.r = f_r
        self.m = f_m
        self.tau = f_tau
        self.k = f_k
        self.kappa = f_kappa
        self.repul_h = f_repul_h
        self.repul_m = f_repul_m
        self.alpha = alpha


class ForcefulHuman(Human):
    def __init__(self, unique_id, model,
                 pos, velocity,
                 target, tmp_div,
                 shared,
                 human_var_inst,
                 space, add_file_name,
                 forceful_human_var_inst,
                 target_id=3,
                 tmp_pos=(0., 0.),
                 in_goal=False, pos_array=[],
                 elapsed_time=0.,  # 経過時間
                 _force_mode = False,
                 ):
        super().__init__(unique_id, model, pos,
                         velocity, target,
                         tmp_div, 
                         shared,
                         human_var_inst,
                         space, add_file_name,
                         target_id,
                         tmp_pos, in_goal,  pos_array,
                         elapsed_time,
                         )
        self.fhspecs = forceful_human_var_inst
        self._force_mode = True

    @property
    def hspecs(self):
        if self.unique_id == 1:
            # print(f"{self._force_mode}")
            pass
        # return self.fhspecs if self._force_mode else self._hspecs
            # print(f"_force_mode={self._force_mode}, hspecs called")
            if self._force_mode:
                # print(f"Returning fhspecs: {self.fhspecs}")
                return self.fhspecs
            else:
                # print(f"Returning _hspecs: {self._hspecs}")
                return self._hspecs
        if self._force_mode:
            if self.unique_id == 1:
                # print(f"fhspecs: {self._force_mode}")
                pass
            return self.fhspecs
        else:
            if self.unique_id == 1:
                pass
            return self._hspecs
    
    @property
    def fmode(self):
        return self._force_mode
    
    @fmode.setter
    def fmode(self, val: bool):
        self._force_mode = val

    def step(self):  # 次の位置を特定するための計算式を書く
        # 67step後(最大約20m進んだのち)にforcefulhumanに変化(naito 10step(20m))
        self.target_update()
        i = 5
        while i:
            self._calculate()
            # target_dis = self.space.get_distance(self.pos, self.target)
            target_dis = abs(self.pos[1] - 14.)
            self.goal_check(target_dis)
            self.tmp_pos[0] = self.pos[0] + \
                self.velocity[0] * self._shared.dt  # 仮の位置を計算
            self.tmp_pos[1] = self.pos[1] + self.velocity[1] * self._shared.dt
            if self.pos_check():
                break
            else:
                i -= 1
                self.reset_target()
        return None

    def write_record(self, path):
        if self.model.csv_plot:
            np.savetxt(f"{path}/csv/id{self.unique_id}_"
                       f"forceful.csv", self.pos_array, delimiter=",")
        if self.in_goal:
            with open(f"{self.add_file_name}/Data/"
                      f"forceful.dat", "a") as f:
                f.write(f"{self.elapsed_time} \n")
            path = path.replace(f"/seed_{self.model.seed}", "")
            df = pd.DataFrame({"m": [self.hspecs.m], "nol_pop": [self.model.population], "seed": [
                self.model.seed], "id": [self.unique_id], "elapsed_time": [self.elapsed_time]})
            df.to_csv(f"{path}/forceful_time.csv",
                      mode="a", header=False)
        return None
    

class Obstacle(mesa.Agent):
    def __init__(self, unique_id, model, pos, dir):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.dir = dir

    def step(self):
        return None


class Wall(mesa.Agent):
    def __init__(self, unique_id, model, pos, wall_r, dir):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.wall_r = wall_r
        self.dir = dir

    def step(self):
        return None


class Goal(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)

    def step(self):
        return None
