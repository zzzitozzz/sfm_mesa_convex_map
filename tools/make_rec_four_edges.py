import os
import numpy as np

tmp_arr = [[2.,2.],[156.,100.], [8.,8.],[48.,38.],  #十字路の全ての辺を作るための座標(辺)
           [54.,8.],[84.,38.], [90.,8.],[150.,38.],
           [8.,44.],[48.,94.], [54.,44.],[84.,94.],
           [90.,44.],[150.,94.]]

tmp_arr = [[8.,8.],[48.,38.],[54.,8.],[84.,38.], # 避難者の初期位置を作るための座標(点)
           [90.,8.],[150.,38.],[8.,44.],[48.,94.], 
           [54.,44.],[84.,94.],[90.,44.],[150.,94.]]
tmp_arr2 = []
i = 0
while 1:
    if i >= len(tmp_arr) - 1:
        break
    point_x1 = tmp_arr[i] # 左上
    point_x2 = tmp_arr[i+1] # 右下
    point_x3 = [point_x2[0], point_x1[1]] # 右上
    point_x4 = [point_x1[0], point_x2[1]] # 左下
    # tmp_arr2.append([point_x1, point_x3]) # 上辺
    # tmp_arr2.append([point_x3, point_x2]) # 右辺 
    # tmp_arr2.append([point_x2, point_x4]) # 下辺
    # tmp_arr2.append([point_x4, point_x1]) # 左辺
    print(f"{point_x1}, {point_x2}, {point_x3}, {point_x4}\n")
    i += 2

print(f"{tmp_arr2}")

