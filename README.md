# sfm_mesa_convex_map
##　ブランチの歴史
1. feature/agent-turning(リファクタリング)
2. perf/wall_dist_vec(今回：壁の計算方法を改良する)

## 今回の最終目的
- 主な壁と避難者の距離の計算方法をベクトル演算のみに変える ← 途中(曲がり角付近の避難者の挙動がおかしいため)
- (できればnumbaのnjitに対応させ，計算効率を上げたい)