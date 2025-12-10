# sfm_mesa_convex_map
##　ブランチの歴史
1. feature/agent-turning(リファクタリング)
2. perf/wall_dist_vec(壁の計算方法を改良する)
3. refactor/waypoint-param(目的地の設定方法をハードコーディングから変数などの柔軟な方法に変える)
4. feature/map-scaling(今回：実践的なマップを作成し，避難者を動かす)

## 今回(このブランチ)の達成事項
- convex_mapからconvex_mapとintersection_mapを選べるように変更\\(ただしconvex_map側は調整しておらず動作するか不明)
- 初期経路選択方法を改良
- なんらかの理由で中間目的地に到達できず，その場にとどまっている場合は、経路を再探索する機能を追加
- 乱数生成方法をnp.randomからGenarator(rng)に変更

## 今回の最終目標
- 実践的なマップを作成し，避難者を動かす ok

## 後々やりたいこと
- (できればnumbaのnjitに対応させ，計算効率を上げたい)
