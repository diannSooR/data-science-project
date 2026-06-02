# 🔍 Reporte de Inspección del Modelo

**Fecha:** 2026-06-02 00:56:57
**Archivo:** `/home/ubuntu/Uploads/outputs/modelo/modelo_final.joblib`

---

## Tipo de Modelo

**AveragingEnsemble**

> Ensemble que promedia probabilidades de múltiples modelos

### Sub-modelos (2)

#### 1. Scikit-learn (GradientBoostingClassifier)

| Parámetro | Valor |
|-----------|-------|
| `ccp_alpha` | `0.0` |
| `criterion` | `friedman_mse` |
| `init` | `None` |
| `learning_rate` | `0.03` |
| `loss` | `log_loss` |
| `max_depth` | `4` |
| `max_features` | `sqrt` |
| `max_leaf_nodes` | `None` |
| `min_impurity_decrease` | `0.0` |
| `min_samples_leaf` | `8` |
| `min_samples_split` | `2` |
| `min_weight_fraction_leaf` | `0.0` |
| `n_estimators` | `800` |
| `n_iter_no_change` | `None` |
| `random_state` | `42` |
| `subsample` | `0.8` |
| `tol` | `0.0001` |
| `validation_fraction` | `0.1` |
| `verbose` | `0` |
| `warm_start` | `False` |

#### 2. XGBoost (XGBClassifier)

| Parámetro | Valor |
|-----------|-------|
| `objective` | `binary:logistic` |
| `base_score` | `None` |
| `booster` | `None` |
| `callbacks` | `None` |
| `colsample_bylevel` | `None` |
| `colsample_bynode` | `None` |
| `colsample_bytree` | `0.7` |
| `device` | `None` |
| `early_stopping_rounds` | `None` |
| `enable_categorical` | `False` |
| `eval_metric` | `auc` |
| `feature_types` | `None` |
| `feature_weights` | `None` |
| `gamma` | `None` |
| `grow_policy` | `None` |
| `importance_type` | `None` |
| `interaction_constraints` | `None` |
| `learning_rate` | `0.03` |
| `max_bin` | `None` |
| `max_cat_threshold` | `None` |
| `max_cat_to_onehot` | `None` |
| `max_delta_step` | `None` |
| `max_depth` | `4` |
| `max_leaves` | `None` |
| `min_child_weight` | `None` |
| `missing` | `nan` |
| `monotone_constraints` | `None` |
| `multi_strategy` | `None` |
| `n_estimators` | `800` |
| `n_jobs` | `-1` |
| `num_parallel_tree` | `None` |
| `random_state` | `42` |
| `reg_alpha` | `None` |
| `reg_lambda` | `2.0` |
| `sampling_method` | `None` |
| `scale_pos_weight` | `None` |
| `subsample` | `0.8` |
| `tree_method` | `None` |
| `validate_parameters` | `None` |
| `verbosity` | `0` |

## Features (25)

1. `riesgo_sismo`
2. `pct_afectacion_sismo`
3. `riesgo_inundacion`
4. `severidad_inundacion`
5. `pct_afectacion_inundacion`
6. `riesgo_laderas`
7. `pct_afectacion_laderas`
8. `fracturas_count`
9. `fracturas_longitud_m`
10. `tipo_suelo`
11. `pob_total`
12. `imu_2020`
13. `area_total`
14. `suma_riesgos`
15. `indice_riesgo_compuesto`
16. `riesgo_general`
17. `sismo_x_inundacion`
18. `fracturas_por_area`
19. `pob_densidad`
20. `log_pob`
21. `log_area`
22. `indice_x_pob`
23. `indice_x_area`
24. `imu_x_indice`
25. `sismo_x_suelo`

## Importancia de Features

| # | Feature | Importancia |
|---|---------|-------------|
| 1 | `indice_x_area` | 0.1286 ████████████ |
| 2 | `indice_riesgo_compuesto` | 0.1214 ████████████ |
| 3 | `imu_x_indice` | 0.0921 █████████ |
| 4 | `area_total` | 0.0637 ██████ |
| 5 | `log_area` | 0.0599 █████ |
| 6 | `riesgo_sismo` | 0.0592 █████ |
| 7 | `riesgo_general` | 0.0566 █████ |
| 8 | `imu_2020` | 0.0432 ████ |
| 9 | `suma_riesgos` | 0.0348 ███ |
| 10 | `indice_x_pob` | 0.0345 ███ |
| 11 | `sismo_x_suelo` | 0.0332 ███ |
| 12 | `log_pob` | 0.0324 ███ |
| 13 | `pct_afectacion_sismo` | 0.0297 ██ |
| 14 | `pob_total` | 0.0245 ██ |
| 15 | `sismo_x_inundacion` | 0.0233 ██ |
| 16 | `pct_afectacion_laderas` | 0.0208 ██ |
| 17 | `fracturas_count` | 0.0199 █ |
| 18 | `pob_densidad` | 0.0188 █ |
| 19 | `fracturas_por_area` | 0.0186 █ |
| 20 | `pct_afectacion_inundacion` | 0.0185 █ |
| 21 | `riesgo_laderas` | 0.0168 █ |
| 22 | `riesgo_inundacion` | 0.0141 █ |
| 23 | `fracturas_longitud_m` | 0.0129 █ |
| 24 | `severidad_inundacion` | 0.0113 █ |
| 25 | `tipo_suelo` | 0.0110 █ |

## Métricas de Evaluación

| Métrica | Valor |
|---------|-------|
| **accuracy** | 0.8941 ✅ |
| **precision** | 0.7426 |
| **recall** | 0.8559 |
| **f1** | 0.7953 |
| **auc_roc** | 0.9422 ✅ |
| **threshold** | 0.2600 |
