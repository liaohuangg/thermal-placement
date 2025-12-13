\* ChipletPlacementGrid *\
Minimize
Objective: 0.1 bbox_area_proxy_t + dx_abs_0_1 + dx_abs_1_2 + dx_abs_1_3
 + dx_abs_2_4 + dy_abs_0_1 + dy_abs_1_2 + dy_abs_1_3 + dy_abs_2_4
Subject To
all_not_same_implies_diff_dist_pair_c5_0_1: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev0 + same_dist_pair_c5_0_1_prev1
 + same_dist_pair_c5_0_1_prev2 + same_dist_pair_c5_0_1_prev3
 + same_dist_pair_c5_0_1_prev4 + same_dist_pair_c5_0_1_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_0_2: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev0 + same_dist_pair_c5_0_2_prev1
 + same_dist_pair_c5_0_2_prev2 + same_dist_pair_c5_0_2_prev3
 + same_dist_pair_c5_0_2_prev4 + same_dist_pair_c5_0_2_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_0_3: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev0 + same_dist_pair_c5_0_3_prev1
 + same_dist_pair_c5_0_3_prev2 + same_dist_pair_c5_0_3_prev3
 + same_dist_pair_c5_0_3_prev4 + same_dist_pair_c5_0_3_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_0_4: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev0 + same_dist_pair_c5_0_4_prev1
 + same_dist_pair_c5_0_4_prev2 + same_dist_pair_c5_0_4_prev3
 + same_dist_pair_c5_0_4_prev4 + same_dist_pair_c5_0_4_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_1_2: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev0 + same_dist_pair_c5_1_2_prev1
 + same_dist_pair_c5_1_2_prev2 + same_dist_pair_c5_1_2_prev3
 + same_dist_pair_c5_1_2_prev4 + same_dist_pair_c5_1_2_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_1_3: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev0 + same_dist_pair_c5_1_3_prev1
 + same_dist_pair_c5_1_3_prev2 + same_dist_pair_c5_1_3_prev3
 + same_dist_pair_c5_1_3_prev4 + same_dist_pair_c5_1_3_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_1_4: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev0 + same_dist_pair_c5_1_4_prev1
 + same_dist_pair_c5_1_4_prev2 + same_dist_pair_c5_1_4_prev3
 + same_dist_pair_c5_1_4_prev4 + same_dist_pair_c5_1_4_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_2_3: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev0 + same_dist_pair_c5_2_3_prev1
 + same_dist_pair_c5_2_3_prev2 + same_dist_pair_c5_2_3_prev3
 + same_dist_pair_c5_2_3_prev4 + same_dist_pair_c5_2_3_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_2_4: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev0 + same_dist_pair_c5_2_4_prev1
 + same_dist_pair_c5_2_4_prev2 + same_dist_pair_c5_2_4_prev3
 + same_dist_pair_c5_2_4_prev4 + same_dist_pair_c5_2_4_prev5 >= 1
all_not_same_implies_diff_dist_pair_c5_3_4: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev0 + same_dist_pair_c5_3_4_prev1
 + same_dist_pair_c5_3_4_prev2 + same_dist_pair_c5_3_4_prev3
 + same_dist_pair_c5_3_4_prev4 + same_dist_pair_c5_3_4_prev5 >= 1
bbox_h_def: bbox_h - bbox_max_y + bbox_min_y = 0
bbox_max_x_0: bbox_max_x - w_0 - x_0 >= 0
bbox_max_x_1: bbox_max_x - w_1 - x_1 >= 0
bbox_max_x_2: bbox_max_x - w_2 - x_2 >= 0
bbox_max_x_3: bbox_max_x - w_3 - x_3 >= 0
bbox_max_x_4: bbox_max_x - w_4 - x_4 >= 0
bbox_max_y_0: bbox_max_y - h_0 - y_0 >= 0
bbox_max_y_1: bbox_max_y - h_1 - y_1 >= 0
bbox_max_y_2: bbox_max_y - h_2 - y_2 >= 0
bbox_max_y_3: bbox_max_y - h_3 - y_3 >= 0
bbox_max_y_4: bbox_max_y - h_4 - y_4 >= 0
bbox_min_x_0: bbox_min_x - x_0 <= 0
bbox_min_x_1: bbox_min_x - x_1 <= 0
bbox_min_x_2: bbox_min_x - x_2 <= 0
bbox_min_x_3: bbox_min_x - x_3 <= 0
bbox_min_x_4: bbox_min_x - x_4 <= 0
bbox_min_y_0: bbox_min_y - y_0 <= 0
bbox_min_y_1: bbox_min_y - y_1 <= 0
bbox_min_y_2: bbox_min_y - y_2 <= 0
bbox_min_y_3: bbox_min_y - y_3 <= 0
bbox_min_y_4: bbox_min_y - y_4 <= 0
bbox_w_def: - bbox_max_x + bbox_min_x + bbox_w = 0
boundary_x_0: w_0 + x_0 <= 60
boundary_x_1: w_1 + x_1 <= 60
boundary_x_2: w_2 + x_2 <= 60
boundary_x_3: w_3 + x_3 <= 60
boundary_x_4: w_4 + x_4 <= 60
boundary_y_0: h_0 + y_0 <= 60
boundary_y_1: h_1 + y_1 <= 60
boundary_y_2: h_2 + y_2 <= 60
boundary_y_3: h_3 + y_3 <= 60
boundary_y_4: h_4 + y_4 <= 60
cx_def_0: cx_0 - 0.5 w_0 - x_0 = 0
cx_def_1: cx_1 - 0.5 w_1 - x_1 = 0
cx_def_2: cx_2 - 0.5 w_2 - x_2 = 0
cx_def_3: cx_3 - 0.5 w_3 - x_3 = 0
cx_def_4: cx_4 - 0.5 w_4 - x_4 = 0
cy_def_0: cy_0 - 0.5 h_0 - y_0 = 0
cy_def_1: cy_1 - 0.5 h_1 - y_1 = 0
cy_def_2: cy_2 - 0.5 h_2 - y_2 = 0
cy_def_3: cy_3 - 0.5 h_3 - y_3 = 0
cy_def_4: cy_4 - 0.5 h_4 - y_4 = 0
diff_dist_pair_implies_not_same_c5_0_1_prev0: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev0 <= 1
diff_dist_pair_implies_not_same_c5_0_1_prev1: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev1 <= 1
diff_dist_pair_implies_not_same_c5_0_1_prev2: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev2 <= 1
diff_dist_pair_implies_not_same_c5_0_1_prev3: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev3 <= 1
diff_dist_pair_implies_not_same_c5_0_1_prev4: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev4 <= 1
diff_dist_pair_implies_not_same_c5_0_1_prev5: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev5 <= 1
diff_dist_pair_implies_not_same_c5_0_2_prev0: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev0 <= 1
diff_dist_pair_implies_not_same_c5_0_2_prev1: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev1 <= 1
diff_dist_pair_implies_not_same_c5_0_2_prev2: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev2 <= 1
diff_dist_pair_implies_not_same_c5_0_2_prev3: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev3 <= 1
diff_dist_pair_implies_not_same_c5_0_2_prev4: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev4 <= 1
diff_dist_pair_implies_not_same_c5_0_2_prev5: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev5 <= 1
diff_dist_pair_implies_not_same_c5_0_3_prev0: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev0 <= 1
diff_dist_pair_implies_not_same_c5_0_3_prev1: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev1 <= 1
diff_dist_pair_implies_not_same_c5_0_3_prev2: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev2 <= 1
diff_dist_pair_implies_not_same_c5_0_3_prev3: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev3 <= 1
diff_dist_pair_implies_not_same_c5_0_3_prev4: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev4 <= 1
diff_dist_pair_implies_not_same_c5_0_3_prev5: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev5 <= 1
diff_dist_pair_implies_not_same_c5_0_4_prev0: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev0 <= 1
diff_dist_pair_implies_not_same_c5_0_4_prev1: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev1 <= 1
diff_dist_pair_implies_not_same_c5_0_4_prev2: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev2 <= 1
diff_dist_pair_implies_not_same_c5_0_4_prev3: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev3 <= 1
diff_dist_pair_implies_not_same_c5_0_4_prev4: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev4 <= 1
diff_dist_pair_implies_not_same_c5_0_4_prev5: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev5 <= 1
diff_dist_pair_implies_not_same_c5_1_2_prev0: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev0 <= 1
diff_dist_pair_implies_not_same_c5_1_2_prev1: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev1 <= 1
diff_dist_pair_implies_not_same_c5_1_2_prev2: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev2 <= 1
diff_dist_pair_implies_not_same_c5_1_2_prev3: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev3 <= 1
diff_dist_pair_implies_not_same_c5_1_2_prev4: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev4 <= 1
diff_dist_pair_implies_not_same_c5_1_2_prev5: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev5 <= 1
diff_dist_pair_implies_not_same_c5_1_3_prev0: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev0 <= 1
diff_dist_pair_implies_not_same_c5_1_3_prev1: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev1 <= 1
diff_dist_pair_implies_not_same_c5_1_3_prev2: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev2 <= 1
diff_dist_pair_implies_not_same_c5_1_3_prev3: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev3 <= 1
diff_dist_pair_implies_not_same_c5_1_3_prev4: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev4 <= 1
diff_dist_pair_implies_not_same_c5_1_3_prev5: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev5 <= 1
diff_dist_pair_implies_not_same_c5_1_4_prev0: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev0 <= 1
diff_dist_pair_implies_not_same_c5_1_4_prev1: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev1 <= 1
diff_dist_pair_implies_not_same_c5_1_4_prev2: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev2 <= 1
diff_dist_pair_implies_not_same_c5_1_4_prev3: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev3 <= 1
diff_dist_pair_implies_not_same_c5_1_4_prev4: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev4 <= 1
diff_dist_pair_implies_not_same_c5_1_4_prev5: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev5 <= 1
diff_dist_pair_implies_not_same_c5_2_3_prev0: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev0 <= 1
diff_dist_pair_implies_not_same_c5_2_3_prev1: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev1 <= 1
diff_dist_pair_implies_not_same_c5_2_3_prev2: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev2 <= 1
diff_dist_pair_implies_not_same_c5_2_3_prev3: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev3 <= 1
diff_dist_pair_implies_not_same_c5_2_3_prev4: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev4 <= 1
diff_dist_pair_implies_not_same_c5_2_3_prev5: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev5 <= 1
diff_dist_pair_implies_not_same_c5_2_4_prev0: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev0 <= 1
diff_dist_pair_implies_not_same_c5_2_4_prev1: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev1 <= 1
diff_dist_pair_implies_not_same_c5_2_4_prev2: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev2 <= 1
diff_dist_pair_implies_not_same_c5_2_4_prev3: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev3 <= 1
diff_dist_pair_implies_not_same_c5_2_4_prev4: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev4 <= 1
diff_dist_pair_implies_not_same_c5_2_4_prev5: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev5 <= 1
diff_dist_pair_implies_not_same_c5_3_4_prev0: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev0 <= 1
diff_dist_pair_implies_not_same_c5_3_4_prev1: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev1 <= 1
diff_dist_pair_implies_not_same_c5_3_4_prev2: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev2 <= 1
diff_dist_pair_implies_not_same_c5_3_4_prev3: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev3 <= 1
diff_dist_pair_implies_not_same_c5_3_4_prev4: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev4 <= 1
diff_dist_pair_implies_not_same_c5_3_4_prev5: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev5 <= 1
dist_curr_pair_def_c5_0_1: dist_curr_pair_c5_0_1 - dx_grid_abs_pair_c5_0_1
 - dy_grid_abs_pair_c5_0_1 = 0
dist_curr_pair_def_c5_0_2: dist_curr_pair_c5_0_2 - dx_grid_abs_pair_c5_0_2
 - dy_grid_abs_pair_c5_0_2 = 0
dist_curr_pair_def_c5_0_3: dist_curr_pair_c5_0_3 - dx_grid_abs_pair_c5_0_3
 - dy_grid_abs_pair_c5_0_3 = 0
dist_curr_pair_def_c5_0_4: dist_curr_pair_c5_0_4 - dx_grid_abs_pair_c5_0_4
 - dy_grid_abs_pair_c5_0_4 = 0
dist_curr_pair_def_c5_1_2: dist_curr_pair_c5_1_2 - dx_grid_abs_pair_c5_1_2
 - dy_grid_abs_pair_c5_1_2 = 0
dist_curr_pair_def_c5_1_3: dist_curr_pair_c5_1_3 - dx_grid_abs_pair_c5_1_3
 - dy_grid_abs_pair_c5_1_3 = 0
dist_curr_pair_def_c5_1_4: dist_curr_pair_c5_1_4 - dx_grid_abs_pair_c5_1_4
 - dy_grid_abs_pair_c5_1_4 = 0
dist_curr_pair_def_c5_2_3: dist_curr_pair_c5_2_3 - dx_grid_abs_pair_c5_2_3
 - dy_grid_abs_pair_c5_2_3 = 0
dist_curr_pair_def_c5_2_4: dist_curr_pair_c5_2_4 - dx_grid_abs_pair_c5_2_4
 - dy_grid_abs_pair_c5_2_4 = 0
dist_curr_pair_def_c5_3_4: dist_curr_pair_c5_3_4 - dx_grid_abs_pair_c5_3_4
 - dy_grid_abs_pair_c5_3_4 = 0
dist_diff_abs_pair_c5_0_1_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev0
 + 240 dist_diff_abs_pair_c5_0_1_prev0_is_positive
 + dist_diff_pair_c5_0_1_prev0 >= 0
dist_diff_abs_pair_c5_0_1_prev0_abs_ge_orig: dist_diff_abs_pair_c5_0_1_prev0
 - 240 dist_diff_abs_pair_c5_0_1_prev0_is_positive
 - dist_diff_pair_c5_0_1_prev0 >= -240
dist_diff_abs_pair_c5_0_1_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev0
 - 240 dist_diff_abs_pair_c5_0_1_prev0_is_positive
 + dist_diff_pair_c5_0_1_prev0 <= 0
dist_diff_abs_pair_c5_0_1_prev0_abs_le_orig: dist_diff_abs_pair_c5_0_1_prev0
 + 240 dist_diff_abs_pair_c5_0_1_prev0_is_positive
 - dist_diff_pair_c5_0_1_prev0 <= 240
dist_diff_abs_pair_c5_0_1_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_0_1_prev0_is_positive
 + dist_diff_pair_c5_0_1_prev0 <= -0.001
dist_diff_abs_pair_c5_0_1_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_0_1_prev0_is_positive
 + dist_diff_pair_c5_0_1_prev0 >= -240
dist_diff_abs_pair_c5_0_1_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev1
 + 240 dist_diff_abs_pair_c5_0_1_prev1_is_positive
 + dist_diff_pair_c5_0_1_prev1 >= 0
dist_diff_abs_pair_c5_0_1_prev1_abs_ge_orig: dist_diff_abs_pair_c5_0_1_prev1
 - 240 dist_diff_abs_pair_c5_0_1_prev1_is_positive
 - dist_diff_pair_c5_0_1_prev1 >= -240
dist_diff_abs_pair_c5_0_1_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev1
 - 240 dist_diff_abs_pair_c5_0_1_prev1_is_positive
 + dist_diff_pair_c5_0_1_prev1 <= 0
dist_diff_abs_pair_c5_0_1_prev1_abs_le_orig: dist_diff_abs_pair_c5_0_1_prev1
 + 240 dist_diff_abs_pair_c5_0_1_prev1_is_positive
 - dist_diff_pair_c5_0_1_prev1 <= 240
dist_diff_abs_pair_c5_0_1_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_0_1_prev1_is_positive
 + dist_diff_pair_c5_0_1_prev1 <= -0.001
dist_diff_abs_pair_c5_0_1_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_0_1_prev1_is_positive
 + dist_diff_pair_c5_0_1_prev1 >= -240
dist_diff_abs_pair_c5_0_1_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev2
 + 240 dist_diff_abs_pair_c5_0_1_prev2_is_positive
 + dist_diff_pair_c5_0_1_prev2 >= 0
dist_diff_abs_pair_c5_0_1_prev2_abs_ge_orig: dist_diff_abs_pair_c5_0_1_prev2
 - 240 dist_diff_abs_pair_c5_0_1_prev2_is_positive
 - dist_diff_pair_c5_0_1_prev2 >= -240
dist_diff_abs_pair_c5_0_1_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev2
 - 240 dist_diff_abs_pair_c5_0_1_prev2_is_positive
 + dist_diff_pair_c5_0_1_prev2 <= 0
dist_diff_abs_pair_c5_0_1_prev2_abs_le_orig: dist_diff_abs_pair_c5_0_1_prev2
 + 240 dist_diff_abs_pair_c5_0_1_prev2_is_positive
 - dist_diff_pair_c5_0_1_prev2 <= 240
dist_diff_abs_pair_c5_0_1_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_0_1_prev2_is_positive
 + dist_diff_pair_c5_0_1_prev2 <= -0.001
dist_diff_abs_pair_c5_0_1_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_0_1_prev2_is_positive
 + dist_diff_pair_c5_0_1_prev2 >= -240
dist_diff_abs_pair_c5_0_1_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev3
 + 240 dist_diff_abs_pair_c5_0_1_prev3_is_positive
 + dist_diff_pair_c5_0_1_prev3 >= 0
dist_diff_abs_pair_c5_0_1_prev3_abs_ge_orig: dist_diff_abs_pair_c5_0_1_prev3
 - 240 dist_diff_abs_pair_c5_0_1_prev3_is_positive
 - dist_diff_pair_c5_0_1_prev3 >= -240
dist_diff_abs_pair_c5_0_1_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev3
 - 240 dist_diff_abs_pair_c5_0_1_prev3_is_positive
 + dist_diff_pair_c5_0_1_prev3 <= 0
dist_diff_abs_pair_c5_0_1_prev3_abs_le_orig: dist_diff_abs_pair_c5_0_1_prev3
 + 240 dist_diff_abs_pair_c5_0_1_prev3_is_positive
 - dist_diff_pair_c5_0_1_prev3 <= 240
dist_diff_abs_pair_c5_0_1_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_0_1_prev3_is_positive
 + dist_diff_pair_c5_0_1_prev3 <= -0.001
dist_diff_abs_pair_c5_0_1_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_0_1_prev3_is_positive
 + dist_diff_pair_c5_0_1_prev3 >= -240
dist_diff_abs_pair_c5_0_1_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev4
 + 240 dist_diff_abs_pair_c5_0_1_prev4_is_positive
 + dist_diff_pair_c5_0_1_prev4 >= 0
dist_diff_abs_pair_c5_0_1_prev4_abs_ge_orig: dist_diff_abs_pair_c5_0_1_prev4
 - 240 dist_diff_abs_pair_c5_0_1_prev4_is_positive
 - dist_diff_pair_c5_0_1_prev4 >= -240
dist_diff_abs_pair_c5_0_1_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev4
 - 240 dist_diff_abs_pair_c5_0_1_prev4_is_positive
 + dist_diff_pair_c5_0_1_prev4 <= 0
dist_diff_abs_pair_c5_0_1_prev4_abs_le_orig: dist_diff_abs_pair_c5_0_1_prev4
 + 240 dist_diff_abs_pair_c5_0_1_prev4_is_positive
 - dist_diff_pair_c5_0_1_prev4 <= 240
dist_diff_abs_pair_c5_0_1_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_0_1_prev4_is_positive
 + dist_diff_pair_c5_0_1_prev4 <= -0.001
dist_diff_abs_pair_c5_0_1_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_0_1_prev4_is_positive
 + dist_diff_pair_c5_0_1_prev4 >= -240
dist_diff_abs_pair_c5_0_1_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev5
 + 240 dist_diff_abs_pair_c5_0_1_prev5_is_positive
 + dist_diff_pair_c5_0_1_prev5 >= 0
dist_diff_abs_pair_c5_0_1_prev5_abs_ge_orig: dist_diff_abs_pair_c5_0_1_prev5
 - 240 dist_diff_abs_pair_c5_0_1_prev5_is_positive
 - dist_diff_pair_c5_0_1_prev5 >= -240
dist_diff_abs_pair_c5_0_1_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_1_prev5
 - 240 dist_diff_abs_pair_c5_0_1_prev5_is_positive
 + dist_diff_pair_c5_0_1_prev5 <= 0
dist_diff_abs_pair_c5_0_1_prev5_abs_le_orig: dist_diff_abs_pair_c5_0_1_prev5
 + 240 dist_diff_abs_pair_c5_0_1_prev5_is_positive
 - dist_diff_pair_c5_0_1_prev5 <= 240
dist_diff_abs_pair_c5_0_1_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_0_1_prev5_is_positive
 + dist_diff_pair_c5_0_1_prev5 <= -0.001
dist_diff_abs_pair_c5_0_1_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_0_1_prev5_is_positive
 + dist_diff_pair_c5_0_1_prev5 >= -240
dist_diff_abs_pair_c5_0_2_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev0
 + 240 dist_diff_abs_pair_c5_0_2_prev0_is_positive
 + dist_diff_pair_c5_0_2_prev0 >= 0
dist_diff_abs_pair_c5_0_2_prev0_abs_ge_orig: dist_diff_abs_pair_c5_0_2_prev0
 - 240 dist_diff_abs_pair_c5_0_2_prev0_is_positive
 - dist_diff_pair_c5_0_2_prev0 >= -240
dist_diff_abs_pair_c5_0_2_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev0
 - 240 dist_diff_abs_pair_c5_0_2_prev0_is_positive
 + dist_diff_pair_c5_0_2_prev0 <= 0
dist_diff_abs_pair_c5_0_2_prev0_abs_le_orig: dist_diff_abs_pair_c5_0_2_prev0
 + 240 dist_diff_abs_pair_c5_0_2_prev0_is_positive
 - dist_diff_pair_c5_0_2_prev0 <= 240
dist_diff_abs_pair_c5_0_2_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_0_2_prev0_is_positive
 + dist_diff_pair_c5_0_2_prev0 <= -0.001
dist_diff_abs_pair_c5_0_2_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_0_2_prev0_is_positive
 + dist_diff_pair_c5_0_2_prev0 >= -240
dist_diff_abs_pair_c5_0_2_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev1
 + 240 dist_diff_abs_pair_c5_0_2_prev1_is_positive
 + dist_diff_pair_c5_0_2_prev1 >= 0
dist_diff_abs_pair_c5_0_2_prev1_abs_ge_orig: dist_diff_abs_pair_c5_0_2_prev1
 - 240 dist_diff_abs_pair_c5_0_2_prev1_is_positive
 - dist_diff_pair_c5_0_2_prev1 >= -240
dist_diff_abs_pair_c5_0_2_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev1
 - 240 dist_diff_abs_pair_c5_0_2_prev1_is_positive
 + dist_diff_pair_c5_0_2_prev1 <= 0
dist_diff_abs_pair_c5_0_2_prev1_abs_le_orig: dist_diff_abs_pair_c5_0_2_prev1
 + 240 dist_diff_abs_pair_c5_0_2_prev1_is_positive
 - dist_diff_pair_c5_0_2_prev1 <= 240
dist_diff_abs_pair_c5_0_2_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_0_2_prev1_is_positive
 + dist_diff_pair_c5_0_2_prev1 <= -0.001
dist_diff_abs_pair_c5_0_2_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_0_2_prev1_is_positive
 + dist_diff_pair_c5_0_2_prev1 >= -240
dist_diff_abs_pair_c5_0_2_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev2
 + 240 dist_diff_abs_pair_c5_0_2_prev2_is_positive
 + dist_diff_pair_c5_0_2_prev2 >= 0
dist_diff_abs_pair_c5_0_2_prev2_abs_ge_orig: dist_diff_abs_pair_c5_0_2_prev2
 - 240 dist_diff_abs_pair_c5_0_2_prev2_is_positive
 - dist_diff_pair_c5_0_2_prev2 >= -240
dist_diff_abs_pair_c5_0_2_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev2
 - 240 dist_diff_abs_pair_c5_0_2_prev2_is_positive
 + dist_diff_pair_c5_0_2_prev2 <= 0
dist_diff_abs_pair_c5_0_2_prev2_abs_le_orig: dist_diff_abs_pair_c5_0_2_prev2
 + 240 dist_diff_abs_pair_c5_0_2_prev2_is_positive
 - dist_diff_pair_c5_0_2_prev2 <= 240
dist_diff_abs_pair_c5_0_2_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_0_2_prev2_is_positive
 + dist_diff_pair_c5_0_2_prev2 <= -0.001
dist_diff_abs_pair_c5_0_2_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_0_2_prev2_is_positive
 + dist_diff_pair_c5_0_2_prev2 >= -240
dist_diff_abs_pair_c5_0_2_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev3
 + 240 dist_diff_abs_pair_c5_0_2_prev3_is_positive
 + dist_diff_pair_c5_0_2_prev3 >= 0
dist_diff_abs_pair_c5_0_2_prev3_abs_ge_orig: dist_diff_abs_pair_c5_0_2_prev3
 - 240 dist_diff_abs_pair_c5_0_2_prev3_is_positive
 - dist_diff_pair_c5_0_2_prev3 >= -240
dist_diff_abs_pair_c5_0_2_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev3
 - 240 dist_diff_abs_pair_c5_0_2_prev3_is_positive
 + dist_diff_pair_c5_0_2_prev3 <= 0
dist_diff_abs_pair_c5_0_2_prev3_abs_le_orig: dist_diff_abs_pair_c5_0_2_prev3
 + 240 dist_diff_abs_pair_c5_0_2_prev3_is_positive
 - dist_diff_pair_c5_0_2_prev3 <= 240
dist_diff_abs_pair_c5_0_2_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_0_2_prev3_is_positive
 + dist_diff_pair_c5_0_2_prev3 <= -0.001
dist_diff_abs_pair_c5_0_2_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_0_2_prev3_is_positive
 + dist_diff_pair_c5_0_2_prev3 >= -240
dist_diff_abs_pair_c5_0_2_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev4
 + 240 dist_diff_abs_pair_c5_0_2_prev4_is_positive
 + dist_diff_pair_c5_0_2_prev4 >= 0
dist_diff_abs_pair_c5_0_2_prev4_abs_ge_orig: dist_diff_abs_pair_c5_0_2_prev4
 - 240 dist_diff_abs_pair_c5_0_2_prev4_is_positive
 - dist_diff_pair_c5_0_2_prev4 >= -240
dist_diff_abs_pair_c5_0_2_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev4
 - 240 dist_diff_abs_pair_c5_0_2_prev4_is_positive
 + dist_diff_pair_c5_0_2_prev4 <= 0
dist_diff_abs_pair_c5_0_2_prev4_abs_le_orig: dist_diff_abs_pair_c5_0_2_prev4
 + 240 dist_diff_abs_pair_c5_0_2_prev4_is_positive
 - dist_diff_pair_c5_0_2_prev4 <= 240
dist_diff_abs_pair_c5_0_2_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_0_2_prev4_is_positive
 + dist_diff_pair_c5_0_2_prev4 <= -0.001
dist_diff_abs_pair_c5_0_2_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_0_2_prev4_is_positive
 + dist_diff_pair_c5_0_2_prev4 >= -240
dist_diff_abs_pair_c5_0_2_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev5
 + 240 dist_diff_abs_pair_c5_0_2_prev5_is_positive
 + dist_diff_pair_c5_0_2_prev5 >= 0
dist_diff_abs_pair_c5_0_2_prev5_abs_ge_orig: dist_diff_abs_pair_c5_0_2_prev5
 - 240 dist_diff_abs_pair_c5_0_2_prev5_is_positive
 - dist_diff_pair_c5_0_2_prev5 >= -240
dist_diff_abs_pair_c5_0_2_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_2_prev5
 - 240 dist_diff_abs_pair_c5_0_2_prev5_is_positive
 + dist_diff_pair_c5_0_2_prev5 <= 0
dist_diff_abs_pair_c5_0_2_prev5_abs_le_orig: dist_diff_abs_pair_c5_0_2_prev5
 + 240 dist_diff_abs_pair_c5_0_2_prev5_is_positive
 - dist_diff_pair_c5_0_2_prev5 <= 240
dist_diff_abs_pair_c5_0_2_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_0_2_prev5_is_positive
 + dist_diff_pair_c5_0_2_prev5 <= -0.001
dist_diff_abs_pair_c5_0_2_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_0_2_prev5_is_positive
 + dist_diff_pair_c5_0_2_prev5 >= -240
dist_diff_abs_pair_c5_0_3_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev0
 + 240 dist_diff_abs_pair_c5_0_3_prev0_is_positive
 + dist_diff_pair_c5_0_3_prev0 >= 0
dist_diff_abs_pair_c5_0_3_prev0_abs_ge_orig: dist_diff_abs_pair_c5_0_3_prev0
 - 240 dist_diff_abs_pair_c5_0_3_prev0_is_positive
 - dist_diff_pair_c5_0_3_prev0 >= -240
dist_diff_abs_pair_c5_0_3_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev0
 - 240 dist_diff_abs_pair_c5_0_3_prev0_is_positive
 + dist_diff_pair_c5_0_3_prev0 <= 0
dist_diff_abs_pair_c5_0_3_prev0_abs_le_orig: dist_diff_abs_pair_c5_0_3_prev0
 + 240 dist_diff_abs_pair_c5_0_3_prev0_is_positive
 - dist_diff_pair_c5_0_3_prev0 <= 240
dist_diff_abs_pair_c5_0_3_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_0_3_prev0_is_positive
 + dist_diff_pair_c5_0_3_prev0 <= -0.001
dist_diff_abs_pair_c5_0_3_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_0_3_prev0_is_positive
 + dist_diff_pair_c5_0_3_prev0 >= -240
dist_diff_abs_pair_c5_0_3_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev1
 + 240 dist_diff_abs_pair_c5_0_3_prev1_is_positive
 + dist_diff_pair_c5_0_3_prev1 >= 0
dist_diff_abs_pair_c5_0_3_prev1_abs_ge_orig: dist_diff_abs_pair_c5_0_3_prev1
 - 240 dist_diff_abs_pair_c5_0_3_prev1_is_positive
 - dist_diff_pair_c5_0_3_prev1 >= -240
dist_diff_abs_pair_c5_0_3_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev1
 - 240 dist_diff_abs_pair_c5_0_3_prev1_is_positive
 + dist_diff_pair_c5_0_3_prev1 <= 0
dist_diff_abs_pair_c5_0_3_prev1_abs_le_orig: dist_diff_abs_pair_c5_0_3_prev1
 + 240 dist_diff_abs_pair_c5_0_3_prev1_is_positive
 - dist_diff_pair_c5_0_3_prev1 <= 240
dist_diff_abs_pair_c5_0_3_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_0_3_prev1_is_positive
 + dist_diff_pair_c5_0_3_prev1 <= -0.001
dist_diff_abs_pair_c5_0_3_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_0_3_prev1_is_positive
 + dist_diff_pair_c5_0_3_prev1 >= -240
dist_diff_abs_pair_c5_0_3_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev2
 + 240 dist_diff_abs_pair_c5_0_3_prev2_is_positive
 + dist_diff_pair_c5_0_3_prev2 >= 0
dist_diff_abs_pair_c5_0_3_prev2_abs_ge_orig: dist_diff_abs_pair_c5_0_3_prev2
 - 240 dist_diff_abs_pair_c5_0_3_prev2_is_positive
 - dist_diff_pair_c5_0_3_prev2 >= -240
dist_diff_abs_pair_c5_0_3_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev2
 - 240 dist_diff_abs_pair_c5_0_3_prev2_is_positive
 + dist_diff_pair_c5_0_3_prev2 <= 0
dist_diff_abs_pair_c5_0_3_prev2_abs_le_orig: dist_diff_abs_pair_c5_0_3_prev2
 + 240 dist_diff_abs_pair_c5_0_3_prev2_is_positive
 - dist_diff_pair_c5_0_3_prev2 <= 240
dist_diff_abs_pair_c5_0_3_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_0_3_prev2_is_positive
 + dist_diff_pair_c5_0_3_prev2 <= -0.001
dist_diff_abs_pair_c5_0_3_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_0_3_prev2_is_positive
 + dist_diff_pair_c5_0_3_prev2 >= -240
dist_diff_abs_pair_c5_0_3_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev3
 + 240 dist_diff_abs_pair_c5_0_3_prev3_is_positive
 + dist_diff_pair_c5_0_3_prev3 >= 0
dist_diff_abs_pair_c5_0_3_prev3_abs_ge_orig: dist_diff_abs_pair_c5_0_3_prev3
 - 240 dist_diff_abs_pair_c5_0_3_prev3_is_positive
 - dist_diff_pair_c5_0_3_prev3 >= -240
dist_diff_abs_pair_c5_0_3_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev3
 - 240 dist_diff_abs_pair_c5_0_3_prev3_is_positive
 + dist_diff_pair_c5_0_3_prev3 <= 0
dist_diff_abs_pair_c5_0_3_prev3_abs_le_orig: dist_diff_abs_pair_c5_0_3_prev3
 + 240 dist_diff_abs_pair_c5_0_3_prev3_is_positive
 - dist_diff_pair_c5_0_3_prev3 <= 240
dist_diff_abs_pair_c5_0_3_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_0_3_prev3_is_positive
 + dist_diff_pair_c5_0_3_prev3 <= -0.001
dist_diff_abs_pair_c5_0_3_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_0_3_prev3_is_positive
 + dist_diff_pair_c5_0_3_prev3 >= -240
dist_diff_abs_pair_c5_0_3_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev4
 + 240 dist_diff_abs_pair_c5_0_3_prev4_is_positive
 + dist_diff_pair_c5_0_3_prev4 >= 0
dist_diff_abs_pair_c5_0_3_prev4_abs_ge_orig: dist_diff_abs_pair_c5_0_3_prev4
 - 240 dist_diff_abs_pair_c5_0_3_prev4_is_positive
 - dist_diff_pair_c5_0_3_prev4 >= -240
dist_diff_abs_pair_c5_0_3_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev4
 - 240 dist_diff_abs_pair_c5_0_3_prev4_is_positive
 + dist_diff_pair_c5_0_3_prev4 <= 0
dist_diff_abs_pair_c5_0_3_prev4_abs_le_orig: dist_diff_abs_pair_c5_0_3_prev4
 + 240 dist_diff_abs_pair_c5_0_3_prev4_is_positive
 - dist_diff_pair_c5_0_3_prev4 <= 240
dist_diff_abs_pair_c5_0_3_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_0_3_prev4_is_positive
 + dist_diff_pair_c5_0_3_prev4 <= -0.001
dist_diff_abs_pair_c5_0_3_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_0_3_prev4_is_positive
 + dist_diff_pair_c5_0_3_prev4 >= -240
dist_diff_abs_pair_c5_0_3_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev5
 + 240 dist_diff_abs_pair_c5_0_3_prev5_is_positive
 + dist_diff_pair_c5_0_3_prev5 >= 0
dist_diff_abs_pair_c5_0_3_prev5_abs_ge_orig: dist_diff_abs_pair_c5_0_3_prev5
 - 240 dist_diff_abs_pair_c5_0_3_prev5_is_positive
 - dist_diff_pair_c5_0_3_prev5 >= -240
dist_diff_abs_pair_c5_0_3_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_3_prev5
 - 240 dist_diff_abs_pair_c5_0_3_prev5_is_positive
 + dist_diff_pair_c5_0_3_prev5 <= 0
dist_diff_abs_pair_c5_0_3_prev5_abs_le_orig: dist_diff_abs_pair_c5_0_3_prev5
 + 240 dist_diff_abs_pair_c5_0_3_prev5_is_positive
 - dist_diff_pair_c5_0_3_prev5 <= 240
dist_diff_abs_pair_c5_0_3_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_0_3_prev5_is_positive
 + dist_diff_pair_c5_0_3_prev5 <= -0.001
dist_diff_abs_pair_c5_0_3_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_0_3_prev5_is_positive
 + dist_diff_pair_c5_0_3_prev5 >= -240
dist_diff_abs_pair_c5_0_4_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev0
 + 240 dist_diff_abs_pair_c5_0_4_prev0_is_positive
 + dist_diff_pair_c5_0_4_prev0 >= 0
dist_diff_abs_pair_c5_0_4_prev0_abs_ge_orig: dist_diff_abs_pair_c5_0_4_prev0
 - 240 dist_diff_abs_pair_c5_0_4_prev0_is_positive
 - dist_diff_pair_c5_0_4_prev0 >= -240
dist_diff_abs_pair_c5_0_4_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev0
 - 240 dist_diff_abs_pair_c5_0_4_prev0_is_positive
 + dist_diff_pair_c5_0_4_prev0 <= 0
dist_diff_abs_pair_c5_0_4_prev0_abs_le_orig: dist_diff_abs_pair_c5_0_4_prev0
 + 240 dist_diff_abs_pair_c5_0_4_prev0_is_positive
 - dist_diff_pair_c5_0_4_prev0 <= 240
dist_diff_abs_pair_c5_0_4_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_0_4_prev0_is_positive
 + dist_diff_pair_c5_0_4_prev0 <= -0.001
dist_diff_abs_pair_c5_0_4_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_0_4_prev0_is_positive
 + dist_diff_pair_c5_0_4_prev0 >= -240
dist_diff_abs_pair_c5_0_4_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev1
 + 240 dist_diff_abs_pair_c5_0_4_prev1_is_positive
 + dist_diff_pair_c5_0_4_prev1 >= 0
dist_diff_abs_pair_c5_0_4_prev1_abs_ge_orig: dist_diff_abs_pair_c5_0_4_prev1
 - 240 dist_diff_abs_pair_c5_0_4_prev1_is_positive
 - dist_diff_pair_c5_0_4_prev1 >= -240
dist_diff_abs_pair_c5_0_4_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev1
 - 240 dist_diff_abs_pair_c5_0_4_prev1_is_positive
 + dist_diff_pair_c5_0_4_prev1 <= 0
dist_diff_abs_pair_c5_0_4_prev1_abs_le_orig: dist_diff_abs_pair_c5_0_4_prev1
 + 240 dist_diff_abs_pair_c5_0_4_prev1_is_positive
 - dist_diff_pair_c5_0_4_prev1 <= 240
dist_diff_abs_pair_c5_0_4_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_0_4_prev1_is_positive
 + dist_diff_pair_c5_0_4_prev1 <= -0.001
dist_diff_abs_pair_c5_0_4_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_0_4_prev1_is_positive
 + dist_diff_pair_c5_0_4_prev1 >= -240
dist_diff_abs_pair_c5_0_4_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev2
 + 240 dist_diff_abs_pair_c5_0_4_prev2_is_positive
 + dist_diff_pair_c5_0_4_prev2 >= 0
dist_diff_abs_pair_c5_0_4_prev2_abs_ge_orig: dist_diff_abs_pair_c5_0_4_prev2
 - 240 dist_diff_abs_pair_c5_0_4_prev2_is_positive
 - dist_diff_pair_c5_0_4_prev2 >= -240
dist_diff_abs_pair_c5_0_4_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev2
 - 240 dist_diff_abs_pair_c5_0_4_prev2_is_positive
 + dist_diff_pair_c5_0_4_prev2 <= 0
dist_diff_abs_pair_c5_0_4_prev2_abs_le_orig: dist_diff_abs_pair_c5_0_4_prev2
 + 240 dist_diff_abs_pair_c5_0_4_prev2_is_positive
 - dist_diff_pair_c5_0_4_prev2 <= 240
dist_diff_abs_pair_c5_0_4_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_0_4_prev2_is_positive
 + dist_diff_pair_c5_0_4_prev2 <= -0.001
dist_diff_abs_pair_c5_0_4_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_0_4_prev2_is_positive
 + dist_diff_pair_c5_0_4_prev2 >= -240
dist_diff_abs_pair_c5_0_4_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev3
 + 240 dist_diff_abs_pair_c5_0_4_prev3_is_positive
 + dist_diff_pair_c5_0_4_prev3 >= 0
dist_diff_abs_pair_c5_0_4_prev3_abs_ge_orig: dist_diff_abs_pair_c5_0_4_prev3
 - 240 dist_diff_abs_pair_c5_0_4_prev3_is_positive
 - dist_diff_pair_c5_0_4_prev3 >= -240
dist_diff_abs_pair_c5_0_4_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev3
 - 240 dist_diff_abs_pair_c5_0_4_prev3_is_positive
 + dist_diff_pair_c5_0_4_prev3 <= 0
dist_diff_abs_pair_c5_0_4_prev3_abs_le_orig: dist_diff_abs_pair_c5_0_4_prev3
 + 240 dist_diff_abs_pair_c5_0_4_prev3_is_positive
 - dist_diff_pair_c5_0_4_prev3 <= 240
dist_diff_abs_pair_c5_0_4_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_0_4_prev3_is_positive
 + dist_diff_pair_c5_0_4_prev3 <= -0.001
dist_diff_abs_pair_c5_0_4_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_0_4_prev3_is_positive
 + dist_diff_pair_c5_0_4_prev3 >= -240
dist_diff_abs_pair_c5_0_4_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev4
 + 240 dist_diff_abs_pair_c5_0_4_prev4_is_positive
 + dist_diff_pair_c5_0_4_prev4 >= 0
dist_diff_abs_pair_c5_0_4_prev4_abs_ge_orig: dist_diff_abs_pair_c5_0_4_prev4
 - 240 dist_diff_abs_pair_c5_0_4_prev4_is_positive
 - dist_diff_pair_c5_0_4_prev4 >= -240
dist_diff_abs_pair_c5_0_4_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev4
 - 240 dist_diff_abs_pair_c5_0_4_prev4_is_positive
 + dist_diff_pair_c5_0_4_prev4 <= 0
dist_diff_abs_pair_c5_0_4_prev4_abs_le_orig: dist_diff_abs_pair_c5_0_4_prev4
 + 240 dist_diff_abs_pair_c5_0_4_prev4_is_positive
 - dist_diff_pair_c5_0_4_prev4 <= 240
dist_diff_abs_pair_c5_0_4_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_0_4_prev4_is_positive
 + dist_diff_pair_c5_0_4_prev4 <= -0.001
dist_diff_abs_pair_c5_0_4_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_0_4_prev4_is_positive
 + dist_diff_pair_c5_0_4_prev4 >= -240
dist_diff_abs_pair_c5_0_4_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev5
 + 240 dist_diff_abs_pair_c5_0_4_prev5_is_positive
 + dist_diff_pair_c5_0_4_prev5 >= 0
dist_diff_abs_pair_c5_0_4_prev5_abs_ge_orig: dist_diff_abs_pair_c5_0_4_prev5
 - 240 dist_diff_abs_pair_c5_0_4_prev5_is_positive
 - dist_diff_pair_c5_0_4_prev5 >= -240
dist_diff_abs_pair_c5_0_4_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_0_4_prev5
 - 240 dist_diff_abs_pair_c5_0_4_prev5_is_positive
 + dist_diff_pair_c5_0_4_prev5 <= 0
dist_diff_abs_pair_c5_0_4_prev5_abs_le_orig: dist_diff_abs_pair_c5_0_4_prev5
 + 240 dist_diff_abs_pair_c5_0_4_prev5_is_positive
 - dist_diff_pair_c5_0_4_prev5 <= 240
dist_diff_abs_pair_c5_0_4_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_0_4_prev5_is_positive
 + dist_diff_pair_c5_0_4_prev5 <= -0.001
dist_diff_abs_pair_c5_0_4_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_0_4_prev5_is_positive
 + dist_diff_pair_c5_0_4_prev5 >= -240
dist_diff_abs_pair_c5_1_2_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev0
 + 240 dist_diff_abs_pair_c5_1_2_prev0_is_positive
 + dist_diff_pair_c5_1_2_prev0 >= 0
dist_diff_abs_pair_c5_1_2_prev0_abs_ge_orig: dist_diff_abs_pair_c5_1_2_prev0
 - 240 dist_diff_abs_pair_c5_1_2_prev0_is_positive
 - dist_diff_pair_c5_1_2_prev0 >= -240
dist_diff_abs_pair_c5_1_2_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev0
 - 240 dist_diff_abs_pair_c5_1_2_prev0_is_positive
 + dist_diff_pair_c5_1_2_prev0 <= 0
dist_diff_abs_pair_c5_1_2_prev0_abs_le_orig: dist_diff_abs_pair_c5_1_2_prev0
 + 240 dist_diff_abs_pair_c5_1_2_prev0_is_positive
 - dist_diff_pair_c5_1_2_prev0 <= 240
dist_diff_abs_pair_c5_1_2_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_1_2_prev0_is_positive
 + dist_diff_pair_c5_1_2_prev0 <= -0.001
dist_diff_abs_pair_c5_1_2_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_1_2_prev0_is_positive
 + dist_diff_pair_c5_1_2_prev0 >= -240
dist_diff_abs_pair_c5_1_2_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev1
 + 240 dist_diff_abs_pair_c5_1_2_prev1_is_positive
 + dist_diff_pair_c5_1_2_prev1 >= 0
dist_diff_abs_pair_c5_1_2_prev1_abs_ge_orig: dist_diff_abs_pair_c5_1_2_prev1
 - 240 dist_diff_abs_pair_c5_1_2_prev1_is_positive
 - dist_diff_pair_c5_1_2_prev1 >= -240
dist_diff_abs_pair_c5_1_2_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev1
 - 240 dist_diff_abs_pair_c5_1_2_prev1_is_positive
 + dist_diff_pair_c5_1_2_prev1 <= 0
dist_diff_abs_pair_c5_1_2_prev1_abs_le_orig: dist_diff_abs_pair_c5_1_2_prev1
 + 240 dist_diff_abs_pair_c5_1_2_prev1_is_positive
 - dist_diff_pair_c5_1_2_prev1 <= 240
dist_diff_abs_pair_c5_1_2_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_1_2_prev1_is_positive
 + dist_diff_pair_c5_1_2_prev1 <= -0.001
dist_diff_abs_pair_c5_1_2_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_1_2_prev1_is_positive
 + dist_diff_pair_c5_1_2_prev1 >= -240
dist_diff_abs_pair_c5_1_2_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev2
 + 240 dist_diff_abs_pair_c5_1_2_prev2_is_positive
 + dist_diff_pair_c5_1_2_prev2 >= 0
dist_diff_abs_pair_c5_1_2_prev2_abs_ge_orig: dist_diff_abs_pair_c5_1_2_prev2
 - 240 dist_diff_abs_pair_c5_1_2_prev2_is_positive
 - dist_diff_pair_c5_1_2_prev2 >= -240
dist_diff_abs_pair_c5_1_2_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev2
 - 240 dist_diff_abs_pair_c5_1_2_prev2_is_positive
 + dist_diff_pair_c5_1_2_prev2 <= 0
dist_diff_abs_pair_c5_1_2_prev2_abs_le_orig: dist_diff_abs_pair_c5_1_2_prev2
 + 240 dist_diff_abs_pair_c5_1_2_prev2_is_positive
 - dist_diff_pair_c5_1_2_prev2 <= 240
dist_diff_abs_pair_c5_1_2_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_1_2_prev2_is_positive
 + dist_diff_pair_c5_1_2_prev2 <= -0.001
dist_diff_abs_pair_c5_1_2_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_1_2_prev2_is_positive
 + dist_diff_pair_c5_1_2_prev2 >= -240
dist_diff_abs_pair_c5_1_2_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev3
 + 240 dist_diff_abs_pair_c5_1_2_prev3_is_positive
 + dist_diff_pair_c5_1_2_prev3 >= 0
dist_diff_abs_pair_c5_1_2_prev3_abs_ge_orig: dist_diff_abs_pair_c5_1_2_prev3
 - 240 dist_diff_abs_pair_c5_1_2_prev3_is_positive
 - dist_diff_pair_c5_1_2_prev3 >= -240
dist_diff_abs_pair_c5_1_2_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev3
 - 240 dist_diff_abs_pair_c5_1_2_prev3_is_positive
 + dist_diff_pair_c5_1_2_prev3 <= 0
dist_diff_abs_pair_c5_1_2_prev3_abs_le_orig: dist_diff_abs_pair_c5_1_2_prev3
 + 240 dist_diff_abs_pair_c5_1_2_prev3_is_positive
 - dist_diff_pair_c5_1_2_prev3 <= 240
dist_diff_abs_pair_c5_1_2_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_1_2_prev3_is_positive
 + dist_diff_pair_c5_1_2_prev3 <= -0.001
dist_diff_abs_pair_c5_1_2_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_1_2_prev3_is_positive
 + dist_diff_pair_c5_1_2_prev3 >= -240
dist_diff_abs_pair_c5_1_2_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev4
 + 240 dist_diff_abs_pair_c5_1_2_prev4_is_positive
 + dist_diff_pair_c5_1_2_prev4 >= 0
dist_diff_abs_pair_c5_1_2_prev4_abs_ge_orig: dist_diff_abs_pair_c5_1_2_prev4
 - 240 dist_diff_abs_pair_c5_1_2_prev4_is_positive
 - dist_diff_pair_c5_1_2_prev4 >= -240
dist_diff_abs_pair_c5_1_2_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev4
 - 240 dist_diff_abs_pair_c5_1_2_prev4_is_positive
 + dist_diff_pair_c5_1_2_prev4 <= 0
dist_diff_abs_pair_c5_1_2_prev4_abs_le_orig: dist_diff_abs_pair_c5_1_2_prev4
 + 240 dist_diff_abs_pair_c5_1_2_prev4_is_positive
 - dist_diff_pair_c5_1_2_prev4 <= 240
dist_diff_abs_pair_c5_1_2_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_1_2_prev4_is_positive
 + dist_diff_pair_c5_1_2_prev4 <= -0.001
dist_diff_abs_pair_c5_1_2_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_1_2_prev4_is_positive
 + dist_diff_pair_c5_1_2_prev4 >= -240
dist_diff_abs_pair_c5_1_2_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev5
 + 240 dist_diff_abs_pair_c5_1_2_prev5_is_positive
 + dist_diff_pair_c5_1_2_prev5 >= 0
dist_diff_abs_pair_c5_1_2_prev5_abs_ge_orig: dist_diff_abs_pair_c5_1_2_prev5
 - 240 dist_diff_abs_pair_c5_1_2_prev5_is_positive
 - dist_diff_pair_c5_1_2_prev5 >= -240
dist_diff_abs_pair_c5_1_2_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_2_prev5
 - 240 dist_diff_abs_pair_c5_1_2_prev5_is_positive
 + dist_diff_pair_c5_1_2_prev5 <= 0
dist_diff_abs_pair_c5_1_2_prev5_abs_le_orig: dist_diff_abs_pair_c5_1_2_prev5
 + 240 dist_diff_abs_pair_c5_1_2_prev5_is_positive
 - dist_diff_pair_c5_1_2_prev5 <= 240
dist_diff_abs_pair_c5_1_2_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_1_2_prev5_is_positive
 + dist_diff_pair_c5_1_2_prev5 <= -0.001
dist_diff_abs_pair_c5_1_2_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_1_2_prev5_is_positive
 + dist_diff_pair_c5_1_2_prev5 >= -240
dist_diff_abs_pair_c5_1_3_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev0
 + 240 dist_diff_abs_pair_c5_1_3_prev0_is_positive
 + dist_diff_pair_c5_1_3_prev0 >= 0
dist_diff_abs_pair_c5_1_3_prev0_abs_ge_orig: dist_diff_abs_pair_c5_1_3_prev0
 - 240 dist_diff_abs_pair_c5_1_3_prev0_is_positive
 - dist_diff_pair_c5_1_3_prev0 >= -240
dist_diff_abs_pair_c5_1_3_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev0
 - 240 dist_diff_abs_pair_c5_1_3_prev0_is_positive
 + dist_diff_pair_c5_1_3_prev0 <= 0
dist_diff_abs_pair_c5_1_3_prev0_abs_le_orig: dist_diff_abs_pair_c5_1_3_prev0
 + 240 dist_diff_abs_pair_c5_1_3_prev0_is_positive
 - dist_diff_pair_c5_1_3_prev0 <= 240
dist_diff_abs_pair_c5_1_3_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_1_3_prev0_is_positive
 + dist_diff_pair_c5_1_3_prev0 <= -0.001
dist_diff_abs_pair_c5_1_3_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_1_3_prev0_is_positive
 + dist_diff_pair_c5_1_3_prev0 >= -240
dist_diff_abs_pair_c5_1_3_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev1
 + 240 dist_diff_abs_pair_c5_1_3_prev1_is_positive
 + dist_diff_pair_c5_1_3_prev1 >= 0
dist_diff_abs_pair_c5_1_3_prev1_abs_ge_orig: dist_diff_abs_pair_c5_1_3_prev1
 - 240 dist_diff_abs_pair_c5_1_3_prev1_is_positive
 - dist_diff_pair_c5_1_3_prev1 >= -240
dist_diff_abs_pair_c5_1_3_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev1
 - 240 dist_diff_abs_pair_c5_1_3_prev1_is_positive
 + dist_diff_pair_c5_1_3_prev1 <= 0
dist_diff_abs_pair_c5_1_3_prev1_abs_le_orig: dist_diff_abs_pair_c5_1_3_prev1
 + 240 dist_diff_abs_pair_c5_1_3_prev1_is_positive
 - dist_diff_pair_c5_1_3_prev1 <= 240
dist_diff_abs_pair_c5_1_3_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_1_3_prev1_is_positive
 + dist_diff_pair_c5_1_3_prev1 <= -0.001
dist_diff_abs_pair_c5_1_3_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_1_3_prev1_is_positive
 + dist_diff_pair_c5_1_3_prev1 >= -240
dist_diff_abs_pair_c5_1_3_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev2
 + 240 dist_diff_abs_pair_c5_1_3_prev2_is_positive
 + dist_diff_pair_c5_1_3_prev2 >= 0
dist_diff_abs_pair_c5_1_3_prev2_abs_ge_orig: dist_diff_abs_pair_c5_1_3_prev2
 - 240 dist_diff_abs_pair_c5_1_3_prev2_is_positive
 - dist_diff_pair_c5_1_3_prev2 >= -240
dist_diff_abs_pair_c5_1_3_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev2
 - 240 dist_diff_abs_pair_c5_1_3_prev2_is_positive
 + dist_diff_pair_c5_1_3_prev2 <= 0
dist_diff_abs_pair_c5_1_3_prev2_abs_le_orig: dist_diff_abs_pair_c5_1_3_prev2
 + 240 dist_diff_abs_pair_c5_1_3_prev2_is_positive
 - dist_diff_pair_c5_1_3_prev2 <= 240
dist_diff_abs_pair_c5_1_3_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_1_3_prev2_is_positive
 + dist_diff_pair_c5_1_3_prev2 <= -0.001
dist_diff_abs_pair_c5_1_3_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_1_3_prev2_is_positive
 + dist_diff_pair_c5_1_3_prev2 >= -240
dist_diff_abs_pair_c5_1_3_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev3
 + 240 dist_diff_abs_pair_c5_1_3_prev3_is_positive
 + dist_diff_pair_c5_1_3_prev3 >= 0
dist_diff_abs_pair_c5_1_3_prev3_abs_ge_orig: dist_diff_abs_pair_c5_1_3_prev3
 - 240 dist_diff_abs_pair_c5_1_3_prev3_is_positive
 - dist_diff_pair_c5_1_3_prev3 >= -240
dist_diff_abs_pair_c5_1_3_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev3
 - 240 dist_diff_abs_pair_c5_1_3_prev3_is_positive
 + dist_diff_pair_c5_1_3_prev3 <= 0
dist_diff_abs_pair_c5_1_3_prev3_abs_le_orig: dist_diff_abs_pair_c5_1_3_prev3
 + 240 dist_diff_abs_pair_c5_1_3_prev3_is_positive
 - dist_diff_pair_c5_1_3_prev3 <= 240
dist_diff_abs_pair_c5_1_3_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_1_3_prev3_is_positive
 + dist_diff_pair_c5_1_3_prev3 <= -0.001
dist_diff_abs_pair_c5_1_3_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_1_3_prev3_is_positive
 + dist_diff_pair_c5_1_3_prev3 >= -240
dist_diff_abs_pair_c5_1_3_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev4
 + 240 dist_diff_abs_pair_c5_1_3_prev4_is_positive
 + dist_diff_pair_c5_1_3_prev4 >= 0
dist_diff_abs_pair_c5_1_3_prev4_abs_ge_orig: dist_diff_abs_pair_c5_1_3_prev4
 - 240 dist_diff_abs_pair_c5_1_3_prev4_is_positive
 - dist_diff_pair_c5_1_3_prev4 >= -240
dist_diff_abs_pair_c5_1_3_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev4
 - 240 dist_diff_abs_pair_c5_1_3_prev4_is_positive
 + dist_diff_pair_c5_1_3_prev4 <= 0
dist_diff_abs_pair_c5_1_3_prev4_abs_le_orig: dist_diff_abs_pair_c5_1_3_prev4
 + 240 dist_diff_abs_pair_c5_1_3_prev4_is_positive
 - dist_diff_pair_c5_1_3_prev4 <= 240
dist_diff_abs_pair_c5_1_3_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_1_3_prev4_is_positive
 + dist_diff_pair_c5_1_3_prev4 <= -0.001
dist_diff_abs_pair_c5_1_3_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_1_3_prev4_is_positive
 + dist_diff_pair_c5_1_3_prev4 >= -240
dist_diff_abs_pair_c5_1_3_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev5
 + 240 dist_diff_abs_pair_c5_1_3_prev5_is_positive
 + dist_diff_pair_c5_1_3_prev5 >= 0
dist_diff_abs_pair_c5_1_3_prev5_abs_ge_orig: dist_diff_abs_pair_c5_1_3_prev5
 - 240 dist_diff_abs_pair_c5_1_3_prev5_is_positive
 - dist_diff_pair_c5_1_3_prev5 >= -240
dist_diff_abs_pair_c5_1_3_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_3_prev5
 - 240 dist_diff_abs_pair_c5_1_3_prev5_is_positive
 + dist_diff_pair_c5_1_3_prev5 <= 0
dist_diff_abs_pair_c5_1_3_prev5_abs_le_orig: dist_diff_abs_pair_c5_1_3_prev5
 + 240 dist_diff_abs_pair_c5_1_3_prev5_is_positive
 - dist_diff_pair_c5_1_3_prev5 <= 240
dist_diff_abs_pair_c5_1_3_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_1_3_prev5_is_positive
 + dist_diff_pair_c5_1_3_prev5 <= -0.001
dist_diff_abs_pair_c5_1_3_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_1_3_prev5_is_positive
 + dist_diff_pair_c5_1_3_prev5 >= -240
dist_diff_abs_pair_c5_1_4_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev0
 + 240 dist_diff_abs_pair_c5_1_4_prev0_is_positive
 + dist_diff_pair_c5_1_4_prev0 >= 0
dist_diff_abs_pair_c5_1_4_prev0_abs_ge_orig: dist_diff_abs_pair_c5_1_4_prev0
 - 240 dist_diff_abs_pair_c5_1_4_prev0_is_positive
 - dist_diff_pair_c5_1_4_prev0 >= -240
dist_diff_abs_pair_c5_1_4_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev0
 - 240 dist_diff_abs_pair_c5_1_4_prev0_is_positive
 + dist_diff_pair_c5_1_4_prev0 <= 0
dist_diff_abs_pair_c5_1_4_prev0_abs_le_orig: dist_diff_abs_pair_c5_1_4_prev0
 + 240 dist_diff_abs_pair_c5_1_4_prev0_is_positive
 - dist_diff_pair_c5_1_4_prev0 <= 240
dist_diff_abs_pair_c5_1_4_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_1_4_prev0_is_positive
 + dist_diff_pair_c5_1_4_prev0 <= -0.001
dist_diff_abs_pair_c5_1_4_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_1_4_prev0_is_positive
 + dist_diff_pair_c5_1_4_prev0 >= -240
dist_diff_abs_pair_c5_1_4_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev1
 + 240 dist_diff_abs_pair_c5_1_4_prev1_is_positive
 + dist_diff_pair_c5_1_4_prev1 >= 0
dist_diff_abs_pair_c5_1_4_prev1_abs_ge_orig: dist_diff_abs_pair_c5_1_4_prev1
 - 240 dist_diff_abs_pair_c5_1_4_prev1_is_positive
 - dist_diff_pair_c5_1_4_prev1 >= -240
dist_diff_abs_pair_c5_1_4_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev1
 - 240 dist_diff_abs_pair_c5_1_4_prev1_is_positive
 + dist_diff_pair_c5_1_4_prev1 <= 0
dist_diff_abs_pair_c5_1_4_prev1_abs_le_orig: dist_diff_abs_pair_c5_1_4_prev1
 + 240 dist_diff_abs_pair_c5_1_4_prev1_is_positive
 - dist_diff_pair_c5_1_4_prev1 <= 240
dist_diff_abs_pair_c5_1_4_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_1_4_prev1_is_positive
 + dist_diff_pair_c5_1_4_prev1 <= -0.001
dist_diff_abs_pair_c5_1_4_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_1_4_prev1_is_positive
 + dist_diff_pair_c5_1_4_prev1 >= -240
dist_diff_abs_pair_c5_1_4_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev2
 + 240 dist_diff_abs_pair_c5_1_4_prev2_is_positive
 + dist_diff_pair_c5_1_4_prev2 >= 0
dist_diff_abs_pair_c5_1_4_prev2_abs_ge_orig: dist_diff_abs_pair_c5_1_4_prev2
 - 240 dist_diff_abs_pair_c5_1_4_prev2_is_positive
 - dist_diff_pair_c5_1_4_prev2 >= -240
dist_diff_abs_pair_c5_1_4_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev2
 - 240 dist_diff_abs_pair_c5_1_4_prev2_is_positive
 + dist_diff_pair_c5_1_4_prev2 <= 0
dist_diff_abs_pair_c5_1_4_prev2_abs_le_orig: dist_diff_abs_pair_c5_1_4_prev2
 + 240 dist_diff_abs_pair_c5_1_4_prev2_is_positive
 - dist_diff_pair_c5_1_4_prev2 <= 240
dist_diff_abs_pair_c5_1_4_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_1_4_prev2_is_positive
 + dist_diff_pair_c5_1_4_prev2 <= -0.001
dist_diff_abs_pair_c5_1_4_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_1_4_prev2_is_positive
 + dist_diff_pair_c5_1_4_prev2 >= -240
dist_diff_abs_pair_c5_1_4_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev3
 + 240 dist_diff_abs_pair_c5_1_4_prev3_is_positive
 + dist_diff_pair_c5_1_4_prev3 >= 0
dist_diff_abs_pair_c5_1_4_prev3_abs_ge_orig: dist_diff_abs_pair_c5_1_4_prev3
 - 240 dist_diff_abs_pair_c5_1_4_prev3_is_positive
 - dist_diff_pair_c5_1_4_prev3 >= -240
dist_diff_abs_pair_c5_1_4_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev3
 - 240 dist_diff_abs_pair_c5_1_4_prev3_is_positive
 + dist_diff_pair_c5_1_4_prev3 <= 0
dist_diff_abs_pair_c5_1_4_prev3_abs_le_orig: dist_diff_abs_pair_c5_1_4_prev3
 + 240 dist_diff_abs_pair_c5_1_4_prev3_is_positive
 - dist_diff_pair_c5_1_4_prev3 <= 240
dist_diff_abs_pair_c5_1_4_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_1_4_prev3_is_positive
 + dist_diff_pair_c5_1_4_prev3 <= -0.001
dist_diff_abs_pair_c5_1_4_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_1_4_prev3_is_positive
 + dist_diff_pair_c5_1_4_prev3 >= -240
dist_diff_abs_pair_c5_1_4_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev4
 + 240 dist_diff_abs_pair_c5_1_4_prev4_is_positive
 + dist_diff_pair_c5_1_4_prev4 >= 0
dist_diff_abs_pair_c5_1_4_prev4_abs_ge_orig: dist_diff_abs_pair_c5_1_4_prev4
 - 240 dist_diff_abs_pair_c5_1_4_prev4_is_positive
 - dist_diff_pair_c5_1_4_prev4 >= -240
dist_diff_abs_pair_c5_1_4_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev4
 - 240 dist_diff_abs_pair_c5_1_4_prev4_is_positive
 + dist_diff_pair_c5_1_4_prev4 <= 0
dist_diff_abs_pair_c5_1_4_prev4_abs_le_orig: dist_diff_abs_pair_c5_1_4_prev4
 + 240 dist_diff_abs_pair_c5_1_4_prev4_is_positive
 - dist_diff_pair_c5_1_4_prev4 <= 240
dist_diff_abs_pair_c5_1_4_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_1_4_prev4_is_positive
 + dist_diff_pair_c5_1_4_prev4 <= -0.001
dist_diff_abs_pair_c5_1_4_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_1_4_prev4_is_positive
 + dist_diff_pair_c5_1_4_prev4 >= -240
dist_diff_abs_pair_c5_1_4_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev5
 + 240 dist_diff_abs_pair_c5_1_4_prev5_is_positive
 + dist_diff_pair_c5_1_4_prev5 >= 0
dist_diff_abs_pair_c5_1_4_prev5_abs_ge_orig: dist_diff_abs_pair_c5_1_4_prev5
 - 240 dist_diff_abs_pair_c5_1_4_prev5_is_positive
 - dist_diff_pair_c5_1_4_prev5 >= -240
dist_diff_abs_pair_c5_1_4_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_1_4_prev5
 - 240 dist_diff_abs_pair_c5_1_4_prev5_is_positive
 + dist_diff_pair_c5_1_4_prev5 <= 0
dist_diff_abs_pair_c5_1_4_prev5_abs_le_orig: dist_diff_abs_pair_c5_1_4_prev5
 + 240 dist_diff_abs_pair_c5_1_4_prev5_is_positive
 - dist_diff_pair_c5_1_4_prev5 <= 240
dist_diff_abs_pair_c5_1_4_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_1_4_prev5_is_positive
 + dist_diff_pair_c5_1_4_prev5 <= -0.001
dist_diff_abs_pair_c5_1_4_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_1_4_prev5_is_positive
 + dist_diff_pair_c5_1_4_prev5 >= -240
dist_diff_abs_pair_c5_2_3_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev0
 + 240 dist_diff_abs_pair_c5_2_3_prev0_is_positive
 + dist_diff_pair_c5_2_3_prev0 >= 0
dist_diff_abs_pair_c5_2_3_prev0_abs_ge_orig: dist_diff_abs_pair_c5_2_3_prev0
 - 240 dist_diff_abs_pair_c5_2_3_prev0_is_positive
 - dist_diff_pair_c5_2_3_prev0 >= -240
dist_diff_abs_pair_c5_2_3_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev0
 - 240 dist_diff_abs_pair_c5_2_3_prev0_is_positive
 + dist_diff_pair_c5_2_3_prev0 <= 0
dist_diff_abs_pair_c5_2_3_prev0_abs_le_orig: dist_diff_abs_pair_c5_2_3_prev0
 + 240 dist_diff_abs_pair_c5_2_3_prev0_is_positive
 - dist_diff_pair_c5_2_3_prev0 <= 240
dist_diff_abs_pair_c5_2_3_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_2_3_prev0_is_positive
 + dist_diff_pair_c5_2_3_prev0 <= -0.001
dist_diff_abs_pair_c5_2_3_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_2_3_prev0_is_positive
 + dist_diff_pair_c5_2_3_prev0 >= -240
dist_diff_abs_pair_c5_2_3_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev1
 + 240 dist_diff_abs_pair_c5_2_3_prev1_is_positive
 + dist_diff_pair_c5_2_3_prev1 >= 0
dist_diff_abs_pair_c5_2_3_prev1_abs_ge_orig: dist_diff_abs_pair_c5_2_3_prev1
 - 240 dist_diff_abs_pair_c5_2_3_prev1_is_positive
 - dist_diff_pair_c5_2_3_prev1 >= -240
dist_diff_abs_pair_c5_2_3_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev1
 - 240 dist_diff_abs_pair_c5_2_3_prev1_is_positive
 + dist_diff_pair_c5_2_3_prev1 <= 0
dist_diff_abs_pair_c5_2_3_prev1_abs_le_orig: dist_diff_abs_pair_c5_2_3_prev1
 + 240 dist_diff_abs_pair_c5_2_3_prev1_is_positive
 - dist_diff_pair_c5_2_3_prev1 <= 240
dist_diff_abs_pair_c5_2_3_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_2_3_prev1_is_positive
 + dist_diff_pair_c5_2_3_prev1 <= -0.001
dist_diff_abs_pair_c5_2_3_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_2_3_prev1_is_positive
 + dist_diff_pair_c5_2_3_prev1 >= -240
dist_diff_abs_pair_c5_2_3_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev2
 + 240 dist_diff_abs_pair_c5_2_3_prev2_is_positive
 + dist_diff_pair_c5_2_3_prev2 >= 0
dist_diff_abs_pair_c5_2_3_prev2_abs_ge_orig: dist_diff_abs_pair_c5_2_3_prev2
 - 240 dist_diff_abs_pair_c5_2_3_prev2_is_positive
 - dist_diff_pair_c5_2_3_prev2 >= -240
dist_diff_abs_pair_c5_2_3_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev2
 - 240 dist_diff_abs_pair_c5_2_3_prev2_is_positive
 + dist_diff_pair_c5_2_3_prev2 <= 0
dist_diff_abs_pair_c5_2_3_prev2_abs_le_orig: dist_diff_abs_pair_c5_2_3_prev2
 + 240 dist_diff_abs_pair_c5_2_3_prev2_is_positive
 - dist_diff_pair_c5_2_3_prev2 <= 240
dist_diff_abs_pair_c5_2_3_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_2_3_prev2_is_positive
 + dist_diff_pair_c5_2_3_prev2 <= -0.001
dist_diff_abs_pair_c5_2_3_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_2_3_prev2_is_positive
 + dist_diff_pair_c5_2_3_prev2 >= -240
dist_diff_abs_pair_c5_2_3_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev3
 + 240 dist_diff_abs_pair_c5_2_3_prev3_is_positive
 + dist_diff_pair_c5_2_3_prev3 >= 0
dist_diff_abs_pair_c5_2_3_prev3_abs_ge_orig: dist_diff_abs_pair_c5_2_3_prev3
 - 240 dist_diff_abs_pair_c5_2_3_prev3_is_positive
 - dist_diff_pair_c5_2_3_prev3 >= -240
dist_diff_abs_pair_c5_2_3_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev3
 - 240 dist_diff_abs_pair_c5_2_3_prev3_is_positive
 + dist_diff_pair_c5_2_3_prev3 <= 0
dist_diff_abs_pair_c5_2_3_prev3_abs_le_orig: dist_diff_abs_pair_c5_2_3_prev3
 + 240 dist_diff_abs_pair_c5_2_3_prev3_is_positive
 - dist_diff_pair_c5_2_3_prev3 <= 240
dist_diff_abs_pair_c5_2_3_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_2_3_prev3_is_positive
 + dist_diff_pair_c5_2_3_prev3 <= -0.001
dist_diff_abs_pair_c5_2_3_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_2_3_prev3_is_positive
 + dist_diff_pair_c5_2_3_prev3 >= -240
dist_diff_abs_pair_c5_2_3_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev4
 + 240 dist_diff_abs_pair_c5_2_3_prev4_is_positive
 + dist_diff_pair_c5_2_3_prev4 >= 0
dist_diff_abs_pair_c5_2_3_prev4_abs_ge_orig: dist_diff_abs_pair_c5_2_3_prev4
 - 240 dist_diff_abs_pair_c5_2_3_prev4_is_positive
 - dist_diff_pair_c5_2_3_prev4 >= -240
dist_diff_abs_pair_c5_2_3_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev4
 - 240 dist_diff_abs_pair_c5_2_3_prev4_is_positive
 + dist_diff_pair_c5_2_3_prev4 <= 0
dist_diff_abs_pair_c5_2_3_prev4_abs_le_orig: dist_diff_abs_pair_c5_2_3_prev4
 + 240 dist_diff_abs_pair_c5_2_3_prev4_is_positive
 - dist_diff_pair_c5_2_3_prev4 <= 240
dist_diff_abs_pair_c5_2_3_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_2_3_prev4_is_positive
 + dist_diff_pair_c5_2_3_prev4 <= -0.001
dist_diff_abs_pair_c5_2_3_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_2_3_prev4_is_positive
 + dist_diff_pair_c5_2_3_prev4 >= -240
dist_diff_abs_pair_c5_2_3_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev5
 + 240 dist_diff_abs_pair_c5_2_3_prev5_is_positive
 + dist_diff_pair_c5_2_3_prev5 >= 0
dist_diff_abs_pair_c5_2_3_prev5_abs_ge_orig: dist_diff_abs_pair_c5_2_3_prev5
 - 240 dist_diff_abs_pair_c5_2_3_prev5_is_positive
 - dist_diff_pair_c5_2_3_prev5 >= -240
dist_diff_abs_pair_c5_2_3_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_3_prev5
 - 240 dist_diff_abs_pair_c5_2_3_prev5_is_positive
 + dist_diff_pair_c5_2_3_prev5 <= 0
dist_diff_abs_pair_c5_2_3_prev5_abs_le_orig: dist_diff_abs_pair_c5_2_3_prev5
 + 240 dist_diff_abs_pair_c5_2_3_prev5_is_positive
 - dist_diff_pair_c5_2_3_prev5 <= 240
dist_diff_abs_pair_c5_2_3_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_2_3_prev5_is_positive
 + dist_diff_pair_c5_2_3_prev5 <= -0.001
dist_diff_abs_pair_c5_2_3_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_2_3_prev5_is_positive
 + dist_diff_pair_c5_2_3_prev5 >= -240
dist_diff_abs_pair_c5_2_4_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev0
 + 240 dist_diff_abs_pair_c5_2_4_prev0_is_positive
 + dist_diff_pair_c5_2_4_prev0 >= 0
dist_diff_abs_pair_c5_2_4_prev0_abs_ge_orig: dist_diff_abs_pair_c5_2_4_prev0
 - 240 dist_diff_abs_pair_c5_2_4_prev0_is_positive
 - dist_diff_pair_c5_2_4_prev0 >= -240
dist_diff_abs_pair_c5_2_4_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev0
 - 240 dist_diff_abs_pair_c5_2_4_prev0_is_positive
 + dist_diff_pair_c5_2_4_prev0 <= 0
dist_diff_abs_pair_c5_2_4_prev0_abs_le_orig: dist_diff_abs_pair_c5_2_4_prev0
 + 240 dist_diff_abs_pair_c5_2_4_prev0_is_positive
 - dist_diff_pair_c5_2_4_prev0 <= 240
dist_diff_abs_pair_c5_2_4_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_2_4_prev0_is_positive
 + dist_diff_pair_c5_2_4_prev0 <= -0.001
dist_diff_abs_pair_c5_2_4_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_2_4_prev0_is_positive
 + dist_diff_pair_c5_2_4_prev0 >= -240
dist_diff_abs_pair_c5_2_4_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev1
 + 240 dist_diff_abs_pair_c5_2_4_prev1_is_positive
 + dist_diff_pair_c5_2_4_prev1 >= 0
dist_diff_abs_pair_c5_2_4_prev1_abs_ge_orig: dist_diff_abs_pair_c5_2_4_prev1
 - 240 dist_diff_abs_pair_c5_2_4_prev1_is_positive
 - dist_diff_pair_c5_2_4_prev1 >= -240
dist_diff_abs_pair_c5_2_4_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev1
 - 240 dist_diff_abs_pair_c5_2_4_prev1_is_positive
 + dist_diff_pair_c5_2_4_prev1 <= 0
dist_diff_abs_pair_c5_2_4_prev1_abs_le_orig: dist_diff_abs_pair_c5_2_4_prev1
 + 240 dist_diff_abs_pair_c5_2_4_prev1_is_positive
 - dist_diff_pair_c5_2_4_prev1 <= 240
dist_diff_abs_pair_c5_2_4_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_2_4_prev1_is_positive
 + dist_diff_pair_c5_2_4_prev1 <= -0.001
dist_diff_abs_pair_c5_2_4_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_2_4_prev1_is_positive
 + dist_diff_pair_c5_2_4_prev1 >= -240
dist_diff_abs_pair_c5_2_4_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev2
 + 240 dist_diff_abs_pair_c5_2_4_prev2_is_positive
 + dist_diff_pair_c5_2_4_prev2 >= 0
dist_diff_abs_pair_c5_2_4_prev2_abs_ge_orig: dist_diff_abs_pair_c5_2_4_prev2
 - 240 dist_diff_abs_pair_c5_2_4_prev2_is_positive
 - dist_diff_pair_c5_2_4_prev2 >= -240
dist_diff_abs_pair_c5_2_4_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev2
 - 240 dist_diff_abs_pair_c5_2_4_prev2_is_positive
 + dist_diff_pair_c5_2_4_prev2 <= 0
dist_diff_abs_pair_c5_2_4_prev2_abs_le_orig: dist_diff_abs_pair_c5_2_4_prev2
 + 240 dist_diff_abs_pair_c5_2_4_prev2_is_positive
 - dist_diff_pair_c5_2_4_prev2 <= 240
dist_diff_abs_pair_c5_2_4_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_2_4_prev2_is_positive
 + dist_diff_pair_c5_2_4_prev2 <= -0.001
dist_diff_abs_pair_c5_2_4_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_2_4_prev2_is_positive
 + dist_diff_pair_c5_2_4_prev2 >= -240
dist_diff_abs_pair_c5_2_4_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev3
 + 240 dist_diff_abs_pair_c5_2_4_prev3_is_positive
 + dist_diff_pair_c5_2_4_prev3 >= 0
dist_diff_abs_pair_c5_2_4_prev3_abs_ge_orig: dist_diff_abs_pair_c5_2_4_prev3
 - 240 dist_diff_abs_pair_c5_2_4_prev3_is_positive
 - dist_diff_pair_c5_2_4_prev3 >= -240
dist_diff_abs_pair_c5_2_4_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev3
 - 240 dist_diff_abs_pair_c5_2_4_prev3_is_positive
 + dist_diff_pair_c5_2_4_prev3 <= 0
dist_diff_abs_pair_c5_2_4_prev3_abs_le_orig: dist_diff_abs_pair_c5_2_4_prev3
 + 240 dist_diff_abs_pair_c5_2_4_prev3_is_positive
 - dist_diff_pair_c5_2_4_prev3 <= 240
dist_diff_abs_pair_c5_2_4_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_2_4_prev3_is_positive
 + dist_diff_pair_c5_2_4_prev3 <= -0.001
dist_diff_abs_pair_c5_2_4_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_2_4_prev3_is_positive
 + dist_diff_pair_c5_2_4_prev3 >= -240
dist_diff_abs_pair_c5_2_4_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev4
 + 240 dist_diff_abs_pair_c5_2_4_prev4_is_positive
 + dist_diff_pair_c5_2_4_prev4 >= 0
dist_diff_abs_pair_c5_2_4_prev4_abs_ge_orig: dist_diff_abs_pair_c5_2_4_prev4
 - 240 dist_diff_abs_pair_c5_2_4_prev4_is_positive
 - dist_diff_pair_c5_2_4_prev4 >= -240
dist_diff_abs_pair_c5_2_4_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev4
 - 240 dist_diff_abs_pair_c5_2_4_prev4_is_positive
 + dist_diff_pair_c5_2_4_prev4 <= 0
dist_diff_abs_pair_c5_2_4_prev4_abs_le_orig: dist_diff_abs_pair_c5_2_4_prev4
 + 240 dist_diff_abs_pair_c5_2_4_prev4_is_positive
 - dist_diff_pair_c5_2_4_prev4 <= 240
dist_diff_abs_pair_c5_2_4_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_2_4_prev4_is_positive
 + dist_diff_pair_c5_2_4_prev4 <= -0.001
dist_diff_abs_pair_c5_2_4_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_2_4_prev4_is_positive
 + dist_diff_pair_c5_2_4_prev4 >= -240
dist_diff_abs_pair_c5_2_4_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev5
 + 240 dist_diff_abs_pair_c5_2_4_prev5_is_positive
 + dist_diff_pair_c5_2_4_prev5 >= 0
dist_diff_abs_pair_c5_2_4_prev5_abs_ge_orig: dist_diff_abs_pair_c5_2_4_prev5
 - 240 dist_diff_abs_pair_c5_2_4_prev5_is_positive
 - dist_diff_pair_c5_2_4_prev5 >= -240
dist_diff_abs_pair_c5_2_4_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_2_4_prev5
 - 240 dist_diff_abs_pair_c5_2_4_prev5_is_positive
 + dist_diff_pair_c5_2_4_prev5 <= 0
dist_diff_abs_pair_c5_2_4_prev5_abs_le_orig: dist_diff_abs_pair_c5_2_4_prev5
 + 240 dist_diff_abs_pair_c5_2_4_prev5_is_positive
 - dist_diff_pair_c5_2_4_prev5 <= 240
dist_diff_abs_pair_c5_2_4_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_2_4_prev5_is_positive
 + dist_diff_pair_c5_2_4_prev5 <= -0.001
dist_diff_abs_pair_c5_2_4_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_2_4_prev5_is_positive
 + dist_diff_pair_c5_2_4_prev5 >= -240
dist_diff_abs_pair_c5_3_4_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev0
 + 240 dist_diff_abs_pair_c5_3_4_prev0_is_positive
 + dist_diff_pair_c5_3_4_prev0 >= 0
dist_diff_abs_pair_c5_3_4_prev0_abs_ge_orig: dist_diff_abs_pair_c5_3_4_prev0
 - 240 dist_diff_abs_pair_c5_3_4_prev0_is_positive
 - dist_diff_pair_c5_3_4_prev0 >= -240
dist_diff_abs_pair_c5_3_4_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev0
 - 240 dist_diff_abs_pair_c5_3_4_prev0_is_positive
 + dist_diff_pair_c5_3_4_prev0 <= 0
dist_diff_abs_pair_c5_3_4_prev0_abs_le_orig: dist_diff_abs_pair_c5_3_4_prev0
 + 240 dist_diff_abs_pair_c5_3_4_prev0_is_positive
 - dist_diff_pair_c5_3_4_prev0 <= 240
dist_diff_abs_pair_c5_3_4_prev0_force_negative:
 - 240 dist_diff_abs_pair_c5_3_4_prev0_is_positive
 + dist_diff_pair_c5_3_4_prev0 <= -0.001
dist_diff_abs_pair_c5_3_4_prev0_force_positive:
 - 240 dist_diff_abs_pair_c5_3_4_prev0_is_positive
 + dist_diff_pair_c5_3_4_prev0 >= -240
dist_diff_abs_pair_c5_3_4_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev1
 + 240 dist_diff_abs_pair_c5_3_4_prev1_is_positive
 + dist_diff_pair_c5_3_4_prev1 >= 0
dist_diff_abs_pair_c5_3_4_prev1_abs_ge_orig: dist_diff_abs_pair_c5_3_4_prev1
 - 240 dist_diff_abs_pair_c5_3_4_prev1_is_positive
 - dist_diff_pair_c5_3_4_prev1 >= -240
dist_diff_abs_pair_c5_3_4_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev1
 - 240 dist_diff_abs_pair_c5_3_4_prev1_is_positive
 + dist_diff_pair_c5_3_4_prev1 <= 0
dist_diff_abs_pair_c5_3_4_prev1_abs_le_orig: dist_diff_abs_pair_c5_3_4_prev1
 + 240 dist_diff_abs_pair_c5_3_4_prev1_is_positive
 - dist_diff_pair_c5_3_4_prev1 <= 240
dist_diff_abs_pair_c5_3_4_prev1_force_negative:
 - 240 dist_diff_abs_pair_c5_3_4_prev1_is_positive
 + dist_diff_pair_c5_3_4_prev1 <= -0.001
dist_diff_abs_pair_c5_3_4_prev1_force_positive:
 - 240 dist_diff_abs_pair_c5_3_4_prev1_is_positive
 + dist_diff_pair_c5_3_4_prev1 >= -240
dist_diff_abs_pair_c5_3_4_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev2
 + 240 dist_diff_abs_pair_c5_3_4_prev2_is_positive
 + dist_diff_pair_c5_3_4_prev2 >= 0
dist_diff_abs_pair_c5_3_4_prev2_abs_ge_orig: dist_diff_abs_pair_c5_3_4_prev2
 - 240 dist_diff_abs_pair_c5_3_4_prev2_is_positive
 - dist_diff_pair_c5_3_4_prev2 >= -240
dist_diff_abs_pair_c5_3_4_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev2
 - 240 dist_diff_abs_pair_c5_3_4_prev2_is_positive
 + dist_diff_pair_c5_3_4_prev2 <= 0
dist_diff_abs_pair_c5_3_4_prev2_abs_le_orig: dist_diff_abs_pair_c5_3_4_prev2
 + 240 dist_diff_abs_pair_c5_3_4_prev2_is_positive
 - dist_diff_pair_c5_3_4_prev2 <= 240
dist_diff_abs_pair_c5_3_4_prev2_force_negative:
 - 240 dist_diff_abs_pair_c5_3_4_prev2_is_positive
 + dist_diff_pair_c5_3_4_prev2 <= -0.001
dist_diff_abs_pair_c5_3_4_prev2_force_positive:
 - 240 dist_diff_abs_pair_c5_3_4_prev2_is_positive
 + dist_diff_pair_c5_3_4_prev2 >= -240
dist_diff_abs_pair_c5_3_4_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev3
 + 240 dist_diff_abs_pair_c5_3_4_prev3_is_positive
 + dist_diff_pair_c5_3_4_prev3 >= 0
dist_diff_abs_pair_c5_3_4_prev3_abs_ge_orig: dist_diff_abs_pair_c5_3_4_prev3
 - 240 dist_diff_abs_pair_c5_3_4_prev3_is_positive
 - dist_diff_pair_c5_3_4_prev3 >= -240
dist_diff_abs_pair_c5_3_4_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev3
 - 240 dist_diff_abs_pair_c5_3_4_prev3_is_positive
 + dist_diff_pair_c5_3_4_prev3 <= 0
dist_diff_abs_pair_c5_3_4_prev3_abs_le_orig: dist_diff_abs_pair_c5_3_4_prev3
 + 240 dist_diff_abs_pair_c5_3_4_prev3_is_positive
 - dist_diff_pair_c5_3_4_prev3 <= 240
dist_diff_abs_pair_c5_3_4_prev3_force_negative:
 - 240 dist_diff_abs_pair_c5_3_4_prev3_is_positive
 + dist_diff_pair_c5_3_4_prev3 <= -0.001
dist_diff_abs_pair_c5_3_4_prev3_force_positive:
 - 240 dist_diff_abs_pair_c5_3_4_prev3_is_positive
 + dist_diff_pair_c5_3_4_prev3 >= -240
dist_diff_abs_pair_c5_3_4_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev4
 + 240 dist_diff_abs_pair_c5_3_4_prev4_is_positive
 + dist_diff_pair_c5_3_4_prev4 >= 0
dist_diff_abs_pair_c5_3_4_prev4_abs_ge_orig: dist_diff_abs_pair_c5_3_4_prev4
 - 240 dist_diff_abs_pair_c5_3_4_prev4_is_positive
 - dist_diff_pair_c5_3_4_prev4 >= -240
dist_diff_abs_pair_c5_3_4_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev4
 - 240 dist_diff_abs_pair_c5_3_4_prev4_is_positive
 + dist_diff_pair_c5_3_4_prev4 <= 0
dist_diff_abs_pair_c5_3_4_prev4_abs_le_orig: dist_diff_abs_pair_c5_3_4_prev4
 + 240 dist_diff_abs_pair_c5_3_4_prev4_is_positive
 - dist_diff_pair_c5_3_4_prev4 <= 240
dist_diff_abs_pair_c5_3_4_prev4_force_negative:
 - 240 dist_diff_abs_pair_c5_3_4_prev4_is_positive
 + dist_diff_pair_c5_3_4_prev4 <= -0.001
dist_diff_abs_pair_c5_3_4_prev4_force_positive:
 - 240 dist_diff_abs_pair_c5_3_4_prev4_is_positive
 + dist_diff_pair_c5_3_4_prev4 >= -240
dist_diff_abs_pair_c5_3_4_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev5
 + 240 dist_diff_abs_pair_c5_3_4_prev5_is_positive
 + dist_diff_pair_c5_3_4_prev5 >= 0
dist_diff_abs_pair_c5_3_4_prev5_abs_ge_orig: dist_diff_abs_pair_c5_3_4_prev5
 - 240 dist_diff_abs_pair_c5_3_4_prev5_is_positive
 - dist_diff_pair_c5_3_4_prev5 >= -240
dist_diff_abs_pair_c5_3_4_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c5_3_4_prev5
 - 240 dist_diff_abs_pair_c5_3_4_prev5_is_positive
 + dist_diff_pair_c5_3_4_prev5 <= 0
dist_diff_abs_pair_c5_3_4_prev5_abs_le_orig: dist_diff_abs_pair_c5_3_4_prev5
 + 240 dist_diff_abs_pair_c5_3_4_prev5_is_positive
 - dist_diff_pair_c5_3_4_prev5 <= 240
dist_diff_abs_pair_c5_3_4_prev5_force_negative:
 - 240 dist_diff_abs_pair_c5_3_4_prev5_is_positive
 + dist_diff_pair_c5_3_4_prev5 <= -0.001
dist_diff_abs_pair_c5_3_4_prev5_force_positive:
 - 240 dist_diff_abs_pair_c5_3_4_prev5_is_positive
 + dist_diff_pair_c5_3_4_prev5 >= -240
dist_diff_pair_def_c5_0_1_prev0: - dist_curr_pair_c5_0_1
 + dist_diff_pair_c5_0_1_prev0 = -38
dist_diff_pair_def_c5_0_1_prev1: - dist_curr_pair_c5_0_1
 + dist_diff_pair_c5_0_1_prev1 = -46
dist_diff_pair_def_c5_0_1_prev2: - dist_curr_pair_c5_0_1
 + dist_diff_pair_c5_0_1_prev2 = -38
dist_diff_pair_def_c5_0_1_prev3: - dist_curr_pair_c5_0_1
 + dist_diff_pair_c5_0_1_prev3 = -46
dist_diff_pair_def_c5_0_1_prev4: - dist_curr_pair_c5_0_1
 + dist_diff_pair_c5_0_1_prev4 = -38
dist_diff_pair_def_c5_0_1_prev5: - dist_curr_pair_c5_0_1
 + dist_diff_pair_c5_0_1_prev5 = -46
dist_diff_pair_def_c5_0_2_prev0: - dist_curr_pair_c5_0_2
 + dist_diff_pair_c5_0_2_prev0 = -58
dist_diff_pair_def_c5_0_2_prev1: - dist_curr_pair_c5_0_2
 + dist_diff_pair_c5_0_2_prev1 = -74
dist_diff_pair_def_c5_0_2_prev2: - dist_curr_pair_c5_0_2
 + dist_diff_pair_c5_0_2_prev2 = -66
dist_diff_pair_def_c5_0_2_prev3: - dist_curr_pair_c5_0_2
 + dist_diff_pair_c5_0_2_prev3 = -78
dist_diff_pair_def_c5_0_2_prev4: - dist_curr_pair_c5_0_2
 + dist_diff_pair_c5_0_2_prev4 = -70
dist_diff_pair_def_c5_0_2_prev5: - dist_curr_pair_c5_0_2
 + dist_diff_pair_c5_0_2_prev5 = -78
dist_diff_pair_def_c5_0_3_prev0: - dist_curr_pair_c5_0_3
 + dist_diff_pair_c5_0_3_prev0 = -58
dist_diff_pair_def_c5_0_3_prev1: - dist_curr_pair_c5_0_3
 + dist_diff_pair_c5_0_3_prev1 = -54
dist_diff_pair_def_c5_0_3_prev2: - dist_curr_pair_c5_0_3
 + dist_diff_pair_c5_0_3_prev2 = -58
dist_diff_pair_def_c5_0_3_prev3: - dist_curr_pair_c5_0_3
 + dist_diff_pair_c5_0_3_prev3 = -58
dist_diff_pair_def_c5_0_3_prev4: - dist_curr_pair_c5_0_3
 + dist_diff_pair_c5_0_3_prev4 = -62
dist_diff_pair_def_c5_0_3_prev5: - dist_curr_pair_c5_0_3
 + dist_diff_pair_c5_0_3_prev5 = -58
dist_diff_pair_def_c5_0_4_prev0: - dist_curr_pair_c5_0_4
 + dist_diff_pair_c5_0_4_prev0 = -66
dist_diff_pair_def_c5_0_4_prev1: - dist_curr_pair_c5_0_4
 + dist_diff_pair_c5_0_4_prev1 = -18
dist_diff_pair_def_c5_0_4_prev2: - dist_curr_pair_c5_0_4
 + dist_diff_pair_c5_0_4_prev2 = -42
dist_diff_pair_def_c5_0_4_prev3: - dist_curr_pair_c5_0_4
 + dist_diff_pair_c5_0_4_prev3 = -21
dist_diff_pair_def_c5_0_4_prev4: - dist_curr_pair_c5_0_4
 + dist_diff_pair_c5_0_4_prev4 = -47
dist_diff_pair_def_c5_0_4_prev5: - dist_curr_pair_c5_0_4
 + dist_diff_pair_c5_0_4_prev5 = -26
dist_diff_pair_def_c5_1_2_prev0: - dist_curr_pair_c5_1_2
 + dist_diff_pair_c5_1_2_prev0 = -20
dist_diff_pair_def_c5_1_2_prev1: - dist_curr_pair_c5_1_2
 + dist_diff_pair_c5_1_2_prev1 = -40
dist_diff_pair_def_c5_1_2_prev2: - dist_curr_pair_c5_1_2
 + dist_diff_pair_c5_1_2_prev2 = -40
dist_diff_pair_def_c5_1_2_prev3: - dist_curr_pair_c5_1_2
 + dist_diff_pair_c5_1_2_prev3 = -44
dist_diff_pair_def_c5_1_2_prev4: - dist_curr_pair_c5_1_2
 + dist_diff_pair_c5_1_2_prev4 = -44
dist_diff_pair_def_c5_1_2_prev5: - dist_curr_pair_c5_1_2
 + dist_diff_pair_c5_1_2_prev5 = -44
dist_diff_pair_def_c5_1_3_prev0: - dist_curr_pair_c5_1_3
 + dist_diff_pair_c5_1_3_prev0 = -32
dist_diff_pair_def_c5_1_3_prev1: - dist_curr_pair_c5_1_3
 + dist_diff_pair_c5_1_3_prev1 = -32
dist_diff_pair_def_c5_1_3_prev2: - dist_curr_pair_c5_1_3
 + dist_diff_pair_c5_1_3_prev2 = -20
dist_diff_pair_def_c5_1_3_prev3: - dist_curr_pair_c5_1_3
 + dist_diff_pair_c5_1_3_prev3 = -28
dist_diff_pair_def_c5_1_3_prev4: - dist_curr_pair_c5_1_3
 + dist_diff_pair_c5_1_3_prev4 = -24
dist_diff_pair_def_c5_1_3_prev5: - dist_curr_pair_c5_1_3
 + dist_diff_pair_c5_1_3_prev5 = -28
dist_diff_pair_def_c5_1_4_prev0: - dist_curr_pair_c5_1_4
 + dist_diff_pair_c5_1_4_prev0 = -44
dist_diff_pair_def_c5_1_4_prev1: - dist_curr_pair_c5_1_4
 + dist_diff_pair_c5_1_4_prev1 = -64
dist_diff_pair_def_c5_1_4_prev2: - dist_curr_pair_c5_1_4
 + dist_diff_pair_c5_1_4_prev2 = -32
dist_diff_pair_def_c5_1_4_prev3: - dist_curr_pair_c5_1_4
 + dist_diff_pair_c5_1_4_prev3 = -59
dist_diff_pair_def_c5_1_4_prev4: - dist_curr_pair_c5_1_4
 + dist_diff_pair_c5_1_4_prev4 = -29
dist_diff_pair_def_c5_1_4_prev5: - dist_curr_pair_c5_1_4
 + dist_diff_pair_c5_1_4_prev5 = -64
dist_diff_pair_def_c5_2_3_prev0: - dist_curr_pair_c5_2_3
 + dist_diff_pair_c5_2_3_prev0 = -52
dist_diff_pair_def_c5_2_3_prev1: - dist_curr_pair_c5_2_3
 + dist_diff_pair_c5_2_3_prev1 = -72
dist_diff_pair_def_c5_2_3_prev2: - dist_curr_pair_c5_2_3
 + dist_diff_pair_c5_2_3_prev2 = -60
dist_diff_pair_def_c5_2_3_prev3: - dist_curr_pair_c5_2_3
 + dist_diff_pair_c5_2_3_prev3 = -72
dist_diff_pair_def_c5_2_3_prev4: - dist_curr_pair_c5_2_3
 + dist_diff_pair_c5_2_3_prev4 = -60
dist_diff_pair_def_c5_2_3_prev5: - dist_curr_pair_c5_2_3
 + dist_diff_pair_c5_2_3_prev5 = -72
dist_diff_pair_def_c5_2_4_prev0: - dist_curr_pair_c5_2_4
 + dist_diff_pair_c5_2_4_prev0 = -24
dist_diff_pair_def_c5_2_4_prev1: - dist_curr_pair_c5_2_4
 + dist_diff_pair_c5_2_4_prev1 = -56
dist_diff_pair_def_c5_2_4_prev2: - dist_curr_pair_c5_2_4
 + dist_diff_pair_c5_2_4_prev2 = -24
dist_diff_pair_def_c5_2_4_prev3: - dist_curr_pair_c5_2_4
 + dist_diff_pair_c5_2_4_prev3 = -57
dist_diff_pair_def_c5_2_4_prev4: - dist_curr_pair_c5_2_4
 + dist_diff_pair_c5_2_4_prev4 = -23
dist_diff_pair_def_c5_2_4_prev5: - dist_curr_pair_c5_2_4
 + dist_diff_pair_c5_2_4_prev5 = -52
dist_diff_pair_def_c5_3_4_prev0: - dist_curr_pair_c5_3_4
 + dist_diff_pair_c5_3_4_prev0 = -76
dist_diff_pair_def_c5_3_4_prev1: - dist_curr_pair_c5_3_4
 + dist_diff_pair_c5_3_4_prev1 = -72
dist_diff_pair_def_c5_3_4_prev2: - dist_curr_pair_c5_3_4
 + dist_diff_pair_c5_3_4_prev2 = -52
dist_diff_pair_def_c5_3_4_prev3: - dist_curr_pair_c5_3_4
 + dist_diff_pair_c5_3_4_prev3 = -71
dist_diff_pair_def_c5_3_4_prev4: - dist_curr_pair_c5_3_4
 + dist_diff_pair_c5_3_4_prev4 = -53
dist_diff_pair_def_c5_3_4_prev5: - dist_curr_pair_c5_3_4
 + dist_diff_pair_c5_3_4_prev5 = -76
dx_abs_0_1_abs_ge_neg_orig: dx_abs_0_1 + 60 dx_abs_0_1_is_positive
 + dx_diff_0_1 >= 0
dx_abs_0_1_abs_ge_orig: dx_abs_0_1 - 60 dx_abs_0_1_is_positive - dx_diff_0_1
 >= -60
dx_abs_0_1_abs_le_neg_orig: dx_abs_0_1 - 60 dx_abs_0_1_is_positive
 + dx_diff_0_1 <= 0
dx_abs_0_1_abs_le_orig: dx_abs_0_1 + 60 dx_abs_0_1_is_positive - dx_diff_0_1
 <= 60
dx_abs_0_1_force_negative: - 60 dx_abs_0_1_is_positive + dx_diff_0_1 <= -0.001
dx_abs_0_1_force_positive: - 60 dx_abs_0_1_is_positive + dx_diff_0_1 >= -60
dx_abs_1_2_abs_ge_neg_orig: dx_abs_1_2 + 60 dx_abs_1_2_is_positive
 + dx_diff_1_2 >= 0
dx_abs_1_2_abs_ge_orig: dx_abs_1_2 - 60 dx_abs_1_2_is_positive - dx_diff_1_2
 >= -60
dx_abs_1_2_abs_le_neg_orig: dx_abs_1_2 - 60 dx_abs_1_2_is_positive
 + dx_diff_1_2 <= 0
dx_abs_1_2_abs_le_orig: dx_abs_1_2 + 60 dx_abs_1_2_is_positive - dx_diff_1_2
 <= 60
dx_abs_1_2_force_negative: - 60 dx_abs_1_2_is_positive + dx_diff_1_2 <= -0.001
dx_abs_1_2_force_positive: - 60 dx_abs_1_2_is_positive + dx_diff_1_2 >= -60
dx_abs_1_3_abs_ge_neg_orig: dx_abs_1_3 + 60 dx_abs_1_3_is_positive
 + dx_diff_1_3 >= 0
dx_abs_1_3_abs_ge_orig: dx_abs_1_3 - 60 dx_abs_1_3_is_positive - dx_diff_1_3
 >= -60
dx_abs_1_3_abs_le_neg_orig: dx_abs_1_3 - 60 dx_abs_1_3_is_positive
 + dx_diff_1_3 <= 0
dx_abs_1_3_abs_le_orig: dx_abs_1_3 + 60 dx_abs_1_3_is_positive - dx_diff_1_3
 <= 60
dx_abs_1_3_force_negative: - 60 dx_abs_1_3_is_positive + dx_diff_1_3 <= -0.001
dx_abs_1_3_force_positive: - 60 dx_abs_1_3_is_positive + dx_diff_1_3 >= -60
dx_abs_2_4_abs_ge_neg_orig: dx_abs_2_4 + 60 dx_abs_2_4_is_positive
 + dx_diff_2_4 >= 0
dx_abs_2_4_abs_ge_orig: dx_abs_2_4 - 60 dx_abs_2_4_is_positive - dx_diff_2_4
 >= -60
dx_abs_2_4_abs_le_neg_orig: dx_abs_2_4 - 60 dx_abs_2_4_is_positive
 + dx_diff_2_4 <= 0
dx_abs_2_4_abs_le_orig: dx_abs_2_4 + 60 dx_abs_2_4_is_positive - dx_diff_2_4
 <= 60
dx_abs_2_4_force_negative: - 60 dx_abs_2_4_is_positive + dx_diff_2_4 <= -0.001
dx_abs_2_4_force_positive: - 60 dx_abs_2_4_is_positive + dx_diff_2_4 >= -60
dx_diff_def_0_1: - cx_0 + cx_1 + dx_diff_0_1 = 0
dx_diff_def_1_2: - cx_1 + cx_2 + dx_diff_1_2 = 0
dx_diff_def_1_3: - cx_1 + cx_3 + dx_diff_1_3 = 0
dx_diff_def_2_4: - cx_2 + cx_4 + dx_diff_2_4 = 0
dx_grid_abs_pair_c5_0_1_abs_ge_neg_orig: dx_grid_abs_pair_c5_0_1
 + 120 dx_grid_abs_pair_c5_0_1_is_positive + dx_grid_diff_c5_0_1 >= 0
dx_grid_abs_pair_c5_0_1_abs_ge_orig: dx_grid_abs_pair_c5_0_1
 - 120 dx_grid_abs_pair_c5_0_1_is_positive - dx_grid_diff_c5_0_1 >= -120
dx_grid_abs_pair_c5_0_1_abs_le_neg_orig: dx_grid_abs_pair_c5_0_1
 - 120 dx_grid_abs_pair_c5_0_1_is_positive + dx_grid_diff_c5_0_1 <= 0
dx_grid_abs_pair_c5_0_1_abs_le_orig: dx_grid_abs_pair_c5_0_1
 + 120 dx_grid_abs_pair_c5_0_1_is_positive - dx_grid_diff_c5_0_1 <= 120
dx_grid_abs_pair_c5_0_1_force_negative:
 - 120 dx_grid_abs_pair_c5_0_1_is_positive + dx_grid_diff_c5_0_1 <= -0.001
dx_grid_abs_pair_c5_0_1_force_positive:
 - 120 dx_grid_abs_pair_c5_0_1_is_positive + dx_grid_diff_c5_0_1 >= -120
dx_grid_abs_pair_c5_0_2_abs_ge_neg_orig: dx_grid_abs_pair_c5_0_2
 + 120 dx_grid_abs_pair_c5_0_2_is_positive + dx_grid_diff_c5_0_2 >= 0
dx_grid_abs_pair_c5_0_2_abs_ge_orig: dx_grid_abs_pair_c5_0_2
 - 120 dx_grid_abs_pair_c5_0_2_is_positive - dx_grid_diff_c5_0_2 >= -120
dx_grid_abs_pair_c5_0_2_abs_le_neg_orig: dx_grid_abs_pair_c5_0_2
 - 120 dx_grid_abs_pair_c5_0_2_is_positive + dx_grid_diff_c5_0_2 <= 0
dx_grid_abs_pair_c5_0_2_abs_le_orig: dx_grid_abs_pair_c5_0_2
 + 120 dx_grid_abs_pair_c5_0_2_is_positive - dx_grid_diff_c5_0_2 <= 120
dx_grid_abs_pair_c5_0_2_force_negative:
 - 120 dx_grid_abs_pair_c5_0_2_is_positive + dx_grid_diff_c5_0_2 <= -0.001
dx_grid_abs_pair_c5_0_2_force_positive:
 - 120 dx_grid_abs_pair_c5_0_2_is_positive + dx_grid_diff_c5_0_2 >= -120
dx_grid_abs_pair_c5_0_3_abs_ge_neg_orig: dx_grid_abs_pair_c5_0_3
 + 120 dx_grid_abs_pair_c5_0_3_is_positive + dx_grid_diff_c5_0_3 >= 0
dx_grid_abs_pair_c5_0_3_abs_ge_orig: dx_grid_abs_pair_c5_0_3
 - 120 dx_grid_abs_pair_c5_0_3_is_positive - dx_grid_diff_c5_0_3 >= -120
dx_grid_abs_pair_c5_0_3_abs_le_neg_orig: dx_grid_abs_pair_c5_0_3
 - 120 dx_grid_abs_pair_c5_0_3_is_positive + dx_grid_diff_c5_0_3 <= 0
dx_grid_abs_pair_c5_0_3_abs_le_orig: dx_grid_abs_pair_c5_0_3
 + 120 dx_grid_abs_pair_c5_0_3_is_positive - dx_grid_diff_c5_0_3 <= 120
dx_grid_abs_pair_c5_0_3_force_negative:
 - 120 dx_grid_abs_pair_c5_0_3_is_positive + dx_grid_diff_c5_0_3 <= -0.001
dx_grid_abs_pair_c5_0_3_force_positive:
 - 120 dx_grid_abs_pair_c5_0_3_is_positive + dx_grid_diff_c5_0_3 >= -120
dx_grid_abs_pair_c5_0_4_abs_ge_neg_orig: dx_grid_abs_pair_c5_0_4
 + 120 dx_grid_abs_pair_c5_0_4_is_positive + dx_grid_diff_c5_0_4 >= 0
dx_grid_abs_pair_c5_0_4_abs_ge_orig: dx_grid_abs_pair_c5_0_4
 - 120 dx_grid_abs_pair_c5_0_4_is_positive - dx_grid_diff_c5_0_4 >= -120
dx_grid_abs_pair_c5_0_4_abs_le_neg_orig: dx_grid_abs_pair_c5_0_4
 - 120 dx_grid_abs_pair_c5_0_4_is_positive + dx_grid_diff_c5_0_4 <= 0
dx_grid_abs_pair_c5_0_4_abs_le_orig: dx_grid_abs_pair_c5_0_4
 + 120 dx_grid_abs_pair_c5_0_4_is_positive - dx_grid_diff_c5_0_4 <= 120
dx_grid_abs_pair_c5_0_4_force_negative:
 - 120 dx_grid_abs_pair_c5_0_4_is_positive + dx_grid_diff_c5_0_4 <= -0.001
dx_grid_abs_pair_c5_0_4_force_positive:
 - 120 dx_grid_abs_pair_c5_0_4_is_positive + dx_grid_diff_c5_0_4 >= -120
dx_grid_abs_pair_c5_1_2_abs_ge_neg_orig: dx_grid_abs_pair_c5_1_2
 + 120 dx_grid_abs_pair_c5_1_2_is_positive + dx_grid_diff_c5_1_2 >= 0
dx_grid_abs_pair_c5_1_2_abs_ge_orig: dx_grid_abs_pair_c5_1_2
 - 120 dx_grid_abs_pair_c5_1_2_is_positive - dx_grid_diff_c5_1_2 >= -120
dx_grid_abs_pair_c5_1_2_abs_le_neg_orig: dx_grid_abs_pair_c5_1_2
 - 120 dx_grid_abs_pair_c5_1_2_is_positive + dx_grid_diff_c5_1_2 <= 0
dx_grid_abs_pair_c5_1_2_abs_le_orig: dx_grid_abs_pair_c5_1_2
 + 120 dx_grid_abs_pair_c5_1_2_is_positive - dx_grid_diff_c5_1_2 <= 120
dx_grid_abs_pair_c5_1_2_force_negative:
 - 120 dx_grid_abs_pair_c5_1_2_is_positive + dx_grid_diff_c5_1_2 <= -0.001
dx_grid_abs_pair_c5_1_2_force_positive:
 - 120 dx_grid_abs_pair_c5_1_2_is_positive + dx_grid_diff_c5_1_2 >= -120
dx_grid_abs_pair_c5_1_3_abs_ge_neg_orig: dx_grid_abs_pair_c5_1_3
 + 120 dx_grid_abs_pair_c5_1_3_is_positive + dx_grid_diff_c5_1_3 >= 0
dx_grid_abs_pair_c5_1_3_abs_ge_orig: dx_grid_abs_pair_c5_1_3
 - 120 dx_grid_abs_pair_c5_1_3_is_positive - dx_grid_diff_c5_1_3 >= -120
dx_grid_abs_pair_c5_1_3_abs_le_neg_orig: dx_grid_abs_pair_c5_1_3
 - 120 dx_grid_abs_pair_c5_1_3_is_positive + dx_grid_diff_c5_1_3 <= 0
dx_grid_abs_pair_c5_1_3_abs_le_orig: dx_grid_abs_pair_c5_1_3
 + 120 dx_grid_abs_pair_c5_1_3_is_positive - dx_grid_diff_c5_1_3 <= 120
dx_grid_abs_pair_c5_1_3_force_negative:
 - 120 dx_grid_abs_pair_c5_1_3_is_positive + dx_grid_diff_c5_1_3 <= -0.001
dx_grid_abs_pair_c5_1_3_force_positive:
 - 120 dx_grid_abs_pair_c5_1_3_is_positive + dx_grid_diff_c5_1_3 >= -120
dx_grid_abs_pair_c5_1_4_abs_ge_neg_orig: dx_grid_abs_pair_c5_1_4
 + 120 dx_grid_abs_pair_c5_1_4_is_positive + dx_grid_diff_c5_1_4 >= 0
dx_grid_abs_pair_c5_1_4_abs_ge_orig: dx_grid_abs_pair_c5_1_4
 - 120 dx_grid_abs_pair_c5_1_4_is_positive - dx_grid_diff_c5_1_4 >= -120
dx_grid_abs_pair_c5_1_4_abs_le_neg_orig: dx_grid_abs_pair_c5_1_4
 - 120 dx_grid_abs_pair_c5_1_4_is_positive + dx_grid_diff_c5_1_4 <= 0
dx_grid_abs_pair_c5_1_4_abs_le_orig: dx_grid_abs_pair_c5_1_4
 + 120 dx_grid_abs_pair_c5_1_4_is_positive - dx_grid_diff_c5_1_4 <= 120
dx_grid_abs_pair_c5_1_4_force_negative:
 - 120 dx_grid_abs_pair_c5_1_4_is_positive + dx_grid_diff_c5_1_4 <= -0.001
dx_grid_abs_pair_c5_1_4_force_positive:
 - 120 dx_grid_abs_pair_c5_1_4_is_positive + dx_grid_diff_c5_1_4 >= -120
dx_grid_abs_pair_c5_2_3_abs_ge_neg_orig: dx_grid_abs_pair_c5_2_3
 + 120 dx_grid_abs_pair_c5_2_3_is_positive + dx_grid_diff_c5_2_3 >= 0
dx_grid_abs_pair_c5_2_3_abs_ge_orig: dx_grid_abs_pair_c5_2_3
 - 120 dx_grid_abs_pair_c5_2_3_is_positive - dx_grid_diff_c5_2_3 >= -120
dx_grid_abs_pair_c5_2_3_abs_le_neg_orig: dx_grid_abs_pair_c5_2_3
 - 120 dx_grid_abs_pair_c5_2_3_is_positive + dx_grid_diff_c5_2_3 <= 0
dx_grid_abs_pair_c5_2_3_abs_le_orig: dx_grid_abs_pair_c5_2_3
 + 120 dx_grid_abs_pair_c5_2_3_is_positive - dx_grid_diff_c5_2_3 <= 120
dx_grid_abs_pair_c5_2_3_force_negative:
 - 120 dx_grid_abs_pair_c5_2_3_is_positive + dx_grid_diff_c5_2_3 <= -0.001
dx_grid_abs_pair_c5_2_3_force_positive:
 - 120 dx_grid_abs_pair_c5_2_3_is_positive + dx_grid_diff_c5_2_3 >= -120
dx_grid_abs_pair_c5_2_4_abs_ge_neg_orig: dx_grid_abs_pair_c5_2_4
 + 120 dx_grid_abs_pair_c5_2_4_is_positive + dx_grid_diff_c5_2_4 >= 0
dx_grid_abs_pair_c5_2_4_abs_ge_orig: dx_grid_abs_pair_c5_2_4
 - 120 dx_grid_abs_pair_c5_2_4_is_positive - dx_grid_diff_c5_2_4 >= -120
dx_grid_abs_pair_c5_2_4_abs_le_neg_orig: dx_grid_abs_pair_c5_2_4
 - 120 dx_grid_abs_pair_c5_2_4_is_positive + dx_grid_diff_c5_2_4 <= 0
dx_grid_abs_pair_c5_2_4_abs_le_orig: dx_grid_abs_pair_c5_2_4
 + 120 dx_grid_abs_pair_c5_2_4_is_positive - dx_grid_diff_c5_2_4 <= 120
dx_grid_abs_pair_c5_2_4_force_negative:
 - 120 dx_grid_abs_pair_c5_2_4_is_positive + dx_grid_diff_c5_2_4 <= -0.001
dx_grid_abs_pair_c5_2_4_force_positive:
 - 120 dx_grid_abs_pair_c5_2_4_is_positive + dx_grid_diff_c5_2_4 >= -120
dx_grid_abs_pair_c5_3_4_abs_ge_neg_orig: dx_grid_abs_pair_c5_3_4
 + 120 dx_grid_abs_pair_c5_3_4_is_positive + dx_grid_diff_c5_3_4 >= 0
dx_grid_abs_pair_c5_3_4_abs_ge_orig: dx_grid_abs_pair_c5_3_4
 - 120 dx_grid_abs_pair_c5_3_4_is_positive - dx_grid_diff_c5_3_4 >= -120
dx_grid_abs_pair_c5_3_4_abs_le_neg_orig: dx_grid_abs_pair_c5_3_4
 - 120 dx_grid_abs_pair_c5_3_4_is_positive + dx_grid_diff_c5_3_4 <= 0
dx_grid_abs_pair_c5_3_4_abs_le_orig: dx_grid_abs_pair_c5_3_4
 + 120 dx_grid_abs_pair_c5_3_4_is_positive - dx_grid_diff_c5_3_4 <= 120
dx_grid_abs_pair_c5_3_4_force_negative:
 - 120 dx_grid_abs_pair_c5_3_4_is_positive + dx_grid_diff_c5_3_4 <= -0.001
dx_grid_abs_pair_c5_3_4_force_positive:
 - 120 dx_grid_abs_pair_c5_3_4_is_positive + dx_grid_diff_c5_3_4 >= -120
dx_grid_diff_def_c5_0_1: dx_grid_diff_c5_0_1 - x_grid_0 + x_grid_1 = 0
dx_grid_diff_def_c5_0_2: dx_grid_diff_c5_0_2 - x_grid_0 + x_grid_2 = 0
dx_grid_diff_def_c5_0_3: dx_grid_diff_c5_0_3 - x_grid_0 + x_grid_3 = 0
dx_grid_diff_def_c5_0_4: dx_grid_diff_c5_0_4 - x_grid_0 + x_grid_4 = 0
dx_grid_diff_def_c5_1_2: dx_grid_diff_c5_1_2 - x_grid_1 + x_grid_2 = 0
dx_grid_diff_def_c5_1_3: dx_grid_diff_c5_1_3 - x_grid_1 + x_grid_3 = 0
dx_grid_diff_def_c5_1_4: dx_grid_diff_c5_1_4 - x_grid_1 + x_grid_4 = 0
dx_grid_diff_def_c5_2_3: dx_grid_diff_c5_2_3 - x_grid_2 + x_grid_3 = 0
dx_grid_diff_def_c5_2_4: dx_grid_diff_c5_2_4 - x_grid_2 + x_grid_4 = 0
dx_grid_diff_def_c5_3_4: dx_grid_diff_c5_3_4 - x_grid_3 + x_grid_4 = 0
dy_abs_0_1_abs_ge_neg_orig: dy_abs_0_1 + 60 dy_abs_0_1_is_positive
 + dy_diff_0_1 >= 0
dy_abs_0_1_abs_ge_orig: dy_abs_0_1 - 60 dy_abs_0_1_is_positive - dy_diff_0_1
 >= -60
dy_abs_0_1_abs_le_neg_orig: dy_abs_0_1 - 60 dy_abs_0_1_is_positive
 + dy_diff_0_1 <= 0
dy_abs_0_1_abs_le_orig: dy_abs_0_1 + 60 dy_abs_0_1_is_positive - dy_diff_0_1
 <= 60
dy_abs_0_1_force_negative: - 60 dy_abs_0_1_is_positive + dy_diff_0_1 <= -0.001
dy_abs_0_1_force_positive: - 60 dy_abs_0_1_is_positive + dy_diff_0_1 >= -60
dy_abs_1_2_abs_ge_neg_orig: dy_abs_1_2 + 60 dy_abs_1_2_is_positive
 + dy_diff_1_2 >= 0
dy_abs_1_2_abs_ge_orig: dy_abs_1_2 - 60 dy_abs_1_2_is_positive - dy_diff_1_2
 >= -60
dy_abs_1_2_abs_le_neg_orig: dy_abs_1_2 - 60 dy_abs_1_2_is_positive
 + dy_diff_1_2 <= 0
dy_abs_1_2_abs_le_orig: dy_abs_1_2 + 60 dy_abs_1_2_is_positive - dy_diff_1_2
 <= 60
dy_abs_1_2_force_negative: - 60 dy_abs_1_2_is_positive + dy_diff_1_2 <= -0.001
dy_abs_1_2_force_positive: - 60 dy_abs_1_2_is_positive + dy_diff_1_2 >= -60
dy_abs_1_3_abs_ge_neg_orig: dy_abs_1_3 + 60 dy_abs_1_3_is_positive
 + dy_diff_1_3 >= 0
dy_abs_1_3_abs_ge_orig: dy_abs_1_3 - 60 dy_abs_1_3_is_positive - dy_diff_1_3
 >= -60
dy_abs_1_3_abs_le_neg_orig: dy_abs_1_3 - 60 dy_abs_1_3_is_positive
 + dy_diff_1_3 <= 0
dy_abs_1_3_abs_le_orig: dy_abs_1_3 + 60 dy_abs_1_3_is_positive - dy_diff_1_3
 <= 60
dy_abs_1_3_force_negative: - 60 dy_abs_1_3_is_positive + dy_diff_1_3 <= -0.001
dy_abs_1_3_force_positive: - 60 dy_abs_1_3_is_positive + dy_diff_1_3 >= -60
dy_abs_2_4_abs_ge_neg_orig: dy_abs_2_4 + 60 dy_abs_2_4_is_positive
 + dy_diff_2_4 >= 0
dy_abs_2_4_abs_ge_orig: dy_abs_2_4 - 60 dy_abs_2_4_is_positive - dy_diff_2_4
 >= -60
dy_abs_2_4_abs_le_neg_orig: dy_abs_2_4 - 60 dy_abs_2_4_is_positive
 + dy_diff_2_4 <= 0
dy_abs_2_4_abs_le_orig: dy_abs_2_4 + 60 dy_abs_2_4_is_positive - dy_diff_2_4
 <= 60
dy_abs_2_4_force_negative: - 60 dy_abs_2_4_is_positive + dy_diff_2_4 <= -0.001
dy_abs_2_4_force_positive: - 60 dy_abs_2_4_is_positive + dy_diff_2_4 >= -60
dy_diff_def_0_1: - cy_0 + cy_1 + dy_diff_0_1 = 0
dy_diff_def_1_2: - cy_1 + cy_2 + dy_diff_1_2 = 0
dy_diff_def_1_3: - cy_1 + cy_3 + dy_diff_1_3 = 0
dy_diff_def_2_4: - cy_2 + cy_4 + dy_diff_2_4 = 0
dy_grid_abs_pair_c5_0_1_abs_ge_neg_orig: dy_grid_abs_pair_c5_0_1
 + 120 dy_grid_abs_pair_c5_0_1_is_positive + dy_grid_diff_c5_0_1 >= 0
dy_grid_abs_pair_c5_0_1_abs_ge_orig: dy_grid_abs_pair_c5_0_1
 - 120 dy_grid_abs_pair_c5_0_1_is_positive - dy_grid_diff_c5_0_1 >= -120
dy_grid_abs_pair_c5_0_1_abs_le_neg_orig: dy_grid_abs_pair_c5_0_1
 - 120 dy_grid_abs_pair_c5_0_1_is_positive + dy_grid_diff_c5_0_1 <= 0
dy_grid_abs_pair_c5_0_1_abs_le_orig: dy_grid_abs_pair_c5_0_1
 + 120 dy_grid_abs_pair_c5_0_1_is_positive - dy_grid_diff_c5_0_1 <= 120
dy_grid_abs_pair_c5_0_1_force_negative:
 - 120 dy_grid_abs_pair_c5_0_1_is_positive + dy_grid_diff_c5_0_1 <= -0.001
dy_grid_abs_pair_c5_0_1_force_positive:
 - 120 dy_grid_abs_pair_c5_0_1_is_positive + dy_grid_diff_c5_0_1 >= -120
dy_grid_abs_pair_c5_0_2_abs_ge_neg_orig: dy_grid_abs_pair_c5_0_2
 + 120 dy_grid_abs_pair_c5_0_2_is_positive + dy_grid_diff_c5_0_2 >= 0
dy_grid_abs_pair_c5_0_2_abs_ge_orig: dy_grid_abs_pair_c5_0_2
 - 120 dy_grid_abs_pair_c5_0_2_is_positive - dy_grid_diff_c5_0_2 >= -120
dy_grid_abs_pair_c5_0_2_abs_le_neg_orig: dy_grid_abs_pair_c5_0_2
 - 120 dy_grid_abs_pair_c5_0_2_is_positive + dy_grid_diff_c5_0_2 <= 0
dy_grid_abs_pair_c5_0_2_abs_le_orig: dy_grid_abs_pair_c5_0_2
 + 120 dy_grid_abs_pair_c5_0_2_is_positive - dy_grid_diff_c5_0_2 <= 120
dy_grid_abs_pair_c5_0_2_force_negative:
 - 120 dy_grid_abs_pair_c5_0_2_is_positive + dy_grid_diff_c5_0_2 <= -0.001
dy_grid_abs_pair_c5_0_2_force_positive:
 - 120 dy_grid_abs_pair_c5_0_2_is_positive + dy_grid_diff_c5_0_2 >= -120
dy_grid_abs_pair_c5_0_3_abs_ge_neg_orig: dy_grid_abs_pair_c5_0_3
 + 120 dy_grid_abs_pair_c5_0_3_is_positive + dy_grid_diff_c5_0_3 >= 0
dy_grid_abs_pair_c5_0_3_abs_ge_orig: dy_grid_abs_pair_c5_0_3
 - 120 dy_grid_abs_pair_c5_0_3_is_positive - dy_grid_diff_c5_0_3 >= -120
dy_grid_abs_pair_c5_0_3_abs_le_neg_orig: dy_grid_abs_pair_c5_0_3
 - 120 dy_grid_abs_pair_c5_0_3_is_positive + dy_grid_diff_c5_0_3 <= 0
dy_grid_abs_pair_c5_0_3_abs_le_orig: dy_grid_abs_pair_c5_0_3
 + 120 dy_grid_abs_pair_c5_0_3_is_positive - dy_grid_diff_c5_0_3 <= 120
dy_grid_abs_pair_c5_0_3_force_negative:
 - 120 dy_grid_abs_pair_c5_0_3_is_positive + dy_grid_diff_c5_0_3 <= -0.001
dy_grid_abs_pair_c5_0_3_force_positive:
 - 120 dy_grid_abs_pair_c5_0_3_is_positive + dy_grid_diff_c5_0_3 >= -120
dy_grid_abs_pair_c5_0_4_abs_ge_neg_orig: dy_grid_abs_pair_c5_0_4
 + 120 dy_grid_abs_pair_c5_0_4_is_positive + dy_grid_diff_c5_0_4 >= 0
dy_grid_abs_pair_c5_0_4_abs_ge_orig: dy_grid_abs_pair_c5_0_4
 - 120 dy_grid_abs_pair_c5_0_4_is_positive - dy_grid_diff_c5_0_4 >= -120
dy_grid_abs_pair_c5_0_4_abs_le_neg_orig: dy_grid_abs_pair_c5_0_4
 - 120 dy_grid_abs_pair_c5_0_4_is_positive + dy_grid_diff_c5_0_4 <= 0
dy_grid_abs_pair_c5_0_4_abs_le_orig: dy_grid_abs_pair_c5_0_4
 + 120 dy_grid_abs_pair_c5_0_4_is_positive - dy_grid_diff_c5_0_4 <= 120
dy_grid_abs_pair_c5_0_4_force_negative:
 - 120 dy_grid_abs_pair_c5_0_4_is_positive + dy_grid_diff_c5_0_4 <= -0.001
dy_grid_abs_pair_c5_0_4_force_positive:
 - 120 dy_grid_abs_pair_c5_0_4_is_positive + dy_grid_diff_c5_0_4 >= -120
dy_grid_abs_pair_c5_1_2_abs_ge_neg_orig: dy_grid_abs_pair_c5_1_2
 + 120 dy_grid_abs_pair_c5_1_2_is_positive + dy_grid_diff_c5_1_2 >= 0
dy_grid_abs_pair_c5_1_2_abs_ge_orig: dy_grid_abs_pair_c5_1_2
 - 120 dy_grid_abs_pair_c5_1_2_is_positive - dy_grid_diff_c5_1_2 >= -120
dy_grid_abs_pair_c5_1_2_abs_le_neg_orig: dy_grid_abs_pair_c5_1_2
 - 120 dy_grid_abs_pair_c5_1_2_is_positive + dy_grid_diff_c5_1_2 <= 0
dy_grid_abs_pair_c5_1_2_abs_le_orig: dy_grid_abs_pair_c5_1_2
 + 120 dy_grid_abs_pair_c5_1_2_is_positive - dy_grid_diff_c5_1_2 <= 120
dy_grid_abs_pair_c5_1_2_force_negative:
 - 120 dy_grid_abs_pair_c5_1_2_is_positive + dy_grid_diff_c5_1_2 <= -0.001
dy_grid_abs_pair_c5_1_2_force_positive:
 - 120 dy_grid_abs_pair_c5_1_2_is_positive + dy_grid_diff_c5_1_2 >= -120
dy_grid_abs_pair_c5_1_3_abs_ge_neg_orig: dy_grid_abs_pair_c5_1_3
 + 120 dy_grid_abs_pair_c5_1_3_is_positive + dy_grid_diff_c5_1_3 >= 0
dy_grid_abs_pair_c5_1_3_abs_ge_orig: dy_grid_abs_pair_c5_1_3
 - 120 dy_grid_abs_pair_c5_1_3_is_positive - dy_grid_diff_c5_1_3 >= -120
dy_grid_abs_pair_c5_1_3_abs_le_neg_orig: dy_grid_abs_pair_c5_1_3
 - 120 dy_grid_abs_pair_c5_1_3_is_positive + dy_grid_diff_c5_1_3 <= 0
dy_grid_abs_pair_c5_1_3_abs_le_orig: dy_grid_abs_pair_c5_1_3
 + 120 dy_grid_abs_pair_c5_1_3_is_positive - dy_grid_diff_c5_1_3 <= 120
dy_grid_abs_pair_c5_1_3_force_negative:
 - 120 dy_grid_abs_pair_c5_1_3_is_positive + dy_grid_diff_c5_1_3 <= -0.001
dy_grid_abs_pair_c5_1_3_force_positive:
 - 120 dy_grid_abs_pair_c5_1_3_is_positive + dy_grid_diff_c5_1_3 >= -120
dy_grid_abs_pair_c5_1_4_abs_ge_neg_orig: dy_grid_abs_pair_c5_1_4
 + 120 dy_grid_abs_pair_c5_1_4_is_positive + dy_grid_diff_c5_1_4 >= 0
dy_grid_abs_pair_c5_1_4_abs_ge_orig: dy_grid_abs_pair_c5_1_4
 - 120 dy_grid_abs_pair_c5_1_4_is_positive - dy_grid_diff_c5_1_4 >= -120
dy_grid_abs_pair_c5_1_4_abs_le_neg_orig: dy_grid_abs_pair_c5_1_4
 - 120 dy_grid_abs_pair_c5_1_4_is_positive + dy_grid_diff_c5_1_4 <= 0
dy_grid_abs_pair_c5_1_4_abs_le_orig: dy_grid_abs_pair_c5_1_4
 + 120 dy_grid_abs_pair_c5_1_4_is_positive - dy_grid_diff_c5_1_4 <= 120
dy_grid_abs_pair_c5_1_4_force_negative:
 - 120 dy_grid_abs_pair_c5_1_4_is_positive + dy_grid_diff_c5_1_4 <= -0.001
dy_grid_abs_pair_c5_1_4_force_positive:
 - 120 dy_grid_abs_pair_c5_1_4_is_positive + dy_grid_diff_c5_1_4 >= -120
dy_grid_abs_pair_c5_2_3_abs_ge_neg_orig: dy_grid_abs_pair_c5_2_3
 + 120 dy_grid_abs_pair_c5_2_3_is_positive + dy_grid_diff_c5_2_3 >= 0
dy_grid_abs_pair_c5_2_3_abs_ge_orig: dy_grid_abs_pair_c5_2_3
 - 120 dy_grid_abs_pair_c5_2_3_is_positive - dy_grid_diff_c5_2_3 >= -120
dy_grid_abs_pair_c5_2_3_abs_le_neg_orig: dy_grid_abs_pair_c5_2_3
 - 120 dy_grid_abs_pair_c5_2_3_is_positive + dy_grid_diff_c5_2_3 <= 0
dy_grid_abs_pair_c5_2_3_abs_le_orig: dy_grid_abs_pair_c5_2_3
 + 120 dy_grid_abs_pair_c5_2_3_is_positive - dy_grid_diff_c5_2_3 <= 120
dy_grid_abs_pair_c5_2_3_force_negative:
 - 120 dy_grid_abs_pair_c5_2_3_is_positive + dy_grid_diff_c5_2_3 <= -0.001
dy_grid_abs_pair_c5_2_3_force_positive:
 - 120 dy_grid_abs_pair_c5_2_3_is_positive + dy_grid_diff_c5_2_3 >= -120
dy_grid_abs_pair_c5_2_4_abs_ge_neg_orig: dy_grid_abs_pair_c5_2_4
 + 120 dy_grid_abs_pair_c5_2_4_is_positive + dy_grid_diff_c5_2_4 >= 0
dy_grid_abs_pair_c5_2_4_abs_ge_orig: dy_grid_abs_pair_c5_2_4
 - 120 dy_grid_abs_pair_c5_2_4_is_positive - dy_grid_diff_c5_2_4 >= -120
dy_grid_abs_pair_c5_2_4_abs_le_neg_orig: dy_grid_abs_pair_c5_2_4
 - 120 dy_grid_abs_pair_c5_2_4_is_positive + dy_grid_diff_c5_2_4 <= 0
dy_grid_abs_pair_c5_2_4_abs_le_orig: dy_grid_abs_pair_c5_2_4
 + 120 dy_grid_abs_pair_c5_2_4_is_positive - dy_grid_diff_c5_2_4 <= 120
dy_grid_abs_pair_c5_2_4_force_negative:
 - 120 dy_grid_abs_pair_c5_2_4_is_positive + dy_grid_diff_c5_2_4 <= -0.001
dy_grid_abs_pair_c5_2_4_force_positive:
 - 120 dy_grid_abs_pair_c5_2_4_is_positive + dy_grid_diff_c5_2_4 >= -120
dy_grid_abs_pair_c5_3_4_abs_ge_neg_orig: dy_grid_abs_pair_c5_3_4
 + 120 dy_grid_abs_pair_c5_3_4_is_positive + dy_grid_diff_c5_3_4 >= 0
dy_grid_abs_pair_c5_3_4_abs_ge_orig: dy_grid_abs_pair_c5_3_4
 - 120 dy_grid_abs_pair_c5_3_4_is_positive - dy_grid_diff_c5_3_4 >= -120
dy_grid_abs_pair_c5_3_4_abs_le_neg_orig: dy_grid_abs_pair_c5_3_4
 - 120 dy_grid_abs_pair_c5_3_4_is_positive + dy_grid_diff_c5_3_4 <= 0
dy_grid_abs_pair_c5_3_4_abs_le_orig: dy_grid_abs_pair_c5_3_4
 + 120 dy_grid_abs_pair_c5_3_4_is_positive - dy_grid_diff_c5_3_4 <= 120
dy_grid_abs_pair_c5_3_4_force_negative:
 - 120 dy_grid_abs_pair_c5_3_4_is_positive + dy_grid_diff_c5_3_4 <= -0.001
dy_grid_abs_pair_c5_3_4_force_positive:
 - 120 dy_grid_abs_pair_c5_3_4_is_positive + dy_grid_diff_c5_3_4 >= -120
dy_grid_diff_def_c5_0_1: dy_grid_diff_c5_0_1 - y_grid_0 + y_grid_1 = 0
dy_grid_diff_def_c5_0_2: dy_grid_diff_c5_0_2 - y_grid_0 + y_grid_2 = 0
dy_grid_diff_def_c5_0_3: dy_grid_diff_c5_0_3 - y_grid_0 + y_grid_3 = 0
dy_grid_diff_def_c5_0_4: dy_grid_diff_c5_0_4 - y_grid_0 + y_grid_4 = 0
dy_grid_diff_def_c5_1_2: dy_grid_diff_c5_1_2 - y_grid_1 + y_grid_2 = 0
dy_grid_diff_def_c5_1_3: dy_grid_diff_c5_1_3 - y_grid_1 + y_grid_3 = 0
dy_grid_diff_def_c5_1_4: dy_grid_diff_c5_1_4 - y_grid_1 + y_grid_4 = 0
dy_grid_diff_def_c5_2_3: dy_grid_diff_c5_2_3 - y_grid_2 + y_grid_3 = 0
dy_grid_diff_def_c5_2_4: dy_grid_diff_c5_2_4 - y_grid_2 + y_grid_4 = 0
dy_grid_diff_def_c5_3_4: dy_grid_diff_c5_3_4 - y_grid_3 + y_grid_4 = 0
exclude_solution_dist_pair_c5: diff_dist_pair_c5_0_1 + diff_dist_pair_c5_0_2
 + diff_dist_pair_c5_0_3 + diff_dist_pair_c5_0_4 + diff_dist_pair_c5_1_2
 + diff_dist_pair_c5_1_3 + diff_dist_pair_c5_1_4 + diff_dist_pair_c5_2_3
 + diff_dist_pair_c5_2_4 + diff_dist_pair_c5_3_4 >= 1
fix_center_x_0: cx_0 = 30
fix_center_y_0: cy_0 = 30
h_align_grid_0: h_0 = 16
h_align_grid_1: h_1 = 20
h_align_grid_2: h_2 = 20
h_align_grid_3: h_3 = 14
h_align_grid_4: h_4 = 4
h_max_0: h_0 <= 60
h_max_1: h_1 <= 60
h_max_2: h_2 <= 60
h_max_3: h_3 <= 60
h_max_4: h_4 <= 60
height_rotation_0: h_0 = 16
height_rotation_1: h_1 + 10 r_1 = 20
height_rotation_2: h_2 = 20
height_rotation_3: h_3 - 2 r_3 = 14
height_rotation_4: h_4 = 4
horizontal_direction_0_1: z1L_0_1 + z1R_0_1 - z1_0_1 = 0
horizontal_direction_1_2: z1L_1_2 + z1R_1_2 - z1_1_2 = 0
horizontal_direction_1_3: z1L_1_3 + z1R_1_3 - z1_1_3 = 0
horizontal_direction_2_4: z1L_2_4 + z1R_2_4 - z1_2_4 = 0
horizontal_left_dist_0_1: - w_0 - x_0 + x_1 + 200 z1L_0_1 <= 200.5
horizontal_left_dist_1_2: - w_1 - x_1 + x_2 + 200 z1L_1_2 <= 200.5
horizontal_left_dist_1_3: - w_1 - x_1 + x_3 + 200 z1L_1_3 <= 200.5
horizontal_left_dist_2_4: - w_2 - x_2 + x_4 + 200 z1L_2_4 <= 200.5
horizontal_left_dist_lb_0_1: - w_0 - x_0 + x_1 - 200 z1L_0_1 >= -200
horizontal_left_dist_lb_1_2: - w_1 - x_1 + x_2 - 200 z1L_1_2 >= -200
horizontal_left_dist_lb_1_3: - w_1 - x_1 + x_3 - 200 z1L_1_3 >= -200
horizontal_left_dist_lb_2_4: - w_2 - x_2 + x_4 - 200 z1L_2_4 >= -200
horizontal_overlap_y1_0_1: - h_1 + y_0 - y_1 + 200 z1_0_1 <= 200
horizontal_overlap_y1_1_2: - h_2 + y_1 - y_2 + 200 z1_1_2 <= 200
horizontal_overlap_y1_1_3: - h_3 + y_1 - y_3 + 200 z1_1_3 <= 200
horizontal_overlap_y1_2_4: - h_4 + y_2 - y_4 + 200 z1_2_4 <= 200
horizontal_overlap_y2_0_1: - h_0 - y_0 + y_1 + 200 z1_0_1 <= 200
horizontal_overlap_y2_1_2: - h_1 - y_1 + y_2 + 200 z1_1_2 <= 200
horizontal_overlap_y2_1_3: - h_1 - y_1 + y_3 + 200 z1_1_3 <= 200
horizontal_overlap_y2_2_4: - h_2 - y_2 + y_4 + 200 z1_2_4 <= 200
horizontal_right_dist_0_1: - w_1 + x_0 - x_1 + 200 z1R_0_1 <= 200.5
horizontal_right_dist_1_2: - w_2 + x_1 - x_2 + 200 z1R_1_2 <= 200.5
horizontal_right_dist_1_3: - w_3 + x_1 - x_3 + 200 z1R_1_3 <= 200.5
horizontal_right_dist_2_4: - w_4 + x_2 - x_4 + 200 z1R_2_4 <= 200.5
horizontal_right_dist_lb_0_1: - w_1 + x_0 - x_1 - 200 z1R_0_1 >= -200
horizontal_right_dist_lb_1_2: - w_2 + x_1 - x_2 - 200 z1R_1_2 >= -200
horizontal_right_dist_lb_1_3: - w_3 + x_1 - x_3 - 200 z1R_1_3 >= -200
horizontal_right_dist_lb_2_4: - w_4 + x_2 - x_4 - 200 z1R_2_4 >= -200
must_adjacent_0_1: z1_0_1 + z2_0_1 = 1
must_adjacent_1_2: z1_1_2 + z2_1_2 = 1
must_adjacent_1_3: z1_1_3 + z2_1_3 = 1
must_adjacent_2_4: z1_2_4 + z2_2_4 = 1
non_overlap_any_0_1: p_down_0_1 + p_left_0_1 + p_right_0_1 + p_up_0_1 = 1
non_overlap_any_0_2: p_down_0_2 + p_left_0_2 + p_right_0_2 + p_up_0_2 = 1
non_overlap_any_0_3: p_down_0_3 + p_left_0_3 + p_right_0_3 + p_up_0_3 = 1
non_overlap_any_0_4: p_down_0_4 + p_left_0_4 + p_right_0_4 + p_up_0_4 = 1
non_overlap_any_1_2: p_down_1_2 + p_left_1_2 + p_right_1_2 + p_up_1_2 = 1
non_overlap_any_1_3: p_down_1_3 + p_left_1_3 + p_right_1_3 + p_up_1_3 = 1
non_overlap_any_1_4: p_down_1_4 + p_left_1_4 + p_right_1_4 + p_up_1_4 = 1
non_overlap_any_2_3: p_down_2_3 + p_left_2_3 + p_right_2_3 + p_up_2_3 = 1
non_overlap_any_2_4: p_down_2_4 + p_left_2_4 + p_right_2_4 + p_up_2_4 = 1
non_overlap_any_3_4: p_down_3_4 + p_left_3_4 + p_right_3_4 + p_up_3_4 = 1
non_overlap_down_0_1: h_0 + 200 p_down_0_1 + y_0 - y_1 <= 200
non_overlap_down_0_2: h_0 + 200 p_down_0_2 + y_0 - y_2 <= 200
non_overlap_down_0_3: h_0 + 200 p_down_0_3 + y_0 - y_3 <= 200
non_overlap_down_0_4: h_0 + 200 p_down_0_4 + y_0 - y_4 <= 200
non_overlap_down_1_2: h_1 + 200 p_down_1_2 + y_1 - y_2 <= 200
non_overlap_down_1_3: h_1 + 200 p_down_1_3 + y_1 - y_3 <= 200
non_overlap_down_1_4: h_1 + 200 p_down_1_4 + y_1 - y_4 <= 200
non_overlap_down_2_3: h_2 + 200 p_down_2_3 + y_2 - y_3 <= 200
non_overlap_down_2_4: h_2 + 200 p_down_2_4 + y_2 - y_4 <= 200
non_overlap_down_3_4: h_3 + 200 p_down_3_4 + y_3 - y_4 <= 200
non_overlap_down_rev_0_1: - h_0 - 200 p_down_0_1 - y_0 + y_1 <= 0
non_overlap_down_rev_0_2: - h_0 - 200 p_down_0_2 - y_0 + y_2 <= 0
non_overlap_down_rev_0_3: - h_0 - 200 p_down_0_3 - y_0 + y_3 <= 0
non_overlap_down_rev_0_4: - h_0 - 200 p_down_0_4 - y_0 + y_4 <= 0
non_overlap_down_rev_1_2: - h_1 - 200 p_down_1_2 - y_1 + y_2 <= 0
non_overlap_down_rev_1_3: - h_1 - 200 p_down_1_3 - y_1 + y_3 <= 0
non_overlap_down_rev_1_4: - h_1 - 200 p_down_1_4 - y_1 + y_4 <= 0
non_overlap_down_rev_2_3: - h_2 - 200 p_down_2_3 - y_2 + y_3 <= 0
non_overlap_down_rev_2_4: - h_2 - 200 p_down_2_4 - y_2 + y_4 <= 0
non_overlap_down_rev_3_4: - h_3 - 200 p_down_3_4 - y_3 + y_4 <= 0
non_overlap_left_0_1: 200 p_left_0_1 + w_0 + x_0 - x_1 <= 200
non_overlap_left_0_2: 200 p_left_0_2 + w_0 + x_0 - x_2 <= 200
non_overlap_left_0_3: 200 p_left_0_3 + w_0 + x_0 - x_3 <= 200
non_overlap_left_0_4: 200 p_left_0_4 + w_0 + x_0 - x_4 <= 200
non_overlap_left_1_2: 200 p_left_1_2 + w_1 + x_1 - x_2 <= 200
non_overlap_left_1_3: 200 p_left_1_3 + w_1 + x_1 - x_3 <= 200
non_overlap_left_1_4: 200 p_left_1_4 + w_1 + x_1 - x_4 <= 200
non_overlap_left_2_3: 200 p_left_2_3 + w_2 + x_2 - x_3 <= 200
non_overlap_left_2_4: 200 p_left_2_4 + w_2 + x_2 - x_4 <= 200
non_overlap_left_3_4: 200 p_left_3_4 + w_3 + x_3 - x_4 <= 200
non_overlap_left_rev_0_1: - 200 p_left_0_1 - w_0 - x_0 + x_1 <= 0
non_overlap_left_rev_0_2: - 200 p_left_0_2 - w_0 - x_0 + x_2 <= 0
non_overlap_left_rev_0_3: - 200 p_left_0_3 - w_0 - x_0 + x_3 <= 0
non_overlap_left_rev_0_4: - 200 p_left_0_4 - w_0 - x_0 + x_4 <= 0
non_overlap_left_rev_1_2: - 200 p_left_1_2 - w_1 - x_1 + x_2 <= 0
non_overlap_left_rev_1_3: - 200 p_left_1_3 - w_1 - x_1 + x_3 <= 0
non_overlap_left_rev_1_4: - 200 p_left_1_4 - w_1 - x_1 + x_4 <= 0
non_overlap_left_rev_2_3: - 200 p_left_2_3 - w_2 - x_2 + x_3 <= 0
non_overlap_left_rev_2_4: - 200 p_left_2_4 - w_2 - x_2 + x_4 <= 0
non_overlap_left_rev_3_4: - 200 p_left_3_4 - w_3 - x_3 + x_4 <= 0
non_overlap_right_0_1: 200 p_right_0_1 + w_1 - x_0 + x_1 <= 200
non_overlap_right_0_2: 200 p_right_0_2 + w_2 - x_0 + x_2 <= 200
non_overlap_right_0_3: 200 p_right_0_3 + w_3 - x_0 + x_3 <= 200
non_overlap_right_0_4: 200 p_right_0_4 + w_4 - x_0 + x_4 <= 200
non_overlap_right_1_2: 200 p_right_1_2 + w_2 - x_1 + x_2 <= 200
non_overlap_right_1_3: 200 p_right_1_3 + w_3 - x_1 + x_3 <= 200
non_overlap_right_1_4: 200 p_right_1_4 + w_4 - x_1 + x_4 <= 200
non_overlap_right_2_3: 200 p_right_2_3 + w_3 - x_2 + x_3 <= 200
non_overlap_right_2_4: 200 p_right_2_4 + w_4 - x_2 + x_4 <= 200
non_overlap_right_3_4: 200 p_right_3_4 + w_4 - x_3 + x_4 <= 200
non_overlap_right_rev_0_1: - 200 p_right_0_1 - w_1 + x_0 - x_1 <= 0
non_overlap_right_rev_0_2: - 200 p_right_0_2 - w_2 + x_0 - x_2 <= 0
non_overlap_right_rev_0_3: - 200 p_right_0_3 - w_3 + x_0 - x_3 <= 0
non_overlap_right_rev_0_4: - 200 p_right_0_4 - w_4 + x_0 - x_4 <= 0
non_overlap_right_rev_1_2: - 200 p_right_1_2 - w_2 + x_1 - x_2 <= 0
non_overlap_right_rev_1_3: - 200 p_right_1_3 - w_3 + x_1 - x_3 <= 0
non_overlap_right_rev_1_4: - 200 p_right_1_4 - w_4 + x_1 - x_4 <= 0
non_overlap_right_rev_2_3: - 200 p_right_2_3 - w_3 + x_2 - x_3 <= 0
non_overlap_right_rev_2_4: - 200 p_right_2_4 - w_4 + x_2 - x_4 <= 0
non_overlap_right_rev_3_4: - 200 p_right_3_4 - w_4 + x_3 - x_4 <= 0
non_overlap_up_0_1: h_1 + 200 p_up_0_1 - y_0 + y_1 <= 200
non_overlap_up_0_2: h_2 + 200 p_up_0_2 - y_0 + y_2 <= 200
non_overlap_up_0_3: h_3 + 200 p_up_0_3 - y_0 + y_3 <= 200
non_overlap_up_0_4: h_4 + 200 p_up_0_4 - y_0 + y_4 <= 200
non_overlap_up_1_2: h_2 + 200 p_up_1_2 - y_1 + y_2 <= 200
non_overlap_up_1_3: h_3 + 200 p_up_1_3 - y_1 + y_3 <= 200
non_overlap_up_1_4: h_4 + 200 p_up_1_4 - y_1 + y_4 <= 200
non_overlap_up_2_3: h_3 + 200 p_up_2_3 - y_2 + y_3 <= 200
non_overlap_up_2_4: h_4 + 200 p_up_2_4 - y_2 + y_4 <= 200
non_overlap_up_3_4: h_4 + 200 p_up_3_4 - y_3 + y_4 <= 200
non_overlap_up_rev_0_1: - h_1 - 200 p_up_0_1 + y_0 - y_1 <= 0
non_overlap_up_rev_0_2: - h_2 - 200 p_up_0_2 + y_0 - y_2 <= 0
non_overlap_up_rev_0_3: - h_3 - 200 p_up_0_3 + y_0 - y_3 <= 0
non_overlap_up_rev_0_4: - h_4 - 200 p_up_0_4 + y_0 - y_4 <= 0
non_overlap_up_rev_1_2: - h_2 - 200 p_up_1_2 + y_1 - y_2 <= 0
non_overlap_up_rev_1_3: - h_3 - 200 p_up_1_3 + y_1 - y_3 <= 0
non_overlap_up_rev_1_4: - h_4 - 200 p_up_1_4 + y_1 - y_4 <= 0
non_overlap_up_rev_2_3: - h_3 - 200 p_up_2_3 + y_2 - y_3 <= 0
non_overlap_up_rev_2_4: - h_4 - 200 p_up_2_4 + y_2 - y_4 <= 0
non_overlap_up_rev_3_4: - h_4 - 200 p_up_3_4 + y_3 - y_4 <= 0
not_same_implies_diff_dist_pair_c5_0_1_prev0: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev0 <= 1
not_same_implies_diff_dist_pair_c5_0_1_prev1: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev1 <= 1
not_same_implies_diff_dist_pair_c5_0_1_prev2: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev2 <= 1
not_same_implies_diff_dist_pair_c5_0_1_prev3: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev3 <= 1
not_same_implies_diff_dist_pair_c5_0_1_prev4: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev4 <= 1
not_same_implies_diff_dist_pair_c5_0_1_prev5: diff_dist_pair_c5_0_1
 + same_dist_pair_c5_0_1_prev5 <= 1
not_same_implies_diff_dist_pair_c5_0_2_prev0: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev0 <= 1
not_same_implies_diff_dist_pair_c5_0_2_prev1: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev1 <= 1
not_same_implies_diff_dist_pair_c5_0_2_prev2: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev2 <= 1
not_same_implies_diff_dist_pair_c5_0_2_prev3: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev3 <= 1
not_same_implies_diff_dist_pair_c5_0_2_prev4: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev4 <= 1
not_same_implies_diff_dist_pair_c5_0_2_prev5: diff_dist_pair_c5_0_2
 + same_dist_pair_c5_0_2_prev5 <= 1
not_same_implies_diff_dist_pair_c5_0_3_prev0: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev0 <= 1
not_same_implies_diff_dist_pair_c5_0_3_prev1: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev1 <= 1
not_same_implies_diff_dist_pair_c5_0_3_prev2: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev2 <= 1
not_same_implies_diff_dist_pair_c5_0_3_prev3: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev3 <= 1
not_same_implies_diff_dist_pair_c5_0_3_prev4: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev4 <= 1
not_same_implies_diff_dist_pair_c5_0_3_prev5: diff_dist_pair_c5_0_3
 + same_dist_pair_c5_0_3_prev5 <= 1
not_same_implies_diff_dist_pair_c5_0_4_prev0: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev0 <= 1
not_same_implies_diff_dist_pair_c5_0_4_prev1: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev1 <= 1
not_same_implies_diff_dist_pair_c5_0_4_prev2: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev2 <= 1
not_same_implies_diff_dist_pair_c5_0_4_prev3: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev3 <= 1
not_same_implies_diff_dist_pair_c5_0_4_prev4: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev4 <= 1
not_same_implies_diff_dist_pair_c5_0_4_prev5: diff_dist_pair_c5_0_4
 + same_dist_pair_c5_0_4_prev5 <= 1
not_same_implies_diff_dist_pair_c5_1_2_prev0: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev0 <= 1
not_same_implies_diff_dist_pair_c5_1_2_prev1: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev1 <= 1
not_same_implies_diff_dist_pair_c5_1_2_prev2: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev2 <= 1
not_same_implies_diff_dist_pair_c5_1_2_prev3: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev3 <= 1
not_same_implies_diff_dist_pair_c5_1_2_prev4: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev4 <= 1
not_same_implies_diff_dist_pair_c5_1_2_prev5: diff_dist_pair_c5_1_2
 + same_dist_pair_c5_1_2_prev5 <= 1
not_same_implies_diff_dist_pair_c5_1_3_prev0: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev0 <= 1
not_same_implies_diff_dist_pair_c5_1_3_prev1: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev1 <= 1
not_same_implies_diff_dist_pair_c5_1_3_prev2: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev2 <= 1
not_same_implies_diff_dist_pair_c5_1_3_prev3: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev3 <= 1
not_same_implies_diff_dist_pair_c5_1_3_prev4: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev4 <= 1
not_same_implies_diff_dist_pair_c5_1_3_prev5: diff_dist_pair_c5_1_3
 + same_dist_pair_c5_1_3_prev5 <= 1
not_same_implies_diff_dist_pair_c5_1_4_prev0: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev0 <= 1
not_same_implies_diff_dist_pair_c5_1_4_prev1: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev1 <= 1
not_same_implies_diff_dist_pair_c5_1_4_prev2: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev2 <= 1
not_same_implies_diff_dist_pair_c5_1_4_prev3: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev3 <= 1
not_same_implies_diff_dist_pair_c5_1_4_prev4: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev4 <= 1
not_same_implies_diff_dist_pair_c5_1_4_prev5: diff_dist_pair_c5_1_4
 + same_dist_pair_c5_1_4_prev5 <= 1
not_same_implies_diff_dist_pair_c5_2_3_prev0: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev0 <= 1
not_same_implies_diff_dist_pair_c5_2_3_prev1: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev1 <= 1
not_same_implies_diff_dist_pair_c5_2_3_prev2: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev2 <= 1
not_same_implies_diff_dist_pair_c5_2_3_prev3: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev3 <= 1
not_same_implies_diff_dist_pair_c5_2_3_prev4: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev4 <= 1
not_same_implies_diff_dist_pair_c5_2_3_prev5: diff_dist_pair_c5_2_3
 + same_dist_pair_c5_2_3_prev5 <= 1
not_same_implies_diff_dist_pair_c5_2_4_prev0: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev0 <= 1
not_same_implies_diff_dist_pair_c5_2_4_prev1: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev1 <= 1
not_same_implies_diff_dist_pair_c5_2_4_prev2: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev2 <= 1
not_same_implies_diff_dist_pair_c5_2_4_prev3: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev3 <= 1
not_same_implies_diff_dist_pair_c5_2_4_prev4: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev4 <= 1
not_same_implies_diff_dist_pair_c5_2_4_prev5: diff_dist_pair_c5_2_4
 + same_dist_pair_c5_2_4_prev5 <= 1
not_same_implies_diff_dist_pair_c5_3_4_prev0: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev0 <= 1
not_same_implies_diff_dist_pair_c5_3_4_prev1: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev1 <= 1
not_same_implies_diff_dist_pair_c5_3_4_prev2: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev2 <= 1
not_same_implies_diff_dist_pair_c5_3_4_prev3: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev3 <= 1
not_same_implies_diff_dist_pair_c5_3_4_prev4: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev4 <= 1
not_same_implies_diff_dist_pair_c5_3_4_prev5: diff_dist_pair_c5_3_4
 + same_dist_pair_c5_3_4_prev5 <= 1
same_dist_pair_lower_c5_0_1_prev0: dist_diff_abs_pair_c5_0_1_prev0
 + 120 same_dist_pair_c5_0_1_prev0 >= 5
same_dist_pair_lower_c5_0_1_prev1: dist_diff_abs_pair_c5_0_1_prev1
 + 120 same_dist_pair_c5_0_1_prev1 >= 5
same_dist_pair_lower_c5_0_1_prev2: dist_diff_abs_pair_c5_0_1_prev2
 + 120 same_dist_pair_c5_0_1_prev2 >= 5
same_dist_pair_lower_c5_0_1_prev3: dist_diff_abs_pair_c5_0_1_prev3
 + 120 same_dist_pair_c5_0_1_prev3 >= 5
same_dist_pair_lower_c5_0_1_prev4: dist_diff_abs_pair_c5_0_1_prev4
 + 120 same_dist_pair_c5_0_1_prev4 >= 5
same_dist_pair_lower_c5_0_1_prev5: dist_diff_abs_pair_c5_0_1_prev5
 + 120 same_dist_pair_c5_0_1_prev5 >= 5
same_dist_pair_lower_c5_0_2_prev0: dist_diff_abs_pair_c5_0_2_prev0
 + 120 same_dist_pair_c5_0_2_prev0 >= 5
same_dist_pair_lower_c5_0_2_prev1: dist_diff_abs_pair_c5_0_2_prev1
 + 120 same_dist_pair_c5_0_2_prev1 >= 5
same_dist_pair_lower_c5_0_2_prev2: dist_diff_abs_pair_c5_0_2_prev2
 + 120 same_dist_pair_c5_0_2_prev2 >= 5
same_dist_pair_lower_c5_0_2_prev3: dist_diff_abs_pair_c5_0_2_prev3
 + 120 same_dist_pair_c5_0_2_prev3 >= 5
same_dist_pair_lower_c5_0_2_prev4: dist_diff_abs_pair_c5_0_2_prev4
 + 120 same_dist_pair_c5_0_2_prev4 >= 5
same_dist_pair_lower_c5_0_2_prev5: dist_diff_abs_pair_c5_0_2_prev5
 + 120 same_dist_pair_c5_0_2_prev5 >= 5
same_dist_pair_lower_c5_0_3_prev0: dist_diff_abs_pair_c5_0_3_prev0
 + 120 same_dist_pair_c5_0_3_prev0 >= 5
same_dist_pair_lower_c5_0_3_prev1: dist_diff_abs_pair_c5_0_3_prev1
 + 120 same_dist_pair_c5_0_3_prev1 >= 5
same_dist_pair_lower_c5_0_3_prev2: dist_diff_abs_pair_c5_0_3_prev2
 + 120 same_dist_pair_c5_0_3_prev2 >= 5
same_dist_pair_lower_c5_0_3_prev3: dist_diff_abs_pair_c5_0_3_prev3
 + 120 same_dist_pair_c5_0_3_prev3 >= 5
same_dist_pair_lower_c5_0_3_prev4: dist_diff_abs_pair_c5_0_3_prev4
 + 120 same_dist_pair_c5_0_3_prev4 >= 5
same_dist_pair_lower_c5_0_3_prev5: dist_diff_abs_pair_c5_0_3_prev5
 + 120 same_dist_pair_c5_0_3_prev5 >= 5
same_dist_pair_lower_c5_0_4_prev0: dist_diff_abs_pair_c5_0_4_prev0
 + 120 same_dist_pair_c5_0_4_prev0 >= 5
same_dist_pair_lower_c5_0_4_prev1: dist_diff_abs_pair_c5_0_4_prev1
 + 120 same_dist_pair_c5_0_4_prev1 >= 5
same_dist_pair_lower_c5_0_4_prev2: dist_diff_abs_pair_c5_0_4_prev2
 + 120 same_dist_pair_c5_0_4_prev2 >= 5
same_dist_pair_lower_c5_0_4_prev3: dist_diff_abs_pair_c5_0_4_prev3
 + 120 same_dist_pair_c5_0_4_prev3 >= 5
same_dist_pair_lower_c5_0_4_prev4: dist_diff_abs_pair_c5_0_4_prev4
 + 120 same_dist_pair_c5_0_4_prev4 >= 5
same_dist_pair_lower_c5_0_4_prev5: dist_diff_abs_pair_c5_0_4_prev5
 + 120 same_dist_pair_c5_0_4_prev5 >= 5
same_dist_pair_lower_c5_1_2_prev0: dist_diff_abs_pair_c5_1_2_prev0
 + 120 same_dist_pair_c5_1_2_prev0 >= 5
same_dist_pair_lower_c5_1_2_prev1: dist_diff_abs_pair_c5_1_2_prev1
 + 120 same_dist_pair_c5_1_2_prev1 >= 5
same_dist_pair_lower_c5_1_2_prev2: dist_diff_abs_pair_c5_1_2_prev2
 + 120 same_dist_pair_c5_1_2_prev2 >= 5
same_dist_pair_lower_c5_1_2_prev3: dist_diff_abs_pair_c5_1_2_prev3
 + 120 same_dist_pair_c5_1_2_prev3 >= 5
same_dist_pair_lower_c5_1_2_prev4: dist_diff_abs_pair_c5_1_2_prev4
 + 120 same_dist_pair_c5_1_2_prev4 >= 5
same_dist_pair_lower_c5_1_2_prev5: dist_diff_abs_pair_c5_1_2_prev5
 + 120 same_dist_pair_c5_1_2_prev5 >= 5
same_dist_pair_lower_c5_1_3_prev0: dist_diff_abs_pair_c5_1_3_prev0
 + 120 same_dist_pair_c5_1_3_prev0 >= 5
same_dist_pair_lower_c5_1_3_prev1: dist_diff_abs_pair_c5_1_3_prev1
 + 120 same_dist_pair_c5_1_3_prev1 >= 5
same_dist_pair_lower_c5_1_3_prev2: dist_diff_abs_pair_c5_1_3_prev2
 + 120 same_dist_pair_c5_1_3_prev2 >= 5
same_dist_pair_lower_c5_1_3_prev3: dist_diff_abs_pair_c5_1_3_prev3
 + 120 same_dist_pair_c5_1_3_prev3 >= 5
same_dist_pair_lower_c5_1_3_prev4: dist_diff_abs_pair_c5_1_3_prev4
 + 120 same_dist_pair_c5_1_3_prev4 >= 5
same_dist_pair_lower_c5_1_3_prev5: dist_diff_abs_pair_c5_1_3_prev5
 + 120 same_dist_pair_c5_1_3_prev5 >= 5
same_dist_pair_lower_c5_1_4_prev0: dist_diff_abs_pair_c5_1_4_prev0
 + 120 same_dist_pair_c5_1_4_prev0 >= 5
same_dist_pair_lower_c5_1_4_prev1: dist_diff_abs_pair_c5_1_4_prev1
 + 120 same_dist_pair_c5_1_4_prev1 >= 5
same_dist_pair_lower_c5_1_4_prev2: dist_diff_abs_pair_c5_1_4_prev2
 + 120 same_dist_pair_c5_1_4_prev2 >= 5
same_dist_pair_lower_c5_1_4_prev3: dist_diff_abs_pair_c5_1_4_prev3
 + 120 same_dist_pair_c5_1_4_prev3 >= 5
same_dist_pair_lower_c5_1_4_prev4: dist_diff_abs_pair_c5_1_4_prev4
 + 120 same_dist_pair_c5_1_4_prev4 >= 5
same_dist_pair_lower_c5_1_4_prev5: dist_diff_abs_pair_c5_1_4_prev5
 + 120 same_dist_pair_c5_1_4_prev5 >= 5
same_dist_pair_lower_c5_2_3_prev0: dist_diff_abs_pair_c5_2_3_prev0
 + 120 same_dist_pair_c5_2_3_prev0 >= 5
same_dist_pair_lower_c5_2_3_prev1: dist_diff_abs_pair_c5_2_3_prev1
 + 120 same_dist_pair_c5_2_3_prev1 >= 5
same_dist_pair_lower_c5_2_3_prev2: dist_diff_abs_pair_c5_2_3_prev2
 + 120 same_dist_pair_c5_2_3_prev2 >= 5
same_dist_pair_lower_c5_2_3_prev3: dist_diff_abs_pair_c5_2_3_prev3
 + 120 same_dist_pair_c5_2_3_prev3 >= 5
same_dist_pair_lower_c5_2_3_prev4: dist_diff_abs_pair_c5_2_3_prev4
 + 120 same_dist_pair_c5_2_3_prev4 >= 5
same_dist_pair_lower_c5_2_3_prev5: dist_diff_abs_pair_c5_2_3_prev5
 + 120 same_dist_pair_c5_2_3_prev5 >= 5
same_dist_pair_lower_c5_2_4_prev0: dist_diff_abs_pair_c5_2_4_prev0
 + 120 same_dist_pair_c5_2_4_prev0 >= 5
same_dist_pair_lower_c5_2_4_prev1: dist_diff_abs_pair_c5_2_4_prev1
 + 120 same_dist_pair_c5_2_4_prev1 >= 5
same_dist_pair_lower_c5_2_4_prev2: dist_diff_abs_pair_c5_2_4_prev2
 + 120 same_dist_pair_c5_2_4_prev2 >= 5
same_dist_pair_lower_c5_2_4_prev3: dist_diff_abs_pair_c5_2_4_prev3
 + 120 same_dist_pair_c5_2_4_prev3 >= 5
same_dist_pair_lower_c5_2_4_prev4: dist_diff_abs_pair_c5_2_4_prev4
 + 120 same_dist_pair_c5_2_4_prev4 >= 5
same_dist_pair_lower_c5_2_4_prev5: dist_diff_abs_pair_c5_2_4_prev5
 + 120 same_dist_pair_c5_2_4_prev5 >= 5
same_dist_pair_lower_c5_3_4_prev0: dist_diff_abs_pair_c5_3_4_prev0
 + 120 same_dist_pair_c5_3_4_prev0 >= 5
same_dist_pair_lower_c5_3_4_prev1: dist_diff_abs_pair_c5_3_4_prev1
 + 120 same_dist_pair_c5_3_4_prev1 >= 5
same_dist_pair_lower_c5_3_4_prev2: dist_diff_abs_pair_c5_3_4_prev2
 + 120 same_dist_pair_c5_3_4_prev2 >= 5
same_dist_pair_lower_c5_3_4_prev3: dist_diff_abs_pair_c5_3_4_prev3
 + 120 same_dist_pair_c5_3_4_prev3 >= 5
same_dist_pair_lower_c5_3_4_prev4: dist_diff_abs_pair_c5_3_4_prev4
 + 120 same_dist_pair_c5_3_4_prev4 >= 5
same_dist_pair_lower_c5_3_4_prev5: dist_diff_abs_pair_c5_3_4_prev5
 + 120 same_dist_pair_c5_3_4_prev5 >= 5
same_dist_pair_upper_c5_0_1_prev0: dist_diff_abs_pair_c5_0_1_prev0
 + 120 same_dist_pair_c5_0_1_prev0 <= 124.95
same_dist_pair_upper_c5_0_1_prev1: dist_diff_abs_pair_c5_0_1_prev1
 + 120 same_dist_pair_c5_0_1_prev1 <= 124.95
same_dist_pair_upper_c5_0_1_prev2: dist_diff_abs_pair_c5_0_1_prev2
 + 120 same_dist_pair_c5_0_1_prev2 <= 124.95
same_dist_pair_upper_c5_0_1_prev3: dist_diff_abs_pair_c5_0_1_prev3
 + 120 same_dist_pair_c5_0_1_prev3 <= 124.95
same_dist_pair_upper_c5_0_1_prev4: dist_diff_abs_pair_c5_0_1_prev4
 + 120 same_dist_pair_c5_0_1_prev4 <= 124.95
same_dist_pair_upper_c5_0_1_prev5: dist_diff_abs_pair_c5_0_1_prev5
 + 120 same_dist_pair_c5_0_1_prev5 <= 124.95
same_dist_pair_upper_c5_0_2_prev0: dist_diff_abs_pair_c5_0_2_prev0
 + 120 same_dist_pair_c5_0_2_prev0 <= 124.95
same_dist_pair_upper_c5_0_2_prev1: dist_diff_abs_pair_c5_0_2_prev1
 + 120 same_dist_pair_c5_0_2_prev1 <= 124.95
same_dist_pair_upper_c5_0_2_prev2: dist_diff_abs_pair_c5_0_2_prev2
 + 120 same_dist_pair_c5_0_2_prev2 <= 124.95
same_dist_pair_upper_c5_0_2_prev3: dist_diff_abs_pair_c5_0_2_prev3
 + 120 same_dist_pair_c5_0_2_prev3 <= 124.95
same_dist_pair_upper_c5_0_2_prev4: dist_diff_abs_pair_c5_0_2_prev4
 + 120 same_dist_pair_c5_0_2_prev4 <= 124.95
same_dist_pair_upper_c5_0_2_prev5: dist_diff_abs_pair_c5_0_2_prev5
 + 120 same_dist_pair_c5_0_2_prev5 <= 124.95
same_dist_pair_upper_c5_0_3_prev0: dist_diff_abs_pair_c5_0_3_prev0
 + 120 same_dist_pair_c5_0_3_prev0 <= 124.95
same_dist_pair_upper_c5_0_3_prev1: dist_diff_abs_pair_c5_0_3_prev1
 + 120 same_dist_pair_c5_0_3_prev1 <= 124.95
same_dist_pair_upper_c5_0_3_prev2: dist_diff_abs_pair_c5_0_3_prev2
 + 120 same_dist_pair_c5_0_3_prev2 <= 124.95
same_dist_pair_upper_c5_0_3_prev3: dist_diff_abs_pair_c5_0_3_prev3
 + 120 same_dist_pair_c5_0_3_prev3 <= 124.95
same_dist_pair_upper_c5_0_3_prev4: dist_diff_abs_pair_c5_0_3_prev4
 + 120 same_dist_pair_c5_0_3_prev4 <= 124.95
same_dist_pair_upper_c5_0_3_prev5: dist_diff_abs_pair_c5_0_3_prev5
 + 120 same_dist_pair_c5_0_3_prev5 <= 124.95
same_dist_pair_upper_c5_0_4_prev0: dist_diff_abs_pair_c5_0_4_prev0
 + 120 same_dist_pair_c5_0_4_prev0 <= 124.95
same_dist_pair_upper_c5_0_4_prev1: dist_diff_abs_pair_c5_0_4_prev1
 + 120 same_dist_pair_c5_0_4_prev1 <= 124.95
same_dist_pair_upper_c5_0_4_prev2: dist_diff_abs_pair_c5_0_4_prev2
 + 120 same_dist_pair_c5_0_4_prev2 <= 124.95
same_dist_pair_upper_c5_0_4_prev3: dist_diff_abs_pair_c5_0_4_prev3
 + 120 same_dist_pair_c5_0_4_prev3 <= 124.95
same_dist_pair_upper_c5_0_4_prev4: dist_diff_abs_pair_c5_0_4_prev4
 + 120 same_dist_pair_c5_0_4_prev4 <= 124.95
same_dist_pair_upper_c5_0_4_prev5: dist_diff_abs_pair_c5_0_4_prev5
 + 120 same_dist_pair_c5_0_4_prev5 <= 124.95
same_dist_pair_upper_c5_1_2_prev0: dist_diff_abs_pair_c5_1_2_prev0
 + 120 same_dist_pair_c5_1_2_prev0 <= 124.95
same_dist_pair_upper_c5_1_2_prev1: dist_diff_abs_pair_c5_1_2_prev1
 + 120 same_dist_pair_c5_1_2_prev1 <= 124.95
same_dist_pair_upper_c5_1_2_prev2: dist_diff_abs_pair_c5_1_2_prev2
 + 120 same_dist_pair_c5_1_2_prev2 <= 124.95
same_dist_pair_upper_c5_1_2_prev3: dist_diff_abs_pair_c5_1_2_prev3
 + 120 same_dist_pair_c5_1_2_prev3 <= 124.95
same_dist_pair_upper_c5_1_2_prev4: dist_diff_abs_pair_c5_1_2_prev4
 + 120 same_dist_pair_c5_1_2_prev4 <= 124.95
same_dist_pair_upper_c5_1_2_prev5: dist_diff_abs_pair_c5_1_2_prev5
 + 120 same_dist_pair_c5_1_2_prev5 <= 124.95
same_dist_pair_upper_c5_1_3_prev0: dist_diff_abs_pair_c5_1_3_prev0
 + 120 same_dist_pair_c5_1_3_prev0 <= 124.95
same_dist_pair_upper_c5_1_3_prev1: dist_diff_abs_pair_c5_1_3_prev1
 + 120 same_dist_pair_c5_1_3_prev1 <= 124.95
same_dist_pair_upper_c5_1_3_prev2: dist_diff_abs_pair_c5_1_3_prev2
 + 120 same_dist_pair_c5_1_3_prev2 <= 124.95
same_dist_pair_upper_c5_1_3_prev3: dist_diff_abs_pair_c5_1_3_prev3
 + 120 same_dist_pair_c5_1_3_prev3 <= 124.95
same_dist_pair_upper_c5_1_3_prev4: dist_diff_abs_pair_c5_1_3_prev4
 + 120 same_dist_pair_c5_1_3_prev4 <= 124.95
same_dist_pair_upper_c5_1_3_prev5: dist_diff_abs_pair_c5_1_3_prev5
 + 120 same_dist_pair_c5_1_3_prev5 <= 124.95
same_dist_pair_upper_c5_1_4_prev0: dist_diff_abs_pair_c5_1_4_prev0
 + 120 same_dist_pair_c5_1_4_prev0 <= 124.95
same_dist_pair_upper_c5_1_4_prev1: dist_diff_abs_pair_c5_1_4_prev1
 + 120 same_dist_pair_c5_1_4_prev1 <= 124.95
same_dist_pair_upper_c5_1_4_prev2: dist_diff_abs_pair_c5_1_4_prev2
 + 120 same_dist_pair_c5_1_4_prev2 <= 124.95
same_dist_pair_upper_c5_1_4_prev3: dist_diff_abs_pair_c5_1_4_prev3
 + 120 same_dist_pair_c5_1_4_prev3 <= 124.95
same_dist_pair_upper_c5_1_4_prev4: dist_diff_abs_pair_c5_1_4_prev4
 + 120 same_dist_pair_c5_1_4_prev4 <= 124.95
same_dist_pair_upper_c5_1_4_prev5: dist_diff_abs_pair_c5_1_4_prev5
 + 120 same_dist_pair_c5_1_4_prev5 <= 124.95
same_dist_pair_upper_c5_2_3_prev0: dist_diff_abs_pair_c5_2_3_prev0
 + 120 same_dist_pair_c5_2_3_prev0 <= 124.95
same_dist_pair_upper_c5_2_3_prev1: dist_diff_abs_pair_c5_2_3_prev1
 + 120 same_dist_pair_c5_2_3_prev1 <= 124.95
same_dist_pair_upper_c5_2_3_prev2: dist_diff_abs_pair_c5_2_3_prev2
 + 120 same_dist_pair_c5_2_3_prev2 <= 124.95
same_dist_pair_upper_c5_2_3_prev3: dist_diff_abs_pair_c5_2_3_prev3
 + 120 same_dist_pair_c5_2_3_prev3 <= 124.95
same_dist_pair_upper_c5_2_3_prev4: dist_diff_abs_pair_c5_2_3_prev4
 + 120 same_dist_pair_c5_2_3_prev4 <= 124.95
same_dist_pair_upper_c5_2_3_prev5: dist_diff_abs_pair_c5_2_3_prev5
 + 120 same_dist_pair_c5_2_3_prev5 <= 124.95
same_dist_pair_upper_c5_2_4_prev0: dist_diff_abs_pair_c5_2_4_prev0
 + 120 same_dist_pair_c5_2_4_prev0 <= 124.95
same_dist_pair_upper_c5_2_4_prev1: dist_diff_abs_pair_c5_2_4_prev1
 + 120 same_dist_pair_c5_2_4_prev1 <= 124.95
same_dist_pair_upper_c5_2_4_prev2: dist_diff_abs_pair_c5_2_4_prev2
 + 120 same_dist_pair_c5_2_4_prev2 <= 124.95
same_dist_pair_upper_c5_2_4_prev3: dist_diff_abs_pair_c5_2_4_prev3
 + 120 same_dist_pair_c5_2_4_prev3 <= 124.95
same_dist_pair_upper_c5_2_4_prev4: dist_diff_abs_pair_c5_2_4_prev4
 + 120 same_dist_pair_c5_2_4_prev4 <= 124.95
same_dist_pair_upper_c5_2_4_prev5: dist_diff_abs_pair_c5_2_4_prev5
 + 120 same_dist_pair_c5_2_4_prev5 <= 124.95
same_dist_pair_upper_c5_3_4_prev0: dist_diff_abs_pair_c5_3_4_prev0
 + 120 same_dist_pair_c5_3_4_prev0 <= 124.95
same_dist_pair_upper_c5_3_4_prev1: dist_diff_abs_pair_c5_3_4_prev1
 + 120 same_dist_pair_c5_3_4_prev1 <= 124.95
same_dist_pair_upper_c5_3_4_prev2: dist_diff_abs_pair_c5_3_4_prev2
 + 120 same_dist_pair_c5_3_4_prev2 <= 124.95
same_dist_pair_upper_c5_3_4_prev3: dist_diff_abs_pair_c5_3_4_prev3
 + 120 same_dist_pair_c5_3_4_prev3 <= 124.95
same_dist_pair_upper_c5_3_4_prev4: dist_diff_abs_pair_c5_3_4_prev4
 + 120 same_dist_pair_c5_3_4_prev4 <= 124.95
same_dist_pair_upper_c5_3_4_prev5: dist_diff_abs_pair_c5_3_4_prev5
 + 120 same_dist_pair_c5_3_4_prev5 <= 124.95
shared_x_min_0_1: shared_x_0_1 - 200 z2_0_1 >= -199.9
shared_x_min_1_2: shared_x_1_2 - 200 z2_1_2 >= -199.9
shared_x_min_1_3: shared_x_1_3 - 200 z2_1_3 >= -199.9
shared_x_min_2_4: shared_x_2_4 - 200 z2_2_4 >= -199.9
shared_x_ub1_0_1: shared_x_0_1 - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
shared_x_ub1_1_2: shared_x_1_2 - w_1 - x_1 + x_2 + 200 z2_1_2 <= 200
shared_x_ub1_1_3: shared_x_1_3 - w_1 - x_1 + x_3 + 200 z2_1_3 <= 200
shared_x_ub1_2_4: shared_x_2_4 - w_2 - x_2 + x_4 + 200 z2_2_4 <= 200
shared_x_ub2_0_1: shared_x_0_1 - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
shared_x_ub2_1_2: shared_x_1_2 - w_2 + x_1 - x_2 + 200 z2_1_2 <= 200
shared_x_ub2_1_3: shared_x_1_3 - w_3 + x_1 - x_3 + 200 z2_1_3 <= 200
shared_x_ub2_2_4: shared_x_2_4 - w_4 + x_2 - x_4 + 200 z2_2_4 <= 200
shared_x_zero_0_1: shared_x_0_1 - 200 z2_0_1 <= 0
shared_x_zero_1_2: shared_x_1_2 - 200 z2_1_2 <= 0
shared_x_zero_1_3: shared_x_1_3 - 200 z2_1_3 <= 0
shared_x_zero_2_4: shared_x_2_4 - 200 z2_2_4 <= 0
shared_y_min_0_1: shared_y_0_1 - 200 z1_0_1 >= -199.9
shared_y_min_1_2: shared_y_1_2 - 200 z1_1_2 >= -199.9
shared_y_min_1_3: shared_y_1_3 - 200 z1_1_3 >= -199.9
shared_y_min_2_4: shared_y_2_4 - 200 z1_2_4 >= -199.9
shared_y_ub1_0_1: - h_0 + shared_y_0_1 - y_0 + y_1 + 200 z1_0_1 <= 200
shared_y_ub1_1_2: - h_1 + shared_y_1_2 - y_1 + y_2 + 200 z1_1_2 <= 200
shared_y_ub1_1_3: - h_1 + shared_y_1_3 - y_1 + y_3 + 200 z1_1_3 <= 200
shared_y_ub1_2_4: - h_2 + shared_y_2_4 - y_2 + y_4 + 200 z1_2_4 <= 200
shared_y_ub2_0_1: - h_1 + shared_y_0_1 + y_0 - y_1 + 200 z1_0_1 <= 200
shared_y_ub2_1_2: - h_2 + shared_y_1_2 + y_1 - y_2 + 200 z1_1_2 <= 200
shared_y_ub2_1_3: - h_3 + shared_y_1_3 + y_1 - y_3 + 200 z1_1_3 <= 200
shared_y_ub2_2_4: - h_4 + shared_y_2_4 + y_2 - y_4 + 200 z1_2_4 <= 200
shared_y_zero_0_1: shared_y_0_1 - 200 z1_0_1 <= 0
shared_y_zero_1_2: shared_y_1_2 - 200 z1_1_2 <= 0
shared_y_zero_1_3: shared_y_1_3 - 200 z1_1_3 <= 0
shared_y_zero_2_4: shared_y_2_4 - 200 z1_2_4 <= 0
t_ge_height: bbox_area_proxy_t - bbox_h >= 0
t_ge_scaled_mean: bbox_area_proxy_t - 0.8 bbox_h - 0.8 bbox_w >= 0
t_ge_width: bbox_area_proxy_t - bbox_w >= 0
vertical_direction_0_1: z2D_0_1 + z2U_0_1 - z2_0_1 = 0
vertical_direction_1_2: z2D_1_2 + z2U_1_2 - z2_1_2 = 0
vertical_direction_1_3: z2D_1_3 + z2U_1_3 - z2_1_3 = 0
vertical_direction_2_4: z2D_2_4 + z2U_2_4 - z2_2_4 = 0
vertical_down_dist_0_1: - h_0 - y_0 + y_1 + 200 z2D_0_1 <= 200.5
vertical_down_dist_1_2: - h_1 - y_1 + y_2 + 200 z2D_1_2 <= 200.5
vertical_down_dist_1_3: - h_1 - y_1 + y_3 + 200 z2D_1_3 <= 200.5
vertical_down_dist_2_4: - h_2 - y_2 + y_4 + 200 z2D_2_4 <= 200.5
vertical_down_dist_lb_0_1: - h_0 - y_0 + y_1 - 200 z2D_0_1 >= -200
vertical_down_dist_lb_1_2: - h_1 - y_1 + y_2 - 200 z2D_1_2 >= -200
vertical_down_dist_lb_1_3: - h_1 - y_1 + y_3 - 200 z2D_1_3 >= -200
vertical_down_dist_lb_2_4: - h_2 - y_2 + y_4 - 200 z2D_2_4 >= -200
vertical_overlap_x1_0_1: - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
vertical_overlap_x1_1_2: - w_2 + x_1 - x_2 + 200 z2_1_2 <= 200
vertical_overlap_x1_1_3: - w_3 + x_1 - x_3 + 200 z2_1_3 <= 200
vertical_overlap_x1_2_4: - w_4 + x_2 - x_4 + 200 z2_2_4 <= 200
vertical_overlap_x2_0_1: - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
vertical_overlap_x2_1_2: - w_1 - x_1 + x_2 + 200 z2_1_2 <= 200
vertical_overlap_x2_1_3: - w_1 - x_1 + x_3 + 200 z2_1_3 <= 200
vertical_overlap_x2_2_4: - w_2 - x_2 + x_4 + 200 z2_2_4 <= 200
vertical_up_dist_0_1: - h_1 + y_0 - y_1 + 200 z2U_0_1 <= 200.5
vertical_up_dist_1_2: - h_2 + y_1 - y_2 + 200 z2U_1_2 <= 200.5
vertical_up_dist_1_3: - h_3 + y_1 - y_3 + 200 z2U_1_3 <= 200.5
vertical_up_dist_2_4: - h_4 + y_2 - y_4 + 200 z2U_2_4 <= 200.5
vertical_up_dist_lb_0_1: - h_1 + y_0 - y_1 - 200 z2U_0_1 >= -200
vertical_up_dist_lb_1_2: - h_2 + y_1 - y_2 - 200 z2U_1_2 >= -200
vertical_up_dist_lb_1_3: - h_3 + y_1 - y_3 - 200 z2U_1_3 >= -200
vertical_up_dist_lb_2_4: - h_4 + y_2 - y_4 - 200 z2U_2_4 >= -200
w_align_grid_0: w_0 = 16
w_align_grid_1: w_1 = 10
w_align_grid_2: w_2 = 20
w_align_grid_3: w_3 = 16
w_align_grid_4: w_4 = 4
w_max_0: w_0 <= 60
w_max_1: w_1 <= 60
w_max_2: w_2 <= 60
w_max_3: w_3 <= 60
w_max_4: w_4 <= 60
width_rotation_0: w_0 = 16
width_rotation_1: - 10 r_1 + w_1 = 10
width_rotation_2: w_2 = 20
width_rotation_3: 2 r_3 + w_3 = 16
width_rotation_4: w_4 = 4
x_grid_to_real_0: x_0 - 0.5 x_grid_0 = 0
x_grid_to_real_1: x_1 - 0.5 x_grid_1 = 0
x_grid_to_real_2: x_2 - 0.5 x_grid_2 = 0
x_grid_to_real_3: x_3 - 0.5 x_grid_3 = 0
x_grid_to_real_4: x_4 - 0.5 x_grid_4 = 0
y_grid_to_real_0: y_0 - 0.5 y_grid_0 = 0
y_grid_to_real_1: y_1 - 0.5 y_grid_1 = 0
y_grid_to_real_2: y_2 - 0.5 y_grid_2 = 0
y_grid_to_real_3: y_3 - 0.5 y_grid_3 = 0
y_grid_to_real_4: y_4 - 0.5 y_grid_4 = 0
Bounds
 bbox_area_proxy_t <= 120
 bbox_h <= 60
 bbox_max_x <= 60
 bbox_max_y <= 60
 bbox_min_x <= 60
 bbox_min_y <= 60
 bbox_w <= 60
 cx_0 <= 60
 cx_1 <= 60
 cx_2 <= 60
 cx_3 <= 60
 cx_4 <= 60
 cy_0 <= 60
 cy_1 <= 60
 cy_2 <= 60
 cy_3 <= 60
 cy_4 <= 60
 dist_curr_pair_c5_0_1 <= 240
 dist_curr_pair_c5_0_2 <= 240
 dist_curr_pair_c5_0_3 <= 240
 dist_curr_pair_c5_0_4 <= 240
 dist_curr_pair_c5_1_2 <= 240
 dist_curr_pair_c5_1_3 <= 240
 dist_curr_pair_c5_1_4 <= 240
 dist_curr_pair_c5_2_3 <= 240
 dist_curr_pair_c5_2_4 <= 240
 dist_curr_pair_c5_3_4 <= 240
 dist_diff_abs_pair_c5_0_1_prev0 <= 240
 dist_diff_abs_pair_c5_0_1_prev1 <= 240
 dist_diff_abs_pair_c5_0_1_prev2 <= 240
 dist_diff_abs_pair_c5_0_1_prev3 <= 240
 dist_diff_abs_pair_c5_0_1_prev4 <= 240
 dist_diff_abs_pair_c5_0_1_prev5 <= 240
 dist_diff_abs_pair_c5_0_2_prev0 <= 240
 dist_diff_abs_pair_c5_0_2_prev1 <= 240
 dist_diff_abs_pair_c5_0_2_prev2 <= 240
 dist_diff_abs_pair_c5_0_2_prev3 <= 240
 dist_diff_abs_pair_c5_0_2_prev4 <= 240
 dist_diff_abs_pair_c5_0_2_prev5 <= 240
 dist_diff_abs_pair_c5_0_3_prev0 <= 240
 dist_diff_abs_pair_c5_0_3_prev1 <= 240
 dist_diff_abs_pair_c5_0_3_prev2 <= 240
 dist_diff_abs_pair_c5_0_3_prev3 <= 240
 dist_diff_abs_pair_c5_0_3_prev4 <= 240
 dist_diff_abs_pair_c5_0_3_prev5 <= 240
 dist_diff_abs_pair_c5_0_4_prev0 <= 240
 dist_diff_abs_pair_c5_0_4_prev1 <= 240
 dist_diff_abs_pair_c5_0_4_prev2 <= 240
 dist_diff_abs_pair_c5_0_4_prev3 <= 240
 dist_diff_abs_pair_c5_0_4_prev4 <= 240
 dist_diff_abs_pair_c5_0_4_prev5 <= 240
 dist_diff_abs_pair_c5_1_2_prev0 <= 240
 dist_diff_abs_pair_c5_1_2_prev1 <= 240
 dist_diff_abs_pair_c5_1_2_prev2 <= 240
 dist_diff_abs_pair_c5_1_2_prev3 <= 240
 dist_diff_abs_pair_c5_1_2_prev4 <= 240
 dist_diff_abs_pair_c5_1_2_prev5 <= 240
 dist_diff_abs_pair_c5_1_3_prev0 <= 240
 dist_diff_abs_pair_c5_1_3_prev1 <= 240
 dist_diff_abs_pair_c5_1_3_prev2 <= 240
 dist_diff_abs_pair_c5_1_3_prev3 <= 240
 dist_diff_abs_pair_c5_1_3_prev4 <= 240
 dist_diff_abs_pair_c5_1_3_prev5 <= 240
 dist_diff_abs_pair_c5_1_4_prev0 <= 240
 dist_diff_abs_pair_c5_1_4_prev1 <= 240
 dist_diff_abs_pair_c5_1_4_prev2 <= 240
 dist_diff_abs_pair_c5_1_4_prev3 <= 240
 dist_diff_abs_pair_c5_1_4_prev4 <= 240
 dist_diff_abs_pair_c5_1_4_prev5 <= 240
 dist_diff_abs_pair_c5_2_3_prev0 <= 240
 dist_diff_abs_pair_c5_2_3_prev1 <= 240
 dist_diff_abs_pair_c5_2_3_prev2 <= 240
 dist_diff_abs_pair_c5_2_3_prev3 <= 240
 dist_diff_abs_pair_c5_2_3_prev4 <= 240
 dist_diff_abs_pair_c5_2_3_prev5 <= 240
 dist_diff_abs_pair_c5_2_4_prev0 <= 240
 dist_diff_abs_pair_c5_2_4_prev1 <= 240
 dist_diff_abs_pair_c5_2_4_prev2 <= 240
 dist_diff_abs_pair_c5_2_4_prev3 <= 240
 dist_diff_abs_pair_c5_2_4_prev4 <= 240
 dist_diff_abs_pair_c5_2_4_prev5 <= 240
 dist_diff_abs_pair_c5_3_4_prev0 <= 240
 dist_diff_abs_pair_c5_3_4_prev1 <= 240
 dist_diff_abs_pair_c5_3_4_prev2 <= 240
 dist_diff_abs_pair_c5_3_4_prev3 <= 240
 dist_diff_abs_pair_c5_3_4_prev4 <= 240
 dist_diff_abs_pair_c5_3_4_prev5 <= 240
 -240 <= dist_diff_pair_c5_0_1_prev0 <= 240
 -240 <= dist_diff_pair_c5_0_1_prev1 <= 240
 -240 <= dist_diff_pair_c5_0_1_prev2 <= 240
 -240 <= dist_diff_pair_c5_0_1_prev3 <= 240
 -240 <= dist_diff_pair_c5_0_1_prev4 <= 240
 -240 <= dist_diff_pair_c5_0_1_prev5 <= 240
 -240 <= dist_diff_pair_c5_0_2_prev0 <= 240
 -240 <= dist_diff_pair_c5_0_2_prev1 <= 240
 -240 <= dist_diff_pair_c5_0_2_prev2 <= 240
 -240 <= dist_diff_pair_c5_0_2_prev3 <= 240
 -240 <= dist_diff_pair_c5_0_2_prev4 <= 240
 -240 <= dist_diff_pair_c5_0_2_prev5 <= 240
 -240 <= dist_diff_pair_c5_0_3_prev0 <= 240
 -240 <= dist_diff_pair_c5_0_3_prev1 <= 240
 -240 <= dist_diff_pair_c5_0_3_prev2 <= 240
 -240 <= dist_diff_pair_c5_0_3_prev3 <= 240
 -240 <= dist_diff_pair_c5_0_3_prev4 <= 240
 -240 <= dist_diff_pair_c5_0_3_prev5 <= 240
 -240 <= dist_diff_pair_c5_0_4_prev0 <= 240
 -240 <= dist_diff_pair_c5_0_4_prev1 <= 240
 -240 <= dist_diff_pair_c5_0_4_prev2 <= 240
 -240 <= dist_diff_pair_c5_0_4_prev3 <= 240
 -240 <= dist_diff_pair_c5_0_4_prev4 <= 240
 -240 <= dist_diff_pair_c5_0_4_prev5 <= 240
 -240 <= dist_diff_pair_c5_1_2_prev0 <= 240
 -240 <= dist_diff_pair_c5_1_2_prev1 <= 240
 -240 <= dist_diff_pair_c5_1_2_prev2 <= 240
 -240 <= dist_diff_pair_c5_1_2_prev3 <= 240
 -240 <= dist_diff_pair_c5_1_2_prev4 <= 240
 -240 <= dist_diff_pair_c5_1_2_prev5 <= 240
 -240 <= dist_diff_pair_c5_1_3_prev0 <= 240
 -240 <= dist_diff_pair_c5_1_3_prev1 <= 240
 -240 <= dist_diff_pair_c5_1_3_prev2 <= 240
 -240 <= dist_diff_pair_c5_1_3_prev3 <= 240
 -240 <= dist_diff_pair_c5_1_3_prev4 <= 240
 -240 <= dist_diff_pair_c5_1_3_prev5 <= 240
 -240 <= dist_diff_pair_c5_1_4_prev0 <= 240
 -240 <= dist_diff_pair_c5_1_4_prev1 <= 240
 -240 <= dist_diff_pair_c5_1_4_prev2 <= 240
 -240 <= dist_diff_pair_c5_1_4_prev3 <= 240
 -240 <= dist_diff_pair_c5_1_4_prev4 <= 240
 -240 <= dist_diff_pair_c5_1_4_prev5 <= 240
 -240 <= dist_diff_pair_c5_2_3_prev0 <= 240
 -240 <= dist_diff_pair_c5_2_3_prev1 <= 240
 -240 <= dist_diff_pair_c5_2_3_prev2 <= 240
 -240 <= dist_diff_pair_c5_2_3_prev3 <= 240
 -240 <= dist_diff_pair_c5_2_3_prev4 <= 240
 -240 <= dist_diff_pair_c5_2_3_prev5 <= 240
 -240 <= dist_diff_pair_c5_2_4_prev0 <= 240
 -240 <= dist_diff_pair_c5_2_4_prev1 <= 240
 -240 <= dist_diff_pair_c5_2_4_prev2 <= 240
 -240 <= dist_diff_pair_c5_2_4_prev3 <= 240
 -240 <= dist_diff_pair_c5_2_4_prev4 <= 240
 -240 <= dist_diff_pair_c5_2_4_prev5 <= 240
 -240 <= dist_diff_pair_c5_3_4_prev0 <= 240
 -240 <= dist_diff_pair_c5_3_4_prev1 <= 240
 -240 <= dist_diff_pair_c5_3_4_prev2 <= 240
 -240 <= dist_diff_pair_c5_3_4_prev3 <= 240
 -240 <= dist_diff_pair_c5_3_4_prev4 <= 240
 -240 <= dist_diff_pair_c5_3_4_prev5 <= 240
 -60 <= dx_diff_0_1 <= 60
 -60 <= dx_diff_1_2 <= 60
 -60 <= dx_diff_1_3 <= 60
 -60 <= dx_diff_2_4 <= 60
 dx_grid_abs_pair_c5_0_1 <= 119
 dx_grid_abs_pair_c5_0_2 <= 119
 dx_grid_abs_pair_c5_0_3 <= 119
 dx_grid_abs_pair_c5_0_4 <= 119
 dx_grid_abs_pair_c5_1_2 <= 119
 dx_grid_abs_pair_c5_1_3 <= 119
 dx_grid_abs_pair_c5_1_4 <= 119
 dx_grid_abs_pair_c5_2_3 <= 119
 dx_grid_abs_pair_c5_2_4 <= 119
 dx_grid_abs_pair_c5_3_4 <= 119
 -119 <= dx_grid_diff_c5_0_1 <= 119
 -119 <= dx_grid_diff_c5_0_2 <= 119
 -119 <= dx_grid_diff_c5_0_3 <= 119
 -119 <= dx_grid_diff_c5_0_4 <= 119
 -119 <= dx_grid_diff_c5_1_2 <= 119
 -119 <= dx_grid_diff_c5_1_3 <= 119
 -119 <= dx_grid_diff_c5_1_4 <= 119
 -119 <= dx_grid_diff_c5_2_3 <= 119
 -119 <= dx_grid_diff_c5_2_4 <= 119
 -119 <= dx_grid_diff_c5_3_4 <= 119
 -60 <= dy_diff_0_1 <= 60
 -60 <= dy_diff_1_2 <= 60
 -60 <= dy_diff_1_3 <= 60
 -60 <= dy_diff_2_4 <= 60
 dy_grid_abs_pair_c5_0_1 <= 119
 dy_grid_abs_pair_c5_0_2 <= 119
 dy_grid_abs_pair_c5_0_3 <= 119
 dy_grid_abs_pair_c5_0_4 <= 119
 dy_grid_abs_pair_c5_1_2 <= 119
 dy_grid_abs_pair_c5_1_3 <= 119
 dy_grid_abs_pair_c5_1_4 <= 119
 dy_grid_abs_pair_c5_2_3 <= 119
 dy_grid_abs_pair_c5_2_4 <= 119
 dy_grid_abs_pair_c5_3_4 <= 119
 -119 <= dy_grid_diff_c5_0_1 <= 119
 -119 <= dy_grid_diff_c5_0_2 <= 119
 -119 <= dy_grid_diff_c5_0_3 <= 119
 -119 <= dy_grid_diff_c5_0_4 <= 119
 -119 <= dy_grid_diff_c5_1_2 <= 119
 -119 <= dy_grid_diff_c5_1_3 <= 119
 -119 <= dy_grid_diff_c5_1_4 <= 119
 -119 <= dy_grid_diff_c5_2_3 <= 119
 -119 <= dy_grid_diff_c5_2_4 <= 119
 -119 <= dy_grid_diff_c5_3_4 <= 119
 h_0 = 16
 10 <= h_1 <= 20
 h_2 = 20
 14 <= h_3 <= 16
 h_4 = 4
 shared_x_0_1 <= 16
 shared_x_1_2 <= 20
 shared_x_1_3 <= 16
 shared_x_2_4 <= 20
 shared_y_0_1 <= 20
 shared_y_1_2 <= 20
 shared_y_1_3 <= 20
 shared_y_2_4 <= 20
 w_0 = 16
 10 <= w_1 <= 20
 w_2 = 20
 14 <= w_3 <= 16
 w_4 = 4
 x_0 <= 60
 x_1 <= 60
 x_2 <= 60
 x_3 <= 60
 x_4 <= 60
 0 <= x_grid_0 <= 119
 0 <= x_grid_1 <= 119
 0 <= x_grid_2 <= 119
 0 <= x_grid_3 <= 119
 0 <= x_grid_4 <= 119
 y_0 <= 60
 y_1 <= 60
 y_2 <= 60
 y_3 <= 60
 y_4 <= 60
 0 <= y_grid_0 <= 119
 0 <= y_grid_1 <= 119
 0 <= y_grid_2 <= 119
 0 <= y_grid_3 <= 119
 0 <= y_grid_4 <= 119
Generals
x_grid_0
x_grid_1
x_grid_2
x_grid_3
x_grid_4
y_grid_0
y_grid_1
y_grid_2
y_grid_3
y_grid_4
Binaries
diff_dist_pair_c5_0_1
diff_dist_pair_c5_0_2
diff_dist_pair_c5_0_3
diff_dist_pair_c5_0_4
diff_dist_pair_c5_1_2
diff_dist_pair_c5_1_3
diff_dist_pair_c5_1_4
diff_dist_pair_c5_2_3
diff_dist_pair_c5_2_4
diff_dist_pair_c5_3_4
dist_diff_abs_pair_c5_0_1_prev0_is_positive
dist_diff_abs_pair_c5_0_1_prev1_is_positive
dist_diff_abs_pair_c5_0_1_prev2_is_positive
dist_diff_abs_pair_c5_0_1_prev3_is_positive
dist_diff_abs_pair_c5_0_1_prev4_is_positive
dist_diff_abs_pair_c5_0_1_prev5_is_positive
dist_diff_abs_pair_c5_0_2_prev0_is_positive
dist_diff_abs_pair_c5_0_2_prev1_is_positive
dist_diff_abs_pair_c5_0_2_prev2_is_positive
dist_diff_abs_pair_c5_0_2_prev3_is_positive
dist_diff_abs_pair_c5_0_2_prev4_is_positive
dist_diff_abs_pair_c5_0_2_prev5_is_positive
dist_diff_abs_pair_c5_0_3_prev0_is_positive
dist_diff_abs_pair_c5_0_3_prev1_is_positive
dist_diff_abs_pair_c5_0_3_prev2_is_positive
dist_diff_abs_pair_c5_0_3_prev3_is_positive
dist_diff_abs_pair_c5_0_3_prev4_is_positive
dist_diff_abs_pair_c5_0_3_prev5_is_positive
dist_diff_abs_pair_c5_0_4_prev0_is_positive
dist_diff_abs_pair_c5_0_4_prev1_is_positive
dist_diff_abs_pair_c5_0_4_prev2_is_positive
dist_diff_abs_pair_c5_0_4_prev3_is_positive
dist_diff_abs_pair_c5_0_4_prev4_is_positive
dist_diff_abs_pair_c5_0_4_prev5_is_positive
dist_diff_abs_pair_c5_1_2_prev0_is_positive
dist_diff_abs_pair_c5_1_2_prev1_is_positive
dist_diff_abs_pair_c5_1_2_prev2_is_positive
dist_diff_abs_pair_c5_1_2_prev3_is_positive
dist_diff_abs_pair_c5_1_2_prev4_is_positive
dist_diff_abs_pair_c5_1_2_prev5_is_positive
dist_diff_abs_pair_c5_1_3_prev0_is_positive
dist_diff_abs_pair_c5_1_3_prev1_is_positive
dist_diff_abs_pair_c5_1_3_prev2_is_positive
dist_diff_abs_pair_c5_1_3_prev3_is_positive
dist_diff_abs_pair_c5_1_3_prev4_is_positive
dist_diff_abs_pair_c5_1_3_prev5_is_positive
dist_diff_abs_pair_c5_1_4_prev0_is_positive
dist_diff_abs_pair_c5_1_4_prev1_is_positive
dist_diff_abs_pair_c5_1_4_prev2_is_positive
dist_diff_abs_pair_c5_1_4_prev3_is_positive
dist_diff_abs_pair_c5_1_4_prev4_is_positive
dist_diff_abs_pair_c5_1_4_prev5_is_positive
dist_diff_abs_pair_c5_2_3_prev0_is_positive
dist_diff_abs_pair_c5_2_3_prev1_is_positive
dist_diff_abs_pair_c5_2_3_prev2_is_positive
dist_diff_abs_pair_c5_2_3_prev3_is_positive
dist_diff_abs_pair_c5_2_3_prev4_is_positive
dist_diff_abs_pair_c5_2_3_prev5_is_positive
dist_diff_abs_pair_c5_2_4_prev0_is_positive
dist_diff_abs_pair_c5_2_4_prev1_is_positive
dist_diff_abs_pair_c5_2_4_prev2_is_positive
dist_diff_abs_pair_c5_2_4_prev3_is_positive
dist_diff_abs_pair_c5_2_4_prev4_is_positive
dist_diff_abs_pair_c5_2_4_prev5_is_positive
dist_diff_abs_pair_c5_3_4_prev0_is_positive
dist_diff_abs_pair_c5_3_4_prev1_is_positive
dist_diff_abs_pair_c5_3_4_prev2_is_positive
dist_diff_abs_pair_c5_3_4_prev3_is_positive
dist_diff_abs_pair_c5_3_4_prev4_is_positive
dist_diff_abs_pair_c5_3_4_prev5_is_positive
dx_abs_0_1_is_positive
dx_abs_1_2_is_positive
dx_abs_1_3_is_positive
dx_abs_2_4_is_positive
dx_grid_abs_pair_c5_0_1_is_positive
dx_grid_abs_pair_c5_0_2_is_positive
dx_grid_abs_pair_c5_0_3_is_positive
dx_grid_abs_pair_c5_0_4_is_positive
dx_grid_abs_pair_c5_1_2_is_positive
dx_grid_abs_pair_c5_1_3_is_positive
dx_grid_abs_pair_c5_1_4_is_positive
dx_grid_abs_pair_c5_2_3_is_positive
dx_grid_abs_pair_c5_2_4_is_positive
dx_grid_abs_pair_c5_3_4_is_positive
dy_abs_0_1_is_positive
dy_abs_1_2_is_positive
dy_abs_1_3_is_positive
dy_abs_2_4_is_positive
dy_grid_abs_pair_c5_0_1_is_positive
dy_grid_abs_pair_c5_0_2_is_positive
dy_grid_abs_pair_c5_0_3_is_positive
dy_grid_abs_pair_c5_0_4_is_positive
dy_grid_abs_pair_c5_1_2_is_positive
dy_grid_abs_pair_c5_1_3_is_positive
dy_grid_abs_pair_c5_1_4_is_positive
dy_grid_abs_pair_c5_2_3_is_positive
dy_grid_abs_pair_c5_2_4_is_positive
dy_grid_abs_pair_c5_3_4_is_positive
p_down_0_1
p_down_0_2
p_down_0_3
p_down_0_4
p_down_1_2
p_down_1_3
p_down_1_4
p_down_2_3
p_down_2_4
p_down_3_4
p_left_0_1
p_left_0_2
p_left_0_3
p_left_0_4
p_left_1_2
p_left_1_3
p_left_1_4
p_left_2_3
p_left_2_4
p_left_3_4
p_right_0_1
p_right_0_2
p_right_0_3
p_right_0_4
p_right_1_2
p_right_1_3
p_right_1_4
p_right_2_3
p_right_2_4
p_right_3_4
p_up_0_1
p_up_0_2
p_up_0_3
p_up_0_4
p_up_1_2
p_up_1_3
p_up_1_4
p_up_2_3
p_up_2_4
p_up_3_4
r_1
r_3
same_dist_pair_c5_0_1_prev0
same_dist_pair_c5_0_1_prev1
same_dist_pair_c5_0_1_prev2
same_dist_pair_c5_0_1_prev3
same_dist_pair_c5_0_1_prev4
same_dist_pair_c5_0_1_prev5
same_dist_pair_c5_0_2_prev0
same_dist_pair_c5_0_2_prev1
same_dist_pair_c5_0_2_prev2
same_dist_pair_c5_0_2_prev3
same_dist_pair_c5_0_2_prev4
same_dist_pair_c5_0_2_prev5
same_dist_pair_c5_0_3_prev0
same_dist_pair_c5_0_3_prev1
same_dist_pair_c5_0_3_prev2
same_dist_pair_c5_0_3_prev3
same_dist_pair_c5_0_3_prev4
same_dist_pair_c5_0_3_prev5
same_dist_pair_c5_0_4_prev0
same_dist_pair_c5_0_4_prev1
same_dist_pair_c5_0_4_prev2
same_dist_pair_c5_0_4_prev3
same_dist_pair_c5_0_4_prev4
same_dist_pair_c5_0_4_prev5
same_dist_pair_c5_1_2_prev0
same_dist_pair_c5_1_2_prev1
same_dist_pair_c5_1_2_prev2
same_dist_pair_c5_1_2_prev3
same_dist_pair_c5_1_2_prev4
same_dist_pair_c5_1_2_prev5
same_dist_pair_c5_1_3_prev0
same_dist_pair_c5_1_3_prev1
same_dist_pair_c5_1_3_prev2
same_dist_pair_c5_1_3_prev3
same_dist_pair_c5_1_3_prev4
same_dist_pair_c5_1_3_prev5
same_dist_pair_c5_1_4_prev0
same_dist_pair_c5_1_4_prev1
same_dist_pair_c5_1_4_prev2
same_dist_pair_c5_1_4_prev3
same_dist_pair_c5_1_4_prev4
same_dist_pair_c5_1_4_prev5
same_dist_pair_c5_2_3_prev0
same_dist_pair_c5_2_3_prev1
same_dist_pair_c5_2_3_prev2
same_dist_pair_c5_2_3_prev3
same_dist_pair_c5_2_3_prev4
same_dist_pair_c5_2_3_prev5
same_dist_pair_c5_2_4_prev0
same_dist_pair_c5_2_4_prev1
same_dist_pair_c5_2_4_prev2
same_dist_pair_c5_2_4_prev3
same_dist_pair_c5_2_4_prev4
same_dist_pair_c5_2_4_prev5
same_dist_pair_c5_3_4_prev0
same_dist_pair_c5_3_4_prev1
same_dist_pair_c5_3_4_prev2
same_dist_pair_c5_3_4_prev3
same_dist_pair_c5_3_4_prev4
same_dist_pair_c5_3_4_prev5
z1L_0_1
z1L_1_2
z1L_1_3
z1L_2_4
z1R_0_1
z1R_1_2
z1R_1_3
z1R_2_4
z1_0_1
z1_1_2
z1_1_3
z1_2_4
z2D_0_1
z2D_1_2
z2D_1_3
z2D_2_4
z2U_0_1
z2U_1_2
z2U_1_3
z2U_2_4
z2_0_1
z2_1_2
z2_1_3
z2_2_4
End
