\* ChipletPlacementGrid *\
Minimize
Objective: 0.1 bbox_area_proxy_t + dx_abs_0_1 + dy_abs_0_1
Subject To
all_not_same_implies_diff_dist_pair_c6_0_1: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev0 + same_dist_pair_c6_0_1_prev1
 + same_dist_pair_c6_0_1_prev2 + same_dist_pair_c6_0_1_prev3
 + same_dist_pair_c6_0_1_prev4 + same_dist_pair_c6_0_1_prev5
 + same_dist_pair_c6_0_1_prev6 >= 1
bbox_h_def: bbox_h - bbox_max_y + bbox_min_y = 0
bbox_max_x_0: bbox_max_x - w_0 - x_0 >= 0
bbox_max_x_1: bbox_max_x - w_1 - x_1 >= 0
bbox_max_y_0: bbox_max_y - h_0 - y_0 >= 0
bbox_max_y_1: bbox_max_y - h_1 - y_1 >= 0
bbox_min_x_0: bbox_min_x - x_0 <= 0
bbox_min_x_1: bbox_min_x - x_1 <= 0
bbox_min_y_0: bbox_min_y - y_0 <= 0
bbox_min_y_1: bbox_min_y - y_1 <= 0
bbox_w_def: - bbox_max_x + bbox_min_x + bbox_w = 0
boundary_x_0: w_0 + x_0 <= 45
boundary_x_1: w_1 + x_1 <= 45
boundary_y_0: h_0 + y_0 <= 30
boundary_y_1: h_1 + y_1 <= 30
cx_def_0: cx_0 - 0.5 w_0 - x_0 = 0
cx_def_1: cx_1 - 0.5 w_1 - x_1 = 0
cy_def_0: cy_0 - 0.5 h_0 - y_0 = 0
cy_def_1: cy_1 - 0.5 h_1 - y_1 = 0
diff_dist_pair_implies_not_same_c6_0_1_prev0: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev0 <= 1
diff_dist_pair_implies_not_same_c6_0_1_prev1: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev1 <= 1
diff_dist_pair_implies_not_same_c6_0_1_prev2: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev2 <= 1
diff_dist_pair_implies_not_same_c6_0_1_prev3: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev3 <= 1
diff_dist_pair_implies_not_same_c6_0_1_prev4: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev4 <= 1
diff_dist_pair_implies_not_same_c6_0_1_prev5: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev5 <= 1
diff_dist_pair_implies_not_same_c6_0_1_prev6: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev6 <= 1
dist_curr_pair_def_c6_0_1: dist_curr_pair_c6_0_1 - dx_grid_abs_pair_c6_0_1
 - dy_grid_abs_pair_c6_0_1 = 0
dist_diff_abs_pair_c6_0_1_prev0_abs_ge_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev0
 + 150 dist_diff_abs_pair_c6_0_1_prev0_is_positive
 + dist_diff_pair_c6_0_1_prev0 >= 0
dist_diff_abs_pair_c6_0_1_prev0_abs_ge_orig: dist_diff_abs_pair_c6_0_1_prev0
 - 150 dist_diff_abs_pair_c6_0_1_prev0_is_positive
 - dist_diff_pair_c6_0_1_prev0 >= -150
dist_diff_abs_pair_c6_0_1_prev0_abs_le_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev0
 - 150 dist_diff_abs_pair_c6_0_1_prev0_is_positive
 + dist_diff_pair_c6_0_1_prev0 <= 0
dist_diff_abs_pair_c6_0_1_prev0_abs_le_orig: dist_diff_abs_pair_c6_0_1_prev0
 + 150 dist_diff_abs_pair_c6_0_1_prev0_is_positive
 - dist_diff_pair_c6_0_1_prev0 <= 150
dist_diff_abs_pair_c6_0_1_prev0_force_negative:
 - 150 dist_diff_abs_pair_c6_0_1_prev0_is_positive
 + dist_diff_pair_c6_0_1_prev0 <= -0.001
dist_diff_abs_pair_c6_0_1_prev0_force_positive:
 - 150 dist_diff_abs_pair_c6_0_1_prev0_is_positive
 + dist_diff_pair_c6_0_1_prev0 >= -150
dist_diff_abs_pair_c6_0_1_prev1_abs_ge_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev1
 + 150 dist_diff_abs_pair_c6_0_1_prev1_is_positive
 + dist_diff_pair_c6_0_1_prev1 >= 0
dist_diff_abs_pair_c6_0_1_prev1_abs_ge_orig: dist_diff_abs_pair_c6_0_1_prev1
 - 150 dist_diff_abs_pair_c6_0_1_prev1_is_positive
 - dist_diff_pair_c6_0_1_prev1 >= -150
dist_diff_abs_pair_c6_0_1_prev1_abs_le_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev1
 - 150 dist_diff_abs_pair_c6_0_1_prev1_is_positive
 + dist_diff_pair_c6_0_1_prev1 <= 0
dist_diff_abs_pair_c6_0_1_prev1_abs_le_orig: dist_diff_abs_pair_c6_0_1_prev1
 + 150 dist_diff_abs_pair_c6_0_1_prev1_is_positive
 - dist_diff_pair_c6_0_1_prev1 <= 150
dist_diff_abs_pair_c6_0_1_prev1_force_negative:
 - 150 dist_diff_abs_pair_c6_0_1_prev1_is_positive
 + dist_diff_pair_c6_0_1_prev1 <= -0.001
dist_diff_abs_pair_c6_0_1_prev1_force_positive:
 - 150 dist_diff_abs_pair_c6_0_1_prev1_is_positive
 + dist_diff_pair_c6_0_1_prev1 >= -150
dist_diff_abs_pair_c6_0_1_prev2_abs_ge_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev2
 + 150 dist_diff_abs_pair_c6_0_1_prev2_is_positive
 + dist_diff_pair_c6_0_1_prev2 >= 0
dist_diff_abs_pair_c6_0_1_prev2_abs_ge_orig: dist_diff_abs_pair_c6_0_1_prev2
 - 150 dist_diff_abs_pair_c6_0_1_prev2_is_positive
 - dist_diff_pair_c6_0_1_prev2 >= -150
dist_diff_abs_pair_c6_0_1_prev2_abs_le_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev2
 - 150 dist_diff_abs_pair_c6_0_1_prev2_is_positive
 + dist_diff_pair_c6_0_1_prev2 <= 0
dist_diff_abs_pair_c6_0_1_prev2_abs_le_orig: dist_diff_abs_pair_c6_0_1_prev2
 + 150 dist_diff_abs_pair_c6_0_1_prev2_is_positive
 - dist_diff_pair_c6_0_1_prev2 <= 150
dist_diff_abs_pair_c6_0_1_prev2_force_negative:
 - 150 dist_diff_abs_pair_c6_0_1_prev2_is_positive
 + dist_diff_pair_c6_0_1_prev2 <= -0.001
dist_diff_abs_pair_c6_0_1_prev2_force_positive:
 - 150 dist_diff_abs_pair_c6_0_1_prev2_is_positive
 + dist_diff_pair_c6_0_1_prev2 >= -150
dist_diff_abs_pair_c6_0_1_prev3_abs_ge_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev3
 + 150 dist_diff_abs_pair_c6_0_1_prev3_is_positive
 + dist_diff_pair_c6_0_1_prev3 >= 0
dist_diff_abs_pair_c6_0_1_prev3_abs_ge_orig: dist_diff_abs_pair_c6_0_1_prev3
 - 150 dist_diff_abs_pair_c6_0_1_prev3_is_positive
 - dist_diff_pair_c6_0_1_prev3 >= -150
dist_diff_abs_pair_c6_0_1_prev3_abs_le_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev3
 - 150 dist_diff_abs_pair_c6_0_1_prev3_is_positive
 + dist_diff_pair_c6_0_1_prev3 <= 0
dist_diff_abs_pair_c6_0_1_prev3_abs_le_orig: dist_diff_abs_pair_c6_0_1_prev3
 + 150 dist_diff_abs_pair_c6_0_1_prev3_is_positive
 - dist_diff_pair_c6_0_1_prev3 <= 150
dist_diff_abs_pair_c6_0_1_prev3_force_negative:
 - 150 dist_diff_abs_pair_c6_0_1_prev3_is_positive
 + dist_diff_pair_c6_0_1_prev3 <= -0.001
dist_diff_abs_pair_c6_0_1_prev3_force_positive:
 - 150 dist_diff_abs_pair_c6_0_1_prev3_is_positive
 + dist_diff_pair_c6_0_1_prev3 >= -150
dist_diff_abs_pair_c6_0_1_prev4_abs_ge_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev4
 + 150 dist_diff_abs_pair_c6_0_1_prev4_is_positive
 + dist_diff_pair_c6_0_1_prev4 >= 0
dist_diff_abs_pair_c6_0_1_prev4_abs_ge_orig: dist_diff_abs_pair_c6_0_1_prev4
 - 150 dist_diff_abs_pair_c6_0_1_prev4_is_positive
 - dist_diff_pair_c6_0_1_prev4 >= -150
dist_diff_abs_pair_c6_0_1_prev4_abs_le_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev4
 - 150 dist_diff_abs_pair_c6_0_1_prev4_is_positive
 + dist_diff_pair_c6_0_1_prev4 <= 0
dist_diff_abs_pair_c6_0_1_prev4_abs_le_orig: dist_diff_abs_pair_c6_0_1_prev4
 + 150 dist_diff_abs_pair_c6_0_1_prev4_is_positive
 - dist_diff_pair_c6_0_1_prev4 <= 150
dist_diff_abs_pair_c6_0_1_prev4_force_negative:
 - 150 dist_diff_abs_pair_c6_0_1_prev4_is_positive
 + dist_diff_pair_c6_0_1_prev4 <= -0.001
dist_diff_abs_pair_c6_0_1_prev4_force_positive:
 - 150 dist_diff_abs_pair_c6_0_1_prev4_is_positive
 + dist_diff_pair_c6_0_1_prev4 >= -150
dist_diff_abs_pair_c6_0_1_prev5_abs_ge_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev5
 + 150 dist_diff_abs_pair_c6_0_1_prev5_is_positive
 + dist_diff_pair_c6_0_1_prev5 >= 0
dist_diff_abs_pair_c6_0_1_prev5_abs_ge_orig: dist_diff_abs_pair_c6_0_1_prev5
 - 150 dist_diff_abs_pair_c6_0_1_prev5_is_positive
 - dist_diff_pair_c6_0_1_prev5 >= -150
dist_diff_abs_pair_c6_0_1_prev5_abs_le_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev5
 - 150 dist_diff_abs_pair_c6_0_1_prev5_is_positive
 + dist_diff_pair_c6_0_1_prev5 <= 0
dist_diff_abs_pair_c6_0_1_prev5_abs_le_orig: dist_diff_abs_pair_c6_0_1_prev5
 + 150 dist_diff_abs_pair_c6_0_1_prev5_is_positive
 - dist_diff_pair_c6_0_1_prev5 <= 150
dist_diff_abs_pair_c6_0_1_prev5_force_negative:
 - 150 dist_diff_abs_pair_c6_0_1_prev5_is_positive
 + dist_diff_pair_c6_0_1_prev5 <= -0.001
dist_diff_abs_pair_c6_0_1_prev5_force_positive:
 - 150 dist_diff_abs_pair_c6_0_1_prev5_is_positive
 + dist_diff_pair_c6_0_1_prev5 >= -150
dist_diff_abs_pair_c6_0_1_prev6_abs_ge_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev6
 + 150 dist_diff_abs_pair_c6_0_1_prev6_is_positive
 + dist_diff_pair_c6_0_1_prev6 >= 0
dist_diff_abs_pair_c6_0_1_prev6_abs_ge_orig: dist_diff_abs_pair_c6_0_1_prev6
 - 150 dist_diff_abs_pair_c6_0_1_prev6_is_positive
 - dist_diff_pair_c6_0_1_prev6 >= -150
dist_diff_abs_pair_c6_0_1_prev6_abs_le_neg_orig:
 dist_diff_abs_pair_c6_0_1_prev6
 - 150 dist_diff_abs_pair_c6_0_1_prev6_is_positive
 + dist_diff_pair_c6_0_1_prev6 <= 0
dist_diff_abs_pair_c6_0_1_prev6_abs_le_orig: dist_diff_abs_pair_c6_0_1_prev6
 + 150 dist_diff_abs_pair_c6_0_1_prev6_is_positive
 - dist_diff_pair_c6_0_1_prev6 <= 150
dist_diff_abs_pair_c6_0_1_prev6_force_negative:
 - 150 dist_diff_abs_pair_c6_0_1_prev6_is_positive
 + dist_diff_pair_c6_0_1_prev6 <= -0.001
dist_diff_abs_pair_c6_0_1_prev6_force_positive:
 - 150 dist_diff_abs_pair_c6_0_1_prev6_is_positive
 + dist_diff_pair_c6_0_1_prev6 >= -150
dist_diff_pair_def_c6_0_1_prev0: - dist_curr_pair_c6_0_1
 + dist_diff_pair_c6_0_1_prev0 = -25
dist_diff_pair_def_c6_0_1_prev1: - dist_curr_pair_c6_0_1
 + dist_diff_pair_c6_0_1_prev1 = -26
dist_diff_pair_def_c6_0_1_prev2: - dist_curr_pair_c6_0_1
 + dist_diff_pair_c6_0_1_prev2 = -24
dist_diff_pair_def_c6_0_1_prev3: - dist_curr_pair_c6_0_1
 + dist_diff_pair_c6_0_1_prev3 = -27
dist_diff_pair_def_c6_0_1_prev4: - dist_curr_pair_c6_0_1
 + dist_diff_pair_c6_0_1_prev4 = -23
dist_diff_pair_def_c6_0_1_prev5: - dist_curr_pair_c6_0_1
 + dist_diff_pair_c6_0_1_prev5 = -28
dist_diff_pair_def_c6_0_1_prev6: - dist_curr_pair_c6_0_1
 + dist_diff_pair_c6_0_1_prev6 = -22
dx_abs_0_1_abs_ge_neg_orig: dx_abs_0_1 + 45 dx_abs_0_1_is_positive
 + dx_diff_0_1 >= 0
dx_abs_0_1_abs_ge_orig: dx_abs_0_1 - 45 dx_abs_0_1_is_positive - dx_diff_0_1
 >= -45
dx_abs_0_1_abs_le_neg_orig: dx_abs_0_1 - 45 dx_abs_0_1_is_positive
 + dx_diff_0_1 <= 0
dx_abs_0_1_abs_le_orig: dx_abs_0_1 + 45 dx_abs_0_1_is_positive - dx_diff_0_1
 <= 45
dx_abs_0_1_force_negative: - 45 dx_abs_0_1_is_positive + dx_diff_0_1 <= -0.001
dx_abs_0_1_force_positive: - 45 dx_abs_0_1_is_positive + dx_diff_0_1 >= -45
dx_diff_def_0_1: - cx_0 + cx_1 + dx_diff_0_1 = 0
dx_grid_abs_pair_c6_0_1_abs_ge_neg_orig: dx_grid_abs_pair_c6_0_1
 + 90 dx_grid_abs_pair_c6_0_1_is_positive + dx_grid_diff_c6_0_1 >= 0
dx_grid_abs_pair_c6_0_1_abs_ge_orig: dx_grid_abs_pair_c6_0_1
 - 90 dx_grid_abs_pair_c6_0_1_is_positive - dx_grid_diff_c6_0_1 >= -90
dx_grid_abs_pair_c6_0_1_abs_le_neg_orig: dx_grid_abs_pair_c6_0_1
 - 90 dx_grid_abs_pair_c6_0_1_is_positive + dx_grid_diff_c6_0_1 <= 0
dx_grid_abs_pair_c6_0_1_abs_le_orig: dx_grid_abs_pair_c6_0_1
 + 90 dx_grid_abs_pair_c6_0_1_is_positive - dx_grid_diff_c6_0_1 <= 90
dx_grid_abs_pair_c6_0_1_force_negative:
 - 90 dx_grid_abs_pair_c6_0_1_is_positive + dx_grid_diff_c6_0_1 <= -0.001
dx_grid_abs_pair_c6_0_1_force_positive:
 - 90 dx_grid_abs_pair_c6_0_1_is_positive + dx_grid_diff_c6_0_1 >= -90
dx_grid_diff_def_c6_0_1: dx_grid_diff_c6_0_1 - x_grid_0 + x_grid_1 = 0
dy_abs_0_1_abs_ge_neg_orig: dy_abs_0_1 + 30 dy_abs_0_1_is_positive
 + dy_diff_0_1 >= 0
dy_abs_0_1_abs_ge_orig: dy_abs_0_1 - 30 dy_abs_0_1_is_positive - dy_diff_0_1
 >= -30
dy_abs_0_1_abs_le_neg_orig: dy_abs_0_1 - 30 dy_abs_0_1_is_positive
 + dy_diff_0_1 <= 0
dy_abs_0_1_abs_le_orig: dy_abs_0_1 + 30 dy_abs_0_1_is_positive - dy_diff_0_1
 <= 30
dy_abs_0_1_force_negative: - 30 dy_abs_0_1_is_positive + dy_diff_0_1 <= -0.001
dy_abs_0_1_force_positive: - 30 dy_abs_0_1_is_positive + dy_diff_0_1 >= -30
dy_diff_def_0_1: - cy_0 + cy_1 + dy_diff_0_1 = 0
dy_grid_abs_pair_c6_0_1_abs_ge_neg_orig: dy_grid_abs_pair_c6_0_1
 + 60 dy_grid_abs_pair_c6_0_1_is_positive + dy_grid_diff_c6_0_1 >= 0
dy_grid_abs_pair_c6_0_1_abs_ge_orig: dy_grid_abs_pair_c6_0_1
 - 60 dy_grid_abs_pair_c6_0_1_is_positive - dy_grid_diff_c6_0_1 >= -60
dy_grid_abs_pair_c6_0_1_abs_le_neg_orig: dy_grid_abs_pair_c6_0_1
 - 60 dy_grid_abs_pair_c6_0_1_is_positive + dy_grid_diff_c6_0_1 <= 0
dy_grid_abs_pair_c6_0_1_abs_le_orig: dy_grid_abs_pair_c6_0_1
 + 60 dy_grid_abs_pair_c6_0_1_is_positive - dy_grid_diff_c6_0_1 <= 60
dy_grid_abs_pair_c6_0_1_force_negative:
 - 60 dy_grid_abs_pair_c6_0_1_is_positive + dy_grid_diff_c6_0_1 <= -0.001
dy_grid_abs_pair_c6_0_1_force_positive:
 - 60 dy_grid_abs_pair_c6_0_1_is_positive + dy_grid_diff_c6_0_1 >= -60
dy_grid_diff_def_c6_0_1: dy_grid_diff_c6_0_1 - y_grid_0 + y_grid_1 = 0
exclude_solution_dist_pair_c6: diff_dist_pair_c6_0_1 >= 1
fix_center_x_0: cx_0 = 22.5
fix_center_y_0: cy_0 = 15
h_align_grid_0: h_0 = 10
h_align_grid_1: h_1 = 10
h_max_0: h_0 <= 30
h_max_1: h_1 <= 30
height_rotation_0: h_0 - 5 r_0 = 10
height_rotation_1: h_1 = 10
horizontal_direction_0_1: z1L_0_1 + z1R_0_1 - z1_0_1 = 0
horizontal_left_dist_0_1: - w_0 - x_0 + x_1 + 200 z1L_0_1 <= 200.5
horizontal_left_dist_lb_0_1: - w_0 - x_0 + x_1 - 200 z1L_0_1 >= -200
horizontal_overlap_y1_0_1: - h_1 + y_0 - y_1 + 200 z1_0_1 <= 200
horizontal_overlap_y2_0_1: - h_0 - y_0 + y_1 + 200 z1_0_1 <= 200
horizontal_right_dist_0_1: - w_1 + x_0 - x_1 + 200 z1R_0_1 <= 200.5
horizontal_right_dist_lb_0_1: - w_1 + x_0 - x_1 - 200 z1R_0_1 >= -200
must_adjacent_0_1: z1_0_1 + z2_0_1 = 1
non_overlap_any_0_1: p_down_0_1 + p_left_0_1 + p_right_0_1 + p_up_0_1 = 1
non_overlap_down_0_1: h_0 + 200 p_down_0_1 + y_0 - y_1 <= 200
non_overlap_down_rev_0_1: - h_0 - 200 p_down_0_1 - y_0 + y_1 <= 0
non_overlap_left_0_1: 200 p_left_0_1 + w_0 + x_0 - x_1 <= 200
non_overlap_left_rev_0_1: - 200 p_left_0_1 - w_0 - x_0 + x_1 <= 0
non_overlap_right_0_1: 200 p_right_0_1 + w_1 - x_0 + x_1 <= 200
non_overlap_right_rev_0_1: - 200 p_right_0_1 - w_1 + x_0 - x_1 <= 0
non_overlap_up_0_1: h_1 + 200 p_up_0_1 - y_0 + y_1 <= 200
non_overlap_up_rev_0_1: - h_1 - 200 p_up_0_1 + y_0 - y_1 <= 0
not_same_implies_diff_dist_pair_c6_0_1_prev0: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev0 <= 1
not_same_implies_diff_dist_pair_c6_0_1_prev1: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev1 <= 1
not_same_implies_diff_dist_pair_c6_0_1_prev2: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev2 <= 1
not_same_implies_diff_dist_pair_c6_0_1_prev3: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev3 <= 1
not_same_implies_diff_dist_pair_c6_0_1_prev4: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev4 <= 1
not_same_implies_diff_dist_pair_c6_0_1_prev5: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev5 <= 1
not_same_implies_diff_dist_pair_c6_0_1_prev6: diff_dist_pair_c6_0_1
 + same_dist_pair_c6_0_1_prev6 <= 1
same_dist_pair_lower_c6_0_1_prev0: dist_diff_abs_pair_c6_0_1_prev0
 + 90 same_dist_pair_c6_0_1_prev0 >= 1
same_dist_pair_lower_c6_0_1_prev1: dist_diff_abs_pair_c6_0_1_prev1
 + 90 same_dist_pair_c6_0_1_prev1 >= 1
same_dist_pair_lower_c6_0_1_prev2: dist_diff_abs_pair_c6_0_1_prev2
 + 90 same_dist_pair_c6_0_1_prev2 >= 1
same_dist_pair_lower_c6_0_1_prev3: dist_diff_abs_pair_c6_0_1_prev3
 + 90 same_dist_pair_c6_0_1_prev3 >= 1
same_dist_pair_lower_c6_0_1_prev4: dist_diff_abs_pair_c6_0_1_prev4
 + 90 same_dist_pair_c6_0_1_prev4 >= 1
same_dist_pair_lower_c6_0_1_prev5: dist_diff_abs_pair_c6_0_1_prev5
 + 90 same_dist_pair_c6_0_1_prev5 >= 1
same_dist_pair_lower_c6_0_1_prev6: dist_diff_abs_pair_c6_0_1_prev6
 + 90 same_dist_pair_c6_0_1_prev6 >= 1
same_dist_pair_upper_c6_0_1_prev0: dist_diff_abs_pair_c6_0_1_prev0
 + 90 same_dist_pair_c6_0_1_prev0 <= 90.99
same_dist_pair_upper_c6_0_1_prev1: dist_diff_abs_pair_c6_0_1_prev1
 + 90 same_dist_pair_c6_0_1_prev1 <= 90.99
same_dist_pair_upper_c6_0_1_prev2: dist_diff_abs_pair_c6_0_1_prev2
 + 90 same_dist_pair_c6_0_1_prev2 <= 90.99
same_dist_pair_upper_c6_0_1_prev3: dist_diff_abs_pair_c6_0_1_prev3
 + 90 same_dist_pair_c6_0_1_prev3 <= 90.99
same_dist_pair_upper_c6_0_1_prev4: dist_diff_abs_pair_c6_0_1_prev4
 + 90 same_dist_pair_c6_0_1_prev4 <= 90.99
same_dist_pair_upper_c6_0_1_prev5: dist_diff_abs_pair_c6_0_1_prev5
 + 90 same_dist_pair_c6_0_1_prev5 <= 90.99
same_dist_pair_upper_c6_0_1_prev6: dist_diff_abs_pair_c6_0_1_prev6
 + 90 same_dist_pair_c6_0_1_prev6 <= 90.99
shared_x_min_0_1: shared_x_0_1 - 200 z2_0_1 >= -199.9
shared_x_ub1_0_1: shared_x_0_1 - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
shared_x_ub2_0_1: shared_x_0_1 - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
shared_x_zero_0_1: shared_x_0_1 - 200 z2_0_1 <= 0
shared_y_min_0_1: shared_y_0_1 - 200 z1_0_1 >= -199.9
shared_y_ub1_0_1: - h_0 + shared_y_0_1 - y_0 + y_1 + 200 z1_0_1 <= 200
shared_y_ub2_0_1: - h_1 + shared_y_0_1 + y_0 - y_1 + 200 z1_0_1 <= 200
shared_y_zero_0_1: shared_y_0_1 - 200 z1_0_1 <= 0
t_ge_height: bbox_area_proxy_t - bbox_h >= 0
t_ge_scaled_mean: bbox_area_proxy_t - 0.8 bbox_h - 0.8 bbox_w >= 0
t_ge_width: bbox_area_proxy_t - bbox_w >= 0
vertical_direction_0_1: z2D_0_1 + z2U_0_1 - z2_0_1 = 0
vertical_down_dist_0_1: - h_0 - y_0 + y_1 + 200 z2D_0_1 <= 200.5
vertical_down_dist_lb_0_1: - h_0 - y_0 + y_1 - 200 z2D_0_1 >= -200
vertical_overlap_x1_0_1: - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
vertical_overlap_x2_0_1: - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
vertical_up_dist_0_1: - h_1 + y_0 - y_1 + 200 z2U_0_1 <= 200.5
vertical_up_dist_lb_0_1: - h_1 + y_0 - y_1 - 200 z2U_0_1 >= -200
w_align_grid_0: w_0 = 15
w_align_grid_1: w_1 = 10
w_max_0: w_0 <= 45
w_max_1: w_1 <= 45
width_rotation_0: 5 r_0 + w_0 = 15
width_rotation_1: w_1 = 10
x_grid_to_real_0: x_0 - 0.5 x_grid_0 = 0
x_grid_to_real_1: x_1 - 0.5 x_grid_1 = 0
y_grid_to_real_0: y_0 - 0.5 y_grid_0 = 0
y_grid_to_real_1: y_1 - 0.5 y_grid_1 = 0
Bounds
 bbox_area_proxy_t <= 75
 bbox_h <= 30
 bbox_max_x <= 45
 bbox_max_y <= 30
 bbox_min_x <= 45
 bbox_min_y <= 30
 bbox_w <= 45
 cx_0 <= 45
 cx_1 <= 45
 cy_0 <= 30
 cy_1 <= 30
 dist_curr_pair_c6_0_1 <= 150
 dist_diff_abs_pair_c6_0_1_prev0 <= 150
 dist_diff_abs_pair_c6_0_1_prev1 <= 150
 dist_diff_abs_pair_c6_0_1_prev2 <= 150
 dist_diff_abs_pair_c6_0_1_prev3 <= 150
 dist_diff_abs_pair_c6_0_1_prev4 <= 150
 dist_diff_abs_pair_c6_0_1_prev5 <= 150
 dist_diff_abs_pair_c6_0_1_prev6 <= 150
 -150 <= dist_diff_pair_c6_0_1_prev0 <= 150
 -150 <= dist_diff_pair_c6_0_1_prev1 <= 150
 -150 <= dist_diff_pair_c6_0_1_prev2 <= 150
 -150 <= dist_diff_pair_c6_0_1_prev3 <= 150
 -150 <= dist_diff_pair_c6_0_1_prev4 <= 150
 -150 <= dist_diff_pair_c6_0_1_prev5 <= 150
 -150 <= dist_diff_pair_c6_0_1_prev6 <= 150
 -45 <= dx_diff_0_1 <= 45
 dx_grid_abs_pair_c6_0_1 <= 89
 -89 <= dx_grid_diff_c6_0_1 <= 89
 -30 <= dy_diff_0_1 <= 30
 dy_grid_abs_pair_c6_0_1 <= 59
 -59 <= dy_grid_diff_c6_0_1 <= 59
 10 <= h_0 <= 15
 h_1 = 10
 shared_x_0_1 <= 15
 shared_y_0_1 <= 10
 10 <= w_0 <= 15
 w_1 = 10
 x_0 <= 45
 x_1 <= 45
 0 <= x_grid_0 <= 89
 0 <= x_grid_1 <= 89
 y_0 <= 30
 y_1 <= 30
 0 <= y_grid_0 <= 59
 0 <= y_grid_1 <= 59
Generals
x_grid_0
x_grid_1
y_grid_0
y_grid_1
Binaries
diff_dist_pair_c6_0_1
dist_diff_abs_pair_c6_0_1_prev0_is_positive
dist_diff_abs_pair_c6_0_1_prev1_is_positive
dist_diff_abs_pair_c6_0_1_prev2_is_positive
dist_diff_abs_pair_c6_0_1_prev3_is_positive
dist_diff_abs_pair_c6_0_1_prev4_is_positive
dist_diff_abs_pair_c6_0_1_prev5_is_positive
dist_diff_abs_pair_c6_0_1_prev6_is_positive
dx_abs_0_1_is_positive
dx_grid_abs_pair_c6_0_1_is_positive
dy_abs_0_1_is_positive
dy_grid_abs_pair_c6_0_1_is_positive
p_down_0_1
p_left_0_1
p_right_0_1
p_up_0_1
r_0
same_dist_pair_c6_0_1_prev0
same_dist_pair_c6_0_1_prev1
same_dist_pair_c6_0_1_prev2
same_dist_pair_c6_0_1_prev3
same_dist_pair_c6_0_1_prev4
same_dist_pair_c6_0_1_prev5
same_dist_pair_c6_0_1_prev6
z1L_0_1
z1R_0_1
z1_0_1
z2D_0_1
z2U_0_1
z2_0_1
End
