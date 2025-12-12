\* ChipletPlacementGrid *\
Minimize
Objective: 0.1 bbox_area_proxy_t + dx_abs_0_1 + dx_abs_1_2 + dx_abs_2_3
 + dx_abs_2_4 + dx_abs_2_5 + dy_abs_0_1 + dy_abs_1_2 + dy_abs_2_3 + dy_abs_2_4
 + dy_abs_2_5
Subject To
bbox_h_def: bbox_h - bbox_max_y + bbox_min_y = 0
bbox_max_x_0: bbox_max_x - w_0 - x_0 >= 0
bbox_max_x_1: bbox_max_x - w_1 - x_1 >= 0
bbox_max_x_2: bbox_max_x - w_2 - x_2 >= 0
bbox_max_x_3: bbox_max_x - w_3 - x_3 >= 0
bbox_max_x_4: bbox_max_x - w_4 - x_4 >= 0
bbox_max_x_5: bbox_max_x - w_5 - x_5 >= 0
bbox_max_y_0: bbox_max_y - h_0 - y_0 >= 0
bbox_max_y_1: bbox_max_y - h_1 - y_1 >= 0
bbox_max_y_2: bbox_max_y - h_2 - y_2 >= 0
bbox_max_y_3: bbox_max_y - h_3 - y_3 >= 0
bbox_max_y_4: bbox_max_y - h_4 - y_4 >= 0
bbox_max_y_5: bbox_max_y - h_5 - y_5 >= 0
bbox_min_x_0: bbox_min_x - x_0 <= 0
bbox_min_x_1: bbox_min_x - x_1 <= 0
bbox_min_x_2: bbox_min_x - x_2 <= 0
bbox_min_x_3: bbox_min_x - x_3 <= 0
bbox_min_x_4: bbox_min_x - x_4 <= 0
bbox_min_x_5: bbox_min_x - x_5 <= 0
bbox_min_y_0: bbox_min_y - y_0 <= 0
bbox_min_y_1: bbox_min_y - y_1 <= 0
bbox_min_y_2: bbox_min_y - y_2 <= 0
bbox_min_y_3: bbox_min_y - y_3 <= 0
bbox_min_y_4: bbox_min_y - y_4 <= 0
bbox_min_y_5: bbox_min_y - y_5 <= 0
bbox_w_def: - bbox_max_x + bbox_min_x + bbox_w = 0
boundary_x_0: w_0 + x_0 <= 30
boundary_x_1: w_1 + x_1 <= 30
boundary_x_2: w_2 + x_2 <= 30
boundary_x_3: w_3 + x_3 <= 30
boundary_x_4: w_4 + x_4 <= 30
boundary_x_5: w_5 + x_5 <= 30
boundary_y_0: h_0 + y_0 <= 30
boundary_y_1: h_1 + y_1 <= 30
boundary_y_2: h_2 + y_2 <= 30
boundary_y_3: h_3 + y_3 <= 30
boundary_y_4: h_4 + y_4 <= 30
boundary_y_5: h_5 + y_5 <= 30
cx_def_0: cx_0 - 0.5 w_0 - x_0 = 0
cx_def_1: cx_1 - 0.5 w_1 - x_1 = 0
cx_def_2: cx_2 - 0.5 w_2 - x_2 = 0
cx_def_3: cx_3 - 0.5 w_3 - x_3 = 0
cx_def_4: cx_4 - 0.5 w_4 - x_4 = 0
cx_def_5: cx_5 - 0.5 w_5 - x_5 = 0
cy_def_0: cy_0 - 0.5 h_0 - y_0 = 0
cy_def_1: cy_1 - 0.5 h_1 - y_1 = 0
cy_def_2: cy_2 - 0.5 h_2 - y_2 = 0
cy_def_3: cy_3 - 0.5 h_3 - y_3 = 0
cy_def_4: cy_4 - 0.5 h_4 - y_4 = 0
cy_def_5: cy_5 - 0.5 h_5 - y_5 = 0
dx_abs_0_1_abs_ge_neg_orig: dx_abs_0_1 + 30 dx_abs_0_1_is_positive
 + dx_diff_0_1 >= 0
dx_abs_0_1_abs_ge_orig: dx_abs_0_1 - 30 dx_abs_0_1_is_positive - dx_diff_0_1
 >= -30
dx_abs_0_1_abs_le_neg_orig: dx_abs_0_1 - 30 dx_abs_0_1_is_positive
 + dx_diff_0_1 <= 0
dx_abs_0_1_abs_le_orig: dx_abs_0_1 + 30 dx_abs_0_1_is_positive - dx_diff_0_1
 <= 30
dx_abs_0_1_force_negative: - 30 dx_abs_0_1_is_positive + dx_diff_0_1 <= -0.001
dx_abs_0_1_force_positive: - 30 dx_abs_0_1_is_positive + dx_diff_0_1 >= -30
dx_abs_1_2_abs_ge_neg_orig: dx_abs_1_2 + 30 dx_abs_1_2_is_positive
 + dx_diff_1_2 >= 0
dx_abs_1_2_abs_ge_orig: dx_abs_1_2 - 30 dx_abs_1_2_is_positive - dx_diff_1_2
 >= -30
dx_abs_1_2_abs_le_neg_orig: dx_abs_1_2 - 30 dx_abs_1_2_is_positive
 + dx_diff_1_2 <= 0
dx_abs_1_2_abs_le_orig: dx_abs_1_2 + 30 dx_abs_1_2_is_positive - dx_diff_1_2
 <= 30
dx_abs_1_2_force_negative: - 30 dx_abs_1_2_is_positive + dx_diff_1_2 <= -0.001
dx_abs_1_2_force_positive: - 30 dx_abs_1_2_is_positive + dx_diff_1_2 >= -30
dx_abs_2_3_abs_ge_neg_orig: dx_abs_2_3 + 30 dx_abs_2_3_is_positive
 + dx_diff_2_3 >= 0
dx_abs_2_3_abs_ge_orig: dx_abs_2_3 - 30 dx_abs_2_3_is_positive - dx_diff_2_3
 >= -30
dx_abs_2_3_abs_le_neg_orig: dx_abs_2_3 - 30 dx_abs_2_3_is_positive
 + dx_diff_2_3 <= 0
dx_abs_2_3_abs_le_orig: dx_abs_2_3 + 30 dx_abs_2_3_is_positive - dx_diff_2_3
 <= 30
dx_abs_2_3_force_negative: - 30 dx_abs_2_3_is_positive + dx_diff_2_3 <= -0.001
dx_abs_2_3_force_positive: - 30 dx_abs_2_3_is_positive + dx_diff_2_3 >= -30
dx_abs_2_4_abs_ge_neg_orig: dx_abs_2_4 + 30 dx_abs_2_4_is_positive
 + dx_diff_2_4 >= 0
dx_abs_2_4_abs_ge_orig: dx_abs_2_4 - 30 dx_abs_2_4_is_positive - dx_diff_2_4
 >= -30
dx_abs_2_4_abs_le_neg_orig: dx_abs_2_4 - 30 dx_abs_2_4_is_positive
 + dx_diff_2_4 <= 0
dx_abs_2_4_abs_le_orig: dx_abs_2_4 + 30 dx_abs_2_4_is_positive - dx_diff_2_4
 <= 30
dx_abs_2_4_force_negative: - 30 dx_abs_2_4_is_positive + dx_diff_2_4 <= -0.001
dx_abs_2_4_force_positive: - 30 dx_abs_2_4_is_positive + dx_diff_2_4 >= -30
dx_abs_2_5_abs_ge_neg_orig: dx_abs_2_5 + 30 dx_abs_2_5_is_positive
 + dx_diff_2_5 >= 0
dx_abs_2_5_abs_ge_orig: dx_abs_2_5 - 30 dx_abs_2_5_is_positive - dx_diff_2_5
 >= -30
dx_abs_2_5_abs_le_neg_orig: dx_abs_2_5 - 30 dx_abs_2_5_is_positive
 + dx_diff_2_5 <= 0
dx_abs_2_5_abs_le_orig: dx_abs_2_5 + 30 dx_abs_2_5_is_positive - dx_diff_2_5
 <= 30
dx_abs_2_5_force_negative: - 30 dx_abs_2_5_is_positive + dx_diff_2_5 <= -0.001
dx_abs_2_5_force_positive: - 30 dx_abs_2_5_is_positive + dx_diff_2_5 >= -30
dx_diff_def_0_1: - cx_0 + cx_1 + dx_diff_0_1 = 0
dx_diff_def_1_2: - cx_1 + cx_2 + dx_diff_1_2 = 0
dx_diff_def_2_3: - cx_2 + cx_3 + dx_diff_2_3 = 0
dx_diff_def_2_4: - cx_2 + cx_4 + dx_diff_2_4 = 0
dx_diff_def_2_5: - cx_2 + cx_5 + dx_diff_2_5 = 0
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
dy_abs_1_2_abs_ge_neg_orig: dy_abs_1_2 + 30 dy_abs_1_2_is_positive
 + dy_diff_1_2 >= 0
dy_abs_1_2_abs_ge_orig: dy_abs_1_2 - 30 dy_abs_1_2_is_positive - dy_diff_1_2
 >= -30
dy_abs_1_2_abs_le_neg_orig: dy_abs_1_2 - 30 dy_abs_1_2_is_positive
 + dy_diff_1_2 <= 0
dy_abs_1_2_abs_le_orig: dy_abs_1_2 + 30 dy_abs_1_2_is_positive - dy_diff_1_2
 <= 30
dy_abs_1_2_force_negative: - 30 dy_abs_1_2_is_positive + dy_diff_1_2 <= -0.001
dy_abs_1_2_force_positive: - 30 dy_abs_1_2_is_positive + dy_diff_1_2 >= -30
dy_abs_2_3_abs_ge_neg_orig: dy_abs_2_3 + 30 dy_abs_2_3_is_positive
 + dy_diff_2_3 >= 0
dy_abs_2_3_abs_ge_orig: dy_abs_2_3 - 30 dy_abs_2_3_is_positive - dy_diff_2_3
 >= -30
dy_abs_2_3_abs_le_neg_orig: dy_abs_2_3 - 30 dy_abs_2_3_is_positive
 + dy_diff_2_3 <= 0
dy_abs_2_3_abs_le_orig: dy_abs_2_3 + 30 dy_abs_2_3_is_positive - dy_diff_2_3
 <= 30
dy_abs_2_3_force_negative: - 30 dy_abs_2_3_is_positive + dy_diff_2_3 <= -0.001
dy_abs_2_3_force_positive: - 30 dy_abs_2_3_is_positive + dy_diff_2_3 >= -30
dy_abs_2_4_abs_ge_neg_orig: dy_abs_2_4 + 30 dy_abs_2_4_is_positive
 + dy_diff_2_4 >= 0
dy_abs_2_4_abs_ge_orig: dy_abs_2_4 - 30 dy_abs_2_4_is_positive - dy_diff_2_4
 >= -30
dy_abs_2_4_abs_le_neg_orig: dy_abs_2_4 - 30 dy_abs_2_4_is_positive
 + dy_diff_2_4 <= 0
dy_abs_2_4_abs_le_orig: dy_abs_2_4 + 30 dy_abs_2_4_is_positive - dy_diff_2_4
 <= 30
dy_abs_2_4_force_negative: - 30 dy_abs_2_4_is_positive + dy_diff_2_4 <= -0.001
dy_abs_2_4_force_positive: - 30 dy_abs_2_4_is_positive + dy_diff_2_4 >= -30
dy_abs_2_5_abs_ge_neg_orig: dy_abs_2_5 + 30 dy_abs_2_5_is_positive
 + dy_diff_2_5 >= 0
dy_abs_2_5_abs_ge_orig: dy_abs_2_5 - 30 dy_abs_2_5_is_positive - dy_diff_2_5
 >= -30
dy_abs_2_5_abs_le_neg_orig: dy_abs_2_5 - 30 dy_abs_2_5_is_positive
 + dy_diff_2_5 <= 0
dy_abs_2_5_abs_le_orig: dy_abs_2_5 + 30 dy_abs_2_5_is_positive - dy_diff_2_5
 <= 30
dy_abs_2_5_force_negative: - 30 dy_abs_2_5_is_positive + dy_diff_2_5 <= -0.001
dy_abs_2_5_force_positive: - 30 dy_abs_2_5_is_positive + dy_diff_2_5 >= -30
dy_diff_def_0_1: - cy_0 + cy_1 + dy_diff_0_1 = 0
dy_diff_def_1_2: - cy_1 + cy_2 + dy_diff_1_2 = 0
dy_diff_def_2_3: - cy_2 + cy_3 + dy_diff_2_3 = 0
dy_diff_def_2_4: - cy_2 + cy_4 + dy_diff_2_4 = 0
dy_diff_def_2_5: - cy_2 + cy_5 + dy_diff_2_5 = 0
fix_center_x_0: cx_0 = 15
fix_center_y_0: cy_0 = 15
h_align_grid_0: h_0 = 10
h_align_grid_1: h_1 = 5
h_align_grid_2: h_2 = 10
h_align_grid_3: h_3 = 5
h_align_grid_4: h_4 = 5
h_align_grid_5: h_5 = 5
h_max_0: h_0 <= 30
h_max_1: h_1 <= 30
h_max_2: h_2 <= 30
h_max_3: h_3 <= 30
h_max_4: h_4 <= 30
h_max_5: h_5 <= 30
height_rotation_0: h_0 = 10
height_rotation_1: h_1 = 5
height_rotation_2: h_2 = 10
height_rotation_3: h_3 = 5
height_rotation_4: h_4 = 5
height_rotation_5: h_5 = 5
horizontal_direction_0_1: z1L_0_1 + z1R_0_1 - z1_0_1 = 0
horizontal_direction_1_2: z1L_1_2 + z1R_1_2 - z1_1_2 = 0
horizontal_direction_2_3: z1L_2_3 + z1R_2_3 - z1_2_3 = 0
horizontal_direction_2_4: z1L_2_4 + z1R_2_4 - z1_2_4 = 0
horizontal_direction_2_5: z1L_2_5 + z1R_2_5 - z1_2_5 = 0
horizontal_left_dist_0_1: - w_0 - x_0 + x_1 + 200 z1L_0_1 <= 200.5
horizontal_left_dist_1_2: - w_1 - x_1 + x_2 + 200 z1L_1_2 <= 200.5
horizontal_left_dist_2_3: - w_2 - x_2 + x_3 + 200 z1L_2_3 <= 200.5
horizontal_left_dist_2_4: - w_2 - x_2 + x_4 + 200 z1L_2_4 <= 200.5
horizontal_left_dist_2_5: - w_2 - x_2 + x_5 + 200 z1L_2_5 <= 200.5
horizontal_left_dist_lb_0_1: - w_0 - x_0 + x_1 - 200 z1L_0_1 >= -200
horizontal_left_dist_lb_1_2: - w_1 - x_1 + x_2 - 200 z1L_1_2 >= -200
horizontal_left_dist_lb_2_3: - w_2 - x_2 + x_3 - 200 z1L_2_3 >= -200
horizontal_left_dist_lb_2_4: - w_2 - x_2 + x_4 - 200 z1L_2_4 >= -200
horizontal_left_dist_lb_2_5: - w_2 - x_2 + x_5 - 200 z1L_2_5 >= -200
horizontal_overlap_y1_0_1: - h_1 + y_0 - y_1 + 200 z1_0_1 <= 200
horizontal_overlap_y1_1_2: - h_2 + y_1 - y_2 + 200 z1_1_2 <= 200
horizontal_overlap_y1_2_3: - h_3 + y_2 - y_3 + 200 z1_2_3 <= 200
horizontal_overlap_y1_2_4: - h_4 + y_2 - y_4 + 200 z1_2_4 <= 200
horizontal_overlap_y1_2_5: - h_5 + y_2 - y_5 + 200 z1_2_5 <= 200
horizontal_overlap_y2_0_1: - h_0 - y_0 + y_1 + 200 z1_0_1 <= 200
horizontal_overlap_y2_1_2: - h_1 - y_1 + y_2 + 200 z1_1_2 <= 200
horizontal_overlap_y2_2_3: - h_2 - y_2 + y_3 + 200 z1_2_3 <= 200
horizontal_overlap_y2_2_4: - h_2 - y_2 + y_4 + 200 z1_2_4 <= 200
horizontal_overlap_y2_2_5: - h_2 - y_2 + y_5 + 200 z1_2_5 <= 200
horizontal_right_dist_0_1: - w_1 + x_0 - x_1 + 200 z1R_0_1 <= 200.5
horizontal_right_dist_1_2: - w_2 + x_1 - x_2 + 200 z1R_1_2 <= 200.5
horizontal_right_dist_2_3: - w_3 + x_2 - x_3 + 200 z1R_2_3 <= 200.5
horizontal_right_dist_2_4: - w_4 + x_2 - x_4 + 200 z1R_2_4 <= 200.5
horizontal_right_dist_2_5: - w_5 + x_2 - x_5 + 200 z1R_2_5 <= 200.5
horizontal_right_dist_lb_0_1: - w_1 + x_0 - x_1 - 200 z1R_0_1 >= -200
horizontal_right_dist_lb_1_2: - w_2 + x_1 - x_2 - 200 z1R_1_2 >= -200
horizontal_right_dist_lb_2_3: - w_3 + x_2 - x_3 - 200 z1R_2_3 >= -200
horizontal_right_dist_lb_2_4: - w_4 + x_2 - x_4 - 200 z1R_2_4 >= -200
horizontal_right_dist_lb_2_5: - w_5 + x_2 - x_5 - 200 z1R_2_5 >= -200
must_adjacent_0_1: z1_0_1 + z2_0_1 = 1
must_adjacent_1_2: z1_1_2 + z2_1_2 = 1
must_adjacent_2_3: z1_2_3 + z2_2_3 = 1
must_adjacent_2_4: z1_2_4 + z2_2_4 = 1
must_adjacent_2_5: z1_2_5 + z2_2_5 = 1
non_overlap_any_0_1: p_down_0_1 + p_left_0_1 + p_right_0_1 + p_up_0_1 = 1
non_overlap_any_0_2: p_down_0_2 + p_left_0_2 + p_right_0_2 + p_up_0_2 = 1
non_overlap_any_0_3: p_down_0_3 + p_left_0_3 + p_right_0_3 + p_up_0_3 = 1
non_overlap_any_0_4: p_down_0_4 + p_left_0_4 + p_right_0_4 + p_up_0_4 = 1
non_overlap_any_0_5: p_down_0_5 + p_left_0_5 + p_right_0_5 + p_up_0_5 = 1
non_overlap_any_1_2: p_down_1_2 + p_left_1_2 + p_right_1_2 + p_up_1_2 = 1
non_overlap_any_1_3: p_down_1_3 + p_left_1_3 + p_right_1_3 + p_up_1_3 = 1
non_overlap_any_1_4: p_down_1_4 + p_left_1_4 + p_right_1_4 + p_up_1_4 = 1
non_overlap_any_1_5: p_down_1_5 + p_left_1_5 + p_right_1_5 + p_up_1_5 = 1
non_overlap_any_2_3: p_down_2_3 + p_left_2_3 + p_right_2_3 + p_up_2_3 = 1
non_overlap_any_2_4: p_down_2_4 + p_left_2_4 + p_right_2_4 + p_up_2_4 = 1
non_overlap_any_2_5: p_down_2_5 + p_left_2_5 + p_right_2_5 + p_up_2_5 = 1
non_overlap_any_3_4: p_down_3_4 + p_left_3_4 + p_right_3_4 + p_up_3_4 = 1
non_overlap_any_3_5: p_down_3_5 + p_left_3_5 + p_right_3_5 + p_up_3_5 = 1
non_overlap_any_4_5: p_down_4_5 + p_left_4_5 + p_right_4_5 + p_up_4_5 = 1
non_overlap_down_0_1: h_0 + 200 p_down_0_1 + y_0 - y_1 <= 200
non_overlap_down_0_2: h_0 + 200 p_down_0_2 + y_0 - y_2 <= 200
non_overlap_down_0_3: h_0 + 200 p_down_0_3 + y_0 - y_3 <= 200
non_overlap_down_0_4: h_0 + 200 p_down_0_4 + y_0 - y_4 <= 200
non_overlap_down_0_5: h_0 + 200 p_down_0_5 + y_0 - y_5 <= 200
non_overlap_down_1_2: h_1 + 200 p_down_1_2 + y_1 - y_2 <= 200
non_overlap_down_1_3: h_1 + 200 p_down_1_3 + y_1 - y_3 <= 200
non_overlap_down_1_4: h_1 + 200 p_down_1_4 + y_1 - y_4 <= 200
non_overlap_down_1_5: h_1 + 200 p_down_1_5 + y_1 - y_5 <= 200
non_overlap_down_2_3: h_2 + 200 p_down_2_3 + y_2 - y_3 <= 200
non_overlap_down_2_4: h_2 + 200 p_down_2_4 + y_2 - y_4 <= 200
non_overlap_down_2_5: h_2 + 200 p_down_2_5 + y_2 - y_5 <= 200
non_overlap_down_3_4: h_3 + 200 p_down_3_4 + y_3 - y_4 <= 200
non_overlap_down_3_5: h_3 + 200 p_down_3_5 + y_3 - y_5 <= 200
non_overlap_down_4_5: h_4 + 200 p_down_4_5 + y_4 - y_5 <= 200
non_overlap_down_rev_0_1: - h_0 - 200 p_down_0_1 - y_0 + y_1 <= 0
non_overlap_down_rev_0_2: - h_0 - 200 p_down_0_2 - y_0 + y_2 <= 0
non_overlap_down_rev_0_3: - h_0 - 200 p_down_0_3 - y_0 + y_3 <= 0
non_overlap_down_rev_0_4: - h_0 - 200 p_down_0_4 - y_0 + y_4 <= 0
non_overlap_down_rev_0_5: - h_0 - 200 p_down_0_5 - y_0 + y_5 <= 0
non_overlap_down_rev_1_2: - h_1 - 200 p_down_1_2 - y_1 + y_2 <= 0
non_overlap_down_rev_1_3: - h_1 - 200 p_down_1_3 - y_1 + y_3 <= 0
non_overlap_down_rev_1_4: - h_1 - 200 p_down_1_4 - y_1 + y_4 <= 0
non_overlap_down_rev_1_5: - h_1 - 200 p_down_1_5 - y_1 + y_5 <= 0
non_overlap_down_rev_2_3: - h_2 - 200 p_down_2_3 - y_2 + y_3 <= 0
non_overlap_down_rev_2_4: - h_2 - 200 p_down_2_4 - y_2 + y_4 <= 0
non_overlap_down_rev_2_5: - h_2 - 200 p_down_2_5 - y_2 + y_5 <= 0
non_overlap_down_rev_3_4: - h_3 - 200 p_down_3_4 - y_3 + y_4 <= 0
non_overlap_down_rev_3_5: - h_3 - 200 p_down_3_5 - y_3 + y_5 <= 0
non_overlap_down_rev_4_5: - h_4 - 200 p_down_4_5 - y_4 + y_5 <= 0
non_overlap_left_0_1: 200 p_left_0_1 + w_0 + x_0 - x_1 <= 200
non_overlap_left_0_2: 200 p_left_0_2 + w_0 + x_0 - x_2 <= 200
non_overlap_left_0_3: 200 p_left_0_3 + w_0 + x_0 - x_3 <= 200
non_overlap_left_0_4: 200 p_left_0_4 + w_0 + x_0 - x_4 <= 200
non_overlap_left_0_5: 200 p_left_0_5 + w_0 + x_0 - x_5 <= 200
non_overlap_left_1_2: 200 p_left_1_2 + w_1 + x_1 - x_2 <= 200
non_overlap_left_1_3: 200 p_left_1_3 + w_1 + x_1 - x_3 <= 200
non_overlap_left_1_4: 200 p_left_1_4 + w_1 + x_1 - x_4 <= 200
non_overlap_left_1_5: 200 p_left_1_5 + w_1 + x_1 - x_5 <= 200
non_overlap_left_2_3: 200 p_left_2_3 + w_2 + x_2 - x_3 <= 200
non_overlap_left_2_4: 200 p_left_2_4 + w_2 + x_2 - x_4 <= 200
non_overlap_left_2_5: 200 p_left_2_5 + w_2 + x_2 - x_5 <= 200
non_overlap_left_3_4: 200 p_left_3_4 + w_3 + x_3 - x_4 <= 200
non_overlap_left_3_5: 200 p_left_3_5 + w_3 + x_3 - x_5 <= 200
non_overlap_left_4_5: 200 p_left_4_5 + w_4 + x_4 - x_5 <= 200
non_overlap_left_rev_0_1: - 200 p_left_0_1 - w_0 - x_0 + x_1 <= 0
non_overlap_left_rev_0_2: - 200 p_left_0_2 - w_0 - x_0 + x_2 <= 0
non_overlap_left_rev_0_3: - 200 p_left_0_3 - w_0 - x_0 + x_3 <= 0
non_overlap_left_rev_0_4: - 200 p_left_0_4 - w_0 - x_0 + x_4 <= 0
non_overlap_left_rev_0_5: - 200 p_left_0_5 - w_0 - x_0 + x_5 <= 0
non_overlap_left_rev_1_2: - 200 p_left_1_2 - w_1 - x_1 + x_2 <= 0
non_overlap_left_rev_1_3: - 200 p_left_1_3 - w_1 - x_1 + x_3 <= 0
non_overlap_left_rev_1_4: - 200 p_left_1_4 - w_1 - x_1 + x_4 <= 0
non_overlap_left_rev_1_5: - 200 p_left_1_5 - w_1 - x_1 + x_5 <= 0
non_overlap_left_rev_2_3: - 200 p_left_2_3 - w_2 - x_2 + x_3 <= 0
non_overlap_left_rev_2_4: - 200 p_left_2_4 - w_2 - x_2 + x_4 <= 0
non_overlap_left_rev_2_5: - 200 p_left_2_5 - w_2 - x_2 + x_5 <= 0
non_overlap_left_rev_3_4: - 200 p_left_3_4 - w_3 - x_3 + x_4 <= 0
non_overlap_left_rev_3_5: - 200 p_left_3_5 - w_3 - x_3 + x_5 <= 0
non_overlap_left_rev_4_5: - 200 p_left_4_5 - w_4 - x_4 + x_5 <= 0
non_overlap_right_0_1: 200 p_right_0_1 + w_1 - x_0 + x_1 <= 200
non_overlap_right_0_2: 200 p_right_0_2 + w_2 - x_0 + x_2 <= 200
non_overlap_right_0_3: 200 p_right_0_3 + w_3 - x_0 + x_3 <= 200
non_overlap_right_0_4: 200 p_right_0_4 + w_4 - x_0 + x_4 <= 200
non_overlap_right_0_5: 200 p_right_0_5 + w_5 - x_0 + x_5 <= 200
non_overlap_right_1_2: 200 p_right_1_2 + w_2 - x_1 + x_2 <= 200
non_overlap_right_1_3: 200 p_right_1_3 + w_3 - x_1 + x_3 <= 200
non_overlap_right_1_4: 200 p_right_1_4 + w_4 - x_1 + x_4 <= 200
non_overlap_right_1_5: 200 p_right_1_5 + w_5 - x_1 + x_5 <= 200
non_overlap_right_2_3: 200 p_right_2_3 + w_3 - x_2 + x_3 <= 200
non_overlap_right_2_4: 200 p_right_2_4 + w_4 - x_2 + x_4 <= 200
non_overlap_right_2_5: 200 p_right_2_5 + w_5 - x_2 + x_5 <= 200
non_overlap_right_3_4: 200 p_right_3_4 + w_4 - x_3 + x_4 <= 200
non_overlap_right_3_5: 200 p_right_3_5 + w_5 - x_3 + x_5 <= 200
non_overlap_right_4_5: 200 p_right_4_5 + w_5 - x_4 + x_5 <= 200
non_overlap_right_rev_0_1: - 200 p_right_0_1 - w_1 + x_0 - x_1 <= 0
non_overlap_right_rev_0_2: - 200 p_right_0_2 - w_2 + x_0 - x_2 <= 0
non_overlap_right_rev_0_3: - 200 p_right_0_3 - w_3 + x_0 - x_3 <= 0
non_overlap_right_rev_0_4: - 200 p_right_0_4 - w_4 + x_0 - x_4 <= 0
non_overlap_right_rev_0_5: - 200 p_right_0_5 - w_5 + x_0 - x_5 <= 0
non_overlap_right_rev_1_2: - 200 p_right_1_2 - w_2 + x_1 - x_2 <= 0
non_overlap_right_rev_1_3: - 200 p_right_1_3 - w_3 + x_1 - x_3 <= 0
non_overlap_right_rev_1_4: - 200 p_right_1_4 - w_4 + x_1 - x_4 <= 0
non_overlap_right_rev_1_5: - 200 p_right_1_5 - w_5 + x_1 - x_5 <= 0
non_overlap_right_rev_2_3: - 200 p_right_2_3 - w_3 + x_2 - x_3 <= 0
non_overlap_right_rev_2_4: - 200 p_right_2_4 - w_4 + x_2 - x_4 <= 0
non_overlap_right_rev_2_5: - 200 p_right_2_5 - w_5 + x_2 - x_5 <= 0
non_overlap_right_rev_3_4: - 200 p_right_3_4 - w_4 + x_3 - x_4 <= 0
non_overlap_right_rev_3_5: - 200 p_right_3_5 - w_5 + x_3 - x_5 <= 0
non_overlap_right_rev_4_5: - 200 p_right_4_5 - w_5 + x_4 - x_5 <= 0
non_overlap_up_0_1: h_1 + 200 p_up_0_1 - y_0 + y_1 <= 200
non_overlap_up_0_2: h_2 + 200 p_up_0_2 - y_0 + y_2 <= 200
non_overlap_up_0_3: h_3 + 200 p_up_0_3 - y_0 + y_3 <= 200
non_overlap_up_0_4: h_4 + 200 p_up_0_4 - y_0 + y_4 <= 200
non_overlap_up_0_5: h_5 + 200 p_up_0_5 - y_0 + y_5 <= 200
non_overlap_up_1_2: h_2 + 200 p_up_1_2 - y_1 + y_2 <= 200
non_overlap_up_1_3: h_3 + 200 p_up_1_3 - y_1 + y_3 <= 200
non_overlap_up_1_4: h_4 + 200 p_up_1_4 - y_1 + y_4 <= 200
non_overlap_up_1_5: h_5 + 200 p_up_1_5 - y_1 + y_5 <= 200
non_overlap_up_2_3: h_3 + 200 p_up_2_3 - y_2 + y_3 <= 200
non_overlap_up_2_4: h_4 + 200 p_up_2_4 - y_2 + y_4 <= 200
non_overlap_up_2_5: h_5 + 200 p_up_2_5 - y_2 + y_5 <= 200
non_overlap_up_3_4: h_4 + 200 p_up_3_4 - y_3 + y_4 <= 200
non_overlap_up_3_5: h_5 + 200 p_up_3_5 - y_3 + y_5 <= 200
non_overlap_up_4_5: h_5 + 200 p_up_4_5 - y_4 + y_5 <= 200
non_overlap_up_rev_0_1: - h_1 - 200 p_up_0_1 + y_0 - y_1 <= 0
non_overlap_up_rev_0_2: - h_2 - 200 p_up_0_2 + y_0 - y_2 <= 0
non_overlap_up_rev_0_3: - h_3 - 200 p_up_0_3 + y_0 - y_3 <= 0
non_overlap_up_rev_0_4: - h_4 - 200 p_up_0_4 + y_0 - y_4 <= 0
non_overlap_up_rev_0_5: - h_5 - 200 p_up_0_5 + y_0 - y_5 <= 0
non_overlap_up_rev_1_2: - h_2 - 200 p_up_1_2 + y_1 - y_2 <= 0
non_overlap_up_rev_1_3: - h_3 - 200 p_up_1_3 + y_1 - y_3 <= 0
non_overlap_up_rev_1_4: - h_4 - 200 p_up_1_4 + y_1 - y_4 <= 0
non_overlap_up_rev_1_5: - h_5 - 200 p_up_1_5 + y_1 - y_5 <= 0
non_overlap_up_rev_2_3: - h_3 - 200 p_up_2_3 + y_2 - y_3 <= 0
non_overlap_up_rev_2_4: - h_4 - 200 p_up_2_4 + y_2 - y_4 <= 0
non_overlap_up_rev_2_5: - h_5 - 200 p_up_2_5 + y_2 - y_5 <= 0
non_overlap_up_rev_3_4: - h_4 - 200 p_up_3_4 + y_3 - y_4 <= 0
non_overlap_up_rev_3_5: - h_5 - 200 p_up_3_5 + y_3 - y_5 <= 0
non_overlap_up_rev_4_5: - h_5 - 200 p_up_4_5 + y_4 - y_5 <= 0
shared_x_min_0_1: shared_x_0_1 - 200 z2_0_1 >= -199.9
shared_x_min_1_2: shared_x_1_2 - 200 z2_1_2 >= -199.9
shared_x_min_2_3: shared_x_2_3 - 200 z2_2_3 >= -199.9
shared_x_min_2_4: shared_x_2_4 - 200 z2_2_4 >= -199.9
shared_x_min_2_5: shared_x_2_5 - 200 z2_2_5 >= -199.9
shared_x_ub1_0_1: shared_x_0_1 - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
shared_x_ub1_1_2: shared_x_1_2 - w_1 - x_1 + x_2 + 200 z2_1_2 <= 200
shared_x_ub1_2_3: shared_x_2_3 - w_2 - x_2 + x_3 + 200 z2_2_3 <= 200
shared_x_ub1_2_4: shared_x_2_4 - w_2 - x_2 + x_4 + 200 z2_2_4 <= 200
shared_x_ub1_2_5: shared_x_2_5 - w_2 - x_2 + x_5 + 200 z2_2_5 <= 200
shared_x_ub2_0_1: shared_x_0_1 - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
shared_x_ub2_1_2: shared_x_1_2 - w_2 + x_1 - x_2 + 200 z2_1_2 <= 200
shared_x_ub2_2_3: shared_x_2_3 - w_3 + x_2 - x_3 + 200 z2_2_3 <= 200
shared_x_ub2_2_4: shared_x_2_4 - w_4 + x_2 - x_4 + 200 z2_2_4 <= 200
shared_x_ub2_2_5: shared_x_2_5 - w_5 + x_2 - x_5 + 200 z2_2_5 <= 200
shared_x_zero_0_1: shared_x_0_1 - 200 z2_0_1 <= 0
shared_x_zero_1_2: shared_x_1_2 - 200 z2_1_2 <= 0
shared_x_zero_2_3: shared_x_2_3 - 200 z2_2_3 <= 0
shared_x_zero_2_4: shared_x_2_4 - 200 z2_2_4 <= 0
shared_x_zero_2_5: shared_x_2_5 - 200 z2_2_5 <= 0
shared_y_min_0_1: shared_y_0_1 - 200 z1_0_1 >= -199.9
shared_y_min_1_2: shared_y_1_2 - 200 z1_1_2 >= -199.9
shared_y_min_2_3: shared_y_2_3 - 200 z1_2_3 >= -199.9
shared_y_min_2_4: shared_y_2_4 - 200 z1_2_4 >= -199.9
shared_y_min_2_5: shared_y_2_5 - 200 z1_2_5 >= -199.9
shared_y_ub1_0_1: - h_0 + shared_y_0_1 - y_0 + y_1 + 200 z1_0_1 <= 200
shared_y_ub1_1_2: - h_1 + shared_y_1_2 - y_1 + y_2 + 200 z1_1_2 <= 200
shared_y_ub1_2_3: - h_2 + shared_y_2_3 - y_2 + y_3 + 200 z1_2_3 <= 200
shared_y_ub1_2_4: - h_2 + shared_y_2_4 - y_2 + y_4 + 200 z1_2_4 <= 200
shared_y_ub1_2_5: - h_2 + shared_y_2_5 - y_2 + y_5 + 200 z1_2_5 <= 200
shared_y_ub2_0_1: - h_1 + shared_y_0_1 + y_0 - y_1 + 200 z1_0_1 <= 200
shared_y_ub2_1_2: - h_2 + shared_y_1_2 + y_1 - y_2 + 200 z1_1_2 <= 200
shared_y_ub2_2_3: - h_3 + shared_y_2_3 + y_2 - y_3 + 200 z1_2_3 <= 200
shared_y_ub2_2_4: - h_4 + shared_y_2_4 + y_2 - y_4 + 200 z1_2_4 <= 200
shared_y_ub2_2_5: - h_5 + shared_y_2_5 + y_2 - y_5 + 200 z1_2_5 <= 200
shared_y_zero_0_1: shared_y_0_1 - 200 z1_0_1 <= 0
shared_y_zero_1_2: shared_y_1_2 - 200 z1_1_2 <= 0
shared_y_zero_2_3: shared_y_2_3 - 200 z1_2_3 <= 0
shared_y_zero_2_4: shared_y_2_4 - 200 z1_2_4 <= 0
shared_y_zero_2_5: shared_y_2_5 - 200 z1_2_5 <= 0
t_ge_height: bbox_area_proxy_t - bbox_h >= 0
t_ge_scaled_mean: bbox_area_proxy_t - 0.8 bbox_h - 0.8 bbox_w >= 0
t_ge_width: bbox_area_proxy_t - bbox_w >= 0
vertical_direction_0_1: z2D_0_1 + z2U_0_1 - z2_0_1 = 0
vertical_direction_1_2: z2D_1_2 + z2U_1_2 - z2_1_2 = 0
vertical_direction_2_3: z2D_2_3 + z2U_2_3 - z2_2_3 = 0
vertical_direction_2_4: z2D_2_4 + z2U_2_4 - z2_2_4 = 0
vertical_direction_2_5: z2D_2_5 + z2U_2_5 - z2_2_5 = 0
vertical_down_dist_0_1: - h_0 - y_0 + y_1 + 200 z2D_0_1 <= 200.5
vertical_down_dist_1_2: - h_1 - y_1 + y_2 + 200 z2D_1_2 <= 200.5
vertical_down_dist_2_3: - h_2 - y_2 + y_3 + 200 z2D_2_3 <= 200.5
vertical_down_dist_2_4: - h_2 - y_2 + y_4 + 200 z2D_2_4 <= 200.5
vertical_down_dist_2_5: - h_2 - y_2 + y_5 + 200 z2D_2_5 <= 200.5
vertical_down_dist_lb_0_1: - h_0 - y_0 + y_1 - 200 z2D_0_1 >= -200
vertical_down_dist_lb_1_2: - h_1 - y_1 + y_2 - 200 z2D_1_2 >= -200
vertical_down_dist_lb_2_3: - h_2 - y_2 + y_3 - 200 z2D_2_3 >= -200
vertical_down_dist_lb_2_4: - h_2 - y_2 + y_4 - 200 z2D_2_4 >= -200
vertical_down_dist_lb_2_5: - h_2 - y_2 + y_5 - 200 z2D_2_5 >= -200
vertical_overlap_x1_0_1: - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
vertical_overlap_x1_1_2: - w_2 + x_1 - x_2 + 200 z2_1_2 <= 200
vertical_overlap_x1_2_3: - w_3 + x_2 - x_3 + 200 z2_2_3 <= 200
vertical_overlap_x1_2_4: - w_4 + x_2 - x_4 + 200 z2_2_4 <= 200
vertical_overlap_x1_2_5: - w_5 + x_2 - x_5 + 200 z2_2_5 <= 200
vertical_overlap_x2_0_1: - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
vertical_overlap_x2_1_2: - w_1 - x_1 + x_2 + 200 z2_1_2 <= 200
vertical_overlap_x2_2_3: - w_2 - x_2 + x_3 + 200 z2_2_3 <= 200
vertical_overlap_x2_2_4: - w_2 - x_2 + x_4 + 200 z2_2_4 <= 200
vertical_overlap_x2_2_5: - w_2 - x_2 + x_5 + 200 z2_2_5 <= 200
vertical_up_dist_0_1: - h_1 + y_0 - y_1 + 200 z2U_0_1 <= 200.5
vertical_up_dist_1_2: - h_2 + y_1 - y_2 + 200 z2U_1_2 <= 200.5
vertical_up_dist_2_3: - h_3 + y_2 - y_3 + 200 z2U_2_3 <= 200.5
vertical_up_dist_2_4: - h_4 + y_2 - y_4 + 200 z2U_2_4 <= 200.5
vertical_up_dist_2_5: - h_5 + y_2 - y_5 + 200 z2U_2_5 <= 200.5
vertical_up_dist_lb_0_1: - h_1 + y_0 - y_1 - 200 z2U_0_1 >= -200
vertical_up_dist_lb_1_2: - h_2 + y_1 - y_2 - 200 z2U_1_2 >= -200
vertical_up_dist_lb_2_3: - h_3 + y_2 - y_3 - 200 z2U_2_3 >= -200
vertical_up_dist_lb_2_4: - h_4 + y_2 - y_4 - 200 z2U_2_4 >= -200
vertical_up_dist_lb_2_5: - h_5 + y_2 - y_5 - 200 z2U_2_5 >= -200
w_align_grid_0: w_0 = 10
w_align_grid_1: w_1 = 5
w_align_grid_2: w_2 = 10
w_align_grid_3: w_3 = 5
w_align_grid_4: w_4 = 5
w_align_grid_5: w_5 = 5
w_max_0: w_0 <= 30
w_max_1: w_1 <= 30
w_max_2: w_2 <= 30
w_max_3: w_3 <= 30
w_max_4: w_4 <= 30
w_max_5: w_5 <= 30
width_rotation_0: w_0 = 10
width_rotation_1: w_1 = 5
width_rotation_2: w_2 = 10
width_rotation_3: w_3 = 5
width_rotation_4: w_4 = 5
width_rotation_5: w_5 = 5
x_grid_to_real_0: x_0 - 0.5 x_grid_0 = 0
x_grid_to_real_1: x_1 - 0.5 x_grid_1 = 0
x_grid_to_real_2: x_2 - 0.5 x_grid_2 = 0
x_grid_to_real_3: x_3 - 0.5 x_grid_3 = 0
x_grid_to_real_4: x_4 - 0.5 x_grid_4 = 0
x_grid_to_real_5: x_5 - 0.5 x_grid_5 = 0
y_grid_to_real_0: y_0 - 0.5 y_grid_0 = 0
y_grid_to_real_1: y_1 - 0.5 y_grid_1 = 0
y_grid_to_real_2: y_2 - 0.5 y_grid_2 = 0
y_grid_to_real_3: y_3 - 0.5 y_grid_3 = 0
y_grid_to_real_4: y_4 - 0.5 y_grid_4 = 0
y_grid_to_real_5: y_5 - 0.5 y_grid_5 = 0
Bounds
 bbox_area_proxy_t <= 60
 bbox_h <= 30
 bbox_max_x <= 30
 bbox_max_y <= 30
 bbox_min_x <= 30
 bbox_min_y <= 30
 bbox_w <= 30
 cx_0 <= 30
 cx_1 <= 30
 cx_2 <= 30
 cx_3 <= 30
 cx_4 <= 30
 cx_5 <= 30
 cy_0 <= 30
 cy_1 <= 30
 cy_2 <= 30
 cy_3 <= 30
 cy_4 <= 30
 cy_5 <= 30
 -30 <= dx_diff_0_1 <= 30
 -30 <= dx_diff_1_2 <= 30
 -30 <= dx_diff_2_3 <= 30
 -30 <= dx_diff_2_4 <= 30
 -30 <= dx_diff_2_5 <= 30
 -30 <= dy_diff_0_1 <= 30
 -30 <= dy_diff_1_2 <= 30
 -30 <= dy_diff_2_3 <= 30
 -30 <= dy_diff_2_4 <= 30
 -30 <= dy_diff_2_5 <= 30
 h_0 = 10
 h_1 = 5
 h_2 = 10
 h_3 = 5
 h_4 = 5
 h_5 = 5
 shared_x_0_1 <= 10
 shared_x_1_2 <= 10
 shared_x_2_3 <= 10
 shared_x_2_4 <= 10
 shared_x_2_5 <= 10
 shared_y_0_1 <= 10
 shared_y_1_2 <= 10
 shared_y_2_3 <= 10
 shared_y_2_4 <= 10
 shared_y_2_5 <= 10
 w_0 = 10
 w_1 = 5
 w_2 = 10
 w_3 = 5
 w_4 = 5
 w_5 = 5
 x_0 <= 30
 x_1 <= 30
 x_2 <= 30
 x_3 <= 30
 x_4 <= 30
 x_5 <= 30
 0 <= x_grid_0 <= 59
 0 <= x_grid_1 <= 59
 0 <= x_grid_2 <= 59
 0 <= x_grid_3 <= 59
 0 <= x_grid_4 <= 59
 0 <= x_grid_5 <= 59
 y_0 <= 30
 y_1 <= 30
 y_2 <= 30
 y_3 <= 30
 y_4 <= 30
 y_5 <= 30
 0 <= y_grid_0 <= 59
 0 <= y_grid_1 <= 59
 0 <= y_grid_2 <= 59
 0 <= y_grid_3 <= 59
 0 <= y_grid_4 <= 59
 0 <= y_grid_5 <= 59
Generals
x_grid_0
x_grid_1
x_grid_2
x_grid_3
x_grid_4
x_grid_5
y_grid_0
y_grid_1
y_grid_2
y_grid_3
y_grid_4
y_grid_5
Binaries
dx_abs_0_1_is_positive
dx_abs_1_2_is_positive
dx_abs_2_3_is_positive
dx_abs_2_4_is_positive
dx_abs_2_5_is_positive
dy_abs_0_1_is_positive
dy_abs_1_2_is_positive
dy_abs_2_3_is_positive
dy_abs_2_4_is_positive
dy_abs_2_5_is_positive
p_down_0_1
p_down_0_2
p_down_0_3
p_down_0_4
p_down_0_5
p_down_1_2
p_down_1_3
p_down_1_4
p_down_1_5
p_down_2_3
p_down_2_4
p_down_2_5
p_down_3_4
p_down_3_5
p_down_4_5
p_left_0_1
p_left_0_2
p_left_0_3
p_left_0_4
p_left_0_5
p_left_1_2
p_left_1_3
p_left_1_4
p_left_1_5
p_left_2_3
p_left_2_4
p_left_2_5
p_left_3_4
p_left_3_5
p_left_4_5
p_right_0_1
p_right_0_2
p_right_0_3
p_right_0_4
p_right_0_5
p_right_1_2
p_right_1_3
p_right_1_4
p_right_1_5
p_right_2_3
p_right_2_4
p_right_2_5
p_right_3_4
p_right_3_5
p_right_4_5
p_up_0_1
p_up_0_2
p_up_0_3
p_up_0_4
p_up_0_5
p_up_1_2
p_up_1_3
p_up_1_4
p_up_1_5
p_up_2_3
p_up_2_4
p_up_2_5
p_up_3_4
p_up_3_5
p_up_4_5
z1L_0_1
z1L_1_2
z1L_2_3
z1L_2_4
z1L_2_5
z1R_0_1
z1R_1_2
z1R_2_3
z1R_2_4
z1R_2_5
z1_0_1
z1_1_2
z1_2_3
z1_2_4
z1_2_5
z2D_0_1
z2D_1_2
z2D_2_3
z2D_2_4
z2D_2_5
z2U_0_1
z2U_1_2
z2U_2_3
z2U_2_4
z2U_2_5
z2_0_1
z2_1_2
z2_2_3
z2_2_4
z2_2_5
End
