\* ChipletPlacementGrid *\
Minimize
Objective: 0.1 bbox_area_proxy_t + dx_abs_0_1 + dy_abs_0_1
Subject To
area_proxy_am_gm1: bbox_area_proxy_t - 0.5 bbox_h - 0.5 bbox_w <= 0
area_proxy_am_gm2: - 30 bbox_area_proxy_t + bbox_w <= 0
area_proxy_am_gm3: - 30 bbox_area_proxy_t + bbox_h <= 0
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
boundary_x_0: w_0 + x_0 <= 30
boundary_x_1: w_1 + x_1 <= 30
boundary_y_0: h_0 + y_0 <= 30
boundary_y_1: h_1 + y_1 <= 30
cx_def_0: cx_0 - 0.5 w_0 - x_0 = 0
cx_def_1: cx_1 - 0.5 w_1 - x_1 = 0
cy_def_0: cy_0 - 0.5 h_0 - y_0 = 0
cy_def_1: cy_1 - 0.5 h_1 - y_1 = 0
diff_pos_fixed_c10_p0_0: diff_pos_c10_p0_0 = 0
diff_pos_fixed_c11_p1_0: diff_pos_c11_p1_0 = 0
diff_pos_fixed_c12_p2_0: diff_pos_c12_p2_0 = 0
diff_pos_fixed_c13_p3_0: diff_pos_c13_p3_0 = 0
diff_pos_fixed_c14_p4_0: diff_pos_c14_p4_0 = 0
diff_pos_from_x_c10_p0_1: diff_pos_c10_p0_1 - x_diff_c10_p0_1 >= 0
diff_pos_from_x_c11_p1_1: diff_pos_c11_p1_1 - x_diff_c11_p1_1 >= 0
diff_pos_from_x_c12_p2_1: diff_pos_c12_p2_1 - x_diff_c12_p2_1 >= 0
diff_pos_from_x_c13_p3_1: diff_pos_c13_p3_1 - x_diff_c13_p3_1 >= 0
diff_pos_from_x_c14_p4_1: diff_pos_c14_p4_1 - x_diff_c14_p4_1 >= 0
diff_pos_from_y_c10_p0_1: diff_pos_c10_p0_1 - y_diff_c10_p0_1 >= 0
diff_pos_from_y_c11_p1_1: diff_pos_c11_p1_1 - y_diff_c11_p1_1 >= 0
diff_pos_from_y_c12_p2_1: diff_pos_c12_p2_1 - y_diff_c12_p2_1 >= 0
diff_pos_from_y_c13_p3_1: diff_pos_c13_p3_1 - y_diff_c13_p3_1 >= 0
diff_pos_from_y_c14_p4_1: diff_pos_c14_p4_1 - y_diff_c14_p4_1 >= 0
diff_pos_upper_c10_p0_1: diff_pos_c10_p0_1 - x_diff_c10_p0_1 - y_diff_c10_p0_1
 <= 0
diff_pos_upper_c11_p1_1: diff_pos_c11_p1_1 - x_diff_c11_p1_1 - y_diff_c11_p1_1
 <= 0
diff_pos_upper_c12_p2_1: diff_pos_c12_p2_1 - x_diff_c12_p2_1 - y_diff_c12_p2_1
 <= 0
diff_pos_upper_c13_p3_1: diff_pos_c13_p3_1 - x_diff_c13_p3_1 - y_diff_c13_p3_1
 <= 0
diff_pos_upper_c14_p4_1: diff_pos_c14_p4_1 - x_diff_c14_p4_1 - y_diff_c14_p4_1
 <= 0
dist_diff_abs_pair_lower_c10_p0_0_1: - 60 diff_dist_pair_c10_p0_0_1
 + dist_diff_abs_pair_c10_p0_0_1 >= -57
dist_diff_abs_pair_lower_c11_p1_0_1: - 60 diff_dist_pair_c11_p1_0_1
 + dist_diff_abs_pair_c11_p1_0_1 >= -57
dist_diff_abs_pair_lower_c12_p2_0_1: - 60 diff_dist_pair_c12_p2_0_1
 + dist_diff_abs_pair_c12_p2_0_1 >= -57
dist_diff_abs_pair_lower_c13_p3_0_1: - 60 diff_dist_pair_c13_p3_0_1
 + dist_diff_abs_pair_c13_p3_0_1 >= -57
dist_diff_abs_pair_lower_c14_p4_0_1: - 60 diff_dist_pair_c14_p4_0_1
 + dist_diff_abs_pair_c14_p4_0_1 >= -57
dist_diff_abs_pair_neg_c10_p0_0_1: dist_diff_abs_pair_c10_p0_0_1
 + dist_diff_pair_c10_p0_0_1 >= 0
dist_diff_abs_pair_neg_c11_p1_0_1: dist_diff_abs_pair_c11_p1_0_1
 + dist_diff_pair_c11_p1_0_1 >= 0
dist_diff_abs_pair_neg_c12_p2_0_1: dist_diff_abs_pair_c12_p2_0_1
 + dist_diff_pair_c12_p2_0_1 >= 0
dist_diff_abs_pair_neg_c13_p3_0_1: dist_diff_abs_pair_c13_p3_0_1
 + dist_diff_pair_c13_p3_0_1 >= 0
dist_diff_abs_pair_neg_c14_p4_0_1: dist_diff_abs_pair_c14_p4_0_1
 + dist_diff_pair_c14_p4_0_1 >= 0
dist_diff_abs_pair_pos_c10_p0_0_1: dist_diff_abs_pair_c10_p0_0_1
 - dist_diff_pair_c10_p0_0_1 >= 0
dist_diff_abs_pair_pos_c11_p1_0_1: dist_diff_abs_pair_c11_p1_0_1
 - dist_diff_pair_c11_p1_0_1 >= 0
dist_diff_abs_pair_pos_c12_p2_0_1: dist_diff_abs_pair_c12_p2_0_1
 - dist_diff_pair_c12_p2_0_1 >= 0
dist_diff_abs_pair_pos_c13_p3_0_1: dist_diff_abs_pair_c13_p3_0_1
 - dist_diff_pair_c13_p3_0_1 >= 0
dist_diff_abs_pair_pos_c14_p4_0_1: dist_diff_abs_pair_c14_p4_0_1
 - dist_diff_pair_c14_p4_0_1 >= 0
dist_diff_abs_pair_upper_c10_p0_0_1: - 60 diff_dist_pair_c10_p0_0_1
 + dist_diff_abs_pair_c10_p0_0_1 <= 2.97
dist_diff_abs_pair_upper_c11_p1_0_1: - 60 diff_dist_pair_c11_p1_0_1
 + dist_diff_abs_pair_c11_p1_0_1 <= 2.97
dist_diff_abs_pair_upper_c12_p2_0_1: - 60 diff_dist_pair_c12_p2_0_1
 + dist_diff_abs_pair_c12_p2_0_1 <= 2.97
dist_diff_abs_pair_upper_c13_p3_0_1: - 60 diff_dist_pair_c13_p3_0_1
 + dist_diff_abs_pair_c13_p3_0_1 <= 2.97
dist_diff_abs_pair_upper_c14_p4_0_1: - 60 diff_dist_pair_c14_p4_0_1
 + dist_diff_abs_pair_c14_p4_0_1 <= 2.97
dist_diff_pair_def_c10_p0_0_1: dist_diff_pair_c10_p0_0_1
 - dx_abs_pair_c10_p0_0_1 - dy_abs_pair_c10_p0_0_1 = -10
dist_diff_pair_def_c11_p1_0_1: dist_diff_pair_c11_p1_0_1
 - dx_abs_pair_c11_p1_0_1 - dy_abs_pair_c11_p1_0_1 = -10
dist_diff_pair_def_c12_p2_0_1: dist_diff_pair_c12_p2_0_1
 - dx_abs_pair_c12_p2_0_1 - dy_abs_pair_c12_p2_0_1 = -10
dist_diff_pair_def_c13_p3_0_1: dist_diff_pair_c13_p3_0_1
 - dx_abs_pair_c13_p3_0_1 - dy_abs_pair_c13_p3_0_1 = -10
dist_diff_pair_def_c14_p4_0_1: dist_diff_pair_c14_p4_0_1
 - dx_abs_pair_c14_p4_0_1 - dy_abs_pair_c14_p4_0_1 = -13
dx_abs1_0_1: - cx_0 + cx_1 + dx_abs_0_1 >= 0
dx_abs2_0_1: cx_0 - cx_1 + dx_abs_0_1 >= 0
dx_abs_pair_neg_c10_p0_0_1: cx_0 - cx_1 + dx_abs_pair_c10_p0_0_1 >= 0
dx_abs_pair_neg_c11_p1_0_1: cx_0 - cx_1 + dx_abs_pair_c11_p1_0_1 >= 0
dx_abs_pair_neg_c12_p2_0_1: cx_0 - cx_1 + dx_abs_pair_c12_p2_0_1 >= 0
dx_abs_pair_neg_c13_p3_0_1: cx_0 - cx_1 + dx_abs_pair_c13_p3_0_1 >= 0
dx_abs_pair_neg_c14_p4_0_1: cx_0 - cx_1 + dx_abs_pair_c14_p4_0_1 >= 0
dx_abs_pair_pos_c10_p0_0_1: - cx_0 + cx_1 + dx_abs_pair_c10_p0_0_1 >= 0
dx_abs_pair_pos_c11_p1_0_1: - cx_0 + cx_1 + dx_abs_pair_c11_p1_0_1 >= 0
dx_abs_pair_pos_c12_p2_0_1: - cx_0 + cx_1 + dx_abs_pair_c12_p2_0_1 >= 0
dx_abs_pair_pos_c13_p3_0_1: - cx_0 + cx_1 + dx_abs_pair_c13_p3_0_1 >= 0
dx_abs_pair_pos_c14_p4_0_1: - cx_0 + cx_1 + dx_abs_pair_c14_p4_0_1 >= 0
dy_abs1_0_1: - cy_0 + cy_1 + dy_abs_0_1 >= 0
dy_abs2_0_1: cy_0 - cy_1 + dy_abs_0_1 >= 0
dy_abs_pair_neg_c10_p0_0_1: cy_0 - cy_1 + dy_abs_pair_c10_p0_0_1 >= 0
dy_abs_pair_neg_c11_p1_0_1: cy_0 - cy_1 + dy_abs_pair_c11_p1_0_1 >= 0
dy_abs_pair_neg_c12_p2_0_1: cy_0 - cy_1 + dy_abs_pair_c12_p2_0_1 >= 0
dy_abs_pair_neg_c13_p3_0_1: cy_0 - cy_1 + dy_abs_pair_c13_p3_0_1 >= 0
dy_abs_pair_neg_c14_p4_0_1: cy_0 - cy_1 + dy_abs_pair_c14_p4_0_1 >= 0
dy_abs_pair_pos_c10_p0_0_1: - cy_0 + cy_1 + dy_abs_pair_c10_p0_0_1 >= 0
dy_abs_pair_pos_c11_p1_0_1: - cy_0 + cy_1 + dy_abs_pair_c11_p1_0_1 >= 0
dy_abs_pair_pos_c12_p2_0_1: - cy_0 + cy_1 + dy_abs_pair_c12_p2_0_1 >= 0
dy_abs_pair_pos_c13_p3_0_1: - cy_0 + cy_1 + dy_abs_pair_c13_p3_0_1 >= 0
dy_abs_pair_pos_c14_p4_0_1: - cy_0 + cy_1 + dy_abs_pair_c14_p4_0_1 >= 0
exclude_solution_dist_pair_c10_p0: diff_dist_pair_c10_p0_0_1 >= 1
exclude_solution_dist_pair_c11_p1: diff_dist_pair_c11_p1_0_1 >= 1
exclude_solution_dist_pair_c12_p2: diff_dist_pair_c12_p2_0_1 >= 1
exclude_solution_dist_pair_c13_p3: diff_dist_pair_c13_p3_0_1 >= 1
exclude_solution_dist_pair_c14_p4: diff_dist_pair_c14_p4_0_1 >= 1
exclude_solution_pos_c10_p0: diff_pos_c10_p0_1 >= 1
exclude_solution_pos_c11_p1: diff_pos_c11_p1_1 >= 1
exclude_solution_pos_c12_p2: diff_pos_c12_p2_1 >= 1
exclude_solution_pos_c13_p3: diff_pos_c13_p3_1 >= 1
exclude_solution_pos_c14_p4: diff_pos_c14_p4_1 >= 1
fix_center_x_0: cx_0 = 15
fix_center_y_0: cy_0 = 15
h_align_grid_0: h_0 - 0.5 h_grid_0 = 0
h_align_grid_1: h_1 - 0.5 h_grid_1 = 0
h_grid_lb_0: h_grid_0 >= 19.999
h_grid_lb_1: h_grid_1 >= 19.999
height_rotation_0: h_0 = 10
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
non_overlap_down_0_1: h_0 + 200 p_down_0_1 + y_0 - y_1 <= 200.001
non_overlap_down_rev_0_1: - h_0 - 200 p_down_0_1 - y_0 + y_1 <= 0.001
non_overlap_left_0_1: 200 p_left_0_1 + w_0 + x_0 - x_1 <= 200.001
non_overlap_left_rev_0_1: - 200 p_left_0_1 - w_0 - x_0 + x_1 <= 0.001
non_overlap_right_0_1: 200 p_right_0_1 + w_1 - x_0 + x_1 <= 200.001
non_overlap_right_rev_0_1: - 200 p_right_0_1 - w_1 + x_0 - x_1 <= 0.001
non_overlap_up_0_1: h_1 + 200 p_up_0_1 - y_0 + y_1 <= 200.001
non_overlap_up_rev_0_1: - h_1 - 200 p_up_0_1 + y_0 - y_1 <= 0.001
shared_x_min_0_1: shared_x_0_1 - 200 z2_0_1 >= -199.9
shared_x_ub1_0_1: shared_x_0_1 - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
shared_x_ub2_0_1: shared_x_0_1 - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
shared_x_zero_0_1: shared_x_0_1 - 200 z2_0_1 <= 0
shared_y_min_0_1: shared_y_0_1 - 200 z1_0_1 >= -199.9
shared_y_ub1_0_1: - h_0 + shared_y_0_1 - y_0 + y_1 + 200 z1_0_1 <= 200
shared_y_ub2_0_1: - h_1 + shared_y_0_1 + y_0 - y_1 + 200 z1_0_1 <= 200
shared_y_zero_0_1: shared_y_0_1 - 200 z1_0_1 <= 0
vertical_direction_0_1: z2D_0_1 + z2U_0_1 - z2_0_1 = 0
vertical_down_dist_0_1: - h_0 - y_0 + y_1 + 200 z2D_0_1 <= 200.5
vertical_down_dist_lb_0_1: - h_0 - y_0 + y_1 - 200 z2D_0_1 >= -200
vertical_overlap_x1_0_1: - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
vertical_overlap_x2_0_1: - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
vertical_up_dist_0_1: - h_1 + y_0 - y_1 + 200 z2U_0_1 <= 200.5
vertical_up_dist_lb_0_1: - h_1 + y_0 - y_1 - 200 z2U_0_1 >= -200
w_align_grid_0: w_0 - 0.5 w_grid_0 = 0
w_align_grid_1: w_1 - 0.5 w_grid_1 = 0
w_grid_lb_0: w_grid_0 >= 19.999
w_grid_lb_1: w_grid_1 >= 19.999
width_rotation_0: w_0 = 10
width_rotation_1: w_1 = 10
x_diff_from_minus_c10_p0_1: x_diff_c10_p0_1 - x_diff_minus_c10_p0_1 >= 0
x_diff_from_minus_c11_p1_1: x_diff_c11_p1_1 - x_diff_minus_c11_p1_1 >= 0
x_diff_from_minus_c12_p2_1: x_diff_c12_p2_1 - x_diff_minus_c12_p2_1 >= 0
x_diff_from_minus_c13_p3_1: x_diff_c13_p3_1 - x_diff_minus_c13_p3_1 >= 0
x_diff_from_minus_c14_p4_1: x_diff_c14_p4_1 - x_diff_minus_c14_p4_1 >= 0
x_diff_from_plus_c10_p0_1: x_diff_c10_p0_1 - x_diff_plus_c10_p0_1 >= 0
x_diff_from_plus_c11_p1_1: x_diff_c11_p1_1 - x_diff_plus_c11_p1_1 >= 0
x_diff_from_plus_c12_p2_1: x_diff_c12_p2_1 - x_diff_plus_c12_p2_1 >= 0
x_diff_from_plus_c13_p3_1: x_diff_c13_p3_1 - x_diff_plus_c13_p3_1 >= 0
x_diff_from_plus_c14_p4_1: x_diff_c14_p4_1 - x_diff_plus_c14_p4_1 >= 0
x_diff_minus_lower_c10_p0_1: x_1 + 60 x_diff_minus_c10_p0_1 <= 67
x_diff_minus_lower_c11_p1_1: x_1 + 60 x_diff_minus_c11_p1_1 <= 57
x_diff_minus_lower_c12_p2_1: x_1 + 60 x_diff_minus_c12_p2_1 <= 67
x_diff_minus_lower_c13_p3_1: x_1 + 60 x_diff_minus_c13_p3_1 <= 77
x_diff_minus_lower_c14_p4_1: x_1 + 60 x_diff_minus_c14_p4_1 <= 70
x_diff_minus_upper_c10_p0_1: x_1 + 60 x_diff_minus_c10_p0_1 >= 7.001
x_diff_minus_upper_c11_p1_1: x_1 + 60 x_diff_minus_c11_p1_1 >= -2.999
x_diff_minus_upper_c12_p2_1: x_1 + 60 x_diff_minus_c12_p2_1 >= 7.001
x_diff_minus_upper_c13_p3_1: x_1 + 60 x_diff_minus_c13_p3_1 >= 17.001
x_diff_minus_upper_c14_p4_1: x_1 + 60 x_diff_minus_c14_p4_1 >= 10.001
x_diff_plus_lower_c10_p0_1: x_1 - 60 x_diff_plus_c10_p0_1 >= -47
x_diff_plus_lower_c11_p1_1: x_1 - 60 x_diff_plus_c11_p1_1 >= -57
x_diff_plus_lower_c12_p2_1: x_1 - 60 x_diff_plus_c12_p2_1 >= -47
x_diff_plus_lower_c13_p3_1: x_1 - 60 x_diff_plus_c13_p3_1 >= -37
x_diff_plus_lower_c14_p4_1: x_1 - 60 x_diff_plus_c14_p4_1 >= -44
x_diff_plus_upper_c10_p0_1: x_1 - 60 x_diff_plus_c10_p0_1 <= 12.999
x_diff_plus_upper_c11_p1_1: x_1 - 60 x_diff_plus_c11_p1_1 <= 2.999
x_diff_plus_upper_c12_p2_1: x_1 - 60 x_diff_plus_c12_p2_1 <= 12.999
x_diff_plus_upper_c13_p3_1: x_1 - 60 x_diff_plus_c13_p3_1 <= 22.999
x_diff_plus_upper_c14_p4_1: x_1 - 60 x_diff_plus_c14_p4_1 <= 15.999
x_diff_upper_c10_p0_1: x_diff_c10_p0_1 - x_diff_minus_c10_p0_1
 - x_diff_plus_c10_p0_1 <= 0
x_diff_upper_c11_p1_1: x_diff_c11_p1_1 - x_diff_minus_c11_p1_1
 - x_diff_plus_c11_p1_1 <= 0
x_diff_upper_c12_p2_1: x_diff_c12_p2_1 - x_diff_minus_c12_p2_1
 - x_diff_plus_c12_p2_1 <= 0
x_diff_upper_c13_p3_1: x_diff_c13_p3_1 - x_diff_minus_c13_p3_1
 - x_diff_plus_c13_p3_1 <= 0
x_diff_upper_c14_p4_1: x_diff_c14_p4_1 - x_diff_minus_c14_p4_1
 - x_diff_plus_c14_p4_1 <= 0
x_grid_to_real_0: x_0 - 0.5 x_grid_0 = 0
x_grid_to_real_1: x_1 - 0.5 x_grid_1 = 0
y_diff_from_minus_c10_p0_1: y_diff_c10_p0_1 - y_diff_minus_c10_p0_1 >= 0
y_diff_from_minus_c11_p1_1: y_diff_c11_p1_1 - y_diff_minus_c11_p1_1 >= 0
y_diff_from_minus_c12_p2_1: y_diff_c12_p2_1 - y_diff_minus_c12_p2_1 >= 0
y_diff_from_minus_c13_p3_1: y_diff_c13_p3_1 - y_diff_minus_c13_p3_1 >= 0
y_diff_from_minus_c14_p4_1: y_diff_c14_p4_1 - y_diff_minus_c14_p4_1 >= 0
y_diff_from_plus_c10_p0_1: y_diff_c10_p0_1 - y_diff_plus_c10_p0_1 >= 0
y_diff_from_plus_c11_p1_1: y_diff_c11_p1_1 - y_diff_plus_c11_p1_1 >= 0
y_diff_from_plus_c12_p2_1: y_diff_c12_p2_1 - y_diff_plus_c12_p2_1 >= 0
y_diff_from_plus_c13_p3_1: y_diff_c13_p3_1 - y_diff_plus_c13_p3_1 >= 0
y_diff_from_plus_c14_p4_1: y_diff_c14_p4_1 - y_diff_plus_c14_p4_1 >= 0
y_diff_minus_lower_c10_p0_1: y_1 + 60 y_diff_minus_c10_p0_1 <= 77
y_diff_minus_lower_c11_p1_1: y_1 + 60 y_diff_minus_c11_p1_1 <= 67
y_diff_minus_lower_c12_p2_1: y_1 + 60 y_diff_minus_c12_p2_1 <= 57
y_diff_minus_lower_c13_p3_1: y_1 + 60 y_diff_minus_c13_p3_1 <= 67
y_diff_minus_lower_c14_p4_1: y_1 + 60 y_diff_minus_c14_p4_1 <= 77
y_diff_minus_upper_c10_p0_1: y_1 + 60 y_diff_minus_c10_p0_1 >= 17.001
y_diff_minus_upper_c11_p1_1: y_1 + 60 y_diff_minus_c11_p1_1 >= 7.001
y_diff_minus_upper_c12_p2_1: y_1 + 60 y_diff_minus_c12_p2_1 >= -2.999
y_diff_minus_upper_c13_p3_1: y_1 + 60 y_diff_minus_c13_p3_1 >= 7.001
y_diff_minus_upper_c14_p4_1: y_1 + 60 y_diff_minus_c14_p4_1 >= 17.001
y_diff_plus_lower_c10_p0_1: y_1 - 60 y_diff_plus_c10_p0_1 >= -37
y_diff_plus_lower_c11_p1_1: y_1 - 60 y_diff_plus_c11_p1_1 >= -47
y_diff_plus_lower_c12_p2_1: y_1 - 60 y_diff_plus_c12_p2_1 >= -57
y_diff_plus_lower_c13_p3_1: y_1 - 60 y_diff_plus_c13_p3_1 >= -47
y_diff_plus_lower_c14_p4_1: y_1 - 60 y_diff_plus_c14_p4_1 >= -37
y_diff_plus_upper_c10_p0_1: y_1 - 60 y_diff_plus_c10_p0_1 <= 22.999
y_diff_plus_upper_c11_p1_1: y_1 - 60 y_diff_plus_c11_p1_1 <= 12.999
y_diff_plus_upper_c12_p2_1: y_1 - 60 y_diff_plus_c12_p2_1 <= 2.999
y_diff_plus_upper_c13_p3_1: y_1 - 60 y_diff_plus_c13_p3_1 <= 12.999
y_diff_plus_upper_c14_p4_1: y_1 - 60 y_diff_plus_c14_p4_1 <= 22.999
y_diff_upper_c10_p0_1: y_diff_c10_p0_1 - y_diff_minus_c10_p0_1
 - y_diff_plus_c10_p0_1 <= 0
y_diff_upper_c11_p1_1: y_diff_c11_p1_1 - y_diff_minus_c11_p1_1
 - y_diff_plus_c11_p1_1 <= 0
y_diff_upper_c12_p2_1: y_diff_c12_p2_1 - y_diff_minus_c12_p2_1
 - y_diff_plus_c12_p2_1 <= 0
y_diff_upper_c13_p3_1: y_diff_c13_p3_1 - y_diff_minus_c13_p3_1
 - y_diff_plus_c13_p3_1 <= 0
y_diff_upper_c14_p4_1: y_diff_c14_p4_1 - y_diff_minus_c14_p4_1
 - y_diff_plus_c14_p4_1 <= 0
y_grid_to_real_0: y_0 - 0.5 y_grid_0 = 0
y_grid_to_real_1: y_1 - 0.5 y_grid_1 = 0
Bounds
 bbox_h <= 30
 bbox_max_x <= 30
 bbox_max_y <= 30
 bbox_min_x <= 30
 bbox_min_y <= 30
 bbox_w <= 30
 cx_0 <= 30
 cx_1 <= 30
 cy_0 <= 30
 cy_1 <= 30
 dist_diff_abs_pair_c10_p0_0_1 <= 60
 dist_diff_abs_pair_c11_p1_0_1 <= 60
 dist_diff_abs_pair_c12_p2_0_1 <= 60
 dist_diff_abs_pair_c13_p3_0_1 <= 60
 dist_diff_abs_pair_c14_p4_0_1 <= 60
 -60 <= dist_diff_pair_c10_p0_0_1 <= 60
 -60 <= dist_diff_pair_c11_p1_0_1 <= 60
 -60 <= dist_diff_pair_c12_p2_0_1 <= 60
 -60 <= dist_diff_pair_c13_p3_0_1 <= 60
 -60 <= dist_diff_pair_c14_p4_0_1 <= 60
 dx_abs_pair_c10_p0_0_1 <= 30
 dx_abs_pair_c11_p1_0_1 <= 30
 dx_abs_pair_c12_p2_0_1 <= 30
 dx_abs_pair_c13_p3_0_1 <= 30
 dx_abs_pair_c14_p4_0_1 <= 30
 dy_abs_pair_c10_p0_0_1 <= 30
 dy_abs_pair_c11_p1_0_1 <= 30
 dy_abs_pair_c12_p2_0_1 <= 30
 dy_abs_pair_c13_p3_0_1 <= 30
 dy_abs_pair_c14_p4_0_1 <= 30
 h_0 = 10
 h_1 = 10
 1 <= h_grid_0 <= 60
 1 <= h_grid_1 <= 60
 shared_x_0_1 <= 10
 shared_y_0_1 <= 10
 w_0 = 10
 w_1 = 10
 1 <= w_grid_0 <= 60
 1 <= w_grid_1 <= 60
 x_0 <= 30
 x_1 <= 30
 0 <= x_grid_0 <= 59
 0 <= x_grid_1 <= 59
 y_0 <= 30
 y_1 <= 30
 0 <= y_grid_0 <= 59
 0 <= y_grid_1 <= 59
Generals
h_grid_0
h_grid_1
w_grid_0
w_grid_1
x_grid_0
x_grid_1
y_grid_0
y_grid_1
Binaries
diff_dist_pair_c10_p0_0_1
diff_dist_pair_c11_p1_0_1
diff_dist_pair_c12_p2_0_1
diff_dist_pair_c13_p3_0_1
diff_dist_pair_c14_p4_0_1
diff_pos_c10_p0_0
diff_pos_c10_p0_1
diff_pos_c11_p1_0
diff_pos_c11_p1_1
diff_pos_c12_p2_0
diff_pos_c12_p2_1
diff_pos_c13_p3_0
diff_pos_c13_p3_1
diff_pos_c14_p4_0
diff_pos_c14_p4_1
p_down_0_1
p_left_0_1
p_right_0_1
p_up_0_1
x_diff_c10_p0_1
x_diff_c11_p1_1
x_diff_c12_p2_1
x_diff_c13_p3_1
x_diff_c14_p4_1
x_diff_minus_c10_p0_1
x_diff_minus_c11_p1_1
x_diff_minus_c12_p2_1
x_diff_minus_c13_p3_1
x_diff_minus_c14_p4_1
x_diff_plus_c10_p0_1
x_diff_plus_c11_p1_1
x_diff_plus_c12_p2_1
x_diff_plus_c13_p3_1
x_diff_plus_c14_p4_1
y_diff_c10_p0_1
y_diff_c11_p1_1
y_diff_c12_p2_1
y_diff_c13_p3_1
y_diff_c14_p4_1
y_diff_minus_c10_p0_1
y_diff_minus_c11_p1_1
y_diff_minus_c12_p2_1
y_diff_minus_c13_p3_1
y_diff_minus_c14_p4_1
y_diff_plus_c10_p0_1
y_diff_plus_c11_p1_1
y_diff_plus_c12_p2_1
y_diff_plus_c13_p3_1
y_diff_plus_c14_p4_1
z1L_0_1
z1R_0_1
z1_0_1
z2D_0_1
z2U_0_1
z2_0_1
End
