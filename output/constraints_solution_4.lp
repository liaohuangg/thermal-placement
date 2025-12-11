\* ChipletPlacementGrid *\
Minimize
Objective: 0.1 bbox_area_proxy_t + dx_abs_0_1 + dx_abs_1_2 + dx_abs_1_3
 + dx_abs_2_4 + dy_abs_0_1 + dy_abs_1_2 + dy_abs_1_3 + dy_abs_2_4
Subject To
area_proxy_am_gm1: bbox_area_proxy_t - 0.5 bbox_h - 0.5 bbox_w <= 0
area_proxy_am_gm2: - 60 bbox_area_proxy_t + bbox_w <= 0
area_proxy_am_gm3: - 60 bbox_area_proxy_t + bbox_h <= 0
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
diff_pos_fixed_c3_p0_0: diff_pos_c3_p0_0 = 0
diff_pos_fixed_c4_p1_0: diff_pos_c4_p1_0 = 0
diff_pos_fixed_c5_p2_0: diff_pos_c5_p2_0 = 0
diff_pos_from_x_c3_p0_1: diff_pos_c3_p0_1 - x_diff_c3_p0_1 >= 0
diff_pos_from_x_c3_p0_2: diff_pos_c3_p0_2 - x_diff_c3_p0_2 >= 0
diff_pos_from_x_c3_p0_3: diff_pos_c3_p0_3 - x_diff_c3_p0_3 >= 0
diff_pos_from_x_c3_p0_4: diff_pos_c3_p0_4 - x_diff_c3_p0_4 >= 0
diff_pos_from_x_c4_p1_1: diff_pos_c4_p1_1 - x_diff_c4_p1_1 >= 0
diff_pos_from_x_c4_p1_2: diff_pos_c4_p1_2 - x_diff_c4_p1_2 >= 0
diff_pos_from_x_c4_p1_3: diff_pos_c4_p1_3 - x_diff_c4_p1_3 >= 0
diff_pos_from_x_c4_p1_4: diff_pos_c4_p1_4 - x_diff_c4_p1_4 >= 0
diff_pos_from_x_c5_p2_1: diff_pos_c5_p2_1 - x_diff_c5_p2_1 >= 0
diff_pos_from_x_c5_p2_2: diff_pos_c5_p2_2 - x_diff_c5_p2_2 >= 0
diff_pos_from_x_c5_p2_3: diff_pos_c5_p2_3 - x_diff_c5_p2_3 >= 0
diff_pos_from_x_c5_p2_4: diff_pos_c5_p2_4 - x_diff_c5_p2_4 >= 0
diff_pos_from_y_c3_p0_1: diff_pos_c3_p0_1 - y_diff_c3_p0_1 >= 0
diff_pos_from_y_c3_p0_2: diff_pos_c3_p0_2 - y_diff_c3_p0_2 >= 0
diff_pos_from_y_c3_p0_3: diff_pos_c3_p0_3 - y_diff_c3_p0_3 >= 0
diff_pos_from_y_c3_p0_4: diff_pos_c3_p0_4 - y_diff_c3_p0_4 >= 0
diff_pos_from_y_c4_p1_1: diff_pos_c4_p1_1 - y_diff_c4_p1_1 >= 0
diff_pos_from_y_c4_p1_2: diff_pos_c4_p1_2 - y_diff_c4_p1_2 >= 0
diff_pos_from_y_c4_p1_3: diff_pos_c4_p1_3 - y_diff_c4_p1_3 >= 0
diff_pos_from_y_c4_p1_4: diff_pos_c4_p1_4 - y_diff_c4_p1_4 >= 0
diff_pos_from_y_c5_p2_1: diff_pos_c5_p2_1 - y_diff_c5_p2_1 >= 0
diff_pos_from_y_c5_p2_2: diff_pos_c5_p2_2 - y_diff_c5_p2_2 >= 0
diff_pos_from_y_c5_p2_3: diff_pos_c5_p2_3 - y_diff_c5_p2_3 >= 0
diff_pos_from_y_c5_p2_4: diff_pos_c5_p2_4 - y_diff_c5_p2_4 >= 0
diff_pos_upper_c3_p0_1: diff_pos_c3_p0_1 - x_diff_c3_p0_1 - y_diff_c3_p0_1
 <= 0
diff_pos_upper_c3_p0_2: diff_pos_c3_p0_2 - x_diff_c3_p0_2 - y_diff_c3_p0_2
 <= 0
diff_pos_upper_c3_p0_3: diff_pos_c3_p0_3 - x_diff_c3_p0_3 - y_diff_c3_p0_3
 <= 0
diff_pos_upper_c3_p0_4: diff_pos_c3_p0_4 - x_diff_c3_p0_4 - y_diff_c3_p0_4
 <= 0
diff_pos_upper_c4_p1_1: diff_pos_c4_p1_1 - x_diff_c4_p1_1 - y_diff_c4_p1_1
 <= 0
diff_pos_upper_c4_p1_2: diff_pos_c4_p1_2 - x_diff_c4_p1_2 - y_diff_c4_p1_2
 <= 0
diff_pos_upper_c4_p1_3: diff_pos_c4_p1_3 - x_diff_c4_p1_3 - y_diff_c4_p1_3
 <= 0
diff_pos_upper_c4_p1_4: diff_pos_c4_p1_4 - x_diff_c4_p1_4 - y_diff_c4_p1_4
 <= 0
diff_pos_upper_c5_p2_1: diff_pos_c5_p2_1 - x_diff_c5_p2_1 - y_diff_c5_p2_1
 <= 0
diff_pos_upper_c5_p2_2: diff_pos_c5_p2_2 - x_diff_c5_p2_2 - y_diff_c5_p2_2
 <= 0
diff_pos_upper_c5_p2_3: diff_pos_c5_p2_3 - x_diff_c5_p2_3 - y_diff_c5_p2_3
 <= 0
diff_pos_upper_c5_p2_4: diff_pos_c5_p2_4 - x_diff_c5_p2_4 - y_diff_c5_p2_4
 <= 0
dist_diff_abs_pair_lower_c3_p0_0_1: - 120 diff_dist_pair_c3_p0_0_1
 + dist_diff_abs_pair_c3_p0_0_1 >= -117
dist_diff_abs_pair_lower_c3_p0_0_2: - 120 diff_dist_pair_c3_p0_0_2
 + dist_diff_abs_pair_c3_p0_0_2 >= -117
dist_diff_abs_pair_lower_c3_p0_0_3: - 120 diff_dist_pair_c3_p0_0_3
 + dist_diff_abs_pair_c3_p0_0_3 >= -117
dist_diff_abs_pair_lower_c3_p0_0_4: - 120 diff_dist_pair_c3_p0_0_4
 + dist_diff_abs_pair_c3_p0_0_4 >= -117
dist_diff_abs_pair_lower_c3_p0_1_2: - 120 diff_dist_pair_c3_p0_1_2
 + dist_diff_abs_pair_c3_p0_1_2 >= -117
dist_diff_abs_pair_lower_c3_p0_1_3: - 120 diff_dist_pair_c3_p0_1_3
 + dist_diff_abs_pair_c3_p0_1_3 >= -117
dist_diff_abs_pair_lower_c3_p0_1_4: - 120 diff_dist_pair_c3_p0_1_4
 + dist_diff_abs_pair_c3_p0_1_4 >= -117
dist_diff_abs_pair_lower_c3_p0_2_3: - 120 diff_dist_pair_c3_p0_2_3
 + dist_diff_abs_pair_c3_p0_2_3 >= -117
dist_diff_abs_pair_lower_c3_p0_2_4: - 120 diff_dist_pair_c3_p0_2_4
 + dist_diff_abs_pair_c3_p0_2_4 >= -117
dist_diff_abs_pair_lower_c3_p0_3_4: - 120 diff_dist_pair_c3_p0_3_4
 + dist_diff_abs_pair_c3_p0_3_4 >= -117
dist_diff_abs_pair_lower_c4_p1_0_1: - 120 diff_dist_pair_c4_p1_0_1
 + dist_diff_abs_pair_c4_p1_0_1 >= -117
dist_diff_abs_pair_lower_c4_p1_0_2: - 120 diff_dist_pair_c4_p1_0_2
 + dist_diff_abs_pair_c4_p1_0_2 >= -117
dist_diff_abs_pair_lower_c4_p1_0_3: - 120 diff_dist_pair_c4_p1_0_3
 + dist_diff_abs_pair_c4_p1_0_3 >= -117
dist_diff_abs_pair_lower_c4_p1_0_4: - 120 diff_dist_pair_c4_p1_0_4
 + dist_diff_abs_pair_c4_p1_0_4 >= -117
dist_diff_abs_pair_lower_c4_p1_1_2: - 120 diff_dist_pair_c4_p1_1_2
 + dist_diff_abs_pair_c4_p1_1_2 >= -117
dist_diff_abs_pair_lower_c4_p1_1_3: - 120 diff_dist_pair_c4_p1_1_3
 + dist_diff_abs_pair_c4_p1_1_3 >= -117
dist_diff_abs_pair_lower_c4_p1_1_4: - 120 diff_dist_pair_c4_p1_1_4
 + dist_diff_abs_pair_c4_p1_1_4 >= -117
dist_diff_abs_pair_lower_c4_p1_2_3: - 120 diff_dist_pair_c4_p1_2_3
 + dist_diff_abs_pair_c4_p1_2_3 >= -117
dist_diff_abs_pair_lower_c4_p1_2_4: - 120 diff_dist_pair_c4_p1_2_4
 + dist_diff_abs_pair_c4_p1_2_4 >= -117
dist_diff_abs_pair_lower_c4_p1_3_4: - 120 diff_dist_pair_c4_p1_3_4
 + dist_diff_abs_pair_c4_p1_3_4 >= -117
dist_diff_abs_pair_lower_c5_p2_0_1: - 120 diff_dist_pair_c5_p2_0_1
 + dist_diff_abs_pair_c5_p2_0_1 >= -117
dist_diff_abs_pair_lower_c5_p2_0_2: - 120 diff_dist_pair_c5_p2_0_2
 + dist_diff_abs_pair_c5_p2_0_2 >= -117
dist_diff_abs_pair_lower_c5_p2_0_3: - 120 diff_dist_pair_c5_p2_0_3
 + dist_diff_abs_pair_c5_p2_0_3 >= -117
dist_diff_abs_pair_lower_c5_p2_0_4: - 120 diff_dist_pair_c5_p2_0_4
 + dist_diff_abs_pair_c5_p2_0_4 >= -117
dist_diff_abs_pair_lower_c5_p2_1_2: - 120 diff_dist_pair_c5_p2_1_2
 + dist_diff_abs_pair_c5_p2_1_2 >= -117
dist_diff_abs_pair_lower_c5_p2_1_3: - 120 diff_dist_pair_c5_p2_1_3
 + dist_diff_abs_pair_c5_p2_1_3 >= -117
dist_diff_abs_pair_lower_c5_p2_1_4: - 120 diff_dist_pair_c5_p2_1_4
 + dist_diff_abs_pair_c5_p2_1_4 >= -117
dist_diff_abs_pair_lower_c5_p2_2_3: - 120 diff_dist_pair_c5_p2_2_3
 + dist_diff_abs_pair_c5_p2_2_3 >= -117
dist_diff_abs_pair_lower_c5_p2_2_4: - 120 diff_dist_pair_c5_p2_2_4
 + dist_diff_abs_pair_c5_p2_2_4 >= -117
dist_diff_abs_pair_lower_c5_p2_3_4: - 120 diff_dist_pair_c5_p2_3_4
 + dist_diff_abs_pair_c5_p2_3_4 >= -117
dist_diff_abs_pair_neg_c3_p0_0_1: dist_diff_abs_pair_c3_p0_0_1
 + dist_diff_pair_c3_p0_0_1 >= 0
dist_diff_abs_pair_neg_c3_p0_0_2: dist_diff_abs_pair_c3_p0_0_2
 + dist_diff_pair_c3_p0_0_2 >= 0
dist_diff_abs_pair_neg_c3_p0_0_3: dist_diff_abs_pair_c3_p0_0_3
 + dist_diff_pair_c3_p0_0_3 >= 0
dist_diff_abs_pair_neg_c3_p0_0_4: dist_diff_abs_pair_c3_p0_0_4
 + dist_diff_pair_c3_p0_0_4 >= 0
dist_diff_abs_pair_neg_c3_p0_1_2: dist_diff_abs_pair_c3_p0_1_2
 + dist_diff_pair_c3_p0_1_2 >= 0
dist_diff_abs_pair_neg_c3_p0_1_3: dist_diff_abs_pair_c3_p0_1_3
 + dist_diff_pair_c3_p0_1_3 >= 0
dist_diff_abs_pair_neg_c3_p0_1_4: dist_diff_abs_pair_c3_p0_1_4
 + dist_diff_pair_c3_p0_1_4 >= 0
dist_diff_abs_pair_neg_c3_p0_2_3: dist_diff_abs_pair_c3_p0_2_3
 + dist_diff_pair_c3_p0_2_3 >= 0
dist_diff_abs_pair_neg_c3_p0_2_4: dist_diff_abs_pair_c3_p0_2_4
 + dist_diff_pair_c3_p0_2_4 >= 0
dist_diff_abs_pair_neg_c3_p0_3_4: dist_diff_abs_pair_c3_p0_3_4
 + dist_diff_pair_c3_p0_3_4 >= 0
dist_diff_abs_pair_neg_c4_p1_0_1: dist_diff_abs_pair_c4_p1_0_1
 + dist_diff_pair_c4_p1_0_1 >= 0
dist_diff_abs_pair_neg_c4_p1_0_2: dist_diff_abs_pair_c4_p1_0_2
 + dist_diff_pair_c4_p1_0_2 >= 0
dist_diff_abs_pair_neg_c4_p1_0_3: dist_diff_abs_pair_c4_p1_0_3
 + dist_diff_pair_c4_p1_0_3 >= 0
dist_diff_abs_pair_neg_c4_p1_0_4: dist_diff_abs_pair_c4_p1_0_4
 + dist_diff_pair_c4_p1_0_4 >= 0
dist_diff_abs_pair_neg_c4_p1_1_2: dist_diff_abs_pair_c4_p1_1_2
 + dist_diff_pair_c4_p1_1_2 >= 0
dist_diff_abs_pair_neg_c4_p1_1_3: dist_diff_abs_pair_c4_p1_1_3
 + dist_diff_pair_c4_p1_1_3 >= 0
dist_diff_abs_pair_neg_c4_p1_1_4: dist_diff_abs_pair_c4_p1_1_4
 + dist_diff_pair_c4_p1_1_4 >= 0
dist_diff_abs_pair_neg_c4_p1_2_3: dist_diff_abs_pair_c4_p1_2_3
 + dist_diff_pair_c4_p1_2_3 >= 0
dist_diff_abs_pair_neg_c4_p1_2_4: dist_diff_abs_pair_c4_p1_2_4
 + dist_diff_pair_c4_p1_2_4 >= 0
dist_diff_abs_pair_neg_c4_p1_3_4: dist_diff_abs_pair_c4_p1_3_4
 + dist_diff_pair_c4_p1_3_4 >= 0
dist_diff_abs_pair_neg_c5_p2_0_1: dist_diff_abs_pair_c5_p2_0_1
 + dist_diff_pair_c5_p2_0_1 >= 0
dist_diff_abs_pair_neg_c5_p2_0_2: dist_diff_abs_pair_c5_p2_0_2
 + dist_diff_pair_c5_p2_0_2 >= 0
dist_diff_abs_pair_neg_c5_p2_0_3: dist_diff_abs_pair_c5_p2_0_3
 + dist_diff_pair_c5_p2_0_3 >= 0
dist_diff_abs_pair_neg_c5_p2_0_4: dist_diff_abs_pair_c5_p2_0_4
 + dist_diff_pair_c5_p2_0_4 >= 0
dist_diff_abs_pair_neg_c5_p2_1_2: dist_diff_abs_pair_c5_p2_1_2
 + dist_diff_pair_c5_p2_1_2 >= 0
dist_diff_abs_pair_neg_c5_p2_1_3: dist_diff_abs_pair_c5_p2_1_3
 + dist_diff_pair_c5_p2_1_3 >= 0
dist_diff_abs_pair_neg_c5_p2_1_4: dist_diff_abs_pair_c5_p2_1_4
 + dist_diff_pair_c5_p2_1_4 >= 0
dist_diff_abs_pair_neg_c5_p2_2_3: dist_diff_abs_pair_c5_p2_2_3
 + dist_diff_pair_c5_p2_2_3 >= 0
dist_diff_abs_pair_neg_c5_p2_2_4: dist_diff_abs_pair_c5_p2_2_4
 + dist_diff_pair_c5_p2_2_4 >= 0
dist_diff_abs_pair_neg_c5_p2_3_4: dist_diff_abs_pair_c5_p2_3_4
 + dist_diff_pair_c5_p2_3_4 >= 0
dist_diff_abs_pair_pos_c3_p0_0_1: dist_diff_abs_pair_c3_p0_0_1
 - dist_diff_pair_c3_p0_0_1 >= 0
dist_diff_abs_pair_pos_c3_p0_0_2: dist_diff_abs_pair_c3_p0_0_2
 - dist_diff_pair_c3_p0_0_2 >= 0
dist_diff_abs_pair_pos_c3_p0_0_3: dist_diff_abs_pair_c3_p0_0_3
 - dist_diff_pair_c3_p0_0_3 >= 0
dist_diff_abs_pair_pos_c3_p0_0_4: dist_diff_abs_pair_c3_p0_0_4
 - dist_diff_pair_c3_p0_0_4 >= 0
dist_diff_abs_pair_pos_c3_p0_1_2: dist_diff_abs_pair_c3_p0_1_2
 - dist_diff_pair_c3_p0_1_2 >= 0
dist_diff_abs_pair_pos_c3_p0_1_3: dist_diff_abs_pair_c3_p0_1_3
 - dist_diff_pair_c3_p0_1_3 >= 0
dist_diff_abs_pair_pos_c3_p0_1_4: dist_diff_abs_pair_c3_p0_1_4
 - dist_diff_pair_c3_p0_1_4 >= 0
dist_diff_abs_pair_pos_c3_p0_2_3: dist_diff_abs_pair_c3_p0_2_3
 - dist_diff_pair_c3_p0_2_3 >= 0
dist_diff_abs_pair_pos_c3_p0_2_4: dist_diff_abs_pair_c3_p0_2_4
 - dist_diff_pair_c3_p0_2_4 >= 0
dist_diff_abs_pair_pos_c3_p0_3_4: dist_diff_abs_pair_c3_p0_3_4
 - dist_diff_pair_c3_p0_3_4 >= 0
dist_diff_abs_pair_pos_c4_p1_0_1: dist_diff_abs_pair_c4_p1_0_1
 - dist_diff_pair_c4_p1_0_1 >= 0
dist_diff_abs_pair_pos_c4_p1_0_2: dist_diff_abs_pair_c4_p1_0_2
 - dist_diff_pair_c4_p1_0_2 >= 0
dist_diff_abs_pair_pos_c4_p1_0_3: dist_diff_abs_pair_c4_p1_0_3
 - dist_diff_pair_c4_p1_0_3 >= 0
dist_diff_abs_pair_pos_c4_p1_0_4: dist_diff_abs_pair_c4_p1_0_4
 - dist_diff_pair_c4_p1_0_4 >= 0
dist_diff_abs_pair_pos_c4_p1_1_2: dist_diff_abs_pair_c4_p1_1_2
 - dist_diff_pair_c4_p1_1_2 >= 0
dist_diff_abs_pair_pos_c4_p1_1_3: dist_diff_abs_pair_c4_p1_1_3
 - dist_diff_pair_c4_p1_1_3 >= 0
dist_diff_abs_pair_pos_c4_p1_1_4: dist_diff_abs_pair_c4_p1_1_4
 - dist_diff_pair_c4_p1_1_4 >= 0
dist_diff_abs_pair_pos_c4_p1_2_3: dist_diff_abs_pair_c4_p1_2_3
 - dist_diff_pair_c4_p1_2_3 >= 0
dist_diff_abs_pair_pos_c4_p1_2_4: dist_diff_abs_pair_c4_p1_2_4
 - dist_diff_pair_c4_p1_2_4 >= 0
dist_diff_abs_pair_pos_c4_p1_3_4: dist_diff_abs_pair_c4_p1_3_4
 - dist_diff_pair_c4_p1_3_4 >= 0
dist_diff_abs_pair_pos_c5_p2_0_1: dist_diff_abs_pair_c5_p2_0_1
 - dist_diff_pair_c5_p2_0_1 >= 0
dist_diff_abs_pair_pos_c5_p2_0_2: dist_diff_abs_pair_c5_p2_0_2
 - dist_diff_pair_c5_p2_0_2 >= 0
dist_diff_abs_pair_pos_c5_p2_0_3: dist_diff_abs_pair_c5_p2_0_3
 - dist_diff_pair_c5_p2_0_3 >= 0
dist_diff_abs_pair_pos_c5_p2_0_4: dist_diff_abs_pair_c5_p2_0_4
 - dist_diff_pair_c5_p2_0_4 >= 0
dist_diff_abs_pair_pos_c5_p2_1_2: dist_diff_abs_pair_c5_p2_1_2
 - dist_diff_pair_c5_p2_1_2 >= 0
dist_diff_abs_pair_pos_c5_p2_1_3: dist_diff_abs_pair_c5_p2_1_3
 - dist_diff_pair_c5_p2_1_3 >= 0
dist_diff_abs_pair_pos_c5_p2_1_4: dist_diff_abs_pair_c5_p2_1_4
 - dist_diff_pair_c5_p2_1_4 >= 0
dist_diff_abs_pair_pos_c5_p2_2_3: dist_diff_abs_pair_c5_p2_2_3
 - dist_diff_pair_c5_p2_2_3 >= 0
dist_diff_abs_pair_pos_c5_p2_2_4: dist_diff_abs_pair_c5_p2_2_4
 - dist_diff_pair_c5_p2_2_4 >= 0
dist_diff_abs_pair_pos_c5_p2_3_4: dist_diff_abs_pair_c5_p2_3_4
 - dist_diff_pair_c5_p2_3_4 >= 0
dist_diff_abs_pair_upper_c3_p0_0_1: - 120 diff_dist_pair_c3_p0_0_1
 + dist_diff_abs_pair_c3_p0_0_1 <= 2.97
dist_diff_abs_pair_upper_c3_p0_0_2: - 120 diff_dist_pair_c3_p0_0_2
 + dist_diff_abs_pair_c3_p0_0_2 <= 2.97
dist_diff_abs_pair_upper_c3_p0_0_3: - 120 diff_dist_pair_c3_p0_0_3
 + dist_diff_abs_pair_c3_p0_0_3 <= 2.97
dist_diff_abs_pair_upper_c3_p0_0_4: - 120 diff_dist_pair_c3_p0_0_4
 + dist_diff_abs_pair_c3_p0_0_4 <= 2.97
dist_diff_abs_pair_upper_c3_p0_1_2: - 120 diff_dist_pair_c3_p0_1_2
 + dist_diff_abs_pair_c3_p0_1_2 <= 2.97
dist_diff_abs_pair_upper_c3_p0_1_3: - 120 diff_dist_pair_c3_p0_1_3
 + dist_diff_abs_pair_c3_p0_1_3 <= 2.97
dist_diff_abs_pair_upper_c3_p0_1_4: - 120 diff_dist_pair_c3_p0_1_4
 + dist_diff_abs_pair_c3_p0_1_4 <= 2.97
dist_diff_abs_pair_upper_c3_p0_2_3: - 120 diff_dist_pair_c3_p0_2_3
 + dist_diff_abs_pair_c3_p0_2_3 <= 2.97
dist_diff_abs_pair_upper_c3_p0_2_4: - 120 diff_dist_pair_c3_p0_2_4
 + dist_diff_abs_pair_c3_p0_2_4 <= 2.97
dist_diff_abs_pair_upper_c3_p0_3_4: - 120 diff_dist_pair_c3_p0_3_4
 + dist_diff_abs_pair_c3_p0_3_4 <= 2.97
dist_diff_abs_pair_upper_c4_p1_0_1: - 120 diff_dist_pair_c4_p1_0_1
 + dist_diff_abs_pair_c4_p1_0_1 <= 2.97
dist_diff_abs_pair_upper_c4_p1_0_2: - 120 diff_dist_pair_c4_p1_0_2
 + dist_diff_abs_pair_c4_p1_0_2 <= 2.97
dist_diff_abs_pair_upper_c4_p1_0_3: - 120 diff_dist_pair_c4_p1_0_3
 + dist_diff_abs_pair_c4_p1_0_3 <= 2.97
dist_diff_abs_pair_upper_c4_p1_0_4: - 120 diff_dist_pair_c4_p1_0_4
 + dist_diff_abs_pair_c4_p1_0_4 <= 2.97
dist_diff_abs_pair_upper_c4_p1_1_2: - 120 diff_dist_pair_c4_p1_1_2
 + dist_diff_abs_pair_c4_p1_1_2 <= 2.97
dist_diff_abs_pair_upper_c4_p1_1_3: - 120 diff_dist_pair_c4_p1_1_3
 + dist_diff_abs_pair_c4_p1_1_3 <= 2.97
dist_diff_abs_pair_upper_c4_p1_1_4: - 120 diff_dist_pair_c4_p1_1_4
 + dist_diff_abs_pair_c4_p1_1_4 <= 2.97
dist_diff_abs_pair_upper_c4_p1_2_3: - 120 diff_dist_pair_c4_p1_2_3
 + dist_diff_abs_pair_c4_p1_2_3 <= 2.97
dist_diff_abs_pair_upper_c4_p1_2_4: - 120 diff_dist_pair_c4_p1_2_4
 + dist_diff_abs_pair_c4_p1_2_4 <= 2.97
dist_diff_abs_pair_upper_c4_p1_3_4: - 120 diff_dist_pair_c4_p1_3_4
 + dist_diff_abs_pair_c4_p1_3_4 <= 2.97
dist_diff_abs_pair_upper_c5_p2_0_1: - 120 diff_dist_pair_c5_p2_0_1
 + dist_diff_abs_pair_c5_p2_0_1 <= 2.97
dist_diff_abs_pair_upper_c5_p2_0_2: - 120 diff_dist_pair_c5_p2_0_2
 + dist_diff_abs_pair_c5_p2_0_2 <= 2.97
dist_diff_abs_pair_upper_c5_p2_0_3: - 120 diff_dist_pair_c5_p2_0_3
 + dist_diff_abs_pair_c5_p2_0_3 <= 2.97
dist_diff_abs_pair_upper_c5_p2_0_4: - 120 diff_dist_pair_c5_p2_0_4
 + dist_diff_abs_pair_c5_p2_0_4 <= 2.97
dist_diff_abs_pair_upper_c5_p2_1_2: - 120 diff_dist_pair_c5_p2_1_2
 + dist_diff_abs_pair_c5_p2_1_2 <= 2.97
dist_diff_abs_pair_upper_c5_p2_1_3: - 120 diff_dist_pair_c5_p2_1_3
 + dist_diff_abs_pair_c5_p2_1_3 <= 2.97
dist_diff_abs_pair_upper_c5_p2_1_4: - 120 diff_dist_pair_c5_p2_1_4
 + dist_diff_abs_pair_c5_p2_1_4 <= 2.97
dist_diff_abs_pair_upper_c5_p2_2_3: - 120 diff_dist_pair_c5_p2_2_3
 + dist_diff_abs_pair_c5_p2_2_3 <= 2.97
dist_diff_abs_pair_upper_c5_p2_2_4: - 120 diff_dist_pair_c5_p2_2_4
 + dist_diff_abs_pair_c5_p2_2_4 <= 2.97
dist_diff_abs_pair_upper_c5_p2_3_4: - 120 diff_dist_pair_c5_p2_3_4
 + dist_diff_abs_pair_c5_p2_3_4 <= 2.97
dist_diff_pair_def_c3_p0_0_1: dist_diff_pair_c3_p0_0_1 - dx_abs_pair_c3_p0_0_1
 - dy_abs_pair_c3_p0_0_1 = -18
dist_diff_pair_def_c3_p0_0_2: dist_diff_pair_c3_p0_0_2 - dx_abs_pair_c3_p0_0_2
 - dy_abs_pair_c3_p0_0_2 = -33
dist_diff_pair_def_c3_p0_0_3: dist_diff_pair_c3_p0_0_3 - dx_abs_pair_c3_p0_0_3
 - dy_abs_pair_c3_p0_0_3 = -28
dist_diff_pair_def_c3_p0_0_4: dist_diff_pair_c3_p0_0_4 - dx_abs_pair_c3_p0_0_4
 - dy_abs_pair_c3_p0_0_4 = -21
dist_diff_pair_def_c3_p0_1_2: dist_diff_pair_c3_p0_1_2 - dx_abs_pair_c3_p0_1_2
 - dy_abs_pair_c3_p0_1_2 = -15
dist_diff_pair_def_c3_p0_1_3: dist_diff_pair_c3_p0_1_3 - dx_abs_pair_c3_p0_1_3
 - dy_abs_pair_c3_p0_1_3 = -16
dist_diff_pair_def_c3_p0_1_4: dist_diff_pair_c3_p0_1_4 - dx_abs_pair_c3_p0_1_4
 - dy_abs_pair_c3_p0_1_4 = -27
dist_diff_pair_def_c3_p0_2_3: dist_diff_pair_c3_p0_2_3 - dx_abs_pair_c3_p0_2_3
 - dy_abs_pair_c3_p0_2_3 = -31
dist_diff_pair_def_c3_p0_2_4: dist_diff_pair_c3_p0_2_4 - dx_abs_pair_c3_p0_2_4
 - dy_abs_pair_c3_p0_2_4 = -12
dist_diff_pair_def_c3_p0_3_4: dist_diff_pair_c3_p0_3_4 - dx_abs_pair_c3_p0_3_4
 - dy_abs_pair_c3_p0_3_4 = -37
dist_diff_pair_def_c4_p1_0_1: dist_diff_pair_c4_p1_0_1 - dx_abs_pair_c4_p1_0_1
 - dy_abs_pair_c4_p1_0_1 = -18
dist_diff_pair_def_c4_p1_0_2: dist_diff_pair_c4_p1_0_2 - dx_abs_pair_c4_p1_0_2
 - dy_abs_pair_c4_p1_0_2 = -35
dist_diff_pair_def_c4_p1_0_3: dist_diff_pair_c4_p1_0_3 - dx_abs_pair_c4_p1_0_3
 - dy_abs_pair_c4_p1_0_3 = -30
dist_diff_pair_def_c4_p1_0_4: dist_diff_pair_c4_p1_0_4 - dx_abs_pair_c4_p1_0_4
 - dy_abs_pair_c4_p1_0_4 = -23
dist_diff_pair_def_c4_p1_1_2: dist_diff_pair_c4_p1_1_2 - dx_abs_pair_c4_p1_1_2
 - dy_abs_pair_c4_p1_1_2 = -17
dist_diff_pair_def_c4_p1_1_3: dist_diff_pair_c4_p1_1_3 - dx_abs_pair_c4_p1_1_3
 - dy_abs_pair_c4_p1_1_3 = -14
dist_diff_pair_def_c4_p1_1_4: dist_diff_pair_c4_p1_1_4 - dx_abs_pair_c4_p1_1_4
 - dy_abs_pair_c4_p1_1_4 = -25
dist_diff_pair_def_c4_p1_2_3: dist_diff_pair_c4_p1_2_3 - dx_abs_pair_c4_p1_2_3
 - dy_abs_pair_c4_p1_2_3 = -31
dist_diff_pair_def_c4_p1_2_4: dist_diff_pair_c4_p1_2_4 - dx_abs_pair_c4_p1_2_4
 - dy_abs_pair_c4_p1_2_4 = -12
dist_diff_pair_def_c4_p1_3_4: dist_diff_pair_c4_p1_3_4 - dx_abs_pair_c4_p1_3_4
 - dy_abs_pair_c4_p1_3_4 = -37
dist_diff_pair_def_c5_p2_0_1: dist_diff_pair_c5_p2_0_1 - dx_abs_pair_c5_p2_0_1
 - dy_abs_pair_c5_p2_0_1 = -18
dist_diff_pair_def_c5_p2_0_2: dist_diff_pair_c5_p2_0_2 - dx_abs_pair_c5_p2_0_2
 - dy_abs_pair_c5_p2_0_2 = -33
dist_diff_pair_def_c5_p2_0_3: dist_diff_pair_c5_p2_0_3 - dx_abs_pair_c5_p2_0_3
 - dy_abs_pair_c5_p2_0_3 = -28
dist_diff_pair_def_c5_p2_0_4: dist_diff_pair_c5_p2_0_4 - dx_abs_pair_c5_p2_0_4
 - dy_abs_pair_c5_p2_0_4 = -21
dist_diff_pair_def_c5_p2_1_2: dist_diff_pair_c5_p2_1_2 - dx_abs_pair_c5_p2_1_2
 - dy_abs_pair_c5_p2_1_2 = -15
dist_diff_pair_def_c5_p2_1_3: dist_diff_pair_c5_p2_1_3 - dx_abs_pair_c5_p2_1_3
 - dy_abs_pair_c5_p2_1_3 = -16
dist_diff_pair_def_c5_p2_1_4: dist_diff_pair_c5_p2_1_4 - dx_abs_pair_c5_p2_1_4
 - dy_abs_pair_c5_p2_1_4 = -27
dist_diff_pair_def_c5_p2_2_3: dist_diff_pair_c5_p2_2_3 - dx_abs_pair_c5_p2_2_3
 - dy_abs_pair_c5_p2_2_3 = -31
dist_diff_pair_def_c5_p2_2_4: dist_diff_pair_c5_p2_2_4 - dx_abs_pair_c5_p2_2_4
 - dy_abs_pair_c5_p2_2_4 = -12
dist_diff_pair_def_c5_p2_3_4: dist_diff_pair_c5_p2_3_4 - dx_abs_pair_c5_p2_3_4
 - dy_abs_pair_c5_p2_3_4 = -37
dx_abs1_0_1: - cx_0 + cx_1 + dx_abs_0_1 >= 0
dx_abs1_1_2: - cx_1 + cx_2 + dx_abs_1_2 >= 0
dx_abs1_1_3: - cx_1 + cx_3 + dx_abs_1_3 >= 0
dx_abs1_2_4: - cx_2 + cx_4 + dx_abs_2_4 >= 0
dx_abs2_0_1: cx_0 - cx_1 + dx_abs_0_1 >= 0
dx_abs2_1_2: cx_1 - cx_2 + dx_abs_1_2 >= 0
dx_abs2_1_3: cx_1 - cx_3 + dx_abs_1_3 >= 0
dx_abs2_2_4: cx_2 - cx_4 + dx_abs_2_4 >= 0
dx_abs_pair_neg_c3_p0_0_1: cx_0 - cx_1 + dx_abs_pair_c3_p0_0_1 >= 0
dx_abs_pair_neg_c3_p0_0_2: cx_0 - cx_2 + dx_abs_pair_c3_p0_0_2 >= 0
dx_abs_pair_neg_c3_p0_0_3: cx_0 - cx_3 + dx_abs_pair_c3_p0_0_3 >= 0
dx_abs_pair_neg_c3_p0_0_4: cx_0 - cx_4 + dx_abs_pair_c3_p0_0_4 >= 0
dx_abs_pair_neg_c3_p0_1_2: cx_1 - cx_2 + dx_abs_pair_c3_p0_1_2 >= 0
dx_abs_pair_neg_c3_p0_1_3: cx_1 - cx_3 + dx_abs_pair_c3_p0_1_3 >= 0
dx_abs_pair_neg_c3_p0_1_4: cx_1 - cx_4 + dx_abs_pair_c3_p0_1_4 >= 0
dx_abs_pair_neg_c3_p0_2_3: cx_2 - cx_3 + dx_abs_pair_c3_p0_2_3 >= 0
dx_abs_pair_neg_c3_p0_2_4: cx_2 - cx_4 + dx_abs_pair_c3_p0_2_4 >= 0
dx_abs_pair_neg_c3_p0_3_4: cx_3 - cx_4 + dx_abs_pair_c3_p0_3_4 >= 0
dx_abs_pair_neg_c4_p1_0_1: cx_0 - cx_1 + dx_abs_pair_c4_p1_0_1 >= 0
dx_abs_pair_neg_c4_p1_0_2: cx_0 - cx_2 + dx_abs_pair_c4_p1_0_2 >= 0
dx_abs_pair_neg_c4_p1_0_3: cx_0 - cx_3 + dx_abs_pair_c4_p1_0_3 >= 0
dx_abs_pair_neg_c4_p1_0_4: cx_0 - cx_4 + dx_abs_pair_c4_p1_0_4 >= 0
dx_abs_pair_neg_c4_p1_1_2: cx_1 - cx_2 + dx_abs_pair_c4_p1_1_2 >= 0
dx_abs_pair_neg_c4_p1_1_3: cx_1 - cx_3 + dx_abs_pair_c4_p1_1_3 >= 0
dx_abs_pair_neg_c4_p1_1_4: cx_1 - cx_4 + dx_abs_pair_c4_p1_1_4 >= 0
dx_abs_pair_neg_c4_p1_2_3: cx_2 - cx_3 + dx_abs_pair_c4_p1_2_3 >= 0
dx_abs_pair_neg_c4_p1_2_4: cx_2 - cx_4 + dx_abs_pair_c4_p1_2_4 >= 0
dx_abs_pair_neg_c4_p1_3_4: cx_3 - cx_4 + dx_abs_pair_c4_p1_3_4 >= 0
dx_abs_pair_neg_c5_p2_0_1: cx_0 - cx_1 + dx_abs_pair_c5_p2_0_1 >= 0
dx_abs_pair_neg_c5_p2_0_2: cx_0 - cx_2 + dx_abs_pair_c5_p2_0_2 >= 0
dx_abs_pair_neg_c5_p2_0_3: cx_0 - cx_3 + dx_abs_pair_c5_p2_0_3 >= 0
dx_abs_pair_neg_c5_p2_0_4: cx_0 - cx_4 + dx_abs_pair_c5_p2_0_4 >= 0
dx_abs_pair_neg_c5_p2_1_2: cx_1 - cx_2 + dx_abs_pair_c5_p2_1_2 >= 0
dx_abs_pair_neg_c5_p2_1_3: cx_1 - cx_3 + dx_abs_pair_c5_p2_1_3 >= 0
dx_abs_pair_neg_c5_p2_1_4: cx_1 - cx_4 + dx_abs_pair_c5_p2_1_4 >= 0
dx_abs_pair_neg_c5_p2_2_3: cx_2 - cx_3 + dx_abs_pair_c5_p2_2_3 >= 0
dx_abs_pair_neg_c5_p2_2_4: cx_2 - cx_4 + dx_abs_pair_c5_p2_2_4 >= 0
dx_abs_pair_neg_c5_p2_3_4: cx_3 - cx_4 + dx_abs_pair_c5_p2_3_4 >= 0
dx_abs_pair_pos_c3_p0_0_1: - cx_0 + cx_1 + dx_abs_pair_c3_p0_0_1 >= 0
dx_abs_pair_pos_c3_p0_0_2: - cx_0 + cx_2 + dx_abs_pair_c3_p0_0_2 >= 0
dx_abs_pair_pos_c3_p0_0_3: - cx_0 + cx_3 + dx_abs_pair_c3_p0_0_3 >= 0
dx_abs_pair_pos_c3_p0_0_4: - cx_0 + cx_4 + dx_abs_pair_c3_p0_0_4 >= 0
dx_abs_pair_pos_c3_p0_1_2: - cx_1 + cx_2 + dx_abs_pair_c3_p0_1_2 >= 0
dx_abs_pair_pos_c3_p0_1_3: - cx_1 + cx_3 + dx_abs_pair_c3_p0_1_3 >= 0
dx_abs_pair_pos_c3_p0_1_4: - cx_1 + cx_4 + dx_abs_pair_c3_p0_1_4 >= 0
dx_abs_pair_pos_c3_p0_2_3: - cx_2 + cx_3 + dx_abs_pair_c3_p0_2_3 >= 0
dx_abs_pair_pos_c3_p0_2_4: - cx_2 + cx_4 + dx_abs_pair_c3_p0_2_4 >= 0
dx_abs_pair_pos_c3_p0_3_4: - cx_3 + cx_4 + dx_abs_pair_c3_p0_3_4 >= 0
dx_abs_pair_pos_c4_p1_0_1: - cx_0 + cx_1 + dx_abs_pair_c4_p1_0_1 >= 0
dx_abs_pair_pos_c4_p1_0_2: - cx_0 + cx_2 + dx_abs_pair_c4_p1_0_2 >= 0
dx_abs_pair_pos_c4_p1_0_3: - cx_0 + cx_3 + dx_abs_pair_c4_p1_0_3 >= 0
dx_abs_pair_pos_c4_p1_0_4: - cx_0 + cx_4 + dx_abs_pair_c4_p1_0_4 >= 0
dx_abs_pair_pos_c4_p1_1_2: - cx_1 + cx_2 + dx_abs_pair_c4_p1_1_2 >= 0
dx_abs_pair_pos_c4_p1_1_3: - cx_1 + cx_3 + dx_abs_pair_c4_p1_1_3 >= 0
dx_abs_pair_pos_c4_p1_1_4: - cx_1 + cx_4 + dx_abs_pair_c4_p1_1_4 >= 0
dx_abs_pair_pos_c4_p1_2_3: - cx_2 + cx_3 + dx_abs_pair_c4_p1_2_3 >= 0
dx_abs_pair_pos_c4_p1_2_4: - cx_2 + cx_4 + dx_abs_pair_c4_p1_2_4 >= 0
dx_abs_pair_pos_c4_p1_3_4: - cx_3 + cx_4 + dx_abs_pair_c4_p1_3_4 >= 0
dx_abs_pair_pos_c5_p2_0_1: - cx_0 + cx_1 + dx_abs_pair_c5_p2_0_1 >= 0
dx_abs_pair_pos_c5_p2_0_2: - cx_0 + cx_2 + dx_abs_pair_c5_p2_0_2 >= 0
dx_abs_pair_pos_c5_p2_0_3: - cx_0 + cx_3 + dx_abs_pair_c5_p2_0_3 >= 0
dx_abs_pair_pos_c5_p2_0_4: - cx_0 + cx_4 + dx_abs_pair_c5_p2_0_4 >= 0
dx_abs_pair_pos_c5_p2_1_2: - cx_1 + cx_2 + dx_abs_pair_c5_p2_1_2 >= 0
dx_abs_pair_pos_c5_p2_1_3: - cx_1 + cx_3 + dx_abs_pair_c5_p2_1_3 >= 0
dx_abs_pair_pos_c5_p2_1_4: - cx_1 + cx_4 + dx_abs_pair_c5_p2_1_4 >= 0
dx_abs_pair_pos_c5_p2_2_3: - cx_2 + cx_3 + dx_abs_pair_c5_p2_2_3 >= 0
dx_abs_pair_pos_c5_p2_2_4: - cx_2 + cx_4 + dx_abs_pair_c5_p2_2_4 >= 0
dx_abs_pair_pos_c5_p2_3_4: - cx_3 + cx_4 + dx_abs_pair_c5_p2_3_4 >= 0
dy_abs1_0_1: - cy_0 + cy_1 + dy_abs_0_1 >= 0
dy_abs1_1_2: - cy_1 + cy_2 + dy_abs_1_2 >= 0
dy_abs1_1_3: - cy_1 + cy_3 + dy_abs_1_3 >= 0
dy_abs1_2_4: - cy_2 + cy_4 + dy_abs_2_4 >= 0
dy_abs2_0_1: cy_0 - cy_1 + dy_abs_0_1 >= 0
dy_abs2_1_2: cy_1 - cy_2 + dy_abs_1_2 >= 0
dy_abs2_1_3: cy_1 - cy_3 + dy_abs_1_3 >= 0
dy_abs2_2_4: cy_2 - cy_4 + dy_abs_2_4 >= 0
dy_abs_pair_neg_c3_p0_0_1: cy_0 - cy_1 + dy_abs_pair_c3_p0_0_1 >= 0
dy_abs_pair_neg_c3_p0_0_2: cy_0 - cy_2 + dy_abs_pair_c3_p0_0_2 >= 0
dy_abs_pair_neg_c3_p0_0_3: cy_0 - cy_3 + dy_abs_pair_c3_p0_0_3 >= 0
dy_abs_pair_neg_c3_p0_0_4: cy_0 - cy_4 + dy_abs_pair_c3_p0_0_4 >= 0
dy_abs_pair_neg_c3_p0_1_2: cy_1 - cy_2 + dy_abs_pair_c3_p0_1_2 >= 0
dy_abs_pair_neg_c3_p0_1_3: cy_1 - cy_3 + dy_abs_pair_c3_p0_1_3 >= 0
dy_abs_pair_neg_c3_p0_1_4: cy_1 - cy_4 + dy_abs_pair_c3_p0_1_4 >= 0
dy_abs_pair_neg_c3_p0_2_3: cy_2 - cy_3 + dy_abs_pair_c3_p0_2_3 >= 0
dy_abs_pair_neg_c3_p0_2_4: cy_2 - cy_4 + dy_abs_pair_c3_p0_2_4 >= 0
dy_abs_pair_neg_c3_p0_3_4: cy_3 - cy_4 + dy_abs_pair_c3_p0_3_4 >= 0
dy_abs_pair_neg_c4_p1_0_1: cy_0 - cy_1 + dy_abs_pair_c4_p1_0_1 >= 0
dy_abs_pair_neg_c4_p1_0_2: cy_0 - cy_2 + dy_abs_pair_c4_p1_0_2 >= 0
dy_abs_pair_neg_c4_p1_0_3: cy_0 - cy_3 + dy_abs_pair_c4_p1_0_3 >= 0
dy_abs_pair_neg_c4_p1_0_4: cy_0 - cy_4 + dy_abs_pair_c4_p1_0_4 >= 0
dy_abs_pair_neg_c4_p1_1_2: cy_1 - cy_2 + dy_abs_pair_c4_p1_1_2 >= 0
dy_abs_pair_neg_c4_p1_1_3: cy_1 - cy_3 + dy_abs_pair_c4_p1_1_3 >= 0
dy_abs_pair_neg_c4_p1_1_4: cy_1 - cy_4 + dy_abs_pair_c4_p1_1_4 >= 0
dy_abs_pair_neg_c4_p1_2_3: cy_2 - cy_3 + dy_abs_pair_c4_p1_2_3 >= 0
dy_abs_pair_neg_c4_p1_2_4: cy_2 - cy_4 + dy_abs_pair_c4_p1_2_4 >= 0
dy_abs_pair_neg_c4_p1_3_4: cy_3 - cy_4 + dy_abs_pair_c4_p1_3_4 >= 0
dy_abs_pair_neg_c5_p2_0_1: cy_0 - cy_1 + dy_abs_pair_c5_p2_0_1 >= 0
dy_abs_pair_neg_c5_p2_0_2: cy_0 - cy_2 + dy_abs_pair_c5_p2_0_2 >= 0
dy_abs_pair_neg_c5_p2_0_3: cy_0 - cy_3 + dy_abs_pair_c5_p2_0_3 >= 0
dy_abs_pair_neg_c5_p2_0_4: cy_0 - cy_4 + dy_abs_pair_c5_p2_0_4 >= 0
dy_abs_pair_neg_c5_p2_1_2: cy_1 - cy_2 + dy_abs_pair_c5_p2_1_2 >= 0
dy_abs_pair_neg_c5_p2_1_3: cy_1 - cy_3 + dy_abs_pair_c5_p2_1_3 >= 0
dy_abs_pair_neg_c5_p2_1_4: cy_1 - cy_4 + dy_abs_pair_c5_p2_1_4 >= 0
dy_abs_pair_neg_c5_p2_2_3: cy_2 - cy_3 + dy_abs_pair_c5_p2_2_3 >= 0
dy_abs_pair_neg_c5_p2_2_4: cy_2 - cy_4 + dy_abs_pair_c5_p2_2_4 >= 0
dy_abs_pair_neg_c5_p2_3_4: cy_3 - cy_4 + dy_abs_pair_c5_p2_3_4 >= 0
dy_abs_pair_pos_c3_p0_0_1: - cy_0 + cy_1 + dy_abs_pair_c3_p0_0_1 >= 0
dy_abs_pair_pos_c3_p0_0_2: - cy_0 + cy_2 + dy_abs_pair_c3_p0_0_2 >= 0
dy_abs_pair_pos_c3_p0_0_3: - cy_0 + cy_3 + dy_abs_pair_c3_p0_0_3 >= 0
dy_abs_pair_pos_c3_p0_0_4: - cy_0 + cy_4 + dy_abs_pair_c3_p0_0_4 >= 0
dy_abs_pair_pos_c3_p0_1_2: - cy_1 + cy_2 + dy_abs_pair_c3_p0_1_2 >= 0
dy_abs_pair_pos_c3_p0_1_3: - cy_1 + cy_3 + dy_abs_pair_c3_p0_1_3 >= 0
dy_abs_pair_pos_c3_p0_1_4: - cy_1 + cy_4 + dy_abs_pair_c3_p0_1_4 >= 0
dy_abs_pair_pos_c3_p0_2_3: - cy_2 + cy_3 + dy_abs_pair_c3_p0_2_3 >= 0
dy_abs_pair_pos_c3_p0_2_4: - cy_2 + cy_4 + dy_abs_pair_c3_p0_2_4 >= 0
dy_abs_pair_pos_c3_p0_3_4: - cy_3 + cy_4 + dy_abs_pair_c3_p0_3_4 >= 0
dy_abs_pair_pos_c4_p1_0_1: - cy_0 + cy_1 + dy_abs_pair_c4_p1_0_1 >= 0
dy_abs_pair_pos_c4_p1_0_2: - cy_0 + cy_2 + dy_abs_pair_c4_p1_0_2 >= 0
dy_abs_pair_pos_c4_p1_0_3: - cy_0 + cy_3 + dy_abs_pair_c4_p1_0_3 >= 0
dy_abs_pair_pos_c4_p1_0_4: - cy_0 + cy_4 + dy_abs_pair_c4_p1_0_4 >= 0
dy_abs_pair_pos_c4_p1_1_2: - cy_1 + cy_2 + dy_abs_pair_c4_p1_1_2 >= 0
dy_abs_pair_pos_c4_p1_1_3: - cy_1 + cy_3 + dy_abs_pair_c4_p1_1_3 >= 0
dy_abs_pair_pos_c4_p1_1_4: - cy_1 + cy_4 + dy_abs_pair_c4_p1_1_4 >= 0
dy_abs_pair_pos_c4_p1_2_3: - cy_2 + cy_3 + dy_abs_pair_c4_p1_2_3 >= 0
dy_abs_pair_pos_c4_p1_2_4: - cy_2 + cy_4 + dy_abs_pair_c4_p1_2_4 >= 0
dy_abs_pair_pos_c4_p1_3_4: - cy_3 + cy_4 + dy_abs_pair_c4_p1_3_4 >= 0
dy_abs_pair_pos_c5_p2_0_1: - cy_0 + cy_1 + dy_abs_pair_c5_p2_0_1 >= 0
dy_abs_pair_pos_c5_p2_0_2: - cy_0 + cy_2 + dy_abs_pair_c5_p2_0_2 >= 0
dy_abs_pair_pos_c5_p2_0_3: - cy_0 + cy_3 + dy_abs_pair_c5_p2_0_3 >= 0
dy_abs_pair_pos_c5_p2_0_4: - cy_0 + cy_4 + dy_abs_pair_c5_p2_0_4 >= 0
dy_abs_pair_pos_c5_p2_1_2: - cy_1 + cy_2 + dy_abs_pair_c5_p2_1_2 >= 0
dy_abs_pair_pos_c5_p2_1_3: - cy_1 + cy_3 + dy_abs_pair_c5_p2_1_3 >= 0
dy_abs_pair_pos_c5_p2_1_4: - cy_1 + cy_4 + dy_abs_pair_c5_p2_1_4 >= 0
dy_abs_pair_pos_c5_p2_2_3: - cy_2 + cy_3 + dy_abs_pair_c5_p2_2_3 >= 0
dy_abs_pair_pos_c5_p2_2_4: - cy_2 + cy_4 + dy_abs_pair_c5_p2_2_4 >= 0
dy_abs_pair_pos_c5_p2_3_4: - cy_3 + cy_4 + dy_abs_pair_c5_p2_3_4 >= 0
exclude_solution_dist_pair_c3_p0: diff_dist_pair_c3_p0_0_1
 + diff_dist_pair_c3_p0_0_2 + diff_dist_pair_c3_p0_0_3
 + diff_dist_pair_c3_p0_0_4 + diff_dist_pair_c3_p0_1_2
 + diff_dist_pair_c3_p0_1_3 + diff_dist_pair_c3_p0_1_4
 + diff_dist_pair_c3_p0_2_3 + diff_dist_pair_c3_p0_2_4
 + diff_dist_pair_c3_p0_3_4 >= 1
exclude_solution_dist_pair_c4_p1: diff_dist_pair_c4_p1_0_1
 + diff_dist_pair_c4_p1_0_2 + diff_dist_pair_c4_p1_0_3
 + diff_dist_pair_c4_p1_0_4 + diff_dist_pair_c4_p1_1_2
 + diff_dist_pair_c4_p1_1_3 + diff_dist_pair_c4_p1_1_4
 + diff_dist_pair_c4_p1_2_3 + diff_dist_pair_c4_p1_2_4
 + diff_dist_pair_c4_p1_3_4 >= 1
exclude_solution_dist_pair_c5_p2: diff_dist_pair_c5_p2_0_1
 + diff_dist_pair_c5_p2_0_2 + diff_dist_pair_c5_p2_0_3
 + diff_dist_pair_c5_p2_0_4 + diff_dist_pair_c5_p2_1_2
 + diff_dist_pair_c5_p2_1_3 + diff_dist_pair_c5_p2_1_4
 + diff_dist_pair_c5_p2_2_3 + diff_dist_pair_c5_p2_2_4
 + diff_dist_pair_c5_p2_3_4 >= 1
exclude_solution_pos_c3_p0: diff_pos_c3_p0_1 + diff_pos_c3_p0_2
 + diff_pos_c3_p0_3 + diff_pos_c3_p0_4 >= 1
exclude_solution_pos_c4_p1: diff_pos_c4_p1_1 + diff_pos_c4_p1_2
 + diff_pos_c4_p1_3 + diff_pos_c4_p1_4 >= 1
exclude_solution_pos_c5_p2: diff_pos_c5_p2_1 + diff_pos_c5_p2_2
 + diff_pos_c5_p2_3 + diff_pos_c5_p2_4 >= 1
fix_center_x_0: cx_0 = 30
fix_center_y_0: cy_0 = 30
h_align_grid_0: h_0 - 0.5 h_grid_0 = 0
h_align_grid_1: h_1 - 0.5 h_grid_1 = 0
h_align_grid_2: h_2 - 0.5 h_grid_2 = 0
h_align_grid_3: h_3 - 0.5 h_grid_3 = 0
h_align_grid_4: h_4 - 0.5 h_grid_4 = 0
h_grid_lb_0: h_grid_0 >= 31.999
h_grid_lb_1: h_grid_1 >= 39.999
h_grid_lb_2: h_grid_2 >= 39.999
h_grid_lb_3: h_grid_3 >= 27.999
h_grid_lb_4: h_grid_4 >= 7.999
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
non_overlap_down_0_1: h_0 + 200 p_down_0_1 + y_0 - y_1 <= 200.001
non_overlap_down_0_2: h_0 + 200 p_down_0_2 + y_0 - y_2 <= 200.001
non_overlap_down_0_3: h_0 + 200 p_down_0_3 + y_0 - y_3 <= 200.001
non_overlap_down_0_4: h_0 + 200 p_down_0_4 + y_0 - y_4 <= 200.001
non_overlap_down_1_2: h_1 + 200 p_down_1_2 + y_1 - y_2 <= 200.001
non_overlap_down_1_3: h_1 + 200 p_down_1_3 + y_1 - y_3 <= 200.001
non_overlap_down_1_4: h_1 + 200 p_down_1_4 + y_1 - y_4 <= 200.001
non_overlap_down_2_3: h_2 + 200 p_down_2_3 + y_2 - y_3 <= 200.001
non_overlap_down_2_4: h_2 + 200 p_down_2_4 + y_2 - y_4 <= 200.001
non_overlap_down_3_4: h_3 + 200 p_down_3_4 + y_3 - y_4 <= 200.001
non_overlap_down_rev_0_1: - h_0 - 200 p_down_0_1 - y_0 + y_1 <= 0.001
non_overlap_down_rev_0_2: - h_0 - 200 p_down_0_2 - y_0 + y_2 <= 0.001
non_overlap_down_rev_0_3: - h_0 - 200 p_down_0_3 - y_0 + y_3 <= 0.001
non_overlap_down_rev_0_4: - h_0 - 200 p_down_0_4 - y_0 + y_4 <= 0.001
non_overlap_down_rev_1_2: - h_1 - 200 p_down_1_2 - y_1 + y_2 <= 0.001
non_overlap_down_rev_1_3: - h_1 - 200 p_down_1_3 - y_1 + y_3 <= 0.001
non_overlap_down_rev_1_4: - h_1 - 200 p_down_1_4 - y_1 + y_4 <= 0.001
non_overlap_down_rev_2_3: - h_2 - 200 p_down_2_3 - y_2 + y_3 <= 0.001
non_overlap_down_rev_2_4: - h_2 - 200 p_down_2_4 - y_2 + y_4 <= 0.001
non_overlap_down_rev_3_4: - h_3 - 200 p_down_3_4 - y_3 + y_4 <= 0.001
non_overlap_left_0_1: 200 p_left_0_1 + w_0 + x_0 - x_1 <= 200.001
non_overlap_left_0_2: 200 p_left_0_2 + w_0 + x_0 - x_2 <= 200.001
non_overlap_left_0_3: 200 p_left_0_3 + w_0 + x_0 - x_3 <= 200.001
non_overlap_left_0_4: 200 p_left_0_4 + w_0 + x_0 - x_4 <= 200.001
non_overlap_left_1_2: 200 p_left_1_2 + w_1 + x_1 - x_2 <= 200.001
non_overlap_left_1_3: 200 p_left_1_3 + w_1 + x_1 - x_3 <= 200.001
non_overlap_left_1_4: 200 p_left_1_4 + w_1 + x_1 - x_4 <= 200.001
non_overlap_left_2_3: 200 p_left_2_3 + w_2 + x_2 - x_3 <= 200.001
non_overlap_left_2_4: 200 p_left_2_4 + w_2 + x_2 - x_4 <= 200.001
non_overlap_left_3_4: 200 p_left_3_4 + w_3 + x_3 - x_4 <= 200.001
non_overlap_left_rev_0_1: - 200 p_left_0_1 - w_0 - x_0 + x_1 <= 0.001
non_overlap_left_rev_0_2: - 200 p_left_0_2 - w_0 - x_0 + x_2 <= 0.001
non_overlap_left_rev_0_3: - 200 p_left_0_3 - w_0 - x_0 + x_3 <= 0.001
non_overlap_left_rev_0_4: - 200 p_left_0_4 - w_0 - x_0 + x_4 <= 0.001
non_overlap_left_rev_1_2: - 200 p_left_1_2 - w_1 - x_1 + x_2 <= 0.001
non_overlap_left_rev_1_3: - 200 p_left_1_3 - w_1 - x_1 + x_3 <= 0.001
non_overlap_left_rev_1_4: - 200 p_left_1_4 - w_1 - x_1 + x_4 <= 0.001
non_overlap_left_rev_2_3: - 200 p_left_2_3 - w_2 - x_2 + x_3 <= 0.001
non_overlap_left_rev_2_4: - 200 p_left_2_4 - w_2 - x_2 + x_4 <= 0.001
non_overlap_left_rev_3_4: - 200 p_left_3_4 - w_3 - x_3 + x_4 <= 0.001
non_overlap_right_0_1: 200 p_right_0_1 + w_1 - x_0 + x_1 <= 200.001
non_overlap_right_0_2: 200 p_right_0_2 + w_2 - x_0 + x_2 <= 200.001
non_overlap_right_0_3: 200 p_right_0_3 + w_3 - x_0 + x_3 <= 200.001
non_overlap_right_0_4: 200 p_right_0_4 + w_4 - x_0 + x_4 <= 200.001
non_overlap_right_1_2: 200 p_right_1_2 + w_2 - x_1 + x_2 <= 200.001
non_overlap_right_1_3: 200 p_right_1_3 + w_3 - x_1 + x_3 <= 200.001
non_overlap_right_1_4: 200 p_right_1_4 + w_4 - x_1 + x_4 <= 200.001
non_overlap_right_2_3: 200 p_right_2_3 + w_3 - x_2 + x_3 <= 200.001
non_overlap_right_2_4: 200 p_right_2_4 + w_4 - x_2 + x_4 <= 200.001
non_overlap_right_3_4: 200 p_right_3_4 + w_4 - x_3 + x_4 <= 200.001
non_overlap_right_rev_0_1: - 200 p_right_0_1 - w_1 + x_0 - x_1 <= 0.001
non_overlap_right_rev_0_2: - 200 p_right_0_2 - w_2 + x_0 - x_2 <= 0.001
non_overlap_right_rev_0_3: - 200 p_right_0_3 - w_3 + x_0 - x_3 <= 0.001
non_overlap_right_rev_0_4: - 200 p_right_0_4 - w_4 + x_0 - x_4 <= 0.001
non_overlap_right_rev_1_2: - 200 p_right_1_2 - w_2 + x_1 - x_2 <= 0.001
non_overlap_right_rev_1_3: - 200 p_right_1_3 - w_3 + x_1 - x_3 <= 0.001
non_overlap_right_rev_1_4: - 200 p_right_1_4 - w_4 + x_1 - x_4 <= 0.001
non_overlap_right_rev_2_3: - 200 p_right_2_3 - w_3 + x_2 - x_3 <= 0.001
non_overlap_right_rev_2_4: - 200 p_right_2_4 - w_4 + x_2 - x_4 <= 0.001
non_overlap_right_rev_3_4: - 200 p_right_3_4 - w_4 + x_3 - x_4 <= 0.001
non_overlap_up_0_1: h_1 + 200 p_up_0_1 - y_0 + y_1 <= 200.001
non_overlap_up_0_2: h_2 + 200 p_up_0_2 - y_0 + y_2 <= 200.001
non_overlap_up_0_3: h_3 + 200 p_up_0_3 - y_0 + y_3 <= 200.001
non_overlap_up_0_4: h_4 + 200 p_up_0_4 - y_0 + y_4 <= 200.001
non_overlap_up_1_2: h_2 + 200 p_up_1_2 - y_1 + y_2 <= 200.001
non_overlap_up_1_3: h_3 + 200 p_up_1_3 - y_1 + y_3 <= 200.001
non_overlap_up_1_4: h_4 + 200 p_up_1_4 - y_1 + y_4 <= 200.001
non_overlap_up_2_3: h_3 + 200 p_up_2_3 - y_2 + y_3 <= 200.001
non_overlap_up_2_4: h_4 + 200 p_up_2_4 - y_2 + y_4 <= 200.001
non_overlap_up_3_4: h_4 + 200 p_up_3_4 - y_3 + y_4 <= 200.001
non_overlap_up_rev_0_1: - h_1 - 200 p_up_0_1 + y_0 - y_1 <= 0.001
non_overlap_up_rev_0_2: - h_2 - 200 p_up_0_2 + y_0 - y_2 <= 0.001
non_overlap_up_rev_0_3: - h_3 - 200 p_up_0_3 + y_0 - y_3 <= 0.001
non_overlap_up_rev_0_4: - h_4 - 200 p_up_0_4 + y_0 - y_4 <= 0.001
non_overlap_up_rev_1_2: - h_2 - 200 p_up_1_2 + y_1 - y_2 <= 0.001
non_overlap_up_rev_1_3: - h_3 - 200 p_up_1_3 + y_1 - y_3 <= 0.001
non_overlap_up_rev_1_4: - h_4 - 200 p_up_1_4 + y_1 - y_4 <= 0.001
non_overlap_up_rev_2_3: - h_3 - 200 p_up_2_3 + y_2 - y_3 <= 0.001
non_overlap_up_rev_2_4: - h_4 - 200 p_up_2_4 + y_2 - y_4 <= 0.001
non_overlap_up_rev_3_4: - h_4 - 200 p_up_3_4 + y_3 - y_4 <= 0.001
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
w_align_grid_0: w_0 - 0.5 w_grid_0 = 0
w_align_grid_1: w_1 - 0.5 w_grid_1 = 0
w_align_grid_2: w_2 - 0.5 w_grid_2 = 0
w_align_grid_3: w_3 - 0.5 w_grid_3 = 0
w_align_grid_4: w_4 - 0.5 w_grid_4 = 0
w_grid_lb_0: w_grid_0 >= 31.999
w_grid_lb_1: w_grid_1 >= 19.999
w_grid_lb_2: w_grid_2 >= 39.999
w_grid_lb_3: w_grid_3 >= 31.999
w_grid_lb_4: w_grid_4 >= 7.999
width_rotation_0: w_0 = 16
width_rotation_1: - 10 r_1 + w_1 = 10
width_rotation_2: w_2 = 20
width_rotation_3: 2 r_3 + w_3 = 16
width_rotation_4: w_4 = 4
x_diff_from_minus_c3_p0_1: x_diff_c3_p0_1 - x_diff_minus_c3_p0_1 >= 0
x_diff_from_minus_c3_p0_2: x_diff_c3_p0_2 - x_diff_minus_c3_p0_2 >= 0
x_diff_from_minus_c3_p0_3: x_diff_c3_p0_3 - x_diff_minus_c3_p0_3 >= 0
x_diff_from_minus_c3_p0_4: x_diff_c3_p0_4 - x_diff_minus_c3_p0_4 >= 0
x_diff_from_minus_c4_p1_1: x_diff_c4_p1_1 - x_diff_minus_c4_p1_1 >= 0
x_diff_from_minus_c4_p1_2: x_diff_c4_p1_2 - x_diff_minus_c4_p1_2 >= 0
x_diff_from_minus_c4_p1_3: x_diff_c4_p1_3 - x_diff_minus_c4_p1_3 >= 0
x_diff_from_minus_c4_p1_4: x_diff_c4_p1_4 - x_diff_minus_c4_p1_4 >= 0
x_diff_from_minus_c5_p2_1: x_diff_c5_p2_1 - x_diff_minus_c5_p2_1 >= 0
x_diff_from_minus_c5_p2_2: x_diff_c5_p2_2 - x_diff_minus_c5_p2_2 >= 0
x_diff_from_minus_c5_p2_3: x_diff_c5_p2_3 - x_diff_minus_c5_p2_3 >= 0
x_diff_from_minus_c5_p2_4: x_diff_c5_p2_4 - x_diff_minus_c5_p2_4 >= 0
x_diff_from_plus_c3_p0_1: x_diff_c3_p0_1 - x_diff_plus_c3_p0_1 >= 0
x_diff_from_plus_c3_p0_2: x_diff_c3_p0_2 - x_diff_plus_c3_p0_2 >= 0
x_diff_from_plus_c3_p0_3: x_diff_c3_p0_3 - x_diff_plus_c3_p0_3 >= 0
x_diff_from_plus_c3_p0_4: x_diff_c3_p0_4 - x_diff_plus_c3_p0_4 >= 0
x_diff_from_plus_c4_p1_1: x_diff_c4_p1_1 - x_diff_plus_c4_p1_1 >= 0
x_diff_from_plus_c4_p1_2: x_diff_c4_p1_2 - x_diff_plus_c4_p1_2 >= 0
x_diff_from_plus_c4_p1_3: x_diff_c4_p1_3 - x_diff_plus_c4_p1_3 >= 0
x_diff_from_plus_c4_p1_4: x_diff_c4_p1_4 - x_diff_plus_c4_p1_4 >= 0
x_diff_from_plus_c5_p2_1: x_diff_c5_p2_1 - x_diff_plus_c5_p2_1 >= 0
x_diff_from_plus_c5_p2_2: x_diff_c5_p2_2 - x_diff_plus_c5_p2_2 >= 0
x_diff_from_plus_c5_p2_3: x_diff_c5_p2_3 - x_diff_plus_c5_p2_3 >= 0
x_diff_from_plus_c5_p2_4: x_diff_c5_p2_4 - x_diff_plus_c5_p2_4 >= 0
x_diff_minus_lower_c3_p0_1: x_1 + 120 x_diff_minus_c3_p0_1 <= 142
x_diff_minus_lower_c3_p0_2: x_2 + 120 x_diff_minus_c3_p0_2 <= 152
x_diff_minus_lower_c3_p0_3: x_3 + 120 x_diff_minus_c3_p0_3 <= 126
x_diff_minus_lower_c3_p0_4: x_4 + 120 x_diff_minus_c3_p0_4 <= 160
x_diff_minus_lower_c4_p1_1: x_1 + 120 x_diff_minus_c4_p1_1 <= 142
x_diff_minus_lower_c4_p1_2: x_2 + 120 x_diff_minus_c4_p1_2 <= 122
x_diff_minus_lower_c4_p1_3: x_3 + 120 x_diff_minus_c4_p1_3 <= 152
x_diff_minus_lower_c4_p1_4: x_4 + 120 x_diff_minus_c4_p1_4 <= 130
x_diff_minus_lower_c5_p2_1: x_1 + 120 x_diff_minus_c5_p2_1 <= 142
x_diff_minus_lower_c5_p2_2: x_2 + 120 x_diff_minus_c5_p2_2 <= 122
x_diff_minus_lower_c5_p2_3: x_3 + 120 x_diff_minus_c5_p2_3 <= 152
x_diff_minus_lower_c5_p2_4: x_4 + 120 x_diff_minus_c5_p2_4 <= 130
x_diff_minus_upper_c3_p0_1: x_1 + 120 x_diff_minus_c3_p0_1 >= 22.001
x_diff_minus_upper_c3_p0_2: x_2 + 120 x_diff_minus_c3_p0_2 >= 32.001
x_diff_minus_upper_c3_p0_3: x_3 + 120 x_diff_minus_c3_p0_3 >= 6.001
x_diff_minus_upper_c3_p0_4: x_4 + 120 x_diff_minus_c3_p0_4 >= 40.001
x_diff_minus_upper_c4_p1_1: x_1 + 120 x_diff_minus_c4_p1_1 >= 22.001
x_diff_minus_upper_c4_p1_2: x_2 + 120 x_diff_minus_c4_p1_2 >= 2.001
x_diff_minus_upper_c4_p1_3: x_3 + 120 x_diff_minus_c4_p1_3 >= 32.001
x_diff_minus_upper_c4_p1_4: x_4 + 120 x_diff_minus_c4_p1_4 >= 10.001
x_diff_minus_upper_c5_p2_1: x_1 + 120 x_diff_minus_c5_p2_1 >= 22.001
x_diff_minus_upper_c5_p2_2: x_2 + 120 x_diff_minus_c5_p2_2 >= 2.001
x_diff_minus_upper_c5_p2_3: x_3 + 120 x_diff_minus_c5_p2_3 >= 32.001
x_diff_minus_upper_c5_p2_4: x_4 + 120 x_diff_minus_c5_p2_4 >= 10.001
x_diff_plus_lower_c3_p0_1: x_1 - 120 x_diff_plus_c3_p0_1 >= -92
x_diff_plus_lower_c3_p0_2: x_2 - 120 x_diff_plus_c3_p0_2 >= -82
x_diff_plus_lower_c3_p0_3: x_3 - 120 x_diff_plus_c3_p0_3 >= -108
x_diff_plus_lower_c3_p0_4: x_4 - 120 x_diff_plus_c3_p0_4 >= -74
x_diff_plus_lower_c4_p1_1: x_1 - 120 x_diff_plus_c4_p1_1 >= -92
x_diff_plus_lower_c4_p1_2: x_2 - 120 x_diff_plus_c4_p1_2 >= -112
x_diff_plus_lower_c4_p1_3: x_3 - 120 x_diff_plus_c4_p1_3 >= -82
x_diff_plus_lower_c4_p1_4: x_4 - 120 x_diff_plus_c4_p1_4 >= -104
x_diff_plus_lower_c5_p2_1: x_1 - 120 x_diff_plus_c5_p2_1 >= -92
x_diff_plus_lower_c5_p2_2: x_2 - 120 x_diff_plus_c5_p2_2 >= -112
x_diff_plus_lower_c5_p2_3: x_3 - 120 x_diff_plus_c5_p2_3 >= -82
x_diff_plus_lower_c5_p2_4: x_4 - 120 x_diff_plus_c5_p2_4 >= -104
x_diff_plus_upper_c3_p0_1: x_1 - 120 x_diff_plus_c3_p0_1 <= 27.999
x_diff_plus_upper_c3_p0_2: x_2 - 120 x_diff_plus_c3_p0_2 <= 37.999
x_diff_plus_upper_c3_p0_3: x_3 - 120 x_diff_plus_c3_p0_3 <= 11.999
x_diff_plus_upper_c3_p0_4: x_4 - 120 x_diff_plus_c3_p0_4 <= 45.999
x_diff_plus_upper_c4_p1_1: x_1 - 120 x_diff_plus_c4_p1_1 <= 27.999
x_diff_plus_upper_c4_p1_2: x_2 - 120 x_diff_plus_c4_p1_2 <= 7.999
x_diff_plus_upper_c4_p1_3: x_3 - 120 x_diff_plus_c4_p1_3 <= 37.999
x_diff_plus_upper_c4_p1_4: x_4 - 120 x_diff_plus_c4_p1_4 <= 15.999
x_diff_plus_upper_c5_p2_1: x_1 - 120 x_diff_plus_c5_p2_1 <= 27.999
x_diff_plus_upper_c5_p2_2: x_2 - 120 x_diff_plus_c5_p2_2 <= 7.999
x_diff_plus_upper_c5_p2_3: x_3 - 120 x_diff_plus_c5_p2_3 <= 37.999
x_diff_plus_upper_c5_p2_4: x_4 - 120 x_diff_plus_c5_p2_4 <= 15.999
x_diff_upper_c3_p0_1: x_diff_c3_p0_1 - x_diff_minus_c3_p0_1
 - x_diff_plus_c3_p0_1 <= 0
x_diff_upper_c3_p0_2: x_diff_c3_p0_2 - x_diff_minus_c3_p0_2
 - x_diff_plus_c3_p0_2 <= 0
x_diff_upper_c3_p0_3: x_diff_c3_p0_3 - x_diff_minus_c3_p0_3
 - x_diff_plus_c3_p0_3 <= 0
x_diff_upper_c3_p0_4: x_diff_c3_p0_4 - x_diff_minus_c3_p0_4
 - x_diff_plus_c3_p0_4 <= 0
x_diff_upper_c4_p1_1: x_diff_c4_p1_1 - x_diff_minus_c4_p1_1
 - x_diff_plus_c4_p1_1 <= 0
x_diff_upper_c4_p1_2: x_diff_c4_p1_2 - x_diff_minus_c4_p1_2
 - x_diff_plus_c4_p1_2 <= 0
x_diff_upper_c4_p1_3: x_diff_c4_p1_3 - x_diff_minus_c4_p1_3
 - x_diff_plus_c4_p1_3 <= 0
x_diff_upper_c4_p1_4: x_diff_c4_p1_4 - x_diff_minus_c4_p1_4
 - x_diff_plus_c4_p1_4 <= 0
x_diff_upper_c5_p2_1: x_diff_c5_p2_1 - x_diff_minus_c5_p2_1
 - x_diff_plus_c5_p2_1 <= 0
x_diff_upper_c5_p2_2: x_diff_c5_p2_2 - x_diff_minus_c5_p2_2
 - x_diff_plus_c5_p2_2 <= 0
x_diff_upper_c5_p2_3: x_diff_c5_p2_3 - x_diff_minus_c5_p2_3
 - x_diff_plus_c5_p2_3 <= 0
x_diff_upper_c5_p2_4: x_diff_c5_p2_4 - x_diff_minus_c5_p2_4
 - x_diff_plus_c5_p2_4 <= 0
x_grid_to_real_0: x_0 - 0.5 x_grid_0 = 0
x_grid_to_real_1: x_1 - 0.5 x_grid_1 = 0
x_grid_to_real_2: x_2 - 0.5 x_grid_2 = 0
x_grid_to_real_3: x_3 - 0.5 x_grid_3 = 0
x_grid_to_real_4: x_4 - 0.5 x_grid_4 = 0
y_diff_from_minus_c3_p0_1: y_diff_c3_p0_1 - y_diff_minus_c3_p0_1 >= 0
y_diff_from_minus_c3_p0_2: y_diff_c3_p0_2 - y_diff_minus_c3_p0_2 >= 0
y_diff_from_minus_c3_p0_3: y_diff_c3_p0_3 - y_diff_minus_c3_p0_3 >= 0
y_diff_from_minus_c3_p0_4: y_diff_c3_p0_4 - y_diff_minus_c3_p0_4 >= 0
y_diff_from_minus_c4_p1_1: y_diff_c4_p1_1 - y_diff_minus_c4_p1_1 >= 0
y_diff_from_minus_c4_p1_2: y_diff_c4_p1_2 - y_diff_minus_c4_p1_2 >= 0
y_diff_from_minus_c4_p1_3: y_diff_c4_p1_3 - y_diff_minus_c4_p1_3 >= 0
y_diff_from_minus_c4_p1_4: y_diff_c4_p1_4 - y_diff_minus_c4_p1_4 >= 0
y_diff_from_minus_c5_p2_1: y_diff_c5_p2_1 - y_diff_minus_c5_p2_1 >= 0
y_diff_from_minus_c5_p2_2: y_diff_c5_p2_2 - y_diff_minus_c5_p2_2 >= 0
y_diff_from_minus_c5_p2_3: y_diff_c5_p2_3 - y_diff_minus_c5_p2_3 >= 0
y_diff_from_minus_c5_p2_4: y_diff_c5_p2_4 - y_diff_minus_c5_p2_4 >= 0
y_diff_from_plus_c3_p0_1: y_diff_c3_p0_1 - y_diff_plus_c3_p0_1 >= 0
y_diff_from_plus_c3_p0_2: y_diff_c3_p0_2 - y_diff_plus_c3_p0_2 >= 0
y_diff_from_plus_c3_p0_3: y_diff_c3_p0_3 - y_diff_plus_c3_p0_3 >= 0
y_diff_from_plus_c3_p0_4: y_diff_c3_p0_4 - y_diff_plus_c3_p0_4 >= 0
y_diff_from_plus_c4_p1_1: y_diff_c4_p1_1 - y_diff_plus_c4_p1_1 >= 0
y_diff_from_plus_c4_p1_2: y_diff_c4_p1_2 - y_diff_plus_c4_p1_2 >= 0
y_diff_from_plus_c4_p1_3: y_diff_c4_p1_3 - y_diff_plus_c4_p1_3 >= 0
y_diff_from_plus_c4_p1_4: y_diff_c4_p1_4 - y_diff_plus_c4_p1_4 >= 0
y_diff_from_plus_c5_p2_1: y_diff_c5_p2_1 - y_diff_plus_c5_p2_1 >= 0
y_diff_from_plus_c5_p2_2: y_diff_c5_p2_2 - y_diff_plus_c5_p2_2 >= 0
y_diff_from_plus_c5_p2_3: y_diff_c5_p2_3 - y_diff_plus_c5_p2_3 >= 0
y_diff_from_plus_c5_p2_4: y_diff_c5_p2_4 - y_diff_plus_c5_p2_4 >= 0
y_diff_minus_lower_c3_p0_1: y_1 + 120 y_diff_minus_c3_p0_1 <= 155
y_diff_minus_lower_c3_p0_2: y_2 + 120 y_diff_minus_c3_p0_2 <= 155
y_diff_minus_lower_c3_p0_3: y_3 + 120 y_diff_minus_c3_p0_3 <= 155
y_diff_minus_lower_c3_p0_4: y_4 + 120 y_diff_minus_c3_p0_4 <= 151
y_diff_minus_lower_c4_p1_1: y_1 + 120 y_diff_minus_c4_p1_1 <= 119
y_diff_minus_lower_c4_p1_2: y_2 + 120 y_diff_minus_c4_p1_2 <= 117
y_diff_minus_lower_c4_p1_3: y_3 + 120 y_diff_minus_c4_p1_3 <= 123
y_diff_minus_lower_c4_p1_4: y_4 + 120 y_diff_minus_c4_p1_4 <= 137
y_diff_minus_lower_c5_p2_1: y_1 + 120 y_diff_minus_c5_p2_1 <= 155
y_diff_minus_lower_c5_p2_2: y_2 + 120 y_diff_minus_c5_p2_2 <= 155
y_diff_minus_lower_c5_p2_3: y_3 + 120 y_diff_minus_c5_p2_3 <= 155
y_diff_minus_lower_c5_p2_4: y_4 + 120 y_diff_minus_c5_p2_4 <= 151
y_diff_minus_upper_c3_p0_1: y_1 + 120 y_diff_minus_c3_p0_1 >= 35.001
y_diff_minus_upper_c3_p0_2: y_2 + 120 y_diff_minus_c3_p0_2 >= 35.001
y_diff_minus_upper_c3_p0_3: y_3 + 120 y_diff_minus_c3_p0_3 >= 35.001
y_diff_minus_upper_c3_p0_4: y_4 + 120 y_diff_minus_c3_p0_4 >= 31.001
y_diff_minus_upper_c4_p1_1: y_1 + 120 y_diff_minus_c4_p1_1 >= -0.999
y_diff_minus_upper_c4_p1_2: y_2 + 120 y_diff_minus_c4_p1_2 >= -2.999
y_diff_minus_upper_c4_p1_3: y_3 + 120 y_diff_minus_c4_p1_3 >= 3.001
y_diff_minus_upper_c4_p1_4: y_4 + 120 y_diff_minus_c4_p1_4 >= 17.001
y_diff_minus_upper_c5_p2_1: y_1 + 120 y_diff_minus_c5_p2_1 >= 35.001
y_diff_minus_upper_c5_p2_2: y_2 + 120 y_diff_minus_c5_p2_2 >= 35.001
y_diff_minus_upper_c5_p2_3: y_3 + 120 y_diff_minus_c5_p2_3 >= 35.001
y_diff_minus_upper_c5_p2_4: y_4 + 120 y_diff_minus_c5_p2_4 >= 31.001
y_diff_plus_lower_c3_p0_1: y_1 - 120 y_diff_plus_c3_p0_1 >= -79
y_diff_plus_lower_c3_p0_2: y_2 - 120 y_diff_plus_c3_p0_2 >= -79
y_diff_plus_lower_c3_p0_3: y_3 - 120 y_diff_plus_c3_p0_3 >= -79
y_diff_plus_lower_c3_p0_4: y_4 - 120 y_diff_plus_c3_p0_4 >= -83
y_diff_plus_lower_c4_p1_1: y_1 - 120 y_diff_plus_c4_p1_1 >= -115
y_diff_plus_lower_c4_p1_2: y_2 - 120 y_diff_plus_c4_p1_2 >= -117
y_diff_plus_lower_c4_p1_3: y_3 - 120 y_diff_plus_c4_p1_3 >= -111
y_diff_plus_lower_c4_p1_4: y_4 - 120 y_diff_plus_c4_p1_4 >= -97
y_diff_plus_lower_c5_p2_1: y_1 - 120 y_diff_plus_c5_p2_1 >= -79
y_diff_plus_lower_c5_p2_2: y_2 - 120 y_diff_plus_c5_p2_2 >= -79
y_diff_plus_lower_c5_p2_3: y_3 - 120 y_diff_plus_c5_p2_3 >= -79
y_diff_plus_lower_c5_p2_4: y_4 - 120 y_diff_plus_c5_p2_4 >= -83
y_diff_plus_upper_c3_p0_1: y_1 - 120 y_diff_plus_c3_p0_1 <= 40.999
y_diff_plus_upper_c3_p0_2: y_2 - 120 y_diff_plus_c3_p0_2 <= 40.999
y_diff_plus_upper_c3_p0_3: y_3 - 120 y_diff_plus_c3_p0_3 <= 40.999
y_diff_plus_upper_c3_p0_4: y_4 - 120 y_diff_plus_c3_p0_4 <= 36.999
y_diff_plus_upper_c4_p1_1: y_1 - 120 y_diff_plus_c4_p1_1 <= 4.999
y_diff_plus_upper_c4_p1_2: y_2 - 120 y_diff_plus_c4_p1_2 <= 2.999
y_diff_plus_upper_c4_p1_3: y_3 - 120 y_diff_plus_c4_p1_3 <= 8.999
y_diff_plus_upper_c4_p1_4: y_4 - 120 y_diff_plus_c4_p1_4 <= 22.999
y_diff_plus_upper_c5_p2_1: y_1 - 120 y_diff_plus_c5_p2_1 <= 40.999
y_diff_plus_upper_c5_p2_2: y_2 - 120 y_diff_plus_c5_p2_2 <= 40.999
y_diff_plus_upper_c5_p2_3: y_3 - 120 y_diff_plus_c5_p2_3 <= 40.999
y_diff_plus_upper_c5_p2_4: y_4 - 120 y_diff_plus_c5_p2_4 <= 36.999
y_diff_upper_c3_p0_1: y_diff_c3_p0_1 - y_diff_minus_c3_p0_1
 - y_diff_plus_c3_p0_1 <= 0
y_diff_upper_c3_p0_2: y_diff_c3_p0_2 - y_diff_minus_c3_p0_2
 - y_diff_plus_c3_p0_2 <= 0
y_diff_upper_c3_p0_3: y_diff_c3_p0_3 - y_diff_minus_c3_p0_3
 - y_diff_plus_c3_p0_3 <= 0
y_diff_upper_c3_p0_4: y_diff_c3_p0_4 - y_diff_minus_c3_p0_4
 - y_diff_plus_c3_p0_4 <= 0
y_diff_upper_c4_p1_1: y_diff_c4_p1_1 - y_diff_minus_c4_p1_1
 - y_diff_plus_c4_p1_1 <= 0
y_diff_upper_c4_p1_2: y_diff_c4_p1_2 - y_diff_minus_c4_p1_2
 - y_diff_plus_c4_p1_2 <= 0
y_diff_upper_c4_p1_3: y_diff_c4_p1_3 - y_diff_minus_c4_p1_3
 - y_diff_plus_c4_p1_3 <= 0
y_diff_upper_c4_p1_4: y_diff_c4_p1_4 - y_diff_minus_c4_p1_4
 - y_diff_plus_c4_p1_4 <= 0
y_diff_upper_c5_p2_1: y_diff_c5_p2_1 - y_diff_minus_c5_p2_1
 - y_diff_plus_c5_p2_1 <= 0
y_diff_upper_c5_p2_2: y_diff_c5_p2_2 - y_diff_minus_c5_p2_2
 - y_diff_plus_c5_p2_2 <= 0
y_diff_upper_c5_p2_3: y_diff_c5_p2_3 - y_diff_minus_c5_p2_3
 - y_diff_plus_c5_p2_3 <= 0
y_diff_upper_c5_p2_4: y_diff_c5_p2_4 - y_diff_minus_c5_p2_4
 - y_diff_plus_c5_p2_4 <= 0
y_grid_to_real_0: y_0 - 0.5 y_grid_0 = 0
y_grid_to_real_1: y_1 - 0.5 y_grid_1 = 0
y_grid_to_real_2: y_2 - 0.5 y_grid_2 = 0
y_grid_to_real_3: y_3 - 0.5 y_grid_3 = 0
y_grid_to_real_4: y_4 - 0.5 y_grid_4 = 0
Bounds
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
 dist_diff_abs_pair_c3_p0_0_1 <= 120
 dist_diff_abs_pair_c3_p0_0_2 <= 120
 dist_diff_abs_pair_c3_p0_0_3 <= 120
 dist_diff_abs_pair_c3_p0_0_4 <= 120
 dist_diff_abs_pair_c3_p0_1_2 <= 120
 dist_diff_abs_pair_c3_p0_1_3 <= 120
 dist_diff_abs_pair_c3_p0_1_4 <= 120
 dist_diff_abs_pair_c3_p0_2_3 <= 120
 dist_diff_abs_pair_c3_p0_2_4 <= 120
 dist_diff_abs_pair_c3_p0_3_4 <= 120
 dist_diff_abs_pair_c4_p1_0_1 <= 120
 dist_diff_abs_pair_c4_p1_0_2 <= 120
 dist_diff_abs_pair_c4_p1_0_3 <= 120
 dist_diff_abs_pair_c4_p1_0_4 <= 120
 dist_diff_abs_pair_c4_p1_1_2 <= 120
 dist_diff_abs_pair_c4_p1_1_3 <= 120
 dist_diff_abs_pair_c4_p1_1_4 <= 120
 dist_diff_abs_pair_c4_p1_2_3 <= 120
 dist_diff_abs_pair_c4_p1_2_4 <= 120
 dist_diff_abs_pair_c4_p1_3_4 <= 120
 dist_diff_abs_pair_c5_p2_0_1 <= 120
 dist_diff_abs_pair_c5_p2_0_2 <= 120
 dist_diff_abs_pair_c5_p2_0_3 <= 120
 dist_diff_abs_pair_c5_p2_0_4 <= 120
 dist_diff_abs_pair_c5_p2_1_2 <= 120
 dist_diff_abs_pair_c5_p2_1_3 <= 120
 dist_diff_abs_pair_c5_p2_1_4 <= 120
 dist_diff_abs_pair_c5_p2_2_3 <= 120
 dist_diff_abs_pair_c5_p2_2_4 <= 120
 dist_diff_abs_pair_c5_p2_3_4 <= 120
 -120 <= dist_diff_pair_c3_p0_0_1 <= 120
 -120 <= dist_diff_pair_c3_p0_0_2 <= 120
 -120 <= dist_diff_pair_c3_p0_0_3 <= 120
 -120 <= dist_diff_pair_c3_p0_0_4 <= 120
 -120 <= dist_diff_pair_c3_p0_1_2 <= 120
 -120 <= dist_diff_pair_c3_p0_1_3 <= 120
 -120 <= dist_diff_pair_c3_p0_1_4 <= 120
 -120 <= dist_diff_pair_c3_p0_2_3 <= 120
 -120 <= dist_diff_pair_c3_p0_2_4 <= 120
 -120 <= dist_diff_pair_c3_p0_3_4 <= 120
 -120 <= dist_diff_pair_c4_p1_0_1 <= 120
 -120 <= dist_diff_pair_c4_p1_0_2 <= 120
 -120 <= dist_diff_pair_c4_p1_0_3 <= 120
 -120 <= dist_diff_pair_c4_p1_0_4 <= 120
 -120 <= dist_diff_pair_c4_p1_1_2 <= 120
 -120 <= dist_diff_pair_c4_p1_1_3 <= 120
 -120 <= dist_diff_pair_c4_p1_1_4 <= 120
 -120 <= dist_diff_pair_c4_p1_2_3 <= 120
 -120 <= dist_diff_pair_c4_p1_2_4 <= 120
 -120 <= dist_diff_pair_c4_p1_3_4 <= 120
 -120 <= dist_diff_pair_c5_p2_0_1 <= 120
 -120 <= dist_diff_pair_c5_p2_0_2 <= 120
 -120 <= dist_diff_pair_c5_p2_0_3 <= 120
 -120 <= dist_diff_pair_c5_p2_0_4 <= 120
 -120 <= dist_diff_pair_c5_p2_1_2 <= 120
 -120 <= dist_diff_pair_c5_p2_1_3 <= 120
 -120 <= dist_diff_pair_c5_p2_1_4 <= 120
 -120 <= dist_diff_pair_c5_p2_2_3 <= 120
 -120 <= dist_diff_pair_c5_p2_2_4 <= 120
 -120 <= dist_diff_pair_c5_p2_3_4 <= 120
 dx_abs_pair_c3_p0_0_1 <= 60
 dx_abs_pair_c3_p0_0_2 <= 60
 dx_abs_pair_c3_p0_0_3 <= 60
 dx_abs_pair_c3_p0_0_4 <= 60
 dx_abs_pair_c3_p0_1_2 <= 60
 dx_abs_pair_c3_p0_1_3 <= 60
 dx_abs_pair_c3_p0_1_4 <= 60
 dx_abs_pair_c3_p0_2_3 <= 60
 dx_abs_pair_c3_p0_2_4 <= 60
 dx_abs_pair_c3_p0_3_4 <= 60
 dx_abs_pair_c4_p1_0_1 <= 60
 dx_abs_pair_c4_p1_0_2 <= 60
 dx_abs_pair_c4_p1_0_3 <= 60
 dx_abs_pair_c4_p1_0_4 <= 60
 dx_abs_pair_c4_p1_1_2 <= 60
 dx_abs_pair_c4_p1_1_3 <= 60
 dx_abs_pair_c4_p1_1_4 <= 60
 dx_abs_pair_c4_p1_2_3 <= 60
 dx_abs_pair_c4_p1_2_4 <= 60
 dx_abs_pair_c4_p1_3_4 <= 60
 dx_abs_pair_c5_p2_0_1 <= 60
 dx_abs_pair_c5_p2_0_2 <= 60
 dx_abs_pair_c5_p2_0_3 <= 60
 dx_abs_pair_c5_p2_0_4 <= 60
 dx_abs_pair_c5_p2_1_2 <= 60
 dx_abs_pair_c5_p2_1_3 <= 60
 dx_abs_pair_c5_p2_1_4 <= 60
 dx_abs_pair_c5_p2_2_3 <= 60
 dx_abs_pair_c5_p2_2_4 <= 60
 dx_abs_pair_c5_p2_3_4 <= 60
 dy_abs_pair_c3_p0_0_1 <= 60
 dy_abs_pair_c3_p0_0_2 <= 60
 dy_abs_pair_c3_p0_0_3 <= 60
 dy_abs_pair_c3_p0_0_4 <= 60
 dy_abs_pair_c3_p0_1_2 <= 60
 dy_abs_pair_c3_p0_1_3 <= 60
 dy_abs_pair_c3_p0_1_4 <= 60
 dy_abs_pair_c3_p0_2_3 <= 60
 dy_abs_pair_c3_p0_2_4 <= 60
 dy_abs_pair_c3_p0_3_4 <= 60
 dy_abs_pair_c4_p1_0_1 <= 60
 dy_abs_pair_c4_p1_0_2 <= 60
 dy_abs_pair_c4_p1_0_3 <= 60
 dy_abs_pair_c4_p1_0_4 <= 60
 dy_abs_pair_c4_p1_1_2 <= 60
 dy_abs_pair_c4_p1_1_3 <= 60
 dy_abs_pair_c4_p1_1_4 <= 60
 dy_abs_pair_c4_p1_2_3 <= 60
 dy_abs_pair_c4_p1_2_4 <= 60
 dy_abs_pair_c4_p1_3_4 <= 60
 dy_abs_pair_c5_p2_0_1 <= 60
 dy_abs_pair_c5_p2_0_2 <= 60
 dy_abs_pair_c5_p2_0_3 <= 60
 dy_abs_pair_c5_p2_0_4 <= 60
 dy_abs_pair_c5_p2_1_2 <= 60
 dy_abs_pair_c5_p2_1_3 <= 60
 dy_abs_pair_c5_p2_1_4 <= 60
 dy_abs_pair_c5_p2_2_3 <= 60
 dy_abs_pair_c5_p2_2_4 <= 60
 dy_abs_pair_c5_p2_3_4 <= 60
 h_0 = 16
 10 <= h_1 <= 20
 h_2 = 20
 14 <= h_3 <= 16
 h_4 = 4
 1 <= h_grid_0 <= 120
 1 <= h_grid_1 <= 120
 1 <= h_grid_2 <= 120
 1 <= h_grid_3 <= 120
 1 <= h_grid_4 <= 120
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
 1 <= w_grid_0 <= 120
 1 <= w_grid_1 <= 120
 1 <= w_grid_2 <= 120
 1 <= w_grid_3 <= 120
 1 <= w_grid_4 <= 120
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
h_grid_0
h_grid_1
h_grid_2
h_grid_3
h_grid_4
w_grid_0
w_grid_1
w_grid_2
w_grid_3
w_grid_4
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
diff_dist_pair_c3_p0_0_1
diff_dist_pair_c3_p0_0_2
diff_dist_pair_c3_p0_0_3
diff_dist_pair_c3_p0_0_4
diff_dist_pair_c3_p0_1_2
diff_dist_pair_c3_p0_1_3
diff_dist_pair_c3_p0_1_4
diff_dist_pair_c3_p0_2_3
diff_dist_pair_c3_p0_2_4
diff_dist_pair_c3_p0_3_4
diff_dist_pair_c4_p1_0_1
diff_dist_pair_c4_p1_0_2
diff_dist_pair_c4_p1_0_3
diff_dist_pair_c4_p1_0_4
diff_dist_pair_c4_p1_1_2
diff_dist_pair_c4_p1_1_3
diff_dist_pair_c4_p1_1_4
diff_dist_pair_c4_p1_2_3
diff_dist_pair_c4_p1_2_4
diff_dist_pair_c4_p1_3_4
diff_dist_pair_c5_p2_0_1
diff_dist_pair_c5_p2_0_2
diff_dist_pair_c5_p2_0_3
diff_dist_pair_c5_p2_0_4
diff_dist_pair_c5_p2_1_2
diff_dist_pair_c5_p2_1_3
diff_dist_pair_c5_p2_1_4
diff_dist_pair_c5_p2_2_3
diff_dist_pair_c5_p2_2_4
diff_dist_pair_c5_p2_3_4
diff_pos_c3_p0_0
diff_pos_c3_p0_1
diff_pos_c3_p0_2
diff_pos_c3_p0_3
diff_pos_c3_p0_4
diff_pos_c4_p1_0
diff_pos_c4_p1_1
diff_pos_c4_p1_2
diff_pos_c4_p1_3
diff_pos_c4_p1_4
diff_pos_c5_p2_0
diff_pos_c5_p2_1
diff_pos_c5_p2_2
diff_pos_c5_p2_3
diff_pos_c5_p2_4
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
x_diff_c3_p0_1
x_diff_c3_p0_2
x_diff_c3_p0_3
x_diff_c3_p0_4
x_diff_c4_p1_1
x_diff_c4_p1_2
x_diff_c4_p1_3
x_diff_c4_p1_4
x_diff_c5_p2_1
x_diff_c5_p2_2
x_diff_c5_p2_3
x_diff_c5_p2_4
x_diff_minus_c3_p0_1
x_diff_minus_c3_p0_2
x_diff_minus_c3_p0_3
x_diff_minus_c3_p0_4
x_diff_minus_c4_p1_1
x_diff_minus_c4_p1_2
x_diff_minus_c4_p1_3
x_diff_minus_c4_p1_4
x_diff_minus_c5_p2_1
x_diff_minus_c5_p2_2
x_diff_minus_c5_p2_3
x_diff_minus_c5_p2_4
x_diff_plus_c3_p0_1
x_diff_plus_c3_p0_2
x_diff_plus_c3_p0_3
x_diff_plus_c3_p0_4
x_diff_plus_c4_p1_1
x_diff_plus_c4_p1_2
x_diff_plus_c4_p1_3
x_diff_plus_c4_p1_4
x_diff_plus_c5_p2_1
x_diff_plus_c5_p2_2
x_diff_plus_c5_p2_3
x_diff_plus_c5_p2_4
y_diff_c3_p0_1
y_diff_c3_p0_2
y_diff_c3_p0_3
y_diff_c3_p0_4
y_diff_c4_p1_1
y_diff_c4_p1_2
y_diff_c4_p1_3
y_diff_c4_p1_4
y_diff_c5_p2_1
y_diff_c5_p2_2
y_diff_c5_p2_3
y_diff_c5_p2_4
y_diff_minus_c3_p0_1
y_diff_minus_c3_p0_2
y_diff_minus_c3_p0_3
y_diff_minus_c3_p0_4
y_diff_minus_c4_p1_1
y_diff_minus_c4_p1_2
y_diff_minus_c4_p1_3
y_diff_minus_c4_p1_4
y_diff_minus_c5_p2_1
y_diff_minus_c5_p2_2
y_diff_minus_c5_p2_3
y_diff_minus_c5_p2_4
y_diff_plus_c3_p0_1
y_diff_plus_c3_p0_2
y_diff_plus_c3_p0_3
y_diff_plus_c3_p0_4
y_diff_plus_c4_p1_1
y_diff_plus_c4_p1_2
y_diff_plus_c4_p1_3
y_diff_plus_c4_p1_4
y_diff_plus_c5_p2_1
y_diff_plus_c5_p2_2
y_diff_plus_c5_p2_3
y_diff_plus_c5_p2_4
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
