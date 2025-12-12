\* ChipletPlacementGrid *\
Minimize
Objective: 0.1 bbox_area_proxy_t + dx_abs_0_1 + dx_abs_0_2 + dx_abs_0_4
 + dx_abs_0_5 + dx_abs_1_3 + dx_abs_2_3 + dx_abs_2_6 + dx_abs_2_7 + dx_abs_3_8
 + dx_abs_3_9 + dy_abs_0_1 + dy_abs_0_2 + dy_abs_0_4 + dy_abs_0_5 + dy_abs_1_3
 + dy_abs_2_3 + dy_abs_2_6 + dy_abs_2_7 + dy_abs_3_8 + dy_abs_3_9
Subject To
bbox_h_def: bbox_h - bbox_max_y + bbox_min_y = 0
bbox_max_x_0: bbox_max_x - w_0 - x_0 >= 0
bbox_max_x_1: bbox_max_x - w_1 - x_1 >= 0
bbox_max_x_2: bbox_max_x - w_2 - x_2 >= 0
bbox_max_x_3: bbox_max_x - w_3 - x_3 >= 0
bbox_max_x_4: bbox_max_x - w_4 - x_4 >= 0
bbox_max_x_5: bbox_max_x - w_5 - x_5 >= 0
bbox_max_x_6: bbox_max_x - w_6 - x_6 >= 0
bbox_max_x_7: bbox_max_x - w_7 - x_7 >= 0
bbox_max_x_8: bbox_max_x - w_8 - x_8 >= 0
bbox_max_x_9: bbox_max_x - w_9 - x_9 >= 0
bbox_max_y_0: bbox_max_y - h_0 - y_0 >= 0
bbox_max_y_1: bbox_max_y - h_1 - y_1 >= 0
bbox_max_y_2: bbox_max_y - h_2 - y_2 >= 0
bbox_max_y_3: bbox_max_y - h_3 - y_3 >= 0
bbox_max_y_4: bbox_max_y - h_4 - y_4 >= 0
bbox_max_y_5: bbox_max_y - h_5 - y_5 >= 0
bbox_max_y_6: bbox_max_y - h_6 - y_6 >= 0
bbox_max_y_7: bbox_max_y - h_7 - y_7 >= 0
bbox_max_y_8: bbox_max_y - h_8 - y_8 >= 0
bbox_max_y_9: bbox_max_y - h_9 - y_9 >= 0
bbox_min_x_0: bbox_min_x - x_0 <= 0
bbox_min_x_1: bbox_min_x - x_1 <= 0
bbox_min_x_2: bbox_min_x - x_2 <= 0
bbox_min_x_3: bbox_min_x - x_3 <= 0
bbox_min_x_4: bbox_min_x - x_4 <= 0
bbox_min_x_5: bbox_min_x - x_5 <= 0
bbox_min_x_6: bbox_min_x - x_6 <= 0
bbox_min_x_7: bbox_min_x - x_7 <= 0
bbox_min_x_8: bbox_min_x - x_8 <= 0
bbox_min_x_9: bbox_min_x - x_9 <= 0
bbox_min_y_0: bbox_min_y - y_0 <= 0
bbox_min_y_1: bbox_min_y - y_1 <= 0
bbox_min_y_2: bbox_min_y - y_2 <= 0
bbox_min_y_3: bbox_min_y - y_3 <= 0
bbox_min_y_4: bbox_min_y - y_4 <= 0
bbox_min_y_5: bbox_min_y - y_5 <= 0
bbox_min_y_6: bbox_min_y - y_6 <= 0
bbox_min_y_7: bbox_min_y - y_7 <= 0
bbox_min_y_8: bbox_min_y - y_8 <= 0
bbox_min_y_9: bbox_min_y - y_9 <= 0
bbox_w_def: - bbox_max_x + bbox_min_x + bbox_w = 0
boundary_x_0: w_0 + x_0 <= 150
boundary_x_1: w_1 + x_1 <= 150
boundary_x_2: w_2 + x_2 <= 150
boundary_x_3: w_3 + x_3 <= 150
boundary_x_4: w_4 + x_4 <= 150
boundary_x_5: w_5 + x_5 <= 150
boundary_x_6: w_6 + x_6 <= 150
boundary_x_7: w_7 + x_7 <= 150
boundary_x_8: w_8 + x_8 <= 150
boundary_x_9: w_9 + x_9 <= 150
boundary_y_0: h_0 + y_0 <= 150
boundary_y_1: h_1 + y_1 <= 150
boundary_y_2: h_2 + y_2 <= 150
boundary_y_3: h_3 + y_3 <= 150
boundary_y_4: h_4 + y_4 <= 150
boundary_y_5: h_5 + y_5 <= 150
boundary_y_6: h_6 + y_6 <= 150
boundary_y_7: h_7 + y_7 <= 150
boundary_y_8: h_8 + y_8 <= 150
boundary_y_9: h_9 + y_9 <= 150
cx_def_0: cx_0 - 0.5 w_0 - x_0 = 0
cx_def_1: cx_1 - 0.5 w_1 - x_1 = 0
cx_def_2: cx_2 - 0.5 w_2 - x_2 = 0
cx_def_3: cx_3 - 0.5 w_3 - x_3 = 0
cx_def_4: cx_4 - 0.5 w_4 - x_4 = 0
cx_def_5: cx_5 - 0.5 w_5 - x_5 = 0
cx_def_6: cx_6 - 0.5 w_6 - x_6 = 0
cx_def_7: cx_7 - 0.5 w_7 - x_7 = 0
cx_def_8: cx_8 - 0.5 w_8 - x_8 = 0
cx_def_9: cx_9 - 0.5 w_9 - x_9 = 0
cy_def_0: cy_0 - 0.5 h_0 - y_0 = 0
cy_def_1: cy_1 - 0.5 h_1 - y_1 = 0
cy_def_2: cy_2 - 0.5 h_2 - y_2 = 0
cy_def_3: cy_3 - 0.5 h_3 - y_3 = 0
cy_def_4: cy_4 - 0.5 h_4 - y_4 = 0
cy_def_5: cy_5 - 0.5 h_5 - y_5 = 0
cy_def_6: cy_6 - 0.5 h_6 - y_6 = 0
cy_def_7: cy_7 - 0.5 h_7 - y_7 = 0
cy_def_8: cy_8 - 0.5 h_8 - y_8 = 0
cy_def_9: cy_9 - 0.5 h_9 - y_9 = 0
dx_abs_0_1_abs_ge_neg_orig: dx_abs_0_1 + 150 dx_abs_0_1_is_positive
 + dx_diff_0_1 >= 0
dx_abs_0_1_abs_ge_orig: dx_abs_0_1 - 150 dx_abs_0_1_is_positive - dx_diff_0_1
 >= -150
dx_abs_0_1_abs_le_neg_orig: dx_abs_0_1 - 150 dx_abs_0_1_is_positive
 + dx_diff_0_1 <= 0
dx_abs_0_1_abs_le_orig: dx_abs_0_1 + 150 dx_abs_0_1_is_positive - dx_diff_0_1
 <= 150
dx_abs_0_1_force_negative: - 150 dx_abs_0_1_is_positive + dx_diff_0_1
 <= -0.001
dx_abs_0_1_force_positive: - 150 dx_abs_0_1_is_positive + dx_diff_0_1 >= -150
dx_abs_0_2_abs_ge_neg_orig: dx_abs_0_2 + 150 dx_abs_0_2_is_positive
 + dx_diff_0_2 >= 0
dx_abs_0_2_abs_ge_orig: dx_abs_0_2 - 150 dx_abs_0_2_is_positive - dx_diff_0_2
 >= -150
dx_abs_0_2_abs_le_neg_orig: dx_abs_0_2 - 150 dx_abs_0_2_is_positive
 + dx_diff_0_2 <= 0
dx_abs_0_2_abs_le_orig: dx_abs_0_2 + 150 dx_abs_0_2_is_positive - dx_diff_0_2
 <= 150
dx_abs_0_2_force_negative: - 150 dx_abs_0_2_is_positive + dx_diff_0_2
 <= -0.001
dx_abs_0_2_force_positive: - 150 dx_abs_0_2_is_positive + dx_diff_0_2 >= -150
dx_abs_0_4_abs_ge_neg_orig: dx_abs_0_4 + 150 dx_abs_0_4_is_positive
 + dx_diff_0_4 >= 0
dx_abs_0_4_abs_ge_orig: dx_abs_0_4 - 150 dx_abs_0_4_is_positive - dx_diff_0_4
 >= -150
dx_abs_0_4_abs_le_neg_orig: dx_abs_0_4 - 150 dx_abs_0_4_is_positive
 + dx_diff_0_4 <= 0
dx_abs_0_4_abs_le_orig: dx_abs_0_4 + 150 dx_abs_0_4_is_positive - dx_diff_0_4
 <= 150
dx_abs_0_4_force_negative: - 150 dx_abs_0_4_is_positive + dx_diff_0_4
 <= -0.001
dx_abs_0_4_force_positive: - 150 dx_abs_0_4_is_positive + dx_diff_0_4 >= -150
dx_abs_0_5_abs_ge_neg_orig: dx_abs_0_5 + 150 dx_abs_0_5_is_positive
 + dx_diff_0_5 >= 0
dx_abs_0_5_abs_ge_orig: dx_abs_0_5 - 150 dx_abs_0_5_is_positive - dx_diff_0_5
 >= -150
dx_abs_0_5_abs_le_neg_orig: dx_abs_0_5 - 150 dx_abs_0_5_is_positive
 + dx_diff_0_5 <= 0
dx_abs_0_5_abs_le_orig: dx_abs_0_5 + 150 dx_abs_0_5_is_positive - dx_diff_0_5
 <= 150
dx_abs_0_5_force_negative: - 150 dx_abs_0_5_is_positive + dx_diff_0_5
 <= -0.001
dx_abs_0_5_force_positive: - 150 dx_abs_0_5_is_positive + dx_diff_0_5 >= -150
dx_abs_1_3_abs_ge_neg_orig: dx_abs_1_3 + 150 dx_abs_1_3_is_positive
 + dx_diff_1_3 >= 0
dx_abs_1_3_abs_ge_orig: dx_abs_1_3 - 150 dx_abs_1_3_is_positive - dx_diff_1_3
 >= -150
dx_abs_1_3_abs_le_neg_orig: dx_abs_1_3 - 150 dx_abs_1_3_is_positive
 + dx_diff_1_3 <= 0
dx_abs_1_3_abs_le_orig: dx_abs_1_3 + 150 dx_abs_1_3_is_positive - dx_diff_1_3
 <= 150
dx_abs_1_3_force_negative: - 150 dx_abs_1_3_is_positive + dx_diff_1_3
 <= -0.001
dx_abs_1_3_force_positive: - 150 dx_abs_1_3_is_positive + dx_diff_1_3 >= -150
dx_abs_2_3_abs_ge_neg_orig: dx_abs_2_3 + 150 dx_abs_2_3_is_positive
 + dx_diff_2_3 >= 0
dx_abs_2_3_abs_ge_orig: dx_abs_2_3 - 150 dx_abs_2_3_is_positive - dx_diff_2_3
 >= -150
dx_abs_2_3_abs_le_neg_orig: dx_abs_2_3 - 150 dx_abs_2_3_is_positive
 + dx_diff_2_3 <= 0
dx_abs_2_3_abs_le_orig: dx_abs_2_3 + 150 dx_abs_2_3_is_positive - dx_diff_2_3
 <= 150
dx_abs_2_3_force_negative: - 150 dx_abs_2_3_is_positive + dx_diff_2_3
 <= -0.001
dx_abs_2_3_force_positive: - 150 dx_abs_2_3_is_positive + dx_diff_2_3 >= -150
dx_abs_2_6_abs_ge_neg_orig: dx_abs_2_6 + 150 dx_abs_2_6_is_positive
 + dx_diff_2_6 >= 0
dx_abs_2_6_abs_ge_orig: dx_abs_2_6 - 150 dx_abs_2_6_is_positive - dx_diff_2_6
 >= -150
dx_abs_2_6_abs_le_neg_orig: dx_abs_2_6 - 150 dx_abs_2_6_is_positive
 + dx_diff_2_6 <= 0
dx_abs_2_6_abs_le_orig: dx_abs_2_6 + 150 dx_abs_2_6_is_positive - dx_diff_2_6
 <= 150
dx_abs_2_6_force_negative: - 150 dx_abs_2_6_is_positive + dx_diff_2_6
 <= -0.001
dx_abs_2_6_force_positive: - 150 dx_abs_2_6_is_positive + dx_diff_2_6 >= -150
dx_abs_2_7_abs_ge_neg_orig: dx_abs_2_7 + 150 dx_abs_2_7_is_positive
 + dx_diff_2_7 >= 0
dx_abs_2_7_abs_ge_orig: dx_abs_2_7 - 150 dx_abs_2_7_is_positive - dx_diff_2_7
 >= -150
dx_abs_2_7_abs_le_neg_orig: dx_abs_2_7 - 150 dx_abs_2_7_is_positive
 + dx_diff_2_7 <= 0
dx_abs_2_7_abs_le_orig: dx_abs_2_7 + 150 dx_abs_2_7_is_positive - dx_diff_2_7
 <= 150
dx_abs_2_7_force_negative: - 150 dx_abs_2_7_is_positive + dx_diff_2_7
 <= -0.001
dx_abs_2_7_force_positive: - 150 dx_abs_2_7_is_positive + dx_diff_2_7 >= -150
dx_abs_3_8_abs_ge_neg_orig: dx_abs_3_8 + 150 dx_abs_3_8_is_positive
 + dx_diff_3_8 >= 0
dx_abs_3_8_abs_ge_orig: dx_abs_3_8 - 150 dx_abs_3_8_is_positive - dx_diff_3_8
 >= -150
dx_abs_3_8_abs_le_neg_orig: dx_abs_3_8 - 150 dx_abs_3_8_is_positive
 + dx_diff_3_8 <= 0
dx_abs_3_8_abs_le_orig: dx_abs_3_8 + 150 dx_abs_3_8_is_positive - dx_diff_3_8
 <= 150
dx_abs_3_8_force_negative: - 150 dx_abs_3_8_is_positive + dx_diff_3_8
 <= -0.001
dx_abs_3_8_force_positive: - 150 dx_abs_3_8_is_positive + dx_diff_3_8 >= -150
dx_abs_3_9_abs_ge_neg_orig: dx_abs_3_9 + 150 dx_abs_3_9_is_positive
 + dx_diff_3_9 >= 0
dx_abs_3_9_abs_ge_orig: dx_abs_3_9 - 150 dx_abs_3_9_is_positive - dx_diff_3_9
 >= -150
dx_abs_3_9_abs_le_neg_orig: dx_abs_3_9 - 150 dx_abs_3_9_is_positive
 + dx_diff_3_9 <= 0
dx_abs_3_9_abs_le_orig: dx_abs_3_9 + 150 dx_abs_3_9_is_positive - dx_diff_3_9
 <= 150
dx_abs_3_9_force_negative: - 150 dx_abs_3_9_is_positive + dx_diff_3_9
 <= -0.001
dx_abs_3_9_force_positive: - 150 dx_abs_3_9_is_positive + dx_diff_3_9 >= -150
dx_diff_def_0_1: - cx_0 + cx_1 + dx_diff_0_1 = 0
dx_diff_def_0_2: - cx_0 + cx_2 + dx_diff_0_2 = 0
dx_diff_def_0_4: - cx_0 + cx_4 + dx_diff_0_4 = 0
dx_diff_def_0_5: - cx_0 + cx_5 + dx_diff_0_5 = 0
dx_diff_def_1_3: - cx_1 + cx_3 + dx_diff_1_3 = 0
dx_diff_def_2_3: - cx_2 + cx_3 + dx_diff_2_3 = 0
dx_diff_def_2_6: - cx_2 + cx_6 + dx_diff_2_6 = 0
dx_diff_def_2_7: - cx_2 + cx_7 + dx_diff_2_7 = 0
dx_diff_def_3_8: - cx_3 + cx_8 + dx_diff_3_8 = 0
dx_diff_def_3_9: - cx_3 + cx_9 + dx_diff_3_9 = 0
dy_abs_0_1_abs_ge_neg_orig: dy_abs_0_1 + 150 dy_abs_0_1_is_positive
 + dy_diff_0_1 >= 0
dy_abs_0_1_abs_ge_orig: dy_abs_0_1 - 150 dy_abs_0_1_is_positive - dy_diff_0_1
 >= -150
dy_abs_0_1_abs_le_neg_orig: dy_abs_0_1 - 150 dy_abs_0_1_is_positive
 + dy_diff_0_1 <= 0
dy_abs_0_1_abs_le_orig: dy_abs_0_1 + 150 dy_abs_0_1_is_positive - dy_diff_0_1
 <= 150
dy_abs_0_1_force_negative: - 150 dy_abs_0_1_is_positive + dy_diff_0_1
 <= -0.001
dy_abs_0_1_force_positive: - 150 dy_abs_0_1_is_positive + dy_diff_0_1 >= -150
dy_abs_0_2_abs_ge_neg_orig: dy_abs_0_2 + 150 dy_abs_0_2_is_positive
 + dy_diff_0_2 >= 0
dy_abs_0_2_abs_ge_orig: dy_abs_0_2 - 150 dy_abs_0_2_is_positive - dy_diff_0_2
 >= -150
dy_abs_0_2_abs_le_neg_orig: dy_abs_0_2 - 150 dy_abs_0_2_is_positive
 + dy_diff_0_2 <= 0
dy_abs_0_2_abs_le_orig: dy_abs_0_2 + 150 dy_abs_0_2_is_positive - dy_diff_0_2
 <= 150
dy_abs_0_2_force_negative: - 150 dy_abs_0_2_is_positive + dy_diff_0_2
 <= -0.001
dy_abs_0_2_force_positive: - 150 dy_abs_0_2_is_positive + dy_diff_0_2 >= -150
dy_abs_0_4_abs_ge_neg_orig: dy_abs_0_4 + 150 dy_abs_0_4_is_positive
 + dy_diff_0_4 >= 0
dy_abs_0_4_abs_ge_orig: dy_abs_0_4 - 150 dy_abs_0_4_is_positive - dy_diff_0_4
 >= -150
dy_abs_0_4_abs_le_neg_orig: dy_abs_0_4 - 150 dy_abs_0_4_is_positive
 + dy_diff_0_4 <= 0
dy_abs_0_4_abs_le_orig: dy_abs_0_4 + 150 dy_abs_0_4_is_positive - dy_diff_0_4
 <= 150
dy_abs_0_4_force_negative: - 150 dy_abs_0_4_is_positive + dy_diff_0_4
 <= -0.001
dy_abs_0_4_force_positive: - 150 dy_abs_0_4_is_positive + dy_diff_0_4 >= -150
dy_abs_0_5_abs_ge_neg_orig: dy_abs_0_5 + 150 dy_abs_0_5_is_positive
 + dy_diff_0_5 >= 0
dy_abs_0_5_abs_ge_orig: dy_abs_0_5 - 150 dy_abs_0_5_is_positive - dy_diff_0_5
 >= -150
dy_abs_0_5_abs_le_neg_orig: dy_abs_0_5 - 150 dy_abs_0_5_is_positive
 + dy_diff_0_5 <= 0
dy_abs_0_5_abs_le_orig: dy_abs_0_5 + 150 dy_abs_0_5_is_positive - dy_diff_0_5
 <= 150
dy_abs_0_5_force_negative: - 150 dy_abs_0_5_is_positive + dy_diff_0_5
 <= -0.001
dy_abs_0_5_force_positive: - 150 dy_abs_0_5_is_positive + dy_diff_0_5 >= -150
dy_abs_1_3_abs_ge_neg_orig: dy_abs_1_3 + 150 dy_abs_1_3_is_positive
 + dy_diff_1_3 >= 0
dy_abs_1_3_abs_ge_orig: dy_abs_1_3 - 150 dy_abs_1_3_is_positive - dy_diff_1_3
 >= -150
dy_abs_1_3_abs_le_neg_orig: dy_abs_1_3 - 150 dy_abs_1_3_is_positive
 + dy_diff_1_3 <= 0
dy_abs_1_3_abs_le_orig: dy_abs_1_3 + 150 dy_abs_1_3_is_positive - dy_diff_1_3
 <= 150
dy_abs_1_3_force_negative: - 150 dy_abs_1_3_is_positive + dy_diff_1_3
 <= -0.001
dy_abs_1_3_force_positive: - 150 dy_abs_1_3_is_positive + dy_diff_1_3 >= -150
dy_abs_2_3_abs_ge_neg_orig: dy_abs_2_3 + 150 dy_abs_2_3_is_positive
 + dy_diff_2_3 >= 0
dy_abs_2_3_abs_ge_orig: dy_abs_2_3 - 150 dy_abs_2_3_is_positive - dy_diff_2_3
 >= -150
dy_abs_2_3_abs_le_neg_orig: dy_abs_2_3 - 150 dy_abs_2_3_is_positive
 + dy_diff_2_3 <= 0
dy_abs_2_3_abs_le_orig: dy_abs_2_3 + 150 dy_abs_2_3_is_positive - dy_diff_2_3
 <= 150
dy_abs_2_3_force_negative: - 150 dy_abs_2_3_is_positive + dy_diff_2_3
 <= -0.001
dy_abs_2_3_force_positive: - 150 dy_abs_2_3_is_positive + dy_diff_2_3 >= -150
dy_abs_2_6_abs_ge_neg_orig: dy_abs_2_6 + 150 dy_abs_2_6_is_positive
 + dy_diff_2_6 >= 0
dy_abs_2_6_abs_ge_orig: dy_abs_2_6 - 150 dy_abs_2_6_is_positive - dy_diff_2_6
 >= -150
dy_abs_2_6_abs_le_neg_orig: dy_abs_2_6 - 150 dy_abs_2_6_is_positive
 + dy_diff_2_6 <= 0
dy_abs_2_6_abs_le_orig: dy_abs_2_6 + 150 dy_abs_2_6_is_positive - dy_diff_2_6
 <= 150
dy_abs_2_6_force_negative: - 150 dy_abs_2_6_is_positive + dy_diff_2_6
 <= -0.001
dy_abs_2_6_force_positive: - 150 dy_abs_2_6_is_positive + dy_diff_2_6 >= -150
dy_abs_2_7_abs_ge_neg_orig: dy_abs_2_7 + 150 dy_abs_2_7_is_positive
 + dy_diff_2_7 >= 0
dy_abs_2_7_abs_ge_orig: dy_abs_2_7 - 150 dy_abs_2_7_is_positive - dy_diff_2_7
 >= -150
dy_abs_2_7_abs_le_neg_orig: dy_abs_2_7 - 150 dy_abs_2_7_is_positive
 + dy_diff_2_7 <= 0
dy_abs_2_7_abs_le_orig: dy_abs_2_7 + 150 dy_abs_2_7_is_positive - dy_diff_2_7
 <= 150
dy_abs_2_7_force_negative: - 150 dy_abs_2_7_is_positive + dy_diff_2_7
 <= -0.001
dy_abs_2_7_force_positive: - 150 dy_abs_2_7_is_positive + dy_diff_2_7 >= -150
dy_abs_3_8_abs_ge_neg_orig: dy_abs_3_8 + 150 dy_abs_3_8_is_positive
 + dy_diff_3_8 >= 0
dy_abs_3_8_abs_ge_orig: dy_abs_3_8 - 150 dy_abs_3_8_is_positive - dy_diff_3_8
 >= -150
dy_abs_3_8_abs_le_neg_orig: dy_abs_3_8 - 150 dy_abs_3_8_is_positive
 + dy_diff_3_8 <= 0
dy_abs_3_8_abs_le_orig: dy_abs_3_8 + 150 dy_abs_3_8_is_positive - dy_diff_3_8
 <= 150
dy_abs_3_8_force_negative: - 150 dy_abs_3_8_is_positive + dy_diff_3_8
 <= -0.001
dy_abs_3_8_force_positive: - 150 dy_abs_3_8_is_positive + dy_diff_3_8 >= -150
dy_abs_3_9_abs_ge_neg_orig: dy_abs_3_9 + 150 dy_abs_3_9_is_positive
 + dy_diff_3_9 >= 0
dy_abs_3_9_abs_ge_orig: dy_abs_3_9 - 150 dy_abs_3_9_is_positive - dy_diff_3_9
 >= -150
dy_abs_3_9_abs_le_neg_orig: dy_abs_3_9 - 150 dy_abs_3_9_is_positive
 + dy_diff_3_9 <= 0
dy_abs_3_9_abs_le_orig: dy_abs_3_9 + 150 dy_abs_3_9_is_positive - dy_diff_3_9
 <= 150
dy_abs_3_9_force_negative: - 150 dy_abs_3_9_is_positive + dy_diff_3_9
 <= -0.001
dy_abs_3_9_force_positive: - 150 dy_abs_3_9_is_positive + dy_diff_3_9 >= -150
dy_diff_def_0_1: - cy_0 + cy_1 + dy_diff_0_1 = 0
dy_diff_def_0_2: - cy_0 + cy_2 + dy_diff_0_2 = 0
dy_diff_def_0_4: - cy_0 + cy_4 + dy_diff_0_4 = 0
dy_diff_def_0_5: - cy_0 + cy_5 + dy_diff_0_5 = 0
dy_diff_def_1_3: - cy_1 + cy_3 + dy_diff_1_3 = 0
dy_diff_def_2_3: - cy_2 + cy_3 + dy_diff_2_3 = 0
dy_diff_def_2_6: - cy_2 + cy_6 + dy_diff_2_6 = 0
dy_diff_def_2_7: - cy_2 + cy_7 + dy_diff_2_7 = 0
dy_diff_def_3_8: - cy_3 + cy_8 + dy_diff_3_8 = 0
dy_diff_def_3_9: - cy_3 + cy_9 + dy_diff_3_9 = 0
fix_center_x_0: cx_0 = 75
fix_center_y_0: cy_0 = 75
h_align_grid_0: h_0 = 30
h_align_grid_1: h_1 = 50
h_align_grid_2: h_2 = 50
h_align_grid_3: h_3 = 50
h_align_grid_4: h_4 = 15
h_align_grid_5: h_5 = 15
h_align_grid_6: h_6 = 15
h_align_grid_7: h_7 = 15
h_align_grid_8: h_8 = 15
h_align_grid_9: h_9 = 15
h_max_0: h_0 <= 150
h_max_1: h_1 <= 150
h_max_2: h_2 <= 150
h_max_3: h_3 <= 150
h_max_4: h_4 <= 150
h_max_5: h_5 <= 150
h_max_6: h_6 <= 150
h_max_7: h_7 <= 150
h_max_8: h_8 <= 150
h_max_9: h_9 <= 150
height_rotation_0: h_0 - 20 r_0 = 30
height_rotation_1: h_1 + 20 r_1 = 50
height_rotation_2: h_2 + 20 r_2 = 50
height_rotation_3: h_3 + 20 r_3 = 50
height_rotation_4: h_4 + 5 r_4 = 15
height_rotation_5: h_5 + 5 r_5 = 15
height_rotation_6: h_6 + 5 r_6 = 15
height_rotation_7: h_7 + 5 r_7 = 15
height_rotation_8: h_8 + 5 r_8 = 15
height_rotation_9: h_9 + 5 r_9 = 15
horizontal_direction_0_1: z1L_0_1 + z1R_0_1 - z1_0_1 = 0
horizontal_direction_0_2: z1L_0_2 + z1R_0_2 - z1_0_2 = 0
horizontal_direction_0_4: z1L_0_4 + z1R_0_4 - z1_0_4 = 0
horizontal_direction_0_5: z1L_0_5 + z1R_0_5 - z1_0_5 = 0
horizontal_direction_1_3: z1L_1_3 + z1R_1_3 - z1_1_3 = 0
horizontal_direction_2_3: z1L_2_3 + z1R_2_3 - z1_2_3 = 0
horizontal_direction_2_6: z1L_2_6 + z1R_2_6 - z1_2_6 = 0
horizontal_direction_2_7: z1L_2_7 + z1R_2_7 - z1_2_7 = 0
horizontal_direction_3_8: z1L_3_8 + z1R_3_8 - z1_3_8 = 0
horizontal_direction_3_9: z1L_3_9 + z1R_3_9 - z1_3_9 = 0
horizontal_left_dist_0_1: - w_0 - x_0 + x_1 + 200 z1L_0_1 <= 200.5
horizontal_left_dist_0_2: - w_0 - x_0 + x_2 + 200 z1L_0_2 <= 200.5
horizontal_left_dist_0_4: - w_0 - x_0 + x_4 + 200 z1L_0_4 <= 200.5
horizontal_left_dist_0_5: - w_0 - x_0 + x_5 + 200 z1L_0_5 <= 200.5
horizontal_left_dist_1_3: - w_1 - x_1 + x_3 + 200 z1L_1_3 <= 200.5
horizontal_left_dist_2_3: - w_2 - x_2 + x_3 + 200 z1L_2_3 <= 200.5
horizontal_left_dist_2_6: - w_2 - x_2 + x_6 + 200 z1L_2_6 <= 200.5
horizontal_left_dist_2_7: - w_2 - x_2 + x_7 + 200 z1L_2_7 <= 200.5
horizontal_left_dist_3_8: - w_3 - x_3 + x_8 + 200 z1L_3_8 <= 200.5
horizontal_left_dist_3_9: - w_3 - x_3 + x_9 + 200 z1L_3_9 <= 200.5
horizontal_left_dist_lb_0_1: - w_0 - x_0 + x_1 - 200 z1L_0_1 >= -200
horizontal_left_dist_lb_0_2: - w_0 - x_0 + x_2 - 200 z1L_0_2 >= -200
horizontal_left_dist_lb_0_4: - w_0 - x_0 + x_4 - 200 z1L_0_4 >= -200
horizontal_left_dist_lb_0_5: - w_0 - x_0 + x_5 - 200 z1L_0_5 >= -200
horizontal_left_dist_lb_1_3: - w_1 - x_1 + x_3 - 200 z1L_1_3 >= -200
horizontal_left_dist_lb_2_3: - w_2 - x_2 + x_3 - 200 z1L_2_3 >= -200
horizontal_left_dist_lb_2_6: - w_2 - x_2 + x_6 - 200 z1L_2_6 >= -200
horizontal_left_dist_lb_2_7: - w_2 - x_2 + x_7 - 200 z1L_2_7 >= -200
horizontal_left_dist_lb_3_8: - w_3 - x_3 + x_8 - 200 z1L_3_8 >= -200
horizontal_left_dist_lb_3_9: - w_3 - x_3 + x_9 - 200 z1L_3_9 >= -200
horizontal_overlap_y1_0_1: - h_1 + y_0 - y_1 + 200 z1_0_1 <= 200
horizontal_overlap_y1_0_2: - h_2 + y_0 - y_2 + 200 z1_0_2 <= 200
horizontal_overlap_y1_0_4: - h_4 + y_0 - y_4 + 200 z1_0_4 <= 200
horizontal_overlap_y1_0_5: - h_5 + y_0 - y_5 + 200 z1_0_5 <= 200
horizontal_overlap_y1_1_3: - h_3 + y_1 - y_3 + 200 z1_1_3 <= 200
horizontal_overlap_y1_2_3: - h_3 + y_2 - y_3 + 200 z1_2_3 <= 200
horizontal_overlap_y1_2_6: - h_6 + y_2 - y_6 + 200 z1_2_6 <= 200
horizontal_overlap_y1_2_7: - h_7 + y_2 - y_7 + 200 z1_2_7 <= 200
horizontal_overlap_y1_3_8: - h_8 + y_3 - y_8 + 200 z1_3_8 <= 200
horizontal_overlap_y1_3_9: - h_9 + y_3 - y_9 + 200 z1_3_9 <= 200
horizontal_overlap_y2_0_1: - h_0 - y_0 + y_1 + 200 z1_0_1 <= 200
horizontal_overlap_y2_0_2: - h_0 - y_0 + y_2 + 200 z1_0_2 <= 200
horizontal_overlap_y2_0_4: - h_0 - y_0 + y_4 + 200 z1_0_4 <= 200
horizontal_overlap_y2_0_5: - h_0 - y_0 + y_5 + 200 z1_0_5 <= 200
horizontal_overlap_y2_1_3: - h_1 - y_1 + y_3 + 200 z1_1_3 <= 200
horizontal_overlap_y2_2_3: - h_2 - y_2 + y_3 + 200 z1_2_3 <= 200
horizontal_overlap_y2_2_6: - h_2 - y_2 + y_6 + 200 z1_2_6 <= 200
horizontal_overlap_y2_2_7: - h_2 - y_2 + y_7 + 200 z1_2_7 <= 200
horizontal_overlap_y2_3_8: - h_3 - y_3 + y_8 + 200 z1_3_8 <= 200
horizontal_overlap_y2_3_9: - h_3 - y_3 + y_9 + 200 z1_3_9 <= 200
horizontal_right_dist_0_1: - w_1 + x_0 - x_1 + 200 z1R_0_1 <= 200.5
horizontal_right_dist_0_2: - w_2 + x_0 - x_2 + 200 z1R_0_2 <= 200.5
horizontal_right_dist_0_4: - w_4 + x_0 - x_4 + 200 z1R_0_4 <= 200.5
horizontal_right_dist_0_5: - w_5 + x_0 - x_5 + 200 z1R_0_5 <= 200.5
horizontal_right_dist_1_3: - w_3 + x_1 - x_3 + 200 z1R_1_3 <= 200.5
horizontal_right_dist_2_3: - w_3 + x_2 - x_3 + 200 z1R_2_3 <= 200.5
horizontal_right_dist_2_6: - w_6 + x_2 - x_6 + 200 z1R_2_6 <= 200.5
horizontal_right_dist_2_7: - w_7 + x_2 - x_7 + 200 z1R_2_7 <= 200.5
horizontal_right_dist_3_8: - w_8 + x_3 - x_8 + 200 z1R_3_8 <= 200.5
horizontal_right_dist_3_9: - w_9 + x_3 - x_9 + 200 z1R_3_9 <= 200.5
horizontal_right_dist_lb_0_1: - w_1 + x_0 - x_1 - 200 z1R_0_1 >= -200
horizontal_right_dist_lb_0_2: - w_2 + x_0 - x_2 - 200 z1R_0_2 >= -200
horizontal_right_dist_lb_0_4: - w_4 + x_0 - x_4 - 200 z1R_0_4 >= -200
horizontal_right_dist_lb_0_5: - w_5 + x_0 - x_5 - 200 z1R_0_5 >= -200
horizontal_right_dist_lb_1_3: - w_3 + x_1 - x_3 - 200 z1R_1_3 >= -200
horizontal_right_dist_lb_2_3: - w_3 + x_2 - x_3 - 200 z1R_2_3 >= -200
horizontal_right_dist_lb_2_6: - w_6 + x_2 - x_6 - 200 z1R_2_6 >= -200
horizontal_right_dist_lb_2_7: - w_7 + x_2 - x_7 - 200 z1R_2_7 >= -200
horizontal_right_dist_lb_3_8: - w_8 + x_3 - x_8 - 200 z1R_3_8 >= -200
horizontal_right_dist_lb_3_9: - w_9 + x_3 - x_9 - 200 z1R_3_9 >= -200
must_adjacent_0_1: z1_0_1 + z2_0_1 = 1
must_adjacent_0_2: z1_0_2 + z2_0_2 = 1
must_adjacent_0_4: z1_0_4 + z2_0_4 = 1
must_adjacent_0_5: z1_0_5 + z2_0_5 = 1
must_adjacent_1_3: z1_1_3 + z2_1_3 = 1
must_adjacent_2_3: z1_2_3 + z2_2_3 = 1
must_adjacent_2_6: z1_2_6 + z2_2_6 = 1
must_adjacent_2_7: z1_2_7 + z2_2_7 = 1
must_adjacent_3_8: z1_3_8 + z2_3_8 = 1
must_adjacent_3_9: z1_3_9 + z2_3_9 = 1
non_overlap_any_0_1: p_down_0_1 + p_left_0_1 + p_right_0_1 + p_up_0_1 = 1
non_overlap_any_0_2: p_down_0_2 + p_left_0_2 + p_right_0_2 + p_up_0_2 = 1
non_overlap_any_0_3: p_down_0_3 + p_left_0_3 + p_right_0_3 + p_up_0_3 = 1
non_overlap_any_0_4: p_down_0_4 + p_left_0_4 + p_right_0_4 + p_up_0_4 = 1
non_overlap_any_0_5: p_down_0_5 + p_left_0_5 + p_right_0_5 + p_up_0_5 = 1
non_overlap_any_0_6: p_down_0_6 + p_left_0_6 + p_right_0_6 + p_up_0_6 = 1
non_overlap_any_0_7: p_down_0_7 + p_left_0_7 + p_right_0_7 + p_up_0_7 = 1
non_overlap_any_0_8: p_down_0_8 + p_left_0_8 + p_right_0_8 + p_up_0_8 = 1
non_overlap_any_0_9: p_down_0_9 + p_left_0_9 + p_right_0_9 + p_up_0_9 = 1
non_overlap_any_1_2: p_down_1_2 + p_left_1_2 + p_right_1_2 + p_up_1_2 = 1
non_overlap_any_1_3: p_down_1_3 + p_left_1_3 + p_right_1_3 + p_up_1_3 = 1
non_overlap_any_1_4: p_down_1_4 + p_left_1_4 + p_right_1_4 + p_up_1_4 = 1
non_overlap_any_1_5: p_down_1_5 + p_left_1_5 + p_right_1_5 + p_up_1_5 = 1
non_overlap_any_1_6: p_down_1_6 + p_left_1_6 + p_right_1_6 + p_up_1_6 = 1
non_overlap_any_1_7: p_down_1_7 + p_left_1_7 + p_right_1_7 + p_up_1_7 = 1
non_overlap_any_1_8: p_down_1_8 + p_left_1_8 + p_right_1_8 + p_up_1_8 = 1
non_overlap_any_1_9: p_down_1_9 + p_left_1_9 + p_right_1_9 + p_up_1_9 = 1
non_overlap_any_2_3: p_down_2_3 + p_left_2_3 + p_right_2_3 + p_up_2_3 = 1
non_overlap_any_2_4: p_down_2_4 + p_left_2_4 + p_right_2_4 + p_up_2_4 = 1
non_overlap_any_2_5: p_down_2_5 + p_left_2_5 + p_right_2_5 + p_up_2_5 = 1
non_overlap_any_2_6: p_down_2_6 + p_left_2_6 + p_right_2_6 + p_up_2_6 = 1
non_overlap_any_2_7: p_down_2_7 + p_left_2_7 + p_right_2_7 + p_up_2_7 = 1
non_overlap_any_2_8: p_down_2_8 + p_left_2_8 + p_right_2_8 + p_up_2_8 = 1
non_overlap_any_2_9: p_down_2_9 + p_left_2_9 + p_right_2_9 + p_up_2_9 = 1
non_overlap_any_3_4: p_down_3_4 + p_left_3_4 + p_right_3_4 + p_up_3_4 = 1
non_overlap_any_3_5: p_down_3_5 + p_left_3_5 + p_right_3_5 + p_up_3_5 = 1
non_overlap_any_3_6: p_down_3_6 + p_left_3_6 + p_right_3_6 + p_up_3_6 = 1
non_overlap_any_3_7: p_down_3_7 + p_left_3_7 + p_right_3_7 + p_up_3_7 = 1
non_overlap_any_3_8: p_down_3_8 + p_left_3_8 + p_right_3_8 + p_up_3_8 = 1
non_overlap_any_3_9: p_down_3_9 + p_left_3_9 + p_right_3_9 + p_up_3_9 = 1
non_overlap_any_4_5: p_down_4_5 + p_left_4_5 + p_right_4_5 + p_up_4_5 = 1
non_overlap_any_4_6: p_down_4_6 + p_left_4_6 + p_right_4_6 + p_up_4_6 = 1
non_overlap_any_4_7: p_down_4_7 + p_left_4_7 + p_right_4_7 + p_up_4_7 = 1
non_overlap_any_4_8: p_down_4_8 + p_left_4_8 + p_right_4_8 + p_up_4_8 = 1
non_overlap_any_4_9: p_down_4_9 + p_left_4_9 + p_right_4_9 + p_up_4_9 = 1
non_overlap_any_5_6: p_down_5_6 + p_left_5_6 + p_right_5_6 + p_up_5_6 = 1
non_overlap_any_5_7: p_down_5_7 + p_left_5_7 + p_right_5_7 + p_up_5_7 = 1
non_overlap_any_5_8: p_down_5_8 + p_left_5_8 + p_right_5_8 + p_up_5_8 = 1
non_overlap_any_5_9: p_down_5_9 + p_left_5_9 + p_right_5_9 + p_up_5_9 = 1
non_overlap_any_6_7: p_down_6_7 + p_left_6_7 + p_right_6_7 + p_up_6_7 = 1
non_overlap_any_6_8: p_down_6_8 + p_left_6_8 + p_right_6_8 + p_up_6_8 = 1
non_overlap_any_6_9: p_down_6_9 + p_left_6_9 + p_right_6_9 + p_up_6_9 = 1
non_overlap_any_7_8: p_down_7_8 + p_left_7_8 + p_right_7_8 + p_up_7_8 = 1
non_overlap_any_7_9: p_down_7_9 + p_left_7_9 + p_right_7_9 + p_up_7_9 = 1
non_overlap_any_8_9: p_down_8_9 + p_left_8_9 + p_right_8_9 + p_up_8_9 = 1
non_overlap_down_0_1: h_0 + 200 p_down_0_1 + y_0 - y_1 <= 200
non_overlap_down_0_2: h_0 + 200 p_down_0_2 + y_0 - y_2 <= 200
non_overlap_down_0_3: h_0 + 200 p_down_0_3 + y_0 - y_3 <= 200
non_overlap_down_0_4: h_0 + 200 p_down_0_4 + y_0 - y_4 <= 200
non_overlap_down_0_5: h_0 + 200 p_down_0_5 + y_0 - y_5 <= 200
non_overlap_down_0_6: h_0 + 200 p_down_0_6 + y_0 - y_6 <= 200
non_overlap_down_0_7: h_0 + 200 p_down_0_7 + y_0 - y_7 <= 200
non_overlap_down_0_8: h_0 + 200 p_down_0_8 + y_0 - y_8 <= 200
non_overlap_down_0_9: h_0 + 200 p_down_0_9 + y_0 - y_9 <= 200
non_overlap_down_1_2: h_1 + 200 p_down_1_2 + y_1 - y_2 <= 200
non_overlap_down_1_3: h_1 + 200 p_down_1_3 + y_1 - y_3 <= 200
non_overlap_down_1_4: h_1 + 200 p_down_1_4 + y_1 - y_4 <= 200
non_overlap_down_1_5: h_1 + 200 p_down_1_5 + y_1 - y_5 <= 200
non_overlap_down_1_6: h_1 + 200 p_down_1_6 + y_1 - y_6 <= 200
non_overlap_down_1_7: h_1 + 200 p_down_1_7 + y_1 - y_7 <= 200
non_overlap_down_1_8: h_1 + 200 p_down_1_8 + y_1 - y_8 <= 200
non_overlap_down_1_9: h_1 + 200 p_down_1_9 + y_1 - y_9 <= 200
non_overlap_down_2_3: h_2 + 200 p_down_2_3 + y_2 - y_3 <= 200
non_overlap_down_2_4: h_2 + 200 p_down_2_4 + y_2 - y_4 <= 200
non_overlap_down_2_5: h_2 + 200 p_down_2_5 + y_2 - y_5 <= 200
non_overlap_down_2_6: h_2 + 200 p_down_2_6 + y_2 - y_6 <= 200
non_overlap_down_2_7: h_2 + 200 p_down_2_7 + y_2 - y_7 <= 200
non_overlap_down_2_8: h_2 + 200 p_down_2_8 + y_2 - y_8 <= 200
non_overlap_down_2_9: h_2 + 200 p_down_2_9 + y_2 - y_9 <= 200
non_overlap_down_3_4: h_3 + 200 p_down_3_4 + y_3 - y_4 <= 200
non_overlap_down_3_5: h_3 + 200 p_down_3_5 + y_3 - y_5 <= 200
non_overlap_down_3_6: h_3 + 200 p_down_3_6 + y_3 - y_6 <= 200
non_overlap_down_3_7: h_3 + 200 p_down_3_7 + y_3 - y_7 <= 200
non_overlap_down_3_8: h_3 + 200 p_down_3_8 + y_3 - y_8 <= 200
non_overlap_down_3_9: h_3 + 200 p_down_3_9 + y_3 - y_9 <= 200
non_overlap_down_4_5: h_4 + 200 p_down_4_5 + y_4 - y_5 <= 200
non_overlap_down_4_6: h_4 + 200 p_down_4_6 + y_4 - y_6 <= 200
non_overlap_down_4_7: h_4 + 200 p_down_4_7 + y_4 - y_7 <= 200
non_overlap_down_4_8: h_4 + 200 p_down_4_8 + y_4 - y_8 <= 200
non_overlap_down_4_9: h_4 + 200 p_down_4_9 + y_4 - y_9 <= 200
non_overlap_down_5_6: h_5 + 200 p_down_5_6 + y_5 - y_6 <= 200
non_overlap_down_5_7: h_5 + 200 p_down_5_7 + y_5 - y_7 <= 200
non_overlap_down_5_8: h_5 + 200 p_down_5_8 + y_5 - y_8 <= 200
non_overlap_down_5_9: h_5 + 200 p_down_5_9 + y_5 - y_9 <= 200
non_overlap_down_6_7: h_6 + 200 p_down_6_7 + y_6 - y_7 <= 200
non_overlap_down_6_8: h_6 + 200 p_down_6_8 + y_6 - y_8 <= 200
non_overlap_down_6_9: h_6 + 200 p_down_6_9 + y_6 - y_9 <= 200
non_overlap_down_7_8: h_7 + 200 p_down_7_8 + y_7 - y_8 <= 200
non_overlap_down_7_9: h_7 + 200 p_down_7_9 + y_7 - y_9 <= 200
non_overlap_down_8_9: h_8 + 200 p_down_8_9 + y_8 - y_9 <= 200
non_overlap_down_rev_0_1: - h_0 - 200 p_down_0_1 - y_0 + y_1 <= 0
non_overlap_down_rev_0_2: - h_0 - 200 p_down_0_2 - y_0 + y_2 <= 0
non_overlap_down_rev_0_3: - h_0 - 200 p_down_0_3 - y_0 + y_3 <= 0
non_overlap_down_rev_0_4: - h_0 - 200 p_down_0_4 - y_0 + y_4 <= 0
non_overlap_down_rev_0_5: - h_0 - 200 p_down_0_5 - y_0 + y_5 <= 0
non_overlap_down_rev_0_6: - h_0 - 200 p_down_0_6 - y_0 + y_6 <= 0
non_overlap_down_rev_0_7: - h_0 - 200 p_down_0_7 - y_0 + y_7 <= 0
non_overlap_down_rev_0_8: - h_0 - 200 p_down_0_8 - y_0 + y_8 <= 0
non_overlap_down_rev_0_9: - h_0 - 200 p_down_0_9 - y_0 + y_9 <= 0
non_overlap_down_rev_1_2: - h_1 - 200 p_down_1_2 - y_1 + y_2 <= 0
non_overlap_down_rev_1_3: - h_1 - 200 p_down_1_3 - y_1 + y_3 <= 0
non_overlap_down_rev_1_4: - h_1 - 200 p_down_1_4 - y_1 + y_4 <= 0
non_overlap_down_rev_1_5: - h_1 - 200 p_down_1_5 - y_1 + y_5 <= 0
non_overlap_down_rev_1_6: - h_1 - 200 p_down_1_6 - y_1 + y_6 <= 0
non_overlap_down_rev_1_7: - h_1 - 200 p_down_1_7 - y_1 + y_7 <= 0
non_overlap_down_rev_1_8: - h_1 - 200 p_down_1_8 - y_1 + y_8 <= 0
non_overlap_down_rev_1_9: - h_1 - 200 p_down_1_9 - y_1 + y_9 <= 0
non_overlap_down_rev_2_3: - h_2 - 200 p_down_2_3 - y_2 + y_3 <= 0
non_overlap_down_rev_2_4: - h_2 - 200 p_down_2_4 - y_2 + y_4 <= 0
non_overlap_down_rev_2_5: - h_2 - 200 p_down_2_5 - y_2 + y_5 <= 0
non_overlap_down_rev_2_6: - h_2 - 200 p_down_2_6 - y_2 + y_6 <= 0
non_overlap_down_rev_2_7: - h_2 - 200 p_down_2_7 - y_2 + y_7 <= 0
non_overlap_down_rev_2_8: - h_2 - 200 p_down_2_8 - y_2 + y_8 <= 0
non_overlap_down_rev_2_9: - h_2 - 200 p_down_2_9 - y_2 + y_9 <= 0
non_overlap_down_rev_3_4: - h_3 - 200 p_down_3_4 - y_3 + y_4 <= 0
non_overlap_down_rev_3_5: - h_3 - 200 p_down_3_5 - y_3 + y_5 <= 0
non_overlap_down_rev_3_6: - h_3 - 200 p_down_3_6 - y_3 + y_6 <= 0
non_overlap_down_rev_3_7: - h_3 - 200 p_down_3_7 - y_3 + y_7 <= 0
non_overlap_down_rev_3_8: - h_3 - 200 p_down_3_8 - y_3 + y_8 <= 0
non_overlap_down_rev_3_9: - h_3 - 200 p_down_3_9 - y_3 + y_9 <= 0
non_overlap_down_rev_4_5: - h_4 - 200 p_down_4_5 - y_4 + y_5 <= 0
non_overlap_down_rev_4_6: - h_4 - 200 p_down_4_6 - y_4 + y_6 <= 0
non_overlap_down_rev_4_7: - h_4 - 200 p_down_4_7 - y_4 + y_7 <= 0
non_overlap_down_rev_4_8: - h_4 - 200 p_down_4_8 - y_4 + y_8 <= 0
non_overlap_down_rev_4_9: - h_4 - 200 p_down_4_9 - y_4 + y_9 <= 0
non_overlap_down_rev_5_6: - h_5 - 200 p_down_5_6 - y_5 + y_6 <= 0
non_overlap_down_rev_5_7: - h_5 - 200 p_down_5_7 - y_5 + y_7 <= 0
non_overlap_down_rev_5_8: - h_5 - 200 p_down_5_8 - y_5 + y_8 <= 0
non_overlap_down_rev_5_9: - h_5 - 200 p_down_5_9 - y_5 + y_9 <= 0
non_overlap_down_rev_6_7: - h_6 - 200 p_down_6_7 - y_6 + y_7 <= 0
non_overlap_down_rev_6_8: - h_6 - 200 p_down_6_8 - y_6 + y_8 <= 0
non_overlap_down_rev_6_9: - h_6 - 200 p_down_6_9 - y_6 + y_9 <= 0
non_overlap_down_rev_7_8: - h_7 - 200 p_down_7_8 - y_7 + y_8 <= 0
non_overlap_down_rev_7_9: - h_7 - 200 p_down_7_9 - y_7 + y_9 <= 0
non_overlap_down_rev_8_9: - h_8 - 200 p_down_8_9 - y_8 + y_9 <= 0
non_overlap_left_0_1: 200 p_left_0_1 + w_0 + x_0 - x_1 <= 200
non_overlap_left_0_2: 200 p_left_0_2 + w_0 + x_0 - x_2 <= 200
non_overlap_left_0_3: 200 p_left_0_3 + w_0 + x_0 - x_3 <= 200
non_overlap_left_0_4: 200 p_left_0_4 + w_0 + x_0 - x_4 <= 200
non_overlap_left_0_5: 200 p_left_0_5 + w_0 + x_0 - x_5 <= 200
non_overlap_left_0_6: 200 p_left_0_6 + w_0 + x_0 - x_6 <= 200
non_overlap_left_0_7: 200 p_left_0_7 + w_0 + x_0 - x_7 <= 200
non_overlap_left_0_8: 200 p_left_0_8 + w_0 + x_0 - x_8 <= 200
non_overlap_left_0_9: 200 p_left_0_9 + w_0 + x_0 - x_9 <= 200
non_overlap_left_1_2: 200 p_left_1_2 + w_1 + x_1 - x_2 <= 200
non_overlap_left_1_3: 200 p_left_1_3 + w_1 + x_1 - x_3 <= 200
non_overlap_left_1_4: 200 p_left_1_4 + w_1 + x_1 - x_4 <= 200
non_overlap_left_1_5: 200 p_left_1_5 + w_1 + x_1 - x_5 <= 200
non_overlap_left_1_6: 200 p_left_1_6 + w_1 + x_1 - x_6 <= 200
non_overlap_left_1_7: 200 p_left_1_7 + w_1 + x_1 - x_7 <= 200
non_overlap_left_1_8: 200 p_left_1_8 + w_1 + x_1 - x_8 <= 200
non_overlap_left_1_9: 200 p_left_1_9 + w_1 + x_1 - x_9 <= 200
non_overlap_left_2_3: 200 p_left_2_3 + w_2 + x_2 - x_3 <= 200
non_overlap_left_2_4: 200 p_left_2_4 + w_2 + x_2 - x_4 <= 200
non_overlap_left_2_5: 200 p_left_2_5 + w_2 + x_2 - x_5 <= 200
non_overlap_left_2_6: 200 p_left_2_6 + w_2 + x_2 - x_6 <= 200
non_overlap_left_2_7: 200 p_left_2_7 + w_2 + x_2 - x_7 <= 200
non_overlap_left_2_8: 200 p_left_2_8 + w_2 + x_2 - x_8 <= 200
non_overlap_left_2_9: 200 p_left_2_9 + w_2 + x_2 - x_9 <= 200
non_overlap_left_3_4: 200 p_left_3_4 + w_3 + x_3 - x_4 <= 200
non_overlap_left_3_5: 200 p_left_3_5 + w_3 + x_3 - x_5 <= 200
non_overlap_left_3_6: 200 p_left_3_6 + w_3 + x_3 - x_6 <= 200
non_overlap_left_3_7: 200 p_left_3_7 + w_3 + x_3 - x_7 <= 200
non_overlap_left_3_8: 200 p_left_3_8 + w_3 + x_3 - x_8 <= 200
non_overlap_left_3_9: 200 p_left_3_9 + w_3 + x_3 - x_9 <= 200
non_overlap_left_4_5: 200 p_left_4_5 + w_4 + x_4 - x_5 <= 200
non_overlap_left_4_6: 200 p_left_4_6 + w_4 + x_4 - x_6 <= 200
non_overlap_left_4_7: 200 p_left_4_7 + w_4 + x_4 - x_7 <= 200
non_overlap_left_4_8: 200 p_left_4_8 + w_4 + x_4 - x_8 <= 200
non_overlap_left_4_9: 200 p_left_4_9 + w_4 + x_4 - x_9 <= 200
non_overlap_left_5_6: 200 p_left_5_6 + w_5 + x_5 - x_6 <= 200
non_overlap_left_5_7: 200 p_left_5_7 + w_5 + x_5 - x_7 <= 200
non_overlap_left_5_8: 200 p_left_5_8 + w_5 + x_5 - x_8 <= 200
non_overlap_left_5_9: 200 p_left_5_9 + w_5 + x_5 - x_9 <= 200
non_overlap_left_6_7: 200 p_left_6_7 + w_6 + x_6 - x_7 <= 200
non_overlap_left_6_8: 200 p_left_6_8 + w_6 + x_6 - x_8 <= 200
non_overlap_left_6_9: 200 p_left_6_9 + w_6 + x_6 - x_9 <= 200
non_overlap_left_7_8: 200 p_left_7_8 + w_7 + x_7 - x_8 <= 200
non_overlap_left_7_9: 200 p_left_7_9 + w_7 + x_7 - x_9 <= 200
non_overlap_left_8_9: 200 p_left_8_9 + w_8 + x_8 - x_9 <= 200
non_overlap_left_rev_0_1: - 200 p_left_0_1 - w_0 - x_0 + x_1 <= 0
non_overlap_left_rev_0_2: - 200 p_left_0_2 - w_0 - x_0 + x_2 <= 0
non_overlap_left_rev_0_3: - 200 p_left_0_3 - w_0 - x_0 + x_3 <= 0
non_overlap_left_rev_0_4: - 200 p_left_0_4 - w_0 - x_0 + x_4 <= 0
non_overlap_left_rev_0_5: - 200 p_left_0_5 - w_0 - x_0 + x_5 <= 0
non_overlap_left_rev_0_6: - 200 p_left_0_6 - w_0 - x_0 + x_6 <= 0
non_overlap_left_rev_0_7: - 200 p_left_0_7 - w_0 - x_0 + x_7 <= 0
non_overlap_left_rev_0_8: - 200 p_left_0_8 - w_0 - x_0 + x_8 <= 0
non_overlap_left_rev_0_9: - 200 p_left_0_9 - w_0 - x_0 + x_9 <= 0
non_overlap_left_rev_1_2: - 200 p_left_1_2 - w_1 - x_1 + x_2 <= 0
non_overlap_left_rev_1_3: - 200 p_left_1_3 - w_1 - x_1 + x_3 <= 0
non_overlap_left_rev_1_4: - 200 p_left_1_4 - w_1 - x_1 + x_4 <= 0
non_overlap_left_rev_1_5: - 200 p_left_1_5 - w_1 - x_1 + x_5 <= 0
non_overlap_left_rev_1_6: - 200 p_left_1_6 - w_1 - x_1 + x_6 <= 0
non_overlap_left_rev_1_7: - 200 p_left_1_7 - w_1 - x_1 + x_7 <= 0
non_overlap_left_rev_1_8: - 200 p_left_1_8 - w_1 - x_1 + x_8 <= 0
non_overlap_left_rev_1_9: - 200 p_left_1_9 - w_1 - x_1 + x_9 <= 0
non_overlap_left_rev_2_3: - 200 p_left_2_3 - w_2 - x_2 + x_3 <= 0
non_overlap_left_rev_2_4: - 200 p_left_2_4 - w_2 - x_2 + x_4 <= 0
non_overlap_left_rev_2_5: - 200 p_left_2_5 - w_2 - x_2 + x_5 <= 0
non_overlap_left_rev_2_6: - 200 p_left_2_6 - w_2 - x_2 + x_6 <= 0
non_overlap_left_rev_2_7: - 200 p_left_2_7 - w_2 - x_2 + x_7 <= 0
non_overlap_left_rev_2_8: - 200 p_left_2_8 - w_2 - x_2 + x_8 <= 0
non_overlap_left_rev_2_9: - 200 p_left_2_9 - w_2 - x_2 + x_9 <= 0
non_overlap_left_rev_3_4: - 200 p_left_3_4 - w_3 - x_3 + x_4 <= 0
non_overlap_left_rev_3_5: - 200 p_left_3_5 - w_3 - x_3 + x_5 <= 0
non_overlap_left_rev_3_6: - 200 p_left_3_6 - w_3 - x_3 + x_6 <= 0
non_overlap_left_rev_3_7: - 200 p_left_3_7 - w_3 - x_3 + x_7 <= 0
non_overlap_left_rev_3_8: - 200 p_left_3_8 - w_3 - x_3 + x_8 <= 0
non_overlap_left_rev_3_9: - 200 p_left_3_9 - w_3 - x_3 + x_9 <= 0
non_overlap_left_rev_4_5: - 200 p_left_4_5 - w_4 - x_4 + x_5 <= 0
non_overlap_left_rev_4_6: - 200 p_left_4_6 - w_4 - x_4 + x_6 <= 0
non_overlap_left_rev_4_7: - 200 p_left_4_7 - w_4 - x_4 + x_7 <= 0
non_overlap_left_rev_4_8: - 200 p_left_4_8 - w_4 - x_4 + x_8 <= 0
non_overlap_left_rev_4_9: - 200 p_left_4_9 - w_4 - x_4 + x_9 <= 0
non_overlap_left_rev_5_6: - 200 p_left_5_6 - w_5 - x_5 + x_6 <= 0
non_overlap_left_rev_5_7: - 200 p_left_5_7 - w_5 - x_5 + x_7 <= 0
non_overlap_left_rev_5_8: - 200 p_left_5_8 - w_5 - x_5 + x_8 <= 0
non_overlap_left_rev_5_9: - 200 p_left_5_9 - w_5 - x_5 + x_9 <= 0
non_overlap_left_rev_6_7: - 200 p_left_6_7 - w_6 - x_6 + x_7 <= 0
non_overlap_left_rev_6_8: - 200 p_left_6_8 - w_6 - x_6 + x_8 <= 0
non_overlap_left_rev_6_9: - 200 p_left_6_9 - w_6 - x_6 + x_9 <= 0
non_overlap_left_rev_7_8: - 200 p_left_7_8 - w_7 - x_7 + x_8 <= 0
non_overlap_left_rev_7_9: - 200 p_left_7_9 - w_7 - x_7 + x_9 <= 0
non_overlap_left_rev_8_9: - 200 p_left_8_9 - w_8 - x_8 + x_9 <= 0
non_overlap_right_0_1: 200 p_right_0_1 + w_1 - x_0 + x_1 <= 200
non_overlap_right_0_2: 200 p_right_0_2 + w_2 - x_0 + x_2 <= 200
non_overlap_right_0_3: 200 p_right_0_3 + w_3 - x_0 + x_3 <= 200
non_overlap_right_0_4: 200 p_right_0_4 + w_4 - x_0 + x_4 <= 200
non_overlap_right_0_5: 200 p_right_0_5 + w_5 - x_0 + x_5 <= 200
non_overlap_right_0_6: 200 p_right_0_6 + w_6 - x_0 + x_6 <= 200
non_overlap_right_0_7: 200 p_right_0_7 + w_7 - x_0 + x_7 <= 200
non_overlap_right_0_8: 200 p_right_0_8 + w_8 - x_0 + x_8 <= 200
non_overlap_right_0_9: 200 p_right_0_9 + w_9 - x_0 + x_9 <= 200
non_overlap_right_1_2: 200 p_right_1_2 + w_2 - x_1 + x_2 <= 200
non_overlap_right_1_3: 200 p_right_1_3 + w_3 - x_1 + x_3 <= 200
non_overlap_right_1_4: 200 p_right_1_4 + w_4 - x_1 + x_4 <= 200
non_overlap_right_1_5: 200 p_right_1_5 + w_5 - x_1 + x_5 <= 200
non_overlap_right_1_6: 200 p_right_1_6 + w_6 - x_1 + x_6 <= 200
non_overlap_right_1_7: 200 p_right_1_7 + w_7 - x_1 + x_7 <= 200
non_overlap_right_1_8: 200 p_right_1_8 + w_8 - x_1 + x_8 <= 200
non_overlap_right_1_9: 200 p_right_1_9 + w_9 - x_1 + x_9 <= 200
non_overlap_right_2_3: 200 p_right_2_3 + w_3 - x_2 + x_3 <= 200
non_overlap_right_2_4: 200 p_right_2_4 + w_4 - x_2 + x_4 <= 200
non_overlap_right_2_5: 200 p_right_2_5 + w_5 - x_2 + x_5 <= 200
non_overlap_right_2_6: 200 p_right_2_6 + w_6 - x_2 + x_6 <= 200
non_overlap_right_2_7: 200 p_right_2_7 + w_7 - x_2 + x_7 <= 200
non_overlap_right_2_8: 200 p_right_2_8 + w_8 - x_2 + x_8 <= 200
non_overlap_right_2_9: 200 p_right_2_9 + w_9 - x_2 + x_9 <= 200
non_overlap_right_3_4: 200 p_right_3_4 + w_4 - x_3 + x_4 <= 200
non_overlap_right_3_5: 200 p_right_3_5 + w_5 - x_3 + x_5 <= 200
non_overlap_right_3_6: 200 p_right_3_6 + w_6 - x_3 + x_6 <= 200
non_overlap_right_3_7: 200 p_right_3_7 + w_7 - x_3 + x_7 <= 200
non_overlap_right_3_8: 200 p_right_3_8 + w_8 - x_3 + x_8 <= 200
non_overlap_right_3_9: 200 p_right_3_9 + w_9 - x_3 + x_9 <= 200
non_overlap_right_4_5: 200 p_right_4_5 + w_5 - x_4 + x_5 <= 200
non_overlap_right_4_6: 200 p_right_4_6 + w_6 - x_4 + x_6 <= 200
non_overlap_right_4_7: 200 p_right_4_7 + w_7 - x_4 + x_7 <= 200
non_overlap_right_4_8: 200 p_right_4_8 + w_8 - x_4 + x_8 <= 200
non_overlap_right_4_9: 200 p_right_4_9 + w_9 - x_4 + x_9 <= 200
non_overlap_right_5_6: 200 p_right_5_6 + w_6 - x_5 + x_6 <= 200
non_overlap_right_5_7: 200 p_right_5_7 + w_7 - x_5 + x_7 <= 200
non_overlap_right_5_8: 200 p_right_5_8 + w_8 - x_5 + x_8 <= 200
non_overlap_right_5_9: 200 p_right_5_9 + w_9 - x_5 + x_9 <= 200
non_overlap_right_6_7: 200 p_right_6_7 + w_7 - x_6 + x_7 <= 200
non_overlap_right_6_8: 200 p_right_6_8 + w_8 - x_6 + x_8 <= 200
non_overlap_right_6_9: 200 p_right_6_9 + w_9 - x_6 + x_9 <= 200
non_overlap_right_7_8: 200 p_right_7_8 + w_8 - x_7 + x_8 <= 200
non_overlap_right_7_9: 200 p_right_7_9 + w_9 - x_7 + x_9 <= 200
non_overlap_right_8_9: 200 p_right_8_9 + w_9 - x_8 + x_9 <= 200
non_overlap_right_rev_0_1: - 200 p_right_0_1 - w_1 + x_0 - x_1 <= 0
non_overlap_right_rev_0_2: - 200 p_right_0_2 - w_2 + x_0 - x_2 <= 0
non_overlap_right_rev_0_3: - 200 p_right_0_3 - w_3 + x_0 - x_3 <= 0
non_overlap_right_rev_0_4: - 200 p_right_0_4 - w_4 + x_0 - x_4 <= 0
non_overlap_right_rev_0_5: - 200 p_right_0_5 - w_5 + x_0 - x_5 <= 0
non_overlap_right_rev_0_6: - 200 p_right_0_6 - w_6 + x_0 - x_6 <= 0
non_overlap_right_rev_0_7: - 200 p_right_0_7 - w_7 + x_0 - x_7 <= 0
non_overlap_right_rev_0_8: - 200 p_right_0_8 - w_8 + x_0 - x_8 <= 0
non_overlap_right_rev_0_9: - 200 p_right_0_9 - w_9 + x_0 - x_9 <= 0
non_overlap_right_rev_1_2: - 200 p_right_1_2 - w_2 + x_1 - x_2 <= 0
non_overlap_right_rev_1_3: - 200 p_right_1_3 - w_3 + x_1 - x_3 <= 0
non_overlap_right_rev_1_4: - 200 p_right_1_4 - w_4 + x_1 - x_4 <= 0
non_overlap_right_rev_1_5: - 200 p_right_1_5 - w_5 + x_1 - x_5 <= 0
non_overlap_right_rev_1_6: - 200 p_right_1_6 - w_6 + x_1 - x_6 <= 0
non_overlap_right_rev_1_7: - 200 p_right_1_7 - w_7 + x_1 - x_7 <= 0
non_overlap_right_rev_1_8: - 200 p_right_1_8 - w_8 + x_1 - x_8 <= 0
non_overlap_right_rev_1_9: - 200 p_right_1_9 - w_9 + x_1 - x_9 <= 0
non_overlap_right_rev_2_3: - 200 p_right_2_3 - w_3 + x_2 - x_3 <= 0
non_overlap_right_rev_2_4: - 200 p_right_2_4 - w_4 + x_2 - x_4 <= 0
non_overlap_right_rev_2_5: - 200 p_right_2_5 - w_5 + x_2 - x_5 <= 0
non_overlap_right_rev_2_6: - 200 p_right_2_6 - w_6 + x_2 - x_6 <= 0
non_overlap_right_rev_2_7: - 200 p_right_2_7 - w_7 + x_2 - x_7 <= 0
non_overlap_right_rev_2_8: - 200 p_right_2_8 - w_8 + x_2 - x_8 <= 0
non_overlap_right_rev_2_9: - 200 p_right_2_9 - w_9 + x_2 - x_9 <= 0
non_overlap_right_rev_3_4: - 200 p_right_3_4 - w_4 + x_3 - x_4 <= 0
non_overlap_right_rev_3_5: - 200 p_right_3_5 - w_5 + x_3 - x_5 <= 0
non_overlap_right_rev_3_6: - 200 p_right_3_6 - w_6 + x_3 - x_6 <= 0
non_overlap_right_rev_3_7: - 200 p_right_3_7 - w_7 + x_3 - x_7 <= 0
non_overlap_right_rev_3_8: - 200 p_right_3_8 - w_8 + x_3 - x_8 <= 0
non_overlap_right_rev_3_9: - 200 p_right_3_9 - w_9 + x_3 - x_9 <= 0
non_overlap_right_rev_4_5: - 200 p_right_4_5 - w_5 + x_4 - x_5 <= 0
non_overlap_right_rev_4_6: - 200 p_right_4_6 - w_6 + x_4 - x_6 <= 0
non_overlap_right_rev_4_7: - 200 p_right_4_7 - w_7 + x_4 - x_7 <= 0
non_overlap_right_rev_4_8: - 200 p_right_4_8 - w_8 + x_4 - x_8 <= 0
non_overlap_right_rev_4_9: - 200 p_right_4_9 - w_9 + x_4 - x_9 <= 0
non_overlap_right_rev_5_6: - 200 p_right_5_6 - w_6 + x_5 - x_6 <= 0
non_overlap_right_rev_5_7: - 200 p_right_5_7 - w_7 + x_5 - x_7 <= 0
non_overlap_right_rev_5_8: - 200 p_right_5_8 - w_8 + x_5 - x_8 <= 0
non_overlap_right_rev_5_9: - 200 p_right_5_9 - w_9 + x_5 - x_9 <= 0
non_overlap_right_rev_6_7: - 200 p_right_6_7 - w_7 + x_6 - x_7 <= 0
non_overlap_right_rev_6_8: - 200 p_right_6_8 - w_8 + x_6 - x_8 <= 0
non_overlap_right_rev_6_9: - 200 p_right_6_9 - w_9 + x_6 - x_9 <= 0
non_overlap_right_rev_7_8: - 200 p_right_7_8 - w_8 + x_7 - x_8 <= 0
non_overlap_right_rev_7_9: - 200 p_right_7_9 - w_9 + x_7 - x_9 <= 0
non_overlap_right_rev_8_9: - 200 p_right_8_9 - w_9 + x_8 - x_9 <= 0
non_overlap_up_0_1: h_1 + 200 p_up_0_1 - y_0 + y_1 <= 200
non_overlap_up_0_2: h_2 + 200 p_up_0_2 - y_0 + y_2 <= 200
non_overlap_up_0_3: h_3 + 200 p_up_0_3 - y_0 + y_3 <= 200
non_overlap_up_0_4: h_4 + 200 p_up_0_4 - y_0 + y_4 <= 200
non_overlap_up_0_5: h_5 + 200 p_up_0_5 - y_0 + y_5 <= 200
non_overlap_up_0_6: h_6 + 200 p_up_0_6 - y_0 + y_6 <= 200
non_overlap_up_0_7: h_7 + 200 p_up_0_7 - y_0 + y_7 <= 200
non_overlap_up_0_8: h_8 + 200 p_up_0_8 - y_0 + y_8 <= 200
non_overlap_up_0_9: h_9 + 200 p_up_0_9 - y_0 + y_9 <= 200
non_overlap_up_1_2: h_2 + 200 p_up_1_2 - y_1 + y_2 <= 200
non_overlap_up_1_3: h_3 + 200 p_up_1_3 - y_1 + y_3 <= 200
non_overlap_up_1_4: h_4 + 200 p_up_1_4 - y_1 + y_4 <= 200
non_overlap_up_1_5: h_5 + 200 p_up_1_5 - y_1 + y_5 <= 200
non_overlap_up_1_6: h_6 + 200 p_up_1_6 - y_1 + y_6 <= 200
non_overlap_up_1_7: h_7 + 200 p_up_1_7 - y_1 + y_7 <= 200
non_overlap_up_1_8: h_8 + 200 p_up_1_8 - y_1 + y_8 <= 200
non_overlap_up_1_9: h_9 + 200 p_up_1_9 - y_1 + y_9 <= 200
non_overlap_up_2_3: h_3 + 200 p_up_2_3 - y_2 + y_3 <= 200
non_overlap_up_2_4: h_4 + 200 p_up_2_4 - y_2 + y_4 <= 200
non_overlap_up_2_5: h_5 + 200 p_up_2_5 - y_2 + y_5 <= 200
non_overlap_up_2_6: h_6 + 200 p_up_2_6 - y_2 + y_6 <= 200
non_overlap_up_2_7: h_7 + 200 p_up_2_7 - y_2 + y_7 <= 200
non_overlap_up_2_8: h_8 + 200 p_up_2_8 - y_2 + y_8 <= 200
non_overlap_up_2_9: h_9 + 200 p_up_2_9 - y_2 + y_9 <= 200
non_overlap_up_3_4: h_4 + 200 p_up_3_4 - y_3 + y_4 <= 200
non_overlap_up_3_5: h_5 + 200 p_up_3_5 - y_3 + y_5 <= 200
non_overlap_up_3_6: h_6 + 200 p_up_3_6 - y_3 + y_6 <= 200
non_overlap_up_3_7: h_7 + 200 p_up_3_7 - y_3 + y_7 <= 200
non_overlap_up_3_8: h_8 + 200 p_up_3_8 - y_3 + y_8 <= 200
non_overlap_up_3_9: h_9 + 200 p_up_3_9 - y_3 + y_9 <= 200
non_overlap_up_4_5: h_5 + 200 p_up_4_5 - y_4 + y_5 <= 200
non_overlap_up_4_6: h_6 + 200 p_up_4_6 - y_4 + y_6 <= 200
non_overlap_up_4_7: h_7 + 200 p_up_4_7 - y_4 + y_7 <= 200
non_overlap_up_4_8: h_8 + 200 p_up_4_8 - y_4 + y_8 <= 200
non_overlap_up_4_9: h_9 + 200 p_up_4_9 - y_4 + y_9 <= 200
non_overlap_up_5_6: h_6 + 200 p_up_5_6 - y_5 + y_6 <= 200
non_overlap_up_5_7: h_7 + 200 p_up_5_7 - y_5 + y_7 <= 200
non_overlap_up_5_8: h_8 + 200 p_up_5_8 - y_5 + y_8 <= 200
non_overlap_up_5_9: h_9 + 200 p_up_5_9 - y_5 + y_9 <= 200
non_overlap_up_6_7: h_7 + 200 p_up_6_7 - y_6 + y_7 <= 200
non_overlap_up_6_8: h_8 + 200 p_up_6_8 - y_6 + y_8 <= 200
non_overlap_up_6_9: h_9 + 200 p_up_6_9 - y_6 + y_9 <= 200
non_overlap_up_7_8: h_8 + 200 p_up_7_8 - y_7 + y_8 <= 200
non_overlap_up_7_9: h_9 + 200 p_up_7_9 - y_7 + y_9 <= 200
non_overlap_up_8_9: h_9 + 200 p_up_8_9 - y_8 + y_9 <= 200
non_overlap_up_rev_0_1: - h_1 - 200 p_up_0_1 + y_0 - y_1 <= 0
non_overlap_up_rev_0_2: - h_2 - 200 p_up_0_2 + y_0 - y_2 <= 0
non_overlap_up_rev_0_3: - h_3 - 200 p_up_0_3 + y_0 - y_3 <= 0
non_overlap_up_rev_0_4: - h_4 - 200 p_up_0_4 + y_0 - y_4 <= 0
non_overlap_up_rev_0_5: - h_5 - 200 p_up_0_5 + y_0 - y_5 <= 0
non_overlap_up_rev_0_6: - h_6 - 200 p_up_0_6 + y_0 - y_6 <= 0
non_overlap_up_rev_0_7: - h_7 - 200 p_up_0_7 + y_0 - y_7 <= 0
non_overlap_up_rev_0_8: - h_8 - 200 p_up_0_8 + y_0 - y_8 <= 0
non_overlap_up_rev_0_9: - h_9 - 200 p_up_0_9 + y_0 - y_9 <= 0
non_overlap_up_rev_1_2: - h_2 - 200 p_up_1_2 + y_1 - y_2 <= 0
non_overlap_up_rev_1_3: - h_3 - 200 p_up_1_3 + y_1 - y_3 <= 0
non_overlap_up_rev_1_4: - h_4 - 200 p_up_1_4 + y_1 - y_4 <= 0
non_overlap_up_rev_1_5: - h_5 - 200 p_up_1_5 + y_1 - y_5 <= 0
non_overlap_up_rev_1_6: - h_6 - 200 p_up_1_6 + y_1 - y_6 <= 0
non_overlap_up_rev_1_7: - h_7 - 200 p_up_1_7 + y_1 - y_7 <= 0
non_overlap_up_rev_1_8: - h_8 - 200 p_up_1_8 + y_1 - y_8 <= 0
non_overlap_up_rev_1_9: - h_9 - 200 p_up_1_9 + y_1 - y_9 <= 0
non_overlap_up_rev_2_3: - h_3 - 200 p_up_2_3 + y_2 - y_3 <= 0
non_overlap_up_rev_2_4: - h_4 - 200 p_up_2_4 + y_2 - y_4 <= 0
non_overlap_up_rev_2_5: - h_5 - 200 p_up_2_5 + y_2 - y_5 <= 0
non_overlap_up_rev_2_6: - h_6 - 200 p_up_2_6 + y_2 - y_6 <= 0
non_overlap_up_rev_2_7: - h_7 - 200 p_up_2_7 + y_2 - y_7 <= 0
non_overlap_up_rev_2_8: - h_8 - 200 p_up_2_8 + y_2 - y_8 <= 0
non_overlap_up_rev_2_9: - h_9 - 200 p_up_2_9 + y_2 - y_9 <= 0
non_overlap_up_rev_3_4: - h_4 - 200 p_up_3_4 + y_3 - y_4 <= 0
non_overlap_up_rev_3_5: - h_5 - 200 p_up_3_5 + y_3 - y_5 <= 0
non_overlap_up_rev_3_6: - h_6 - 200 p_up_3_6 + y_3 - y_6 <= 0
non_overlap_up_rev_3_7: - h_7 - 200 p_up_3_7 + y_3 - y_7 <= 0
non_overlap_up_rev_3_8: - h_8 - 200 p_up_3_8 + y_3 - y_8 <= 0
non_overlap_up_rev_3_9: - h_9 - 200 p_up_3_9 + y_3 - y_9 <= 0
non_overlap_up_rev_4_5: - h_5 - 200 p_up_4_5 + y_4 - y_5 <= 0
non_overlap_up_rev_4_6: - h_6 - 200 p_up_4_6 + y_4 - y_6 <= 0
non_overlap_up_rev_4_7: - h_7 - 200 p_up_4_7 + y_4 - y_7 <= 0
non_overlap_up_rev_4_8: - h_8 - 200 p_up_4_8 + y_4 - y_8 <= 0
non_overlap_up_rev_4_9: - h_9 - 200 p_up_4_9 + y_4 - y_9 <= 0
non_overlap_up_rev_5_6: - h_6 - 200 p_up_5_6 + y_5 - y_6 <= 0
non_overlap_up_rev_5_7: - h_7 - 200 p_up_5_7 + y_5 - y_7 <= 0
non_overlap_up_rev_5_8: - h_8 - 200 p_up_5_8 + y_5 - y_8 <= 0
non_overlap_up_rev_5_9: - h_9 - 200 p_up_5_9 + y_5 - y_9 <= 0
non_overlap_up_rev_6_7: - h_7 - 200 p_up_6_7 + y_6 - y_7 <= 0
non_overlap_up_rev_6_8: - h_8 - 200 p_up_6_8 + y_6 - y_8 <= 0
non_overlap_up_rev_6_9: - h_9 - 200 p_up_6_9 + y_6 - y_9 <= 0
non_overlap_up_rev_7_8: - h_8 - 200 p_up_7_8 + y_7 - y_8 <= 0
non_overlap_up_rev_7_9: - h_9 - 200 p_up_7_9 + y_7 - y_9 <= 0
non_overlap_up_rev_8_9: - h_9 - 200 p_up_8_9 + y_8 - y_9 <= 0
shared_x_min_0_1: shared_x_0_1 - 200 z2_0_1 >= -199.9
shared_x_min_0_2: shared_x_0_2 - 200 z2_0_2 >= -199.9
shared_x_min_0_4: shared_x_0_4 - 200 z2_0_4 >= -199.9
shared_x_min_0_5: shared_x_0_5 - 200 z2_0_5 >= -199.9
shared_x_min_1_3: shared_x_1_3 - 200 z2_1_3 >= -199.9
shared_x_min_2_3: shared_x_2_3 - 200 z2_2_3 >= -199.9
shared_x_min_2_6: shared_x_2_6 - 200 z2_2_6 >= -199.9
shared_x_min_2_7: shared_x_2_7 - 200 z2_2_7 >= -199.9
shared_x_min_3_8: shared_x_3_8 - 200 z2_3_8 >= -199.9
shared_x_min_3_9: shared_x_3_9 - 200 z2_3_9 >= -199.9
shared_x_ub1_0_1: shared_x_0_1 - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
shared_x_ub1_0_2: shared_x_0_2 - w_0 - x_0 + x_2 + 200 z2_0_2 <= 200
shared_x_ub1_0_4: shared_x_0_4 - w_0 - x_0 + x_4 + 200 z2_0_4 <= 200
shared_x_ub1_0_5: shared_x_0_5 - w_0 - x_0 + x_5 + 200 z2_0_5 <= 200
shared_x_ub1_1_3: shared_x_1_3 - w_1 - x_1 + x_3 + 200 z2_1_3 <= 200
shared_x_ub1_2_3: shared_x_2_3 - w_2 - x_2 + x_3 + 200 z2_2_3 <= 200
shared_x_ub1_2_6: shared_x_2_6 - w_2 - x_2 + x_6 + 200 z2_2_6 <= 200
shared_x_ub1_2_7: shared_x_2_7 - w_2 - x_2 + x_7 + 200 z2_2_7 <= 200
shared_x_ub1_3_8: shared_x_3_8 - w_3 - x_3 + x_8 + 200 z2_3_8 <= 200
shared_x_ub1_3_9: shared_x_3_9 - w_3 - x_3 + x_9 + 200 z2_3_9 <= 200
shared_x_ub2_0_1: shared_x_0_1 - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
shared_x_ub2_0_2: shared_x_0_2 - w_2 + x_0 - x_2 + 200 z2_0_2 <= 200
shared_x_ub2_0_4: shared_x_0_4 - w_4 + x_0 - x_4 + 200 z2_0_4 <= 200
shared_x_ub2_0_5: shared_x_0_5 - w_5 + x_0 - x_5 + 200 z2_0_5 <= 200
shared_x_ub2_1_3: shared_x_1_3 - w_3 + x_1 - x_3 + 200 z2_1_3 <= 200
shared_x_ub2_2_3: shared_x_2_3 - w_3 + x_2 - x_3 + 200 z2_2_3 <= 200
shared_x_ub2_2_6: shared_x_2_6 - w_6 + x_2 - x_6 + 200 z2_2_6 <= 200
shared_x_ub2_2_7: shared_x_2_7 - w_7 + x_2 - x_7 + 200 z2_2_7 <= 200
shared_x_ub2_3_8: shared_x_3_8 - w_8 + x_3 - x_8 + 200 z2_3_8 <= 200
shared_x_ub2_3_9: shared_x_3_9 - w_9 + x_3 - x_9 + 200 z2_3_9 <= 200
shared_x_zero_0_1: shared_x_0_1 - 200 z2_0_1 <= 0
shared_x_zero_0_2: shared_x_0_2 - 200 z2_0_2 <= 0
shared_x_zero_0_4: shared_x_0_4 - 200 z2_0_4 <= 0
shared_x_zero_0_5: shared_x_0_5 - 200 z2_0_5 <= 0
shared_x_zero_1_3: shared_x_1_3 - 200 z2_1_3 <= 0
shared_x_zero_2_3: shared_x_2_3 - 200 z2_2_3 <= 0
shared_x_zero_2_6: shared_x_2_6 - 200 z2_2_6 <= 0
shared_x_zero_2_7: shared_x_2_7 - 200 z2_2_7 <= 0
shared_x_zero_3_8: shared_x_3_8 - 200 z2_3_8 <= 0
shared_x_zero_3_9: shared_x_3_9 - 200 z2_3_9 <= 0
shared_y_min_0_1: shared_y_0_1 - 200 z1_0_1 >= -199.9
shared_y_min_0_2: shared_y_0_2 - 200 z1_0_2 >= -199.9
shared_y_min_0_4: shared_y_0_4 - 200 z1_0_4 >= -199.9
shared_y_min_0_5: shared_y_0_5 - 200 z1_0_5 >= -199.9
shared_y_min_1_3: shared_y_1_3 - 200 z1_1_3 >= -199.9
shared_y_min_2_3: shared_y_2_3 - 200 z1_2_3 >= -199.9
shared_y_min_2_6: shared_y_2_6 - 200 z1_2_6 >= -199.9
shared_y_min_2_7: shared_y_2_7 - 200 z1_2_7 >= -199.9
shared_y_min_3_8: shared_y_3_8 - 200 z1_3_8 >= -199.9
shared_y_min_3_9: shared_y_3_9 - 200 z1_3_9 >= -199.9
shared_y_ub1_0_1: - h_0 + shared_y_0_1 - y_0 + y_1 + 200 z1_0_1 <= 200
shared_y_ub1_0_2: - h_0 + shared_y_0_2 - y_0 + y_2 + 200 z1_0_2 <= 200
shared_y_ub1_0_4: - h_0 + shared_y_0_4 - y_0 + y_4 + 200 z1_0_4 <= 200
shared_y_ub1_0_5: - h_0 + shared_y_0_5 - y_0 + y_5 + 200 z1_0_5 <= 200
shared_y_ub1_1_3: - h_1 + shared_y_1_3 - y_1 + y_3 + 200 z1_1_3 <= 200
shared_y_ub1_2_3: - h_2 + shared_y_2_3 - y_2 + y_3 + 200 z1_2_3 <= 200
shared_y_ub1_2_6: - h_2 + shared_y_2_6 - y_2 + y_6 + 200 z1_2_6 <= 200
shared_y_ub1_2_7: - h_2 + shared_y_2_7 - y_2 + y_7 + 200 z1_2_7 <= 200
shared_y_ub1_3_8: - h_3 + shared_y_3_8 - y_3 + y_8 + 200 z1_3_8 <= 200
shared_y_ub1_3_9: - h_3 + shared_y_3_9 - y_3 + y_9 + 200 z1_3_9 <= 200
shared_y_ub2_0_1: - h_1 + shared_y_0_1 + y_0 - y_1 + 200 z1_0_1 <= 200
shared_y_ub2_0_2: - h_2 + shared_y_0_2 + y_0 - y_2 + 200 z1_0_2 <= 200
shared_y_ub2_0_4: - h_4 + shared_y_0_4 + y_0 - y_4 + 200 z1_0_4 <= 200
shared_y_ub2_0_5: - h_5 + shared_y_0_5 + y_0 - y_5 + 200 z1_0_5 <= 200
shared_y_ub2_1_3: - h_3 + shared_y_1_3 + y_1 - y_3 + 200 z1_1_3 <= 200
shared_y_ub2_2_3: - h_3 + shared_y_2_3 + y_2 - y_3 + 200 z1_2_3 <= 200
shared_y_ub2_2_6: - h_6 + shared_y_2_6 + y_2 - y_6 + 200 z1_2_6 <= 200
shared_y_ub2_2_7: - h_7 + shared_y_2_7 + y_2 - y_7 + 200 z1_2_7 <= 200
shared_y_ub2_3_8: - h_8 + shared_y_3_8 + y_3 - y_8 + 200 z1_3_8 <= 200
shared_y_ub2_3_9: - h_9 + shared_y_3_9 + y_3 - y_9 + 200 z1_3_9 <= 200
shared_y_zero_0_1: shared_y_0_1 - 200 z1_0_1 <= 0
shared_y_zero_0_2: shared_y_0_2 - 200 z1_0_2 <= 0
shared_y_zero_0_4: shared_y_0_4 - 200 z1_0_4 <= 0
shared_y_zero_0_5: shared_y_0_5 - 200 z1_0_5 <= 0
shared_y_zero_1_3: shared_y_1_3 - 200 z1_1_3 <= 0
shared_y_zero_2_3: shared_y_2_3 - 200 z1_2_3 <= 0
shared_y_zero_2_6: shared_y_2_6 - 200 z1_2_6 <= 0
shared_y_zero_2_7: shared_y_2_7 - 200 z1_2_7 <= 0
shared_y_zero_3_8: shared_y_3_8 - 200 z1_3_8 <= 0
shared_y_zero_3_9: shared_y_3_9 - 200 z1_3_9 <= 0
t_ge_height: bbox_area_proxy_t - bbox_h >= 0
t_ge_scaled_mean: bbox_area_proxy_t - 0.8 bbox_h - 0.8 bbox_w >= 0
t_ge_width: bbox_area_proxy_t - bbox_w >= 0
vertical_direction_0_1: z2D_0_1 + z2U_0_1 - z2_0_1 = 0
vertical_direction_0_2: z2D_0_2 + z2U_0_2 - z2_0_2 = 0
vertical_direction_0_4: z2D_0_4 + z2U_0_4 - z2_0_4 = 0
vertical_direction_0_5: z2D_0_5 + z2U_0_5 - z2_0_5 = 0
vertical_direction_1_3: z2D_1_3 + z2U_1_3 - z2_1_3 = 0
vertical_direction_2_3: z2D_2_3 + z2U_2_3 - z2_2_3 = 0
vertical_direction_2_6: z2D_2_6 + z2U_2_6 - z2_2_6 = 0
vertical_direction_2_7: z2D_2_7 + z2U_2_7 - z2_2_7 = 0
vertical_direction_3_8: z2D_3_8 + z2U_3_8 - z2_3_8 = 0
vertical_direction_3_9: z2D_3_9 + z2U_3_9 - z2_3_9 = 0
vertical_down_dist_0_1: - h_0 - y_0 + y_1 + 200 z2D_0_1 <= 200.5
vertical_down_dist_0_2: - h_0 - y_0 + y_2 + 200 z2D_0_2 <= 200.5
vertical_down_dist_0_4: - h_0 - y_0 + y_4 + 200 z2D_0_4 <= 200.5
vertical_down_dist_0_5: - h_0 - y_0 + y_5 + 200 z2D_0_5 <= 200.5
vertical_down_dist_1_3: - h_1 - y_1 + y_3 + 200 z2D_1_3 <= 200.5
vertical_down_dist_2_3: - h_2 - y_2 + y_3 + 200 z2D_2_3 <= 200.5
vertical_down_dist_2_6: - h_2 - y_2 + y_6 + 200 z2D_2_6 <= 200.5
vertical_down_dist_2_7: - h_2 - y_2 + y_7 + 200 z2D_2_7 <= 200.5
vertical_down_dist_3_8: - h_3 - y_3 + y_8 + 200 z2D_3_8 <= 200.5
vertical_down_dist_3_9: - h_3 - y_3 + y_9 + 200 z2D_3_9 <= 200.5
vertical_down_dist_lb_0_1: - h_0 - y_0 + y_1 - 200 z2D_0_1 >= -200
vertical_down_dist_lb_0_2: - h_0 - y_0 + y_2 - 200 z2D_0_2 >= -200
vertical_down_dist_lb_0_4: - h_0 - y_0 + y_4 - 200 z2D_0_4 >= -200
vertical_down_dist_lb_0_5: - h_0 - y_0 + y_5 - 200 z2D_0_5 >= -200
vertical_down_dist_lb_1_3: - h_1 - y_1 + y_3 - 200 z2D_1_3 >= -200
vertical_down_dist_lb_2_3: - h_2 - y_2 + y_3 - 200 z2D_2_3 >= -200
vertical_down_dist_lb_2_6: - h_2 - y_2 + y_6 - 200 z2D_2_6 >= -200
vertical_down_dist_lb_2_7: - h_2 - y_2 + y_7 - 200 z2D_2_7 >= -200
vertical_down_dist_lb_3_8: - h_3 - y_3 + y_8 - 200 z2D_3_8 >= -200
vertical_down_dist_lb_3_9: - h_3 - y_3 + y_9 - 200 z2D_3_9 >= -200
vertical_overlap_x1_0_1: - w_1 + x_0 - x_1 + 200 z2_0_1 <= 200
vertical_overlap_x1_0_2: - w_2 + x_0 - x_2 + 200 z2_0_2 <= 200
vertical_overlap_x1_0_4: - w_4 + x_0 - x_4 + 200 z2_0_4 <= 200
vertical_overlap_x1_0_5: - w_5 + x_0 - x_5 + 200 z2_0_5 <= 200
vertical_overlap_x1_1_3: - w_3 + x_1 - x_3 + 200 z2_1_3 <= 200
vertical_overlap_x1_2_3: - w_3 + x_2 - x_3 + 200 z2_2_3 <= 200
vertical_overlap_x1_2_6: - w_6 + x_2 - x_6 + 200 z2_2_6 <= 200
vertical_overlap_x1_2_7: - w_7 + x_2 - x_7 + 200 z2_2_7 <= 200
vertical_overlap_x1_3_8: - w_8 + x_3 - x_8 + 200 z2_3_8 <= 200
vertical_overlap_x1_3_9: - w_9 + x_3 - x_9 + 200 z2_3_9 <= 200
vertical_overlap_x2_0_1: - w_0 - x_0 + x_1 + 200 z2_0_1 <= 200
vertical_overlap_x2_0_2: - w_0 - x_0 + x_2 + 200 z2_0_2 <= 200
vertical_overlap_x2_0_4: - w_0 - x_0 + x_4 + 200 z2_0_4 <= 200
vertical_overlap_x2_0_5: - w_0 - x_0 + x_5 + 200 z2_0_5 <= 200
vertical_overlap_x2_1_3: - w_1 - x_1 + x_3 + 200 z2_1_3 <= 200
vertical_overlap_x2_2_3: - w_2 - x_2 + x_3 + 200 z2_2_3 <= 200
vertical_overlap_x2_2_6: - w_2 - x_2 + x_6 + 200 z2_2_6 <= 200
vertical_overlap_x2_2_7: - w_2 - x_2 + x_7 + 200 z2_2_7 <= 200
vertical_overlap_x2_3_8: - w_3 - x_3 + x_8 + 200 z2_3_8 <= 200
vertical_overlap_x2_3_9: - w_3 - x_3 + x_9 + 200 z2_3_9 <= 200
vertical_up_dist_0_1: - h_1 + y_0 - y_1 + 200 z2U_0_1 <= 200.5
vertical_up_dist_0_2: - h_2 + y_0 - y_2 + 200 z2U_0_2 <= 200.5
vertical_up_dist_0_4: - h_4 + y_0 - y_4 + 200 z2U_0_4 <= 200.5
vertical_up_dist_0_5: - h_5 + y_0 - y_5 + 200 z2U_0_5 <= 200.5
vertical_up_dist_1_3: - h_3 + y_1 - y_3 + 200 z2U_1_3 <= 200.5
vertical_up_dist_2_3: - h_3 + y_2 - y_3 + 200 z2U_2_3 <= 200.5
vertical_up_dist_2_6: - h_6 + y_2 - y_6 + 200 z2U_2_6 <= 200.5
vertical_up_dist_2_7: - h_7 + y_2 - y_7 + 200 z2U_2_7 <= 200.5
vertical_up_dist_3_8: - h_8 + y_3 - y_8 + 200 z2U_3_8 <= 200.5
vertical_up_dist_3_9: - h_9 + y_3 - y_9 + 200 z2U_3_9 <= 200.5
vertical_up_dist_lb_0_1: - h_1 + y_0 - y_1 - 200 z2U_0_1 >= -200
vertical_up_dist_lb_0_2: - h_2 + y_0 - y_2 - 200 z2U_0_2 >= -200
vertical_up_dist_lb_0_4: - h_4 + y_0 - y_4 - 200 z2U_0_4 >= -200
vertical_up_dist_lb_0_5: - h_5 + y_0 - y_5 - 200 z2U_0_5 >= -200
vertical_up_dist_lb_1_3: - h_3 + y_1 - y_3 - 200 z2U_1_3 >= -200
vertical_up_dist_lb_2_3: - h_3 + y_2 - y_3 - 200 z2U_2_3 >= -200
vertical_up_dist_lb_2_6: - h_6 + y_2 - y_6 - 200 z2U_2_6 >= -200
vertical_up_dist_lb_2_7: - h_7 + y_2 - y_7 - 200 z2U_2_7 >= -200
vertical_up_dist_lb_3_8: - h_8 + y_3 - y_8 - 200 z2U_3_8 >= -200
vertical_up_dist_lb_3_9: - h_9 + y_3 - y_9 - 200 z2U_3_9 >= -200
w_align_grid_0: w_0 = 50
w_align_grid_1: w_1 = 30
w_align_grid_2: w_2 = 30
w_align_grid_3: w_3 = 30
w_align_grid_4: w_4 = 10
w_align_grid_5: w_5 = 10
w_align_grid_6: w_6 = 10
w_align_grid_7: w_7 = 10
w_align_grid_8: w_8 = 10
w_align_grid_9: w_9 = 10
w_max_0: w_0 <= 150
w_max_1: w_1 <= 150
w_max_2: w_2 <= 150
w_max_3: w_3 <= 150
w_max_4: w_4 <= 150
w_max_5: w_5 <= 150
w_max_6: w_6 <= 150
w_max_7: w_7 <= 150
w_max_8: w_8 <= 150
w_max_9: w_9 <= 150
width_rotation_0: 20 r_0 + w_0 = 50
width_rotation_1: - 20 r_1 + w_1 = 30
width_rotation_2: - 20 r_2 + w_2 = 30
width_rotation_3: - 20 r_3 + w_3 = 30
width_rotation_4: - 5 r_4 + w_4 = 10
width_rotation_5: - 5 r_5 + w_5 = 10
width_rotation_6: - 5 r_6 + w_6 = 10
width_rotation_7: - 5 r_7 + w_7 = 10
width_rotation_8: - 5 r_8 + w_8 = 10
width_rotation_9: - 5 r_9 + w_9 = 10
x_grid_to_real_0: x_0 - 0.5 x_grid_0 = 0
x_grid_to_real_1: x_1 - 0.5 x_grid_1 = 0
x_grid_to_real_2: x_2 - 0.5 x_grid_2 = 0
x_grid_to_real_3: x_3 - 0.5 x_grid_3 = 0
x_grid_to_real_4: x_4 - 0.5 x_grid_4 = 0
x_grid_to_real_5: x_5 - 0.5 x_grid_5 = 0
x_grid_to_real_6: x_6 - 0.5 x_grid_6 = 0
x_grid_to_real_7: x_7 - 0.5 x_grid_7 = 0
x_grid_to_real_8: x_8 - 0.5 x_grid_8 = 0
x_grid_to_real_9: x_9 - 0.5 x_grid_9 = 0
y_grid_to_real_0: y_0 - 0.5 y_grid_0 = 0
y_grid_to_real_1: y_1 - 0.5 y_grid_1 = 0
y_grid_to_real_2: y_2 - 0.5 y_grid_2 = 0
y_grid_to_real_3: y_3 - 0.5 y_grid_3 = 0
y_grid_to_real_4: y_4 - 0.5 y_grid_4 = 0
y_grid_to_real_5: y_5 - 0.5 y_grid_5 = 0
y_grid_to_real_6: y_6 - 0.5 y_grid_6 = 0
y_grid_to_real_7: y_7 - 0.5 y_grid_7 = 0
y_grid_to_real_8: y_8 - 0.5 y_grid_8 = 0
y_grid_to_real_9: y_9 - 0.5 y_grid_9 = 0
Bounds
 bbox_area_proxy_t <= 300
 bbox_h <= 150
 bbox_max_x <= 150
 bbox_max_y <= 150
 bbox_min_x <= 150
 bbox_min_y <= 150
 bbox_w <= 150
 cx_0 <= 150
 cx_1 <= 150
 cx_2 <= 150
 cx_3 <= 150
 cx_4 <= 150
 cx_5 <= 150
 cx_6 <= 150
 cx_7 <= 150
 cx_8 <= 150
 cx_9 <= 150
 cy_0 <= 150
 cy_1 <= 150
 cy_2 <= 150
 cy_3 <= 150
 cy_4 <= 150
 cy_5 <= 150
 cy_6 <= 150
 cy_7 <= 150
 cy_8 <= 150
 cy_9 <= 150
 -150 <= dx_diff_0_1 <= 150
 -150 <= dx_diff_0_2 <= 150
 -150 <= dx_diff_0_4 <= 150
 -150 <= dx_diff_0_5 <= 150
 -150 <= dx_diff_1_3 <= 150
 -150 <= dx_diff_2_3 <= 150
 -150 <= dx_diff_2_6 <= 150
 -150 <= dx_diff_2_7 <= 150
 -150 <= dx_diff_3_8 <= 150
 -150 <= dx_diff_3_9 <= 150
 -150 <= dy_diff_0_1 <= 150
 -150 <= dy_diff_0_2 <= 150
 -150 <= dy_diff_0_4 <= 150
 -150 <= dy_diff_0_5 <= 150
 -150 <= dy_diff_1_3 <= 150
 -150 <= dy_diff_2_3 <= 150
 -150 <= dy_diff_2_6 <= 150
 -150 <= dy_diff_2_7 <= 150
 -150 <= dy_diff_3_8 <= 150
 -150 <= dy_diff_3_9 <= 150
 30 <= h_0 <= 50
 30 <= h_1 <= 50
 30 <= h_2 <= 50
 30 <= h_3 <= 50
 10 <= h_4 <= 15
 10 <= h_5 <= 15
 10 <= h_6 <= 15
 10 <= h_7 <= 15
 10 <= h_8 <= 15
 10 <= h_9 <= 15
 shared_x_0_1 <= 50
 shared_x_0_2 <= 50
 shared_x_0_4 <= 50
 shared_x_0_5 <= 50
 shared_x_1_3 <= 30
 shared_x_2_3 <= 30
 shared_x_2_6 <= 30
 shared_x_2_7 <= 30
 shared_x_3_8 <= 30
 shared_x_3_9 <= 30
 shared_y_0_1 <= 50
 shared_y_0_2 <= 50
 shared_y_0_4 <= 30
 shared_y_0_5 <= 30
 shared_y_1_3 <= 50
 shared_y_2_3 <= 50
 shared_y_2_6 <= 50
 shared_y_2_7 <= 50
 shared_y_3_8 <= 50
 shared_y_3_9 <= 50
 30 <= w_0 <= 50
 30 <= w_1 <= 50
 30 <= w_2 <= 50
 30 <= w_3 <= 50
 10 <= w_4 <= 15
 10 <= w_5 <= 15
 10 <= w_6 <= 15
 10 <= w_7 <= 15
 10 <= w_8 <= 15
 10 <= w_9 <= 15
 x_0 <= 150
 x_1 <= 150
 x_2 <= 150
 x_3 <= 150
 x_4 <= 150
 x_5 <= 150
 x_6 <= 150
 x_7 <= 150
 x_8 <= 150
 x_9 <= 150
 0 <= x_grid_0 <= 299
 0 <= x_grid_1 <= 299
 0 <= x_grid_2 <= 299
 0 <= x_grid_3 <= 299
 0 <= x_grid_4 <= 299
 0 <= x_grid_5 <= 299
 0 <= x_grid_6 <= 299
 0 <= x_grid_7 <= 299
 0 <= x_grid_8 <= 299
 0 <= x_grid_9 <= 299
 y_0 <= 150
 y_1 <= 150
 y_2 <= 150
 y_3 <= 150
 y_4 <= 150
 y_5 <= 150
 y_6 <= 150
 y_7 <= 150
 y_8 <= 150
 y_9 <= 150
 0 <= y_grid_0 <= 299
 0 <= y_grid_1 <= 299
 0 <= y_grid_2 <= 299
 0 <= y_grid_3 <= 299
 0 <= y_grid_4 <= 299
 0 <= y_grid_5 <= 299
 0 <= y_grid_6 <= 299
 0 <= y_grid_7 <= 299
 0 <= y_grid_8 <= 299
 0 <= y_grid_9 <= 299
Generals
x_grid_0
x_grid_1
x_grid_2
x_grid_3
x_grid_4
x_grid_5
x_grid_6
x_grid_7
x_grid_8
x_grid_9
y_grid_0
y_grid_1
y_grid_2
y_grid_3
y_grid_4
y_grid_5
y_grid_6
y_grid_7
y_grid_8
y_grid_9
Binaries
dx_abs_0_1_is_positive
dx_abs_0_2_is_positive
dx_abs_0_4_is_positive
dx_abs_0_5_is_positive
dx_abs_1_3_is_positive
dx_abs_2_3_is_positive
dx_abs_2_6_is_positive
dx_abs_2_7_is_positive
dx_abs_3_8_is_positive
dx_abs_3_9_is_positive
dy_abs_0_1_is_positive
dy_abs_0_2_is_positive
dy_abs_0_4_is_positive
dy_abs_0_5_is_positive
dy_abs_1_3_is_positive
dy_abs_2_3_is_positive
dy_abs_2_6_is_positive
dy_abs_2_7_is_positive
dy_abs_3_8_is_positive
dy_abs_3_9_is_positive
p_down_0_1
p_down_0_2
p_down_0_3
p_down_0_4
p_down_0_5
p_down_0_6
p_down_0_7
p_down_0_8
p_down_0_9
p_down_1_2
p_down_1_3
p_down_1_4
p_down_1_5
p_down_1_6
p_down_1_7
p_down_1_8
p_down_1_9
p_down_2_3
p_down_2_4
p_down_2_5
p_down_2_6
p_down_2_7
p_down_2_8
p_down_2_9
p_down_3_4
p_down_3_5
p_down_3_6
p_down_3_7
p_down_3_8
p_down_3_9
p_down_4_5
p_down_4_6
p_down_4_7
p_down_4_8
p_down_4_9
p_down_5_6
p_down_5_7
p_down_5_8
p_down_5_9
p_down_6_7
p_down_6_8
p_down_6_9
p_down_7_8
p_down_7_9
p_down_8_9
p_left_0_1
p_left_0_2
p_left_0_3
p_left_0_4
p_left_0_5
p_left_0_6
p_left_0_7
p_left_0_8
p_left_0_9
p_left_1_2
p_left_1_3
p_left_1_4
p_left_1_5
p_left_1_6
p_left_1_7
p_left_1_8
p_left_1_9
p_left_2_3
p_left_2_4
p_left_2_5
p_left_2_6
p_left_2_7
p_left_2_8
p_left_2_9
p_left_3_4
p_left_3_5
p_left_3_6
p_left_3_7
p_left_3_8
p_left_3_9
p_left_4_5
p_left_4_6
p_left_4_7
p_left_4_8
p_left_4_9
p_left_5_6
p_left_5_7
p_left_5_8
p_left_5_9
p_left_6_7
p_left_6_8
p_left_6_9
p_left_7_8
p_left_7_9
p_left_8_9
p_right_0_1
p_right_0_2
p_right_0_3
p_right_0_4
p_right_0_5
p_right_0_6
p_right_0_7
p_right_0_8
p_right_0_9
p_right_1_2
p_right_1_3
p_right_1_4
p_right_1_5
p_right_1_6
p_right_1_7
p_right_1_8
p_right_1_9
p_right_2_3
p_right_2_4
p_right_2_5
p_right_2_6
p_right_2_7
p_right_2_8
p_right_2_9
p_right_3_4
p_right_3_5
p_right_3_6
p_right_3_7
p_right_3_8
p_right_3_9
p_right_4_5
p_right_4_6
p_right_4_7
p_right_4_8
p_right_4_9
p_right_5_6
p_right_5_7
p_right_5_8
p_right_5_9
p_right_6_7
p_right_6_8
p_right_6_9
p_right_7_8
p_right_7_9
p_right_8_9
p_up_0_1
p_up_0_2
p_up_0_3
p_up_0_4
p_up_0_5
p_up_0_6
p_up_0_7
p_up_0_8
p_up_0_9
p_up_1_2
p_up_1_3
p_up_1_4
p_up_1_5
p_up_1_6
p_up_1_7
p_up_1_8
p_up_1_9
p_up_2_3
p_up_2_4
p_up_2_5
p_up_2_6
p_up_2_7
p_up_2_8
p_up_2_9
p_up_3_4
p_up_3_5
p_up_3_6
p_up_3_7
p_up_3_8
p_up_3_9
p_up_4_5
p_up_4_6
p_up_4_7
p_up_4_8
p_up_4_9
p_up_5_6
p_up_5_7
p_up_5_8
p_up_5_9
p_up_6_7
p_up_6_8
p_up_6_9
p_up_7_8
p_up_7_9
p_up_8_9
r_0
r_1
r_2
r_3
r_4
r_5
r_6
r_7
r_8
r_9
z1L_0_1
z1L_0_2
z1L_0_4
z1L_0_5
z1L_1_3
z1L_2_3
z1L_2_6
z1L_2_7
z1L_3_8
z1L_3_9
z1R_0_1
z1R_0_2
z1R_0_4
z1R_0_5
z1R_1_3
z1R_2_3
z1R_2_6
z1R_2_7
z1R_3_8
z1R_3_9
z1_0_1
z1_0_2
z1_0_4
z1_0_5
z1_1_3
z1_2_3
z1_2_6
z1_2_7
z1_3_8
z1_3_9
z2D_0_1
z2D_0_2
z2D_0_4
z2D_0_5
z2D_1_3
z2D_2_3
z2D_2_6
z2D_2_7
z2D_3_8
z2D_3_9
z2U_0_1
z2U_0_2
z2U_0_4
z2U_0_5
z2U_1_3
z2U_2_3
z2U_2_6
z2U_2_7
z2U_3_8
z2U_3_9
z2_0_1
z2_0_2
z2_0_4
z2_0_5
z2_1_3
z2_2_3
z2_2_6
z2_2_7
z2_3_8
z2_3_9
End
