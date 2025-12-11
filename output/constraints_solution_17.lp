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
diff_pos_fixed_c120_p0_0: diff_pos_c120_p0_0 = 0
diff_pos_fixed_c121_p1_0: diff_pos_c121_p1_0 = 0
diff_pos_fixed_c122_p2_0: diff_pos_c122_p2_0 = 0
diff_pos_fixed_c123_p3_0: diff_pos_c123_p3_0 = 0
diff_pos_fixed_c124_p4_0: diff_pos_c124_p4_0 = 0
diff_pos_fixed_c125_p5_0: diff_pos_c125_p5_0 = 0
diff_pos_fixed_c126_p6_0: diff_pos_c126_p6_0 = 0
diff_pos_fixed_c127_p7_0: diff_pos_c127_p7_0 = 0
diff_pos_fixed_c128_p8_0: diff_pos_c128_p8_0 = 0
diff_pos_fixed_c129_p9_0: diff_pos_c129_p9_0 = 0
diff_pos_fixed_c130_p10_0: diff_pos_c130_p10_0 = 0
diff_pos_fixed_c131_p11_0: diff_pos_c131_p11_0 = 0
diff_pos_fixed_c132_p12_0: diff_pos_c132_p12_0 = 0
diff_pos_fixed_c133_p13_0: diff_pos_c133_p13_0 = 0
diff_pos_fixed_c134_p14_0: diff_pos_c134_p14_0 = 0
diff_pos_fixed_c135_p15_0: diff_pos_c135_p15_0 = 0
diff_pos_from_x_c120_p0_1: diff_pos_c120_p0_1 - x_diff_c120_p0_1 >= 0
diff_pos_from_x_c121_p1_1: diff_pos_c121_p1_1 - x_diff_c121_p1_1 >= 0
diff_pos_from_x_c122_p2_1: diff_pos_c122_p2_1 - x_diff_c122_p2_1 >= 0
diff_pos_from_x_c123_p3_1: diff_pos_c123_p3_1 - x_diff_c123_p3_1 >= 0
diff_pos_from_x_c124_p4_1: diff_pos_c124_p4_1 - x_diff_c124_p4_1 >= 0
diff_pos_from_x_c125_p5_1: diff_pos_c125_p5_1 - x_diff_c125_p5_1 >= 0
diff_pos_from_x_c126_p6_1: diff_pos_c126_p6_1 - x_diff_c126_p6_1 >= 0
diff_pos_from_x_c127_p7_1: diff_pos_c127_p7_1 - x_diff_c127_p7_1 >= 0
diff_pos_from_x_c128_p8_1: diff_pos_c128_p8_1 - x_diff_c128_p8_1 >= 0
diff_pos_from_x_c129_p9_1: diff_pos_c129_p9_1 - x_diff_c129_p9_1 >= 0
diff_pos_from_x_c130_p10_1: diff_pos_c130_p10_1 - x_diff_c130_p10_1 >= 0
diff_pos_from_x_c131_p11_1: diff_pos_c131_p11_1 - x_diff_c131_p11_1 >= 0
diff_pos_from_x_c132_p12_1: diff_pos_c132_p12_1 - x_diff_c132_p12_1 >= 0
diff_pos_from_x_c133_p13_1: diff_pos_c133_p13_1 - x_diff_c133_p13_1 >= 0
diff_pos_from_x_c134_p14_1: diff_pos_c134_p14_1 - x_diff_c134_p14_1 >= 0
diff_pos_from_x_c135_p15_1: diff_pos_c135_p15_1 - x_diff_c135_p15_1 >= 0
diff_pos_from_y_c120_p0_1: diff_pos_c120_p0_1 - y_diff_c120_p0_1 >= 0
diff_pos_from_y_c121_p1_1: diff_pos_c121_p1_1 - y_diff_c121_p1_1 >= 0
diff_pos_from_y_c122_p2_1: diff_pos_c122_p2_1 - y_diff_c122_p2_1 >= 0
diff_pos_from_y_c123_p3_1: diff_pos_c123_p3_1 - y_diff_c123_p3_1 >= 0
diff_pos_from_y_c124_p4_1: diff_pos_c124_p4_1 - y_diff_c124_p4_1 >= 0
diff_pos_from_y_c125_p5_1: diff_pos_c125_p5_1 - y_diff_c125_p5_1 >= 0
diff_pos_from_y_c126_p6_1: diff_pos_c126_p6_1 - y_diff_c126_p6_1 >= 0
diff_pos_from_y_c127_p7_1: diff_pos_c127_p7_1 - y_diff_c127_p7_1 >= 0
diff_pos_from_y_c128_p8_1: diff_pos_c128_p8_1 - y_diff_c128_p8_1 >= 0
diff_pos_from_y_c129_p9_1: diff_pos_c129_p9_1 - y_diff_c129_p9_1 >= 0
diff_pos_from_y_c130_p10_1: diff_pos_c130_p10_1 - y_diff_c130_p10_1 >= 0
diff_pos_from_y_c131_p11_1: diff_pos_c131_p11_1 - y_diff_c131_p11_1 >= 0
diff_pos_from_y_c132_p12_1: diff_pos_c132_p12_1 - y_diff_c132_p12_1 >= 0
diff_pos_from_y_c133_p13_1: diff_pos_c133_p13_1 - y_diff_c133_p13_1 >= 0
diff_pos_from_y_c134_p14_1: diff_pos_c134_p14_1 - y_diff_c134_p14_1 >= 0
diff_pos_from_y_c135_p15_1: diff_pos_c135_p15_1 - y_diff_c135_p15_1 >= 0
diff_pos_upper_c120_p0_1: diff_pos_c120_p0_1 - x_diff_c120_p0_1
 - y_diff_c120_p0_1 <= 0
diff_pos_upper_c121_p1_1: diff_pos_c121_p1_1 - x_diff_c121_p1_1
 - y_diff_c121_p1_1 <= 0
diff_pos_upper_c122_p2_1: diff_pos_c122_p2_1 - x_diff_c122_p2_1
 - y_diff_c122_p2_1 <= 0
diff_pos_upper_c123_p3_1: diff_pos_c123_p3_1 - x_diff_c123_p3_1
 - y_diff_c123_p3_1 <= 0
diff_pos_upper_c124_p4_1: diff_pos_c124_p4_1 - x_diff_c124_p4_1
 - y_diff_c124_p4_1 <= 0
diff_pos_upper_c125_p5_1: diff_pos_c125_p5_1 - x_diff_c125_p5_1
 - y_diff_c125_p5_1 <= 0
diff_pos_upper_c126_p6_1: diff_pos_c126_p6_1 - x_diff_c126_p6_1
 - y_diff_c126_p6_1 <= 0
diff_pos_upper_c127_p7_1: diff_pos_c127_p7_1 - x_diff_c127_p7_1
 - y_diff_c127_p7_1 <= 0
diff_pos_upper_c128_p8_1: diff_pos_c128_p8_1 - x_diff_c128_p8_1
 - y_diff_c128_p8_1 <= 0
diff_pos_upper_c129_p9_1: diff_pos_c129_p9_1 - x_diff_c129_p9_1
 - y_diff_c129_p9_1 <= 0
diff_pos_upper_c130_p10_1: diff_pos_c130_p10_1 - x_diff_c130_p10_1
 - y_diff_c130_p10_1 <= 0
diff_pos_upper_c131_p11_1: diff_pos_c131_p11_1 - x_diff_c131_p11_1
 - y_diff_c131_p11_1 <= 0
diff_pos_upper_c132_p12_1: diff_pos_c132_p12_1 - x_diff_c132_p12_1
 - y_diff_c132_p12_1 <= 0
diff_pos_upper_c133_p13_1: diff_pos_c133_p13_1 - x_diff_c133_p13_1
 - y_diff_c133_p13_1 <= 0
diff_pos_upper_c134_p14_1: diff_pos_c134_p14_1 - x_diff_c134_p14_1
 - y_diff_c134_p14_1 <= 0
diff_pos_upper_c135_p15_1: diff_pos_c135_p15_1 - x_diff_c135_p15_1
 - y_diff_c135_p15_1 <= 0
dist_diff_abs_pair_lower_c120_p0_0_1: - 60 diff_dist_pair_c120_p0_0_1
 + dist_diff_abs_pair_c120_p0_0_1 >= -57
dist_diff_abs_pair_lower_c121_p1_0_1: - 60 diff_dist_pair_c121_p1_0_1
 + dist_diff_abs_pair_c121_p1_0_1 >= -57
dist_diff_abs_pair_lower_c122_p2_0_1: - 60 diff_dist_pair_c122_p2_0_1
 + dist_diff_abs_pair_c122_p2_0_1 >= -57
dist_diff_abs_pair_lower_c123_p3_0_1: - 60 diff_dist_pair_c123_p3_0_1
 + dist_diff_abs_pair_c123_p3_0_1 >= -57
dist_diff_abs_pair_lower_c124_p4_0_1: - 60 diff_dist_pair_c124_p4_0_1
 + dist_diff_abs_pair_c124_p4_0_1 >= -57
dist_diff_abs_pair_lower_c125_p5_0_1: - 60 diff_dist_pair_c125_p5_0_1
 + dist_diff_abs_pair_c125_p5_0_1 >= -57
dist_diff_abs_pair_lower_c126_p6_0_1: - 60 diff_dist_pair_c126_p6_0_1
 + dist_diff_abs_pair_c126_p6_0_1 >= -57
dist_diff_abs_pair_lower_c127_p7_0_1: - 60 diff_dist_pair_c127_p7_0_1
 + dist_diff_abs_pair_c127_p7_0_1 >= -57
dist_diff_abs_pair_lower_c128_p8_0_1: - 60 diff_dist_pair_c128_p8_0_1
 + dist_diff_abs_pair_c128_p8_0_1 >= -57
dist_diff_abs_pair_lower_c129_p9_0_1: - 60 diff_dist_pair_c129_p9_0_1
 + dist_diff_abs_pair_c129_p9_0_1 >= -57
dist_diff_abs_pair_lower_c130_p10_0_1: - 60 diff_dist_pair_c130_p10_0_1
 + dist_diff_abs_pair_c130_p10_0_1 >= -57
dist_diff_abs_pair_lower_c131_p11_0_1: - 60 diff_dist_pair_c131_p11_0_1
 + dist_diff_abs_pair_c131_p11_0_1 >= -57
dist_diff_abs_pair_lower_c132_p12_0_1: - 60 diff_dist_pair_c132_p12_0_1
 + dist_diff_abs_pair_c132_p12_0_1 >= -57
dist_diff_abs_pair_lower_c133_p13_0_1: - 60 diff_dist_pair_c133_p13_0_1
 + dist_diff_abs_pair_c133_p13_0_1 >= -57
dist_diff_abs_pair_lower_c134_p14_0_1: - 60 diff_dist_pair_c134_p14_0_1
 + dist_diff_abs_pair_c134_p14_0_1 >= -57
dist_diff_abs_pair_lower_c135_p15_0_1: - 60 diff_dist_pair_c135_p15_0_1
 + dist_diff_abs_pair_c135_p15_0_1 >= -57
dist_diff_abs_pair_neg_c120_p0_0_1: dist_diff_abs_pair_c120_p0_0_1
 + dist_diff_pair_c120_p0_0_1 >= 0
dist_diff_abs_pair_neg_c121_p1_0_1: dist_diff_abs_pair_c121_p1_0_1
 + dist_diff_pair_c121_p1_0_1 >= 0
dist_diff_abs_pair_neg_c122_p2_0_1: dist_diff_abs_pair_c122_p2_0_1
 + dist_diff_pair_c122_p2_0_1 >= 0
dist_diff_abs_pair_neg_c123_p3_0_1: dist_diff_abs_pair_c123_p3_0_1
 + dist_diff_pair_c123_p3_0_1 >= 0
dist_diff_abs_pair_neg_c124_p4_0_1: dist_diff_abs_pair_c124_p4_0_1
 + dist_diff_pair_c124_p4_0_1 >= 0
dist_diff_abs_pair_neg_c125_p5_0_1: dist_diff_abs_pair_c125_p5_0_1
 + dist_diff_pair_c125_p5_0_1 >= 0
dist_diff_abs_pair_neg_c126_p6_0_1: dist_diff_abs_pair_c126_p6_0_1
 + dist_diff_pair_c126_p6_0_1 >= 0
dist_diff_abs_pair_neg_c127_p7_0_1: dist_diff_abs_pair_c127_p7_0_1
 + dist_diff_pair_c127_p7_0_1 >= 0
dist_diff_abs_pair_neg_c128_p8_0_1: dist_diff_abs_pair_c128_p8_0_1
 + dist_diff_pair_c128_p8_0_1 >= 0
dist_diff_abs_pair_neg_c129_p9_0_1: dist_diff_abs_pair_c129_p9_0_1
 + dist_diff_pair_c129_p9_0_1 >= 0
dist_diff_abs_pair_neg_c130_p10_0_1: dist_diff_abs_pair_c130_p10_0_1
 + dist_diff_pair_c130_p10_0_1 >= 0
dist_diff_abs_pair_neg_c131_p11_0_1: dist_diff_abs_pair_c131_p11_0_1
 + dist_diff_pair_c131_p11_0_1 >= 0
dist_diff_abs_pair_neg_c132_p12_0_1: dist_diff_abs_pair_c132_p12_0_1
 + dist_diff_pair_c132_p12_0_1 >= 0
dist_diff_abs_pair_neg_c133_p13_0_1: dist_diff_abs_pair_c133_p13_0_1
 + dist_diff_pair_c133_p13_0_1 >= 0
dist_diff_abs_pair_neg_c134_p14_0_1: dist_diff_abs_pair_c134_p14_0_1
 + dist_diff_pair_c134_p14_0_1 >= 0
dist_diff_abs_pair_neg_c135_p15_0_1: dist_diff_abs_pair_c135_p15_0_1
 + dist_diff_pair_c135_p15_0_1 >= 0
dist_diff_abs_pair_pos_c120_p0_0_1: dist_diff_abs_pair_c120_p0_0_1
 - dist_diff_pair_c120_p0_0_1 >= 0
dist_diff_abs_pair_pos_c121_p1_0_1: dist_diff_abs_pair_c121_p1_0_1
 - dist_diff_pair_c121_p1_0_1 >= 0
dist_diff_abs_pair_pos_c122_p2_0_1: dist_diff_abs_pair_c122_p2_0_1
 - dist_diff_pair_c122_p2_0_1 >= 0
dist_diff_abs_pair_pos_c123_p3_0_1: dist_diff_abs_pair_c123_p3_0_1
 - dist_diff_pair_c123_p3_0_1 >= 0
dist_diff_abs_pair_pos_c124_p4_0_1: dist_diff_abs_pair_c124_p4_0_1
 - dist_diff_pair_c124_p4_0_1 >= 0
dist_diff_abs_pair_pos_c125_p5_0_1: dist_diff_abs_pair_c125_p5_0_1
 - dist_diff_pair_c125_p5_0_1 >= 0
dist_diff_abs_pair_pos_c126_p6_0_1: dist_diff_abs_pair_c126_p6_0_1
 - dist_diff_pair_c126_p6_0_1 >= 0
dist_diff_abs_pair_pos_c127_p7_0_1: dist_diff_abs_pair_c127_p7_0_1
 - dist_diff_pair_c127_p7_0_1 >= 0
dist_diff_abs_pair_pos_c128_p8_0_1: dist_diff_abs_pair_c128_p8_0_1
 - dist_diff_pair_c128_p8_0_1 >= 0
dist_diff_abs_pair_pos_c129_p9_0_1: dist_diff_abs_pair_c129_p9_0_1
 - dist_diff_pair_c129_p9_0_1 >= 0
dist_diff_abs_pair_pos_c130_p10_0_1: dist_diff_abs_pair_c130_p10_0_1
 - dist_diff_pair_c130_p10_0_1 >= 0
dist_diff_abs_pair_pos_c131_p11_0_1: dist_diff_abs_pair_c131_p11_0_1
 - dist_diff_pair_c131_p11_0_1 >= 0
dist_diff_abs_pair_pos_c132_p12_0_1: dist_diff_abs_pair_c132_p12_0_1
 - dist_diff_pair_c132_p12_0_1 >= 0
dist_diff_abs_pair_pos_c133_p13_0_1: dist_diff_abs_pair_c133_p13_0_1
 - dist_diff_pair_c133_p13_0_1 >= 0
dist_diff_abs_pair_pos_c134_p14_0_1: dist_diff_abs_pair_c134_p14_0_1
 - dist_diff_pair_c134_p14_0_1 >= 0
dist_diff_abs_pair_pos_c135_p15_0_1: dist_diff_abs_pair_c135_p15_0_1
 - dist_diff_pair_c135_p15_0_1 >= 0
dist_diff_abs_pair_upper_c120_p0_0_1: - 60 diff_dist_pair_c120_p0_0_1
 + dist_diff_abs_pair_c120_p0_0_1 <= 2.97
dist_diff_abs_pair_upper_c121_p1_0_1: - 60 diff_dist_pair_c121_p1_0_1
 + dist_diff_abs_pair_c121_p1_0_1 <= 2.97
dist_diff_abs_pair_upper_c122_p2_0_1: - 60 diff_dist_pair_c122_p2_0_1
 + dist_diff_abs_pair_c122_p2_0_1 <= 2.97
dist_diff_abs_pair_upper_c123_p3_0_1: - 60 diff_dist_pair_c123_p3_0_1
 + dist_diff_abs_pair_c123_p3_0_1 <= 2.97
dist_diff_abs_pair_upper_c124_p4_0_1: - 60 diff_dist_pair_c124_p4_0_1
 + dist_diff_abs_pair_c124_p4_0_1 <= 2.97
dist_diff_abs_pair_upper_c125_p5_0_1: - 60 diff_dist_pair_c125_p5_0_1
 + dist_diff_abs_pair_c125_p5_0_1 <= 2.97
dist_diff_abs_pair_upper_c126_p6_0_1: - 60 diff_dist_pair_c126_p6_0_1
 + dist_diff_abs_pair_c126_p6_0_1 <= 2.97
dist_diff_abs_pair_upper_c127_p7_0_1: - 60 diff_dist_pair_c127_p7_0_1
 + dist_diff_abs_pair_c127_p7_0_1 <= 2.97
dist_diff_abs_pair_upper_c128_p8_0_1: - 60 diff_dist_pair_c128_p8_0_1
 + dist_diff_abs_pair_c128_p8_0_1 <= 2.97
dist_diff_abs_pair_upper_c129_p9_0_1: - 60 diff_dist_pair_c129_p9_0_1
 + dist_diff_abs_pair_c129_p9_0_1 <= 2.97
dist_diff_abs_pair_upper_c130_p10_0_1: - 60 diff_dist_pair_c130_p10_0_1
 + dist_diff_abs_pair_c130_p10_0_1 <= 2.97
dist_diff_abs_pair_upper_c131_p11_0_1: - 60 diff_dist_pair_c131_p11_0_1
 + dist_diff_abs_pair_c131_p11_0_1 <= 2.97
dist_diff_abs_pair_upper_c132_p12_0_1: - 60 diff_dist_pair_c132_p12_0_1
 + dist_diff_abs_pair_c132_p12_0_1 <= 2.97
dist_diff_abs_pair_upper_c133_p13_0_1: - 60 diff_dist_pair_c133_p13_0_1
 + dist_diff_abs_pair_c133_p13_0_1 <= 2.97
dist_diff_abs_pair_upper_c134_p14_0_1: - 60 diff_dist_pair_c134_p14_0_1
 + dist_diff_abs_pair_c134_p14_0_1 <= 2.97
dist_diff_abs_pair_upper_c135_p15_0_1: - 60 diff_dist_pair_c135_p15_0_1
 + dist_diff_abs_pair_c135_p15_0_1 <= 2.97
dist_diff_pair_def_c120_p0_0_1: dist_diff_pair_c120_p0_0_1
 - dx_abs_pair_c120_p0_0_1 - dy_abs_pair_c120_p0_0_1 = -10
dist_diff_pair_def_c121_p1_0_1: dist_diff_pair_c121_p1_0_1
 - dx_abs_pair_c121_p1_0_1 - dy_abs_pair_c121_p1_0_1 = -10
dist_diff_pair_def_c122_p2_0_1: dist_diff_pair_c122_p2_0_1
 - dx_abs_pair_c122_p2_0_1 - dy_abs_pair_c122_p2_0_1 = -10
dist_diff_pair_def_c123_p3_0_1: dist_diff_pair_c123_p3_0_1
 - dx_abs_pair_c123_p3_0_1 - dy_abs_pair_c123_p3_0_1 = -10
dist_diff_pair_def_c124_p4_0_1: dist_diff_pair_c124_p4_0_1
 - dx_abs_pair_c124_p4_0_1 - dy_abs_pair_c124_p4_0_1 = -13
dist_diff_pair_def_c125_p5_0_1: dist_diff_pair_c125_p5_0_1
 - dx_abs_pair_c125_p5_0_1 - dy_abs_pair_c125_p5_0_1 = -13
dist_diff_pair_def_c126_p6_0_1: dist_diff_pair_c126_p6_0_1
 - dx_abs_pair_c126_p6_0_1 - dy_abs_pair_c126_p6_0_1 = -13
dist_diff_pair_def_c127_p7_0_1: dist_diff_pair_c127_p7_0_1
 - dx_abs_pair_c127_p7_0_1 - dy_abs_pair_c127_p7_0_1 = -13
dist_diff_pair_def_c128_p8_0_1: dist_diff_pair_c128_p8_0_1
 - dx_abs_pair_c128_p8_0_1 - dy_abs_pair_c128_p8_0_1 = -13
dist_diff_pair_def_c129_p9_0_1: dist_diff_pair_c129_p9_0_1
 - dx_abs_pair_c129_p9_0_1 - dy_abs_pair_c129_p9_0_1 = -13
dist_diff_pair_def_c130_p10_0_1: dist_diff_pair_c130_p10_0_1
 - dx_abs_pair_c130_p10_0_1 - dy_abs_pair_c130_p10_0_1 = -13
dist_diff_pair_def_c131_p11_0_1: dist_diff_pair_c131_p11_0_1
 - dx_abs_pair_c131_p11_0_1 - dy_abs_pair_c131_p11_0_1 = -13
dist_diff_pair_def_c132_p12_0_1: dist_diff_pair_c132_p12_0_1
 - dx_abs_pair_c132_p12_0_1 - dy_abs_pair_c132_p12_0_1 = -16
dist_diff_pair_def_c133_p13_0_1: dist_diff_pair_c133_p13_0_1
 - dx_abs_pair_c133_p13_0_1 - dy_abs_pair_c133_p13_0_1 = -16
dist_diff_pair_def_c134_p14_0_1: dist_diff_pair_c134_p14_0_1
 - dx_abs_pair_c134_p14_0_1 - dy_abs_pair_c134_p14_0_1 = -16
dist_diff_pair_def_c135_p15_0_1: dist_diff_pair_c135_p15_0_1
 - dx_abs_pair_c135_p15_0_1 - dy_abs_pair_c135_p15_0_1 = -16
dx_abs1_0_1: - cx_0 + cx_1 + dx_abs_0_1 >= 0
dx_abs2_0_1: cx_0 - cx_1 + dx_abs_0_1 >= 0
dx_abs_pair_neg_c120_p0_0_1: cx_0 - cx_1 + dx_abs_pair_c120_p0_0_1 >= 0
dx_abs_pair_neg_c121_p1_0_1: cx_0 - cx_1 + dx_abs_pair_c121_p1_0_1 >= 0
dx_abs_pair_neg_c122_p2_0_1: cx_0 - cx_1 + dx_abs_pair_c122_p2_0_1 >= 0
dx_abs_pair_neg_c123_p3_0_1: cx_0 - cx_1 + dx_abs_pair_c123_p3_0_1 >= 0
dx_abs_pair_neg_c124_p4_0_1: cx_0 - cx_1 + dx_abs_pair_c124_p4_0_1 >= 0
dx_abs_pair_neg_c125_p5_0_1: cx_0 - cx_1 + dx_abs_pair_c125_p5_0_1 >= 0
dx_abs_pair_neg_c126_p6_0_1: cx_0 - cx_1 + dx_abs_pair_c126_p6_0_1 >= 0
dx_abs_pair_neg_c127_p7_0_1: cx_0 - cx_1 + dx_abs_pair_c127_p7_0_1 >= 0
dx_abs_pair_neg_c128_p8_0_1: cx_0 - cx_1 + dx_abs_pair_c128_p8_0_1 >= 0
dx_abs_pair_neg_c129_p9_0_1: cx_0 - cx_1 + dx_abs_pair_c129_p9_0_1 >= 0
dx_abs_pair_neg_c130_p10_0_1: cx_0 - cx_1 + dx_abs_pair_c130_p10_0_1 >= 0
dx_abs_pair_neg_c131_p11_0_1: cx_0 - cx_1 + dx_abs_pair_c131_p11_0_1 >= 0
dx_abs_pair_neg_c132_p12_0_1: cx_0 - cx_1 + dx_abs_pair_c132_p12_0_1 >= 0
dx_abs_pair_neg_c133_p13_0_1: cx_0 - cx_1 + dx_abs_pair_c133_p13_0_1 >= 0
dx_abs_pair_neg_c134_p14_0_1: cx_0 - cx_1 + dx_abs_pair_c134_p14_0_1 >= 0
dx_abs_pair_neg_c135_p15_0_1: cx_0 - cx_1 + dx_abs_pair_c135_p15_0_1 >= 0
dx_abs_pair_pos_c120_p0_0_1: - cx_0 + cx_1 + dx_abs_pair_c120_p0_0_1 >= 0
dx_abs_pair_pos_c121_p1_0_1: - cx_0 + cx_1 + dx_abs_pair_c121_p1_0_1 >= 0
dx_abs_pair_pos_c122_p2_0_1: - cx_0 + cx_1 + dx_abs_pair_c122_p2_0_1 >= 0
dx_abs_pair_pos_c123_p3_0_1: - cx_0 + cx_1 + dx_abs_pair_c123_p3_0_1 >= 0
dx_abs_pair_pos_c124_p4_0_1: - cx_0 + cx_1 + dx_abs_pair_c124_p4_0_1 >= 0
dx_abs_pair_pos_c125_p5_0_1: - cx_0 + cx_1 + dx_abs_pair_c125_p5_0_1 >= 0
dx_abs_pair_pos_c126_p6_0_1: - cx_0 + cx_1 + dx_abs_pair_c126_p6_0_1 >= 0
dx_abs_pair_pos_c127_p7_0_1: - cx_0 + cx_1 + dx_abs_pair_c127_p7_0_1 >= 0
dx_abs_pair_pos_c128_p8_0_1: - cx_0 + cx_1 + dx_abs_pair_c128_p8_0_1 >= 0
dx_abs_pair_pos_c129_p9_0_1: - cx_0 + cx_1 + dx_abs_pair_c129_p9_0_1 >= 0
dx_abs_pair_pos_c130_p10_0_1: - cx_0 + cx_1 + dx_abs_pair_c130_p10_0_1 >= 0
dx_abs_pair_pos_c131_p11_0_1: - cx_0 + cx_1 + dx_abs_pair_c131_p11_0_1 >= 0
dx_abs_pair_pos_c132_p12_0_1: - cx_0 + cx_1 + dx_abs_pair_c132_p12_0_1 >= 0
dx_abs_pair_pos_c133_p13_0_1: - cx_0 + cx_1 + dx_abs_pair_c133_p13_0_1 >= 0
dx_abs_pair_pos_c134_p14_0_1: - cx_0 + cx_1 + dx_abs_pair_c134_p14_0_1 >= 0
dx_abs_pair_pos_c135_p15_0_1: - cx_0 + cx_1 + dx_abs_pair_c135_p15_0_1 >= 0
dy_abs1_0_1: - cy_0 + cy_1 + dy_abs_0_1 >= 0
dy_abs2_0_1: cy_0 - cy_1 + dy_abs_0_1 >= 0
dy_abs_pair_neg_c120_p0_0_1: cy_0 - cy_1 + dy_abs_pair_c120_p0_0_1 >= 0
dy_abs_pair_neg_c121_p1_0_1: cy_0 - cy_1 + dy_abs_pair_c121_p1_0_1 >= 0
dy_abs_pair_neg_c122_p2_0_1: cy_0 - cy_1 + dy_abs_pair_c122_p2_0_1 >= 0
dy_abs_pair_neg_c123_p3_0_1: cy_0 - cy_1 + dy_abs_pair_c123_p3_0_1 >= 0
dy_abs_pair_neg_c124_p4_0_1: cy_0 - cy_1 + dy_abs_pair_c124_p4_0_1 >= 0
dy_abs_pair_neg_c125_p5_0_1: cy_0 - cy_1 + dy_abs_pair_c125_p5_0_1 >= 0
dy_abs_pair_neg_c126_p6_0_1: cy_0 - cy_1 + dy_abs_pair_c126_p6_0_1 >= 0
dy_abs_pair_neg_c127_p7_0_1: cy_0 - cy_1 + dy_abs_pair_c127_p7_0_1 >= 0
dy_abs_pair_neg_c128_p8_0_1: cy_0 - cy_1 + dy_abs_pair_c128_p8_0_1 >= 0
dy_abs_pair_neg_c129_p9_0_1: cy_0 - cy_1 + dy_abs_pair_c129_p9_0_1 >= 0
dy_abs_pair_neg_c130_p10_0_1: cy_0 - cy_1 + dy_abs_pair_c130_p10_0_1 >= 0
dy_abs_pair_neg_c131_p11_0_1: cy_0 - cy_1 + dy_abs_pair_c131_p11_0_1 >= 0
dy_abs_pair_neg_c132_p12_0_1: cy_0 - cy_1 + dy_abs_pair_c132_p12_0_1 >= 0
dy_abs_pair_neg_c133_p13_0_1: cy_0 - cy_1 + dy_abs_pair_c133_p13_0_1 >= 0
dy_abs_pair_neg_c134_p14_0_1: cy_0 - cy_1 + dy_abs_pair_c134_p14_0_1 >= 0
dy_abs_pair_neg_c135_p15_0_1: cy_0 - cy_1 + dy_abs_pair_c135_p15_0_1 >= 0
dy_abs_pair_pos_c120_p0_0_1: - cy_0 + cy_1 + dy_abs_pair_c120_p0_0_1 >= 0
dy_abs_pair_pos_c121_p1_0_1: - cy_0 + cy_1 + dy_abs_pair_c121_p1_0_1 >= 0
dy_abs_pair_pos_c122_p2_0_1: - cy_0 + cy_1 + dy_abs_pair_c122_p2_0_1 >= 0
dy_abs_pair_pos_c123_p3_0_1: - cy_0 + cy_1 + dy_abs_pair_c123_p3_0_1 >= 0
dy_abs_pair_pos_c124_p4_0_1: - cy_0 + cy_1 + dy_abs_pair_c124_p4_0_1 >= 0
dy_abs_pair_pos_c125_p5_0_1: - cy_0 + cy_1 + dy_abs_pair_c125_p5_0_1 >= 0
dy_abs_pair_pos_c126_p6_0_1: - cy_0 + cy_1 + dy_abs_pair_c126_p6_0_1 >= 0
dy_abs_pair_pos_c127_p7_0_1: - cy_0 + cy_1 + dy_abs_pair_c127_p7_0_1 >= 0
dy_abs_pair_pos_c128_p8_0_1: - cy_0 + cy_1 + dy_abs_pair_c128_p8_0_1 >= 0
dy_abs_pair_pos_c129_p9_0_1: - cy_0 + cy_1 + dy_abs_pair_c129_p9_0_1 >= 0
dy_abs_pair_pos_c130_p10_0_1: - cy_0 + cy_1 + dy_abs_pair_c130_p10_0_1 >= 0
dy_abs_pair_pos_c131_p11_0_1: - cy_0 + cy_1 + dy_abs_pair_c131_p11_0_1 >= 0
dy_abs_pair_pos_c132_p12_0_1: - cy_0 + cy_1 + dy_abs_pair_c132_p12_0_1 >= 0
dy_abs_pair_pos_c133_p13_0_1: - cy_0 + cy_1 + dy_abs_pair_c133_p13_0_1 >= 0
dy_abs_pair_pos_c134_p14_0_1: - cy_0 + cy_1 + dy_abs_pair_c134_p14_0_1 >= 0
dy_abs_pair_pos_c135_p15_0_1: - cy_0 + cy_1 + dy_abs_pair_c135_p15_0_1 >= 0
exclude_solution_dist_pair_c120_p0: diff_dist_pair_c120_p0_0_1 >= 1
exclude_solution_dist_pair_c121_p1: diff_dist_pair_c121_p1_0_1 >= 1
exclude_solution_dist_pair_c122_p2: diff_dist_pair_c122_p2_0_1 >= 1
exclude_solution_dist_pair_c123_p3: diff_dist_pair_c123_p3_0_1 >= 1
exclude_solution_dist_pair_c124_p4: diff_dist_pair_c124_p4_0_1 >= 1
exclude_solution_dist_pair_c125_p5: diff_dist_pair_c125_p5_0_1 >= 1
exclude_solution_dist_pair_c126_p6: diff_dist_pair_c126_p6_0_1 >= 1
exclude_solution_dist_pair_c127_p7: diff_dist_pair_c127_p7_0_1 >= 1
exclude_solution_dist_pair_c128_p8: diff_dist_pair_c128_p8_0_1 >= 1
exclude_solution_dist_pair_c129_p9: diff_dist_pair_c129_p9_0_1 >= 1
exclude_solution_dist_pair_c130_p10: diff_dist_pair_c130_p10_0_1 >= 1
exclude_solution_dist_pair_c131_p11: diff_dist_pair_c131_p11_0_1 >= 1
exclude_solution_dist_pair_c132_p12: diff_dist_pair_c132_p12_0_1 >= 1
exclude_solution_dist_pair_c133_p13: diff_dist_pair_c133_p13_0_1 >= 1
exclude_solution_dist_pair_c134_p14: diff_dist_pair_c134_p14_0_1 >= 1
exclude_solution_dist_pair_c135_p15: diff_dist_pair_c135_p15_0_1 >= 1
exclude_solution_pos_c120_p0: diff_pos_c120_p0_1 >= 1
exclude_solution_pos_c121_p1: diff_pos_c121_p1_1 >= 1
exclude_solution_pos_c122_p2: diff_pos_c122_p2_1 >= 1
exclude_solution_pos_c123_p3: diff_pos_c123_p3_1 >= 1
exclude_solution_pos_c124_p4: diff_pos_c124_p4_1 >= 1
exclude_solution_pos_c125_p5: diff_pos_c125_p5_1 >= 1
exclude_solution_pos_c126_p6: diff_pos_c126_p6_1 >= 1
exclude_solution_pos_c127_p7: diff_pos_c127_p7_1 >= 1
exclude_solution_pos_c128_p8: diff_pos_c128_p8_1 >= 1
exclude_solution_pos_c129_p9: diff_pos_c129_p9_1 >= 1
exclude_solution_pos_c130_p10: diff_pos_c130_p10_1 >= 1
exclude_solution_pos_c131_p11: diff_pos_c131_p11_1 >= 1
exclude_solution_pos_c132_p12: diff_pos_c132_p12_1 >= 1
exclude_solution_pos_c133_p13: diff_pos_c133_p13_1 >= 1
exclude_solution_pos_c134_p14: diff_pos_c134_p14_1 >= 1
exclude_solution_pos_c135_p15: diff_pos_c135_p15_1 >= 1
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
x_diff_from_minus_c120_p0_1: x_diff_c120_p0_1 - x_diff_minus_c120_p0_1 >= 0
x_diff_from_minus_c121_p1_1: x_diff_c121_p1_1 - x_diff_minus_c121_p1_1 >= 0
x_diff_from_minus_c122_p2_1: x_diff_c122_p2_1 - x_diff_minus_c122_p2_1 >= 0
x_diff_from_minus_c123_p3_1: x_diff_c123_p3_1 - x_diff_minus_c123_p3_1 >= 0
x_diff_from_minus_c124_p4_1: x_diff_c124_p4_1 - x_diff_minus_c124_p4_1 >= 0
x_diff_from_minus_c125_p5_1: x_diff_c125_p5_1 - x_diff_minus_c125_p5_1 >= 0
x_diff_from_minus_c126_p6_1: x_diff_c126_p6_1 - x_diff_minus_c126_p6_1 >= 0
x_diff_from_minus_c127_p7_1: x_diff_c127_p7_1 - x_diff_minus_c127_p7_1 >= 0
x_diff_from_minus_c128_p8_1: x_diff_c128_p8_1 - x_diff_minus_c128_p8_1 >= 0
x_diff_from_minus_c129_p9_1: x_diff_c129_p9_1 - x_diff_minus_c129_p9_1 >= 0
x_diff_from_minus_c130_p10_1: x_diff_c130_p10_1 - x_diff_minus_c130_p10_1 >= 0
x_diff_from_minus_c131_p11_1: x_diff_c131_p11_1 - x_diff_minus_c131_p11_1 >= 0
x_diff_from_minus_c132_p12_1: x_diff_c132_p12_1 - x_diff_minus_c132_p12_1 >= 0
x_diff_from_minus_c133_p13_1: x_diff_c133_p13_1 - x_diff_minus_c133_p13_1 >= 0
x_diff_from_minus_c134_p14_1: x_diff_c134_p14_1 - x_diff_minus_c134_p14_1 >= 0
x_diff_from_minus_c135_p15_1: x_diff_c135_p15_1 - x_diff_minus_c135_p15_1 >= 0
x_diff_from_plus_c120_p0_1: x_diff_c120_p0_1 - x_diff_plus_c120_p0_1 >= 0
x_diff_from_plus_c121_p1_1: x_diff_c121_p1_1 - x_diff_plus_c121_p1_1 >= 0
x_diff_from_plus_c122_p2_1: x_diff_c122_p2_1 - x_diff_plus_c122_p2_1 >= 0
x_diff_from_plus_c123_p3_1: x_diff_c123_p3_1 - x_diff_plus_c123_p3_1 >= 0
x_diff_from_plus_c124_p4_1: x_diff_c124_p4_1 - x_diff_plus_c124_p4_1 >= 0
x_diff_from_plus_c125_p5_1: x_diff_c125_p5_1 - x_diff_plus_c125_p5_1 >= 0
x_diff_from_plus_c126_p6_1: x_diff_c126_p6_1 - x_diff_plus_c126_p6_1 >= 0
x_diff_from_plus_c127_p7_1: x_diff_c127_p7_1 - x_diff_plus_c127_p7_1 >= 0
x_diff_from_plus_c128_p8_1: x_diff_c128_p8_1 - x_diff_plus_c128_p8_1 >= 0
x_diff_from_plus_c129_p9_1: x_diff_c129_p9_1 - x_diff_plus_c129_p9_1 >= 0
x_diff_from_plus_c130_p10_1: x_diff_c130_p10_1 - x_diff_plus_c130_p10_1 >= 0
x_diff_from_plus_c131_p11_1: x_diff_c131_p11_1 - x_diff_plus_c131_p11_1 >= 0
x_diff_from_plus_c132_p12_1: x_diff_c132_p12_1 - x_diff_plus_c132_p12_1 >= 0
x_diff_from_plus_c133_p13_1: x_diff_c133_p13_1 - x_diff_plus_c133_p13_1 >= 0
x_diff_from_plus_c134_p14_1: x_diff_c134_p14_1 - x_diff_plus_c134_p14_1 >= 0
x_diff_from_plus_c135_p15_1: x_diff_c135_p15_1 - x_diff_plus_c135_p15_1 >= 0
x_diff_minus_lower_c120_p0_1: x_1 + 60 x_diff_minus_c120_p0_1 <= 67
x_diff_minus_lower_c121_p1_1: x_1 + 60 x_diff_minus_c121_p1_1 <= 57
x_diff_minus_lower_c122_p2_1: x_1 + 60 x_diff_minus_c122_p2_1 <= 67
x_diff_minus_lower_c123_p3_1: x_1 + 60 x_diff_minus_c123_p3_1 <= 77
x_diff_minus_lower_c124_p4_1: x_1 + 60 x_diff_minus_c124_p4_1 <= 70
x_diff_minus_lower_c125_p5_1: x_1 + 60 x_diff_minus_c125_p5_1 <= 64
x_diff_minus_lower_c126_p6_1: x_1 + 60 x_diff_minus_c126_p6_1 <= 70
x_diff_minus_lower_c127_p7_1: x_1 + 60 x_diff_minus_c127_p7_1 <= 77
x_diff_minus_lower_c128_p8_1: x_1 + 60 x_diff_minus_c128_p8_1 <= 77
x_diff_minus_lower_c129_p9_1: x_1 + 60 x_diff_minus_c129_p9_1 <= 57
x_diff_minus_lower_c130_p10_1: x_1 + 60 x_diff_minus_c130_p10_1 <= 57
x_diff_minus_lower_c131_p11_1: x_1 + 60 x_diff_minus_c131_p11_1 <= 64
x_diff_minus_lower_c132_p12_1: x_1 + 60 x_diff_minus_c132_p12_1 <= 77
x_diff_minus_lower_c133_p13_1: x_1 + 60 x_diff_minus_c133_p13_1 <= 57
x_diff_minus_lower_c134_p14_1: x_1 + 60 x_diff_minus_c134_p14_1 <= 73
x_diff_minus_lower_c135_p15_1: x_1 + 60 x_diff_minus_c135_p15_1 <= 73
x_diff_minus_upper_c120_p0_1: x_1 + 60 x_diff_minus_c120_p0_1 >= 7.001
x_diff_minus_upper_c121_p1_1: x_1 + 60 x_diff_minus_c121_p1_1 >= -2.999
x_diff_minus_upper_c122_p2_1: x_1 + 60 x_diff_minus_c122_p2_1 >= 7.001
x_diff_minus_upper_c123_p3_1: x_1 + 60 x_diff_minus_c123_p3_1 >= 17.001
x_diff_minus_upper_c124_p4_1: x_1 + 60 x_diff_minus_c124_p4_1 >= 10.001
x_diff_minus_upper_c125_p5_1: x_1 + 60 x_diff_minus_c125_p5_1 >= 4.001
x_diff_minus_upper_c126_p6_1: x_1 + 60 x_diff_minus_c126_p6_1 >= 10.001
x_diff_minus_upper_c127_p7_1: x_1 + 60 x_diff_minus_c127_p7_1 >= 17.001
x_diff_minus_upper_c128_p8_1: x_1 + 60 x_diff_minus_c128_p8_1 >= 17.001
x_diff_minus_upper_c129_p9_1: x_1 + 60 x_diff_minus_c129_p9_1 >= -2.999
x_diff_minus_upper_c130_p10_1: x_1 + 60 x_diff_minus_c130_p10_1 >= -2.999
x_diff_minus_upper_c131_p11_1: x_1 + 60 x_diff_minus_c131_p11_1 >= 4.001
x_diff_minus_upper_c132_p12_1: x_1 + 60 x_diff_minus_c132_p12_1 >= 17.001
x_diff_minus_upper_c133_p13_1: x_1 + 60 x_diff_minus_c133_p13_1 >= -2.999
x_diff_minus_upper_c134_p14_1: x_1 + 60 x_diff_minus_c134_p14_1 >= 13.001
x_diff_minus_upper_c135_p15_1: x_1 + 60 x_diff_minus_c135_p15_1 >= 13.001
x_diff_plus_lower_c120_p0_1: x_1 - 60 x_diff_plus_c120_p0_1 >= -47
x_diff_plus_lower_c121_p1_1: x_1 - 60 x_diff_plus_c121_p1_1 >= -57
x_diff_plus_lower_c122_p2_1: x_1 - 60 x_diff_plus_c122_p2_1 >= -47
x_diff_plus_lower_c123_p3_1: x_1 - 60 x_diff_plus_c123_p3_1 >= -37
x_diff_plus_lower_c124_p4_1: x_1 - 60 x_diff_plus_c124_p4_1 >= -44
x_diff_plus_lower_c125_p5_1: x_1 - 60 x_diff_plus_c125_p5_1 >= -50
x_diff_plus_lower_c126_p6_1: x_1 - 60 x_diff_plus_c126_p6_1 >= -44
x_diff_plus_lower_c127_p7_1: x_1 - 60 x_diff_plus_c127_p7_1 >= -37
x_diff_plus_lower_c128_p8_1: x_1 - 60 x_diff_plus_c128_p8_1 >= -37
x_diff_plus_lower_c129_p9_1: x_1 - 60 x_diff_plus_c129_p9_1 >= -57
x_diff_plus_lower_c130_p10_1: x_1 - 60 x_diff_plus_c130_p10_1 >= -57
x_diff_plus_lower_c131_p11_1: x_1 - 60 x_diff_plus_c131_p11_1 >= -50
x_diff_plus_lower_c132_p12_1: x_1 - 60 x_diff_plus_c132_p12_1 >= -37
x_diff_plus_lower_c133_p13_1: x_1 - 60 x_diff_plus_c133_p13_1 >= -57
x_diff_plus_lower_c134_p14_1: x_1 - 60 x_diff_plus_c134_p14_1 >= -41
x_diff_plus_lower_c135_p15_1: x_1 - 60 x_diff_plus_c135_p15_1 >= -41
x_diff_plus_upper_c120_p0_1: x_1 - 60 x_diff_plus_c120_p0_1 <= 12.999
x_diff_plus_upper_c121_p1_1: x_1 - 60 x_diff_plus_c121_p1_1 <= 2.999
x_diff_plus_upper_c122_p2_1: x_1 - 60 x_diff_plus_c122_p2_1 <= 12.999
x_diff_plus_upper_c123_p3_1: x_1 - 60 x_diff_plus_c123_p3_1 <= 22.999
x_diff_plus_upper_c124_p4_1: x_1 - 60 x_diff_plus_c124_p4_1 <= 15.999
x_diff_plus_upper_c125_p5_1: x_1 - 60 x_diff_plus_c125_p5_1 <= 9.999
x_diff_plus_upper_c126_p6_1: x_1 - 60 x_diff_plus_c126_p6_1 <= 15.999
x_diff_plus_upper_c127_p7_1: x_1 - 60 x_diff_plus_c127_p7_1 <= 22.999
x_diff_plus_upper_c128_p8_1: x_1 - 60 x_diff_plus_c128_p8_1 <= 22.999
x_diff_plus_upper_c129_p9_1: x_1 - 60 x_diff_plus_c129_p9_1 <= 2.999
x_diff_plus_upper_c130_p10_1: x_1 - 60 x_diff_plus_c130_p10_1 <= 2.999
x_diff_plus_upper_c131_p11_1: x_1 - 60 x_diff_plus_c131_p11_1 <= 9.999
x_diff_plus_upper_c132_p12_1: x_1 - 60 x_diff_plus_c132_p12_1 <= 22.999
x_diff_plus_upper_c133_p13_1: x_1 - 60 x_diff_plus_c133_p13_1 <= 2.999
x_diff_plus_upper_c134_p14_1: x_1 - 60 x_diff_plus_c134_p14_1 <= 18.999
x_diff_plus_upper_c135_p15_1: x_1 - 60 x_diff_plus_c135_p15_1 <= 18.999
x_diff_upper_c120_p0_1: x_diff_c120_p0_1 - x_diff_minus_c120_p0_1
 - x_diff_plus_c120_p0_1 <= 0
x_diff_upper_c121_p1_1: x_diff_c121_p1_1 - x_diff_minus_c121_p1_1
 - x_diff_plus_c121_p1_1 <= 0
x_diff_upper_c122_p2_1: x_diff_c122_p2_1 - x_diff_minus_c122_p2_1
 - x_diff_plus_c122_p2_1 <= 0
x_diff_upper_c123_p3_1: x_diff_c123_p3_1 - x_diff_minus_c123_p3_1
 - x_diff_plus_c123_p3_1 <= 0
x_diff_upper_c124_p4_1: x_diff_c124_p4_1 - x_diff_minus_c124_p4_1
 - x_diff_plus_c124_p4_1 <= 0
x_diff_upper_c125_p5_1: x_diff_c125_p5_1 - x_diff_minus_c125_p5_1
 - x_diff_plus_c125_p5_1 <= 0
x_diff_upper_c126_p6_1: x_diff_c126_p6_1 - x_diff_minus_c126_p6_1
 - x_diff_plus_c126_p6_1 <= 0
x_diff_upper_c127_p7_1: x_diff_c127_p7_1 - x_diff_minus_c127_p7_1
 - x_diff_plus_c127_p7_1 <= 0
x_diff_upper_c128_p8_1: x_diff_c128_p8_1 - x_diff_minus_c128_p8_1
 - x_diff_plus_c128_p8_1 <= 0
x_diff_upper_c129_p9_1: x_diff_c129_p9_1 - x_diff_minus_c129_p9_1
 - x_diff_plus_c129_p9_1 <= 0
x_diff_upper_c130_p10_1: x_diff_c130_p10_1 - x_diff_minus_c130_p10_1
 - x_diff_plus_c130_p10_1 <= 0
x_diff_upper_c131_p11_1: x_diff_c131_p11_1 - x_diff_minus_c131_p11_1
 - x_diff_plus_c131_p11_1 <= 0
x_diff_upper_c132_p12_1: x_diff_c132_p12_1 - x_diff_minus_c132_p12_1
 - x_diff_plus_c132_p12_1 <= 0
x_diff_upper_c133_p13_1: x_diff_c133_p13_1 - x_diff_minus_c133_p13_1
 - x_diff_plus_c133_p13_1 <= 0
x_diff_upper_c134_p14_1: x_diff_c134_p14_1 - x_diff_minus_c134_p14_1
 - x_diff_plus_c134_p14_1 <= 0
x_diff_upper_c135_p15_1: x_diff_c135_p15_1 - x_diff_minus_c135_p15_1
 - x_diff_plus_c135_p15_1 <= 0
x_grid_to_real_0: x_0 - 0.5 x_grid_0 = 0
x_grid_to_real_1: x_1 - 0.5 x_grid_1 = 0
y_diff_from_minus_c120_p0_1: y_diff_c120_p0_1 - y_diff_minus_c120_p0_1 >= 0
y_diff_from_minus_c121_p1_1: y_diff_c121_p1_1 - y_diff_minus_c121_p1_1 >= 0
y_diff_from_minus_c122_p2_1: y_diff_c122_p2_1 - y_diff_minus_c122_p2_1 >= 0
y_diff_from_minus_c123_p3_1: y_diff_c123_p3_1 - y_diff_minus_c123_p3_1 >= 0
y_diff_from_minus_c124_p4_1: y_diff_c124_p4_1 - y_diff_minus_c124_p4_1 >= 0
y_diff_from_minus_c125_p5_1: y_diff_c125_p5_1 - y_diff_minus_c125_p5_1 >= 0
y_diff_from_minus_c126_p6_1: y_diff_c126_p6_1 - y_diff_minus_c126_p6_1 >= 0
y_diff_from_minus_c127_p7_1: y_diff_c127_p7_1 - y_diff_minus_c127_p7_1 >= 0
y_diff_from_minus_c128_p8_1: y_diff_c128_p8_1 - y_diff_minus_c128_p8_1 >= 0
y_diff_from_minus_c129_p9_1: y_diff_c129_p9_1 - y_diff_minus_c129_p9_1 >= 0
y_diff_from_minus_c130_p10_1: y_diff_c130_p10_1 - y_diff_minus_c130_p10_1 >= 0
y_diff_from_minus_c131_p11_1: y_diff_c131_p11_1 - y_diff_minus_c131_p11_1 >= 0
y_diff_from_minus_c132_p12_1: y_diff_c132_p12_1 - y_diff_minus_c132_p12_1 >= 0
y_diff_from_minus_c133_p13_1: y_diff_c133_p13_1 - y_diff_minus_c133_p13_1 >= 0
y_diff_from_minus_c134_p14_1: y_diff_c134_p14_1 - y_diff_minus_c134_p14_1 >= 0
y_diff_from_minus_c135_p15_1: y_diff_c135_p15_1 - y_diff_minus_c135_p15_1 >= 0
y_diff_from_plus_c120_p0_1: y_diff_c120_p0_1 - y_diff_plus_c120_p0_1 >= 0
y_diff_from_plus_c121_p1_1: y_diff_c121_p1_1 - y_diff_plus_c121_p1_1 >= 0
y_diff_from_plus_c122_p2_1: y_diff_c122_p2_1 - y_diff_plus_c122_p2_1 >= 0
y_diff_from_plus_c123_p3_1: y_diff_c123_p3_1 - y_diff_plus_c123_p3_1 >= 0
y_diff_from_plus_c124_p4_1: y_diff_c124_p4_1 - y_diff_plus_c124_p4_1 >= 0
y_diff_from_plus_c125_p5_1: y_diff_c125_p5_1 - y_diff_plus_c125_p5_1 >= 0
y_diff_from_plus_c126_p6_1: y_diff_c126_p6_1 - y_diff_plus_c126_p6_1 >= 0
y_diff_from_plus_c127_p7_1: y_diff_c127_p7_1 - y_diff_plus_c127_p7_1 >= 0
y_diff_from_plus_c128_p8_1: y_diff_c128_p8_1 - y_diff_plus_c128_p8_1 >= 0
y_diff_from_plus_c129_p9_1: y_diff_c129_p9_1 - y_diff_plus_c129_p9_1 >= 0
y_diff_from_plus_c130_p10_1: y_diff_c130_p10_1 - y_diff_plus_c130_p10_1 >= 0
y_diff_from_plus_c131_p11_1: y_diff_c131_p11_1 - y_diff_plus_c131_p11_1 >= 0
y_diff_from_plus_c132_p12_1: y_diff_c132_p12_1 - y_diff_plus_c132_p12_1 >= 0
y_diff_from_plus_c133_p13_1: y_diff_c133_p13_1 - y_diff_plus_c133_p13_1 >= 0
y_diff_from_plus_c134_p14_1: y_diff_c134_p14_1 - y_diff_plus_c134_p14_1 >= 0
y_diff_from_plus_c135_p15_1: y_diff_c135_p15_1 - y_diff_plus_c135_p15_1 >= 0
y_diff_minus_lower_c120_p0_1: y_1 + 60 y_diff_minus_c120_p0_1 <= 77
y_diff_minus_lower_c121_p1_1: y_1 + 60 y_diff_minus_c121_p1_1 <= 67
y_diff_minus_lower_c122_p2_1: y_1 + 60 y_diff_minus_c122_p2_1 <= 57
y_diff_minus_lower_c123_p3_1: y_1 + 60 y_diff_minus_c123_p3_1 <= 67
y_diff_minus_lower_c124_p4_1: y_1 + 60 y_diff_minus_c124_p4_1 <= 77
y_diff_minus_lower_c125_p5_1: y_1 + 60 y_diff_minus_c125_p5_1 <= 77
y_diff_minus_lower_c126_p6_1: y_1 + 60 y_diff_minus_c126_p6_1 <= 57
y_diff_minus_lower_c127_p7_1: y_1 + 60 y_diff_minus_c127_p7_1 <= 64
y_diff_minus_lower_c128_p8_1: y_1 + 60 y_diff_minus_c128_p8_1 <= 70
y_diff_minus_lower_c129_p9_1: y_1 + 60 y_diff_minus_c129_p9_1 <= 70
y_diff_minus_lower_c130_p10_1: y_1 + 60 y_diff_minus_c130_p10_1 <= 64
y_diff_minus_lower_c131_p11_1: y_1 + 60 y_diff_minus_c131_p11_1 <= 57
y_diff_minus_lower_c132_p12_1: y_1 + 60 y_diff_minus_c132_p12_1 <= 61
y_diff_minus_lower_c133_p13_1: y_1 + 60 y_diff_minus_c133_p13_1 <= 73
y_diff_minus_lower_c134_p14_1: y_1 + 60 y_diff_minus_c134_p14_1 <= 57
y_diff_minus_lower_c135_p15_1: y_1 + 60 y_diff_minus_c135_p15_1 <= 77
y_diff_minus_upper_c120_p0_1: y_1 + 60 y_diff_minus_c120_p0_1 >= 17.001
y_diff_minus_upper_c121_p1_1: y_1 + 60 y_diff_minus_c121_p1_1 >= 7.001
y_diff_minus_upper_c122_p2_1: y_1 + 60 y_diff_minus_c122_p2_1 >= -2.999
y_diff_minus_upper_c123_p3_1: y_1 + 60 y_diff_minus_c123_p3_1 >= 7.001
y_diff_minus_upper_c124_p4_1: y_1 + 60 y_diff_minus_c124_p4_1 >= 17.001
y_diff_minus_upper_c125_p5_1: y_1 + 60 y_diff_minus_c125_p5_1 >= 17.001
y_diff_minus_upper_c126_p6_1: y_1 + 60 y_diff_minus_c126_p6_1 >= -2.999
y_diff_minus_upper_c127_p7_1: y_1 + 60 y_diff_minus_c127_p7_1 >= 4.001
y_diff_minus_upper_c128_p8_1: y_1 + 60 y_diff_minus_c128_p8_1 >= 10.001
y_diff_minus_upper_c129_p9_1: y_1 + 60 y_diff_minus_c129_p9_1 >= 10.001
y_diff_minus_upper_c130_p10_1: y_1 + 60 y_diff_minus_c130_p10_1 >= 4.001
y_diff_minus_upper_c131_p11_1: y_1 + 60 y_diff_minus_c131_p11_1 >= -2.999
y_diff_minus_upper_c132_p12_1: y_1 + 60 y_diff_minus_c132_p12_1 >= 1.001
y_diff_minus_upper_c133_p13_1: y_1 + 60 y_diff_minus_c133_p13_1 >= 13.001
y_diff_minus_upper_c134_p14_1: y_1 + 60 y_diff_minus_c134_p14_1 >= -2.999
y_diff_minus_upper_c135_p15_1: y_1 + 60 y_diff_minus_c135_p15_1 >= 17.001
y_diff_plus_lower_c120_p0_1: y_1 - 60 y_diff_plus_c120_p0_1 >= -37
y_diff_plus_lower_c121_p1_1: y_1 - 60 y_diff_plus_c121_p1_1 >= -47
y_diff_plus_lower_c122_p2_1: y_1 - 60 y_diff_plus_c122_p2_1 >= -57
y_diff_plus_lower_c123_p3_1: y_1 - 60 y_diff_plus_c123_p3_1 >= -47
y_diff_plus_lower_c124_p4_1: y_1 - 60 y_diff_plus_c124_p4_1 >= -37
y_diff_plus_lower_c125_p5_1: y_1 - 60 y_diff_plus_c125_p5_1 >= -37
y_diff_plus_lower_c126_p6_1: y_1 - 60 y_diff_plus_c126_p6_1 >= -57
y_diff_plus_lower_c127_p7_1: y_1 - 60 y_diff_plus_c127_p7_1 >= -50
y_diff_plus_lower_c128_p8_1: y_1 - 60 y_diff_plus_c128_p8_1 >= -44
y_diff_plus_lower_c129_p9_1: y_1 - 60 y_diff_plus_c129_p9_1 >= -44
y_diff_plus_lower_c130_p10_1: y_1 - 60 y_diff_plus_c130_p10_1 >= -50
y_diff_plus_lower_c131_p11_1: y_1 - 60 y_diff_plus_c131_p11_1 >= -57
y_diff_plus_lower_c132_p12_1: y_1 - 60 y_diff_plus_c132_p12_1 >= -53
y_diff_plus_lower_c133_p13_1: y_1 - 60 y_diff_plus_c133_p13_1 >= -41
y_diff_plus_lower_c134_p14_1: y_1 - 60 y_diff_plus_c134_p14_1 >= -57
y_diff_plus_lower_c135_p15_1: y_1 - 60 y_diff_plus_c135_p15_1 >= -37
y_diff_plus_upper_c120_p0_1: y_1 - 60 y_diff_plus_c120_p0_1 <= 22.999
y_diff_plus_upper_c121_p1_1: y_1 - 60 y_diff_plus_c121_p1_1 <= 12.999
y_diff_plus_upper_c122_p2_1: y_1 - 60 y_diff_plus_c122_p2_1 <= 2.999
y_diff_plus_upper_c123_p3_1: y_1 - 60 y_diff_plus_c123_p3_1 <= 12.999
y_diff_plus_upper_c124_p4_1: y_1 - 60 y_diff_plus_c124_p4_1 <= 22.999
y_diff_plus_upper_c125_p5_1: y_1 - 60 y_diff_plus_c125_p5_1 <= 22.999
y_diff_plus_upper_c126_p6_1: y_1 - 60 y_diff_plus_c126_p6_1 <= 2.999
y_diff_plus_upper_c127_p7_1: y_1 - 60 y_diff_plus_c127_p7_1 <= 9.999
y_diff_plus_upper_c128_p8_1: y_1 - 60 y_diff_plus_c128_p8_1 <= 15.999
y_diff_plus_upper_c129_p9_1: y_1 - 60 y_diff_plus_c129_p9_1 <= 15.999
y_diff_plus_upper_c130_p10_1: y_1 - 60 y_diff_plus_c130_p10_1 <= 9.999
y_diff_plus_upper_c131_p11_1: y_1 - 60 y_diff_plus_c131_p11_1 <= 2.999
y_diff_plus_upper_c132_p12_1: y_1 - 60 y_diff_plus_c132_p12_1 <= 6.999
y_diff_plus_upper_c133_p13_1: y_1 - 60 y_diff_plus_c133_p13_1 <= 18.999
y_diff_plus_upper_c134_p14_1: y_1 - 60 y_diff_plus_c134_p14_1 <= 2.999
y_diff_plus_upper_c135_p15_1: y_1 - 60 y_diff_plus_c135_p15_1 <= 22.999
y_diff_upper_c120_p0_1: y_diff_c120_p0_1 - y_diff_minus_c120_p0_1
 - y_diff_plus_c120_p0_1 <= 0
y_diff_upper_c121_p1_1: y_diff_c121_p1_1 - y_diff_minus_c121_p1_1
 - y_diff_plus_c121_p1_1 <= 0
y_diff_upper_c122_p2_1: y_diff_c122_p2_1 - y_diff_minus_c122_p2_1
 - y_diff_plus_c122_p2_1 <= 0
y_diff_upper_c123_p3_1: y_diff_c123_p3_1 - y_diff_minus_c123_p3_1
 - y_diff_plus_c123_p3_1 <= 0
y_diff_upper_c124_p4_1: y_diff_c124_p4_1 - y_diff_minus_c124_p4_1
 - y_diff_plus_c124_p4_1 <= 0
y_diff_upper_c125_p5_1: y_diff_c125_p5_1 - y_diff_minus_c125_p5_1
 - y_diff_plus_c125_p5_1 <= 0
y_diff_upper_c126_p6_1: y_diff_c126_p6_1 - y_diff_minus_c126_p6_1
 - y_diff_plus_c126_p6_1 <= 0
y_diff_upper_c127_p7_1: y_diff_c127_p7_1 - y_diff_minus_c127_p7_1
 - y_diff_plus_c127_p7_1 <= 0
y_diff_upper_c128_p8_1: y_diff_c128_p8_1 - y_diff_minus_c128_p8_1
 - y_diff_plus_c128_p8_1 <= 0
y_diff_upper_c129_p9_1: y_diff_c129_p9_1 - y_diff_minus_c129_p9_1
 - y_diff_plus_c129_p9_1 <= 0
y_diff_upper_c130_p10_1: y_diff_c130_p10_1 - y_diff_minus_c130_p10_1
 - y_diff_plus_c130_p10_1 <= 0
y_diff_upper_c131_p11_1: y_diff_c131_p11_1 - y_diff_minus_c131_p11_1
 - y_diff_plus_c131_p11_1 <= 0
y_diff_upper_c132_p12_1: y_diff_c132_p12_1 - y_diff_minus_c132_p12_1
 - y_diff_plus_c132_p12_1 <= 0
y_diff_upper_c133_p13_1: y_diff_c133_p13_1 - y_diff_minus_c133_p13_1
 - y_diff_plus_c133_p13_1 <= 0
y_diff_upper_c134_p14_1: y_diff_c134_p14_1 - y_diff_minus_c134_p14_1
 - y_diff_plus_c134_p14_1 <= 0
y_diff_upper_c135_p15_1: y_diff_c135_p15_1 - y_diff_minus_c135_p15_1
 - y_diff_plus_c135_p15_1 <= 0
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
 dist_diff_abs_pair_c120_p0_0_1 <= 60
 dist_diff_abs_pair_c121_p1_0_1 <= 60
 dist_diff_abs_pair_c122_p2_0_1 <= 60
 dist_diff_abs_pair_c123_p3_0_1 <= 60
 dist_diff_abs_pair_c124_p4_0_1 <= 60
 dist_diff_abs_pair_c125_p5_0_1 <= 60
 dist_diff_abs_pair_c126_p6_0_1 <= 60
 dist_diff_abs_pair_c127_p7_0_1 <= 60
 dist_diff_abs_pair_c128_p8_0_1 <= 60
 dist_diff_abs_pair_c129_p9_0_1 <= 60
 dist_diff_abs_pair_c130_p10_0_1 <= 60
 dist_diff_abs_pair_c131_p11_0_1 <= 60
 dist_diff_abs_pair_c132_p12_0_1 <= 60
 dist_diff_abs_pair_c133_p13_0_1 <= 60
 dist_diff_abs_pair_c134_p14_0_1 <= 60
 dist_diff_abs_pair_c135_p15_0_1 <= 60
 -60 <= dist_diff_pair_c120_p0_0_1 <= 60
 -60 <= dist_diff_pair_c121_p1_0_1 <= 60
 -60 <= dist_diff_pair_c122_p2_0_1 <= 60
 -60 <= dist_diff_pair_c123_p3_0_1 <= 60
 -60 <= dist_diff_pair_c124_p4_0_1 <= 60
 -60 <= dist_diff_pair_c125_p5_0_1 <= 60
 -60 <= dist_diff_pair_c126_p6_0_1 <= 60
 -60 <= dist_diff_pair_c127_p7_0_1 <= 60
 -60 <= dist_diff_pair_c128_p8_0_1 <= 60
 -60 <= dist_diff_pair_c129_p9_0_1 <= 60
 -60 <= dist_diff_pair_c130_p10_0_1 <= 60
 -60 <= dist_diff_pair_c131_p11_0_1 <= 60
 -60 <= dist_diff_pair_c132_p12_0_1 <= 60
 -60 <= dist_diff_pair_c133_p13_0_1 <= 60
 -60 <= dist_diff_pair_c134_p14_0_1 <= 60
 -60 <= dist_diff_pair_c135_p15_0_1 <= 60
 dx_abs_pair_c120_p0_0_1 <= 30
 dx_abs_pair_c121_p1_0_1 <= 30
 dx_abs_pair_c122_p2_0_1 <= 30
 dx_abs_pair_c123_p3_0_1 <= 30
 dx_abs_pair_c124_p4_0_1 <= 30
 dx_abs_pair_c125_p5_0_1 <= 30
 dx_abs_pair_c126_p6_0_1 <= 30
 dx_abs_pair_c127_p7_0_1 <= 30
 dx_abs_pair_c128_p8_0_1 <= 30
 dx_abs_pair_c129_p9_0_1 <= 30
 dx_abs_pair_c130_p10_0_1 <= 30
 dx_abs_pair_c131_p11_0_1 <= 30
 dx_abs_pair_c132_p12_0_1 <= 30
 dx_abs_pair_c133_p13_0_1 <= 30
 dx_abs_pair_c134_p14_0_1 <= 30
 dx_abs_pair_c135_p15_0_1 <= 30
 dy_abs_pair_c120_p0_0_1 <= 30
 dy_abs_pair_c121_p1_0_1 <= 30
 dy_abs_pair_c122_p2_0_1 <= 30
 dy_abs_pair_c123_p3_0_1 <= 30
 dy_abs_pair_c124_p4_0_1 <= 30
 dy_abs_pair_c125_p5_0_1 <= 30
 dy_abs_pair_c126_p6_0_1 <= 30
 dy_abs_pair_c127_p7_0_1 <= 30
 dy_abs_pair_c128_p8_0_1 <= 30
 dy_abs_pair_c129_p9_0_1 <= 30
 dy_abs_pair_c130_p10_0_1 <= 30
 dy_abs_pair_c131_p11_0_1 <= 30
 dy_abs_pair_c132_p12_0_1 <= 30
 dy_abs_pair_c133_p13_0_1 <= 30
 dy_abs_pair_c134_p14_0_1 <= 30
 dy_abs_pair_c135_p15_0_1 <= 30
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
diff_dist_pair_c120_p0_0_1
diff_dist_pair_c121_p1_0_1
diff_dist_pair_c122_p2_0_1
diff_dist_pair_c123_p3_0_1
diff_dist_pair_c124_p4_0_1
diff_dist_pair_c125_p5_0_1
diff_dist_pair_c126_p6_0_1
diff_dist_pair_c127_p7_0_1
diff_dist_pair_c128_p8_0_1
diff_dist_pair_c129_p9_0_1
diff_dist_pair_c130_p10_0_1
diff_dist_pair_c131_p11_0_1
diff_dist_pair_c132_p12_0_1
diff_dist_pair_c133_p13_0_1
diff_dist_pair_c134_p14_0_1
diff_dist_pair_c135_p15_0_1
diff_pos_c120_p0_0
diff_pos_c120_p0_1
diff_pos_c121_p1_0
diff_pos_c121_p1_1
diff_pos_c122_p2_0
diff_pos_c122_p2_1
diff_pos_c123_p3_0
diff_pos_c123_p3_1
diff_pos_c124_p4_0
diff_pos_c124_p4_1
diff_pos_c125_p5_0
diff_pos_c125_p5_1
diff_pos_c126_p6_0
diff_pos_c126_p6_1
diff_pos_c127_p7_0
diff_pos_c127_p7_1
diff_pos_c128_p8_0
diff_pos_c128_p8_1
diff_pos_c129_p9_0
diff_pos_c129_p9_1
diff_pos_c130_p10_0
diff_pos_c130_p10_1
diff_pos_c131_p11_0
diff_pos_c131_p11_1
diff_pos_c132_p12_0
diff_pos_c132_p12_1
diff_pos_c133_p13_0
diff_pos_c133_p13_1
diff_pos_c134_p14_0
diff_pos_c134_p14_1
diff_pos_c135_p15_0
diff_pos_c135_p15_1
p_down_0_1
p_left_0_1
p_right_0_1
p_up_0_1
x_diff_c120_p0_1
x_diff_c121_p1_1
x_diff_c122_p2_1
x_diff_c123_p3_1
x_diff_c124_p4_1
x_diff_c125_p5_1
x_diff_c126_p6_1
x_diff_c127_p7_1
x_diff_c128_p8_1
x_diff_c129_p9_1
x_diff_c130_p10_1
x_diff_c131_p11_1
x_diff_c132_p12_1
x_diff_c133_p13_1
x_diff_c134_p14_1
x_diff_c135_p15_1
x_diff_minus_c120_p0_1
x_diff_minus_c121_p1_1
x_diff_minus_c122_p2_1
x_diff_minus_c123_p3_1
x_diff_minus_c124_p4_1
x_diff_minus_c125_p5_1
x_diff_minus_c126_p6_1
x_diff_minus_c127_p7_1
x_diff_minus_c128_p8_1
x_diff_minus_c129_p9_1
x_diff_minus_c130_p10_1
x_diff_minus_c131_p11_1
x_diff_minus_c132_p12_1
x_diff_minus_c133_p13_1
x_diff_minus_c134_p14_1
x_diff_minus_c135_p15_1
x_diff_plus_c120_p0_1
x_diff_plus_c121_p1_1
x_diff_plus_c122_p2_1
x_diff_plus_c123_p3_1
x_diff_plus_c124_p4_1
x_diff_plus_c125_p5_1
x_diff_plus_c126_p6_1
x_diff_plus_c127_p7_1
x_diff_plus_c128_p8_1
x_diff_plus_c129_p9_1
x_diff_plus_c130_p10_1
x_diff_plus_c131_p11_1
x_diff_plus_c132_p12_1
x_diff_plus_c133_p13_1
x_diff_plus_c134_p14_1
x_diff_plus_c135_p15_1
y_diff_c120_p0_1
y_diff_c121_p1_1
y_diff_c122_p2_1
y_diff_c123_p3_1
y_diff_c124_p4_1
y_diff_c125_p5_1
y_diff_c126_p6_1
y_diff_c127_p7_1
y_diff_c128_p8_1
y_diff_c129_p9_1
y_diff_c130_p10_1
y_diff_c131_p11_1
y_diff_c132_p12_1
y_diff_c133_p13_1
y_diff_c134_p14_1
y_diff_c135_p15_1
y_diff_minus_c120_p0_1
y_diff_minus_c121_p1_1
y_diff_minus_c122_p2_1
y_diff_minus_c123_p3_1
y_diff_minus_c124_p4_1
y_diff_minus_c125_p5_1
y_diff_minus_c126_p6_1
y_diff_minus_c127_p7_1
y_diff_minus_c128_p8_1
y_diff_minus_c129_p9_1
y_diff_minus_c130_p10_1
y_diff_minus_c131_p11_1
y_diff_minus_c132_p12_1
y_diff_minus_c133_p13_1
y_diff_minus_c134_p14_1
y_diff_minus_c135_p15_1
y_diff_plus_c120_p0_1
y_diff_plus_c121_p1_1
y_diff_plus_c122_p2_1
y_diff_plus_c123_p3_1
y_diff_plus_c124_p4_1
y_diff_plus_c125_p5_1
y_diff_plus_c126_p6_1
y_diff_plus_c127_p7_1
y_diff_plus_c128_p8_1
y_diff_plus_c129_p9_1
y_diff_plus_c130_p10_1
y_diff_plus_c131_p11_1
y_diff_plus_c132_p12_1
y_diff_plus_c133_p13_1
y_diff_plus_c134_p14_1
y_diff_plus_c135_p15_1
z1L_0_1
z1R_0_1
z1_0_1
z2D_0_1
z2U_0_1
z2_0_1
End
