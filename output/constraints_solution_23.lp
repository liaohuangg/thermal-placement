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
diff_pos_fixed_c231_p0_0: diff_pos_c231_p0_0 = 0
diff_pos_fixed_c232_p1_0: diff_pos_c232_p1_0 = 0
diff_pos_fixed_c233_p2_0: diff_pos_c233_p2_0 = 0
diff_pos_fixed_c234_p3_0: diff_pos_c234_p3_0 = 0
diff_pos_fixed_c235_p4_0: diff_pos_c235_p4_0 = 0
diff_pos_fixed_c236_p5_0: diff_pos_c236_p5_0 = 0
diff_pos_fixed_c237_p6_0: diff_pos_c237_p6_0 = 0
diff_pos_fixed_c238_p7_0: diff_pos_c238_p7_0 = 0
diff_pos_fixed_c239_p8_0: diff_pos_c239_p8_0 = 0
diff_pos_fixed_c240_p9_0: diff_pos_c240_p9_0 = 0
diff_pos_fixed_c241_p10_0: diff_pos_c241_p10_0 = 0
diff_pos_fixed_c242_p11_0: diff_pos_c242_p11_0 = 0
diff_pos_fixed_c243_p12_0: diff_pos_c243_p12_0 = 0
diff_pos_fixed_c244_p13_0: diff_pos_c244_p13_0 = 0
diff_pos_fixed_c245_p14_0: diff_pos_c245_p14_0 = 0
diff_pos_fixed_c246_p15_0: diff_pos_c246_p15_0 = 0
diff_pos_fixed_c247_p16_0: diff_pos_c247_p16_0 = 0
diff_pos_fixed_c248_p17_0: diff_pos_c248_p17_0 = 0
diff_pos_fixed_c249_p18_0: diff_pos_c249_p18_0 = 0
diff_pos_fixed_c250_p19_0: diff_pos_c250_p19_0 = 0
diff_pos_fixed_c251_p20_0: diff_pos_c251_p20_0 = 0
diff_pos_fixed_c252_p21_0: diff_pos_c252_p21_0 = 0
diff_pos_from_x_c231_p0_1: diff_pos_c231_p0_1 - x_diff_c231_p0_1 >= 0
diff_pos_from_x_c232_p1_1: diff_pos_c232_p1_1 - x_diff_c232_p1_1 >= 0
diff_pos_from_x_c233_p2_1: diff_pos_c233_p2_1 - x_diff_c233_p2_1 >= 0
diff_pos_from_x_c234_p3_1: diff_pos_c234_p3_1 - x_diff_c234_p3_1 >= 0
diff_pos_from_x_c235_p4_1: diff_pos_c235_p4_1 - x_diff_c235_p4_1 >= 0
diff_pos_from_x_c236_p5_1: diff_pos_c236_p5_1 - x_diff_c236_p5_1 >= 0
diff_pos_from_x_c237_p6_1: diff_pos_c237_p6_1 - x_diff_c237_p6_1 >= 0
diff_pos_from_x_c238_p7_1: diff_pos_c238_p7_1 - x_diff_c238_p7_1 >= 0
diff_pos_from_x_c239_p8_1: diff_pos_c239_p8_1 - x_diff_c239_p8_1 >= 0
diff_pos_from_x_c240_p9_1: diff_pos_c240_p9_1 - x_diff_c240_p9_1 >= 0
diff_pos_from_x_c241_p10_1: diff_pos_c241_p10_1 - x_diff_c241_p10_1 >= 0
diff_pos_from_x_c242_p11_1: diff_pos_c242_p11_1 - x_diff_c242_p11_1 >= 0
diff_pos_from_x_c243_p12_1: diff_pos_c243_p12_1 - x_diff_c243_p12_1 >= 0
diff_pos_from_x_c244_p13_1: diff_pos_c244_p13_1 - x_diff_c244_p13_1 >= 0
diff_pos_from_x_c245_p14_1: diff_pos_c245_p14_1 - x_diff_c245_p14_1 >= 0
diff_pos_from_x_c246_p15_1: diff_pos_c246_p15_1 - x_diff_c246_p15_1 >= 0
diff_pos_from_x_c247_p16_1: diff_pos_c247_p16_1 - x_diff_c247_p16_1 >= 0
diff_pos_from_x_c248_p17_1: diff_pos_c248_p17_1 - x_diff_c248_p17_1 >= 0
diff_pos_from_x_c249_p18_1: diff_pos_c249_p18_1 - x_diff_c249_p18_1 >= 0
diff_pos_from_x_c250_p19_1: diff_pos_c250_p19_1 - x_diff_c250_p19_1 >= 0
diff_pos_from_x_c251_p20_1: diff_pos_c251_p20_1 - x_diff_c251_p20_1 >= 0
diff_pos_from_x_c252_p21_1: diff_pos_c252_p21_1 - x_diff_c252_p21_1 >= 0
diff_pos_from_y_c231_p0_1: diff_pos_c231_p0_1 - y_diff_c231_p0_1 >= 0
diff_pos_from_y_c232_p1_1: diff_pos_c232_p1_1 - y_diff_c232_p1_1 >= 0
diff_pos_from_y_c233_p2_1: diff_pos_c233_p2_1 - y_diff_c233_p2_1 >= 0
diff_pos_from_y_c234_p3_1: diff_pos_c234_p3_1 - y_diff_c234_p3_1 >= 0
diff_pos_from_y_c235_p4_1: diff_pos_c235_p4_1 - y_diff_c235_p4_1 >= 0
diff_pos_from_y_c236_p5_1: diff_pos_c236_p5_1 - y_diff_c236_p5_1 >= 0
diff_pos_from_y_c237_p6_1: diff_pos_c237_p6_1 - y_diff_c237_p6_1 >= 0
diff_pos_from_y_c238_p7_1: diff_pos_c238_p7_1 - y_diff_c238_p7_1 >= 0
diff_pos_from_y_c239_p8_1: diff_pos_c239_p8_1 - y_diff_c239_p8_1 >= 0
diff_pos_from_y_c240_p9_1: diff_pos_c240_p9_1 - y_diff_c240_p9_1 >= 0
diff_pos_from_y_c241_p10_1: diff_pos_c241_p10_1 - y_diff_c241_p10_1 >= 0
diff_pos_from_y_c242_p11_1: diff_pos_c242_p11_1 - y_diff_c242_p11_1 >= 0
diff_pos_from_y_c243_p12_1: diff_pos_c243_p12_1 - y_diff_c243_p12_1 >= 0
diff_pos_from_y_c244_p13_1: diff_pos_c244_p13_1 - y_diff_c244_p13_1 >= 0
diff_pos_from_y_c245_p14_1: diff_pos_c245_p14_1 - y_diff_c245_p14_1 >= 0
diff_pos_from_y_c246_p15_1: diff_pos_c246_p15_1 - y_diff_c246_p15_1 >= 0
diff_pos_from_y_c247_p16_1: diff_pos_c247_p16_1 - y_diff_c247_p16_1 >= 0
diff_pos_from_y_c248_p17_1: diff_pos_c248_p17_1 - y_diff_c248_p17_1 >= 0
diff_pos_from_y_c249_p18_1: diff_pos_c249_p18_1 - y_diff_c249_p18_1 >= 0
diff_pos_from_y_c250_p19_1: diff_pos_c250_p19_1 - y_diff_c250_p19_1 >= 0
diff_pos_from_y_c251_p20_1: diff_pos_c251_p20_1 - y_diff_c251_p20_1 >= 0
diff_pos_from_y_c252_p21_1: diff_pos_c252_p21_1 - y_diff_c252_p21_1 >= 0
diff_pos_upper_c231_p0_1: diff_pos_c231_p0_1 - x_diff_c231_p0_1
 - y_diff_c231_p0_1 <= 0
diff_pos_upper_c232_p1_1: diff_pos_c232_p1_1 - x_diff_c232_p1_1
 - y_diff_c232_p1_1 <= 0
diff_pos_upper_c233_p2_1: diff_pos_c233_p2_1 - x_diff_c233_p2_1
 - y_diff_c233_p2_1 <= 0
diff_pos_upper_c234_p3_1: diff_pos_c234_p3_1 - x_diff_c234_p3_1
 - y_diff_c234_p3_1 <= 0
diff_pos_upper_c235_p4_1: diff_pos_c235_p4_1 - x_diff_c235_p4_1
 - y_diff_c235_p4_1 <= 0
diff_pos_upper_c236_p5_1: diff_pos_c236_p5_1 - x_diff_c236_p5_1
 - y_diff_c236_p5_1 <= 0
diff_pos_upper_c237_p6_1: diff_pos_c237_p6_1 - x_diff_c237_p6_1
 - y_diff_c237_p6_1 <= 0
diff_pos_upper_c238_p7_1: diff_pos_c238_p7_1 - x_diff_c238_p7_1
 - y_diff_c238_p7_1 <= 0
diff_pos_upper_c239_p8_1: diff_pos_c239_p8_1 - x_diff_c239_p8_1
 - y_diff_c239_p8_1 <= 0
diff_pos_upper_c240_p9_1: diff_pos_c240_p9_1 - x_diff_c240_p9_1
 - y_diff_c240_p9_1 <= 0
diff_pos_upper_c241_p10_1: diff_pos_c241_p10_1 - x_diff_c241_p10_1
 - y_diff_c241_p10_1 <= 0
diff_pos_upper_c242_p11_1: diff_pos_c242_p11_1 - x_diff_c242_p11_1
 - y_diff_c242_p11_1 <= 0
diff_pos_upper_c243_p12_1: diff_pos_c243_p12_1 - x_diff_c243_p12_1
 - y_diff_c243_p12_1 <= 0
diff_pos_upper_c244_p13_1: diff_pos_c244_p13_1 - x_diff_c244_p13_1
 - y_diff_c244_p13_1 <= 0
diff_pos_upper_c245_p14_1: diff_pos_c245_p14_1 - x_diff_c245_p14_1
 - y_diff_c245_p14_1 <= 0
diff_pos_upper_c246_p15_1: diff_pos_c246_p15_1 - x_diff_c246_p15_1
 - y_diff_c246_p15_1 <= 0
diff_pos_upper_c247_p16_1: diff_pos_c247_p16_1 - x_diff_c247_p16_1
 - y_diff_c247_p16_1 <= 0
diff_pos_upper_c248_p17_1: diff_pos_c248_p17_1 - x_diff_c248_p17_1
 - y_diff_c248_p17_1 <= 0
diff_pos_upper_c249_p18_1: diff_pos_c249_p18_1 - x_diff_c249_p18_1
 - y_diff_c249_p18_1 <= 0
diff_pos_upper_c250_p19_1: diff_pos_c250_p19_1 - x_diff_c250_p19_1
 - y_diff_c250_p19_1 <= 0
diff_pos_upper_c251_p20_1: diff_pos_c251_p20_1 - x_diff_c251_p20_1
 - y_diff_c251_p20_1 <= 0
diff_pos_upper_c252_p21_1: diff_pos_c252_p21_1 - x_diff_c252_p21_1
 - y_diff_c252_p21_1 <= 0
dist_diff_abs_pair_lower_c231_p0_0_1: - 60 diff_dist_pair_c231_p0_0_1
 + dist_diff_abs_pair_c231_p0_0_1 >= -57
dist_diff_abs_pair_lower_c232_p1_0_1: - 60 diff_dist_pair_c232_p1_0_1
 + dist_diff_abs_pair_c232_p1_0_1 >= -57
dist_diff_abs_pair_lower_c233_p2_0_1: - 60 diff_dist_pair_c233_p2_0_1
 + dist_diff_abs_pair_c233_p2_0_1 >= -57
dist_diff_abs_pair_lower_c234_p3_0_1: - 60 diff_dist_pair_c234_p3_0_1
 + dist_diff_abs_pair_c234_p3_0_1 >= -57
dist_diff_abs_pair_lower_c235_p4_0_1: - 60 diff_dist_pair_c235_p4_0_1
 + dist_diff_abs_pair_c235_p4_0_1 >= -57
dist_diff_abs_pair_lower_c236_p5_0_1: - 60 diff_dist_pair_c236_p5_0_1
 + dist_diff_abs_pair_c236_p5_0_1 >= -57
dist_diff_abs_pair_lower_c237_p6_0_1: - 60 diff_dist_pair_c237_p6_0_1
 + dist_diff_abs_pair_c237_p6_0_1 >= -57
dist_diff_abs_pair_lower_c238_p7_0_1: - 60 diff_dist_pair_c238_p7_0_1
 + dist_diff_abs_pair_c238_p7_0_1 >= -57
dist_diff_abs_pair_lower_c239_p8_0_1: - 60 diff_dist_pair_c239_p8_0_1
 + dist_diff_abs_pair_c239_p8_0_1 >= -57
dist_diff_abs_pair_lower_c240_p9_0_1: - 60 diff_dist_pair_c240_p9_0_1
 + dist_diff_abs_pair_c240_p9_0_1 >= -57
dist_diff_abs_pair_lower_c241_p10_0_1: - 60 diff_dist_pair_c241_p10_0_1
 + dist_diff_abs_pair_c241_p10_0_1 >= -57
dist_diff_abs_pair_lower_c242_p11_0_1: - 60 diff_dist_pair_c242_p11_0_1
 + dist_diff_abs_pair_c242_p11_0_1 >= -57
dist_diff_abs_pair_lower_c243_p12_0_1: - 60 diff_dist_pair_c243_p12_0_1
 + dist_diff_abs_pair_c243_p12_0_1 >= -57
dist_diff_abs_pair_lower_c244_p13_0_1: - 60 diff_dist_pair_c244_p13_0_1
 + dist_diff_abs_pair_c244_p13_0_1 >= -57
dist_diff_abs_pair_lower_c245_p14_0_1: - 60 diff_dist_pair_c245_p14_0_1
 + dist_diff_abs_pair_c245_p14_0_1 >= -57
dist_diff_abs_pair_lower_c246_p15_0_1: - 60 diff_dist_pair_c246_p15_0_1
 + dist_diff_abs_pair_c246_p15_0_1 >= -57
dist_diff_abs_pair_lower_c247_p16_0_1: - 60 diff_dist_pair_c247_p16_0_1
 + dist_diff_abs_pair_c247_p16_0_1 >= -57
dist_diff_abs_pair_lower_c248_p17_0_1: - 60 diff_dist_pair_c248_p17_0_1
 + dist_diff_abs_pair_c248_p17_0_1 >= -57
dist_diff_abs_pair_lower_c249_p18_0_1: - 60 diff_dist_pair_c249_p18_0_1
 + dist_diff_abs_pair_c249_p18_0_1 >= -57
dist_diff_abs_pair_lower_c250_p19_0_1: - 60 diff_dist_pair_c250_p19_0_1
 + dist_diff_abs_pair_c250_p19_0_1 >= -57
dist_diff_abs_pair_lower_c251_p20_0_1: - 60 diff_dist_pair_c251_p20_0_1
 + dist_diff_abs_pair_c251_p20_0_1 >= -57
dist_diff_abs_pair_lower_c252_p21_0_1: - 60 diff_dist_pair_c252_p21_0_1
 + dist_diff_abs_pair_c252_p21_0_1 >= -57
dist_diff_abs_pair_neg_c231_p0_0_1: dist_diff_abs_pair_c231_p0_0_1
 + dist_diff_pair_c231_p0_0_1 >= 0
dist_diff_abs_pair_neg_c232_p1_0_1: dist_diff_abs_pair_c232_p1_0_1
 + dist_diff_pair_c232_p1_0_1 >= 0
dist_diff_abs_pair_neg_c233_p2_0_1: dist_diff_abs_pair_c233_p2_0_1
 + dist_diff_pair_c233_p2_0_1 >= 0
dist_diff_abs_pair_neg_c234_p3_0_1: dist_diff_abs_pair_c234_p3_0_1
 + dist_diff_pair_c234_p3_0_1 >= 0
dist_diff_abs_pair_neg_c235_p4_0_1: dist_diff_abs_pair_c235_p4_0_1
 + dist_diff_pair_c235_p4_0_1 >= 0
dist_diff_abs_pair_neg_c236_p5_0_1: dist_diff_abs_pair_c236_p5_0_1
 + dist_diff_pair_c236_p5_0_1 >= 0
dist_diff_abs_pair_neg_c237_p6_0_1: dist_diff_abs_pair_c237_p6_0_1
 + dist_diff_pair_c237_p6_0_1 >= 0
dist_diff_abs_pair_neg_c238_p7_0_1: dist_diff_abs_pair_c238_p7_0_1
 + dist_diff_pair_c238_p7_0_1 >= 0
dist_diff_abs_pair_neg_c239_p8_0_1: dist_diff_abs_pair_c239_p8_0_1
 + dist_diff_pair_c239_p8_0_1 >= 0
dist_diff_abs_pair_neg_c240_p9_0_1: dist_diff_abs_pair_c240_p9_0_1
 + dist_diff_pair_c240_p9_0_1 >= 0
dist_diff_abs_pair_neg_c241_p10_0_1: dist_diff_abs_pair_c241_p10_0_1
 + dist_diff_pair_c241_p10_0_1 >= 0
dist_diff_abs_pair_neg_c242_p11_0_1: dist_diff_abs_pair_c242_p11_0_1
 + dist_diff_pair_c242_p11_0_1 >= 0
dist_diff_abs_pair_neg_c243_p12_0_1: dist_diff_abs_pair_c243_p12_0_1
 + dist_diff_pair_c243_p12_0_1 >= 0
dist_diff_abs_pair_neg_c244_p13_0_1: dist_diff_abs_pair_c244_p13_0_1
 + dist_diff_pair_c244_p13_0_1 >= 0
dist_diff_abs_pair_neg_c245_p14_0_1: dist_diff_abs_pair_c245_p14_0_1
 + dist_diff_pair_c245_p14_0_1 >= 0
dist_diff_abs_pair_neg_c246_p15_0_1: dist_diff_abs_pair_c246_p15_0_1
 + dist_diff_pair_c246_p15_0_1 >= 0
dist_diff_abs_pair_neg_c247_p16_0_1: dist_diff_abs_pair_c247_p16_0_1
 + dist_diff_pair_c247_p16_0_1 >= 0
dist_diff_abs_pair_neg_c248_p17_0_1: dist_diff_abs_pair_c248_p17_0_1
 + dist_diff_pair_c248_p17_0_1 >= 0
dist_diff_abs_pair_neg_c249_p18_0_1: dist_diff_abs_pair_c249_p18_0_1
 + dist_diff_pair_c249_p18_0_1 >= 0
dist_diff_abs_pair_neg_c250_p19_0_1: dist_diff_abs_pair_c250_p19_0_1
 + dist_diff_pair_c250_p19_0_1 >= 0
dist_diff_abs_pair_neg_c251_p20_0_1: dist_diff_abs_pair_c251_p20_0_1
 + dist_diff_pair_c251_p20_0_1 >= 0
dist_diff_abs_pair_neg_c252_p21_0_1: dist_diff_abs_pair_c252_p21_0_1
 + dist_diff_pair_c252_p21_0_1 >= 0
dist_diff_abs_pair_pos_c231_p0_0_1: dist_diff_abs_pair_c231_p0_0_1
 - dist_diff_pair_c231_p0_0_1 >= 0
dist_diff_abs_pair_pos_c232_p1_0_1: dist_diff_abs_pair_c232_p1_0_1
 - dist_diff_pair_c232_p1_0_1 >= 0
dist_diff_abs_pair_pos_c233_p2_0_1: dist_diff_abs_pair_c233_p2_0_1
 - dist_diff_pair_c233_p2_0_1 >= 0
dist_diff_abs_pair_pos_c234_p3_0_1: dist_diff_abs_pair_c234_p3_0_1
 - dist_diff_pair_c234_p3_0_1 >= 0
dist_diff_abs_pair_pos_c235_p4_0_1: dist_diff_abs_pair_c235_p4_0_1
 - dist_diff_pair_c235_p4_0_1 >= 0
dist_diff_abs_pair_pos_c236_p5_0_1: dist_diff_abs_pair_c236_p5_0_1
 - dist_diff_pair_c236_p5_0_1 >= 0
dist_diff_abs_pair_pos_c237_p6_0_1: dist_diff_abs_pair_c237_p6_0_1
 - dist_diff_pair_c237_p6_0_1 >= 0
dist_diff_abs_pair_pos_c238_p7_0_1: dist_diff_abs_pair_c238_p7_0_1
 - dist_diff_pair_c238_p7_0_1 >= 0
dist_diff_abs_pair_pos_c239_p8_0_1: dist_diff_abs_pair_c239_p8_0_1
 - dist_diff_pair_c239_p8_0_1 >= 0
dist_diff_abs_pair_pos_c240_p9_0_1: dist_diff_abs_pair_c240_p9_0_1
 - dist_diff_pair_c240_p9_0_1 >= 0
dist_diff_abs_pair_pos_c241_p10_0_1: dist_diff_abs_pair_c241_p10_0_1
 - dist_diff_pair_c241_p10_0_1 >= 0
dist_diff_abs_pair_pos_c242_p11_0_1: dist_diff_abs_pair_c242_p11_0_1
 - dist_diff_pair_c242_p11_0_1 >= 0
dist_diff_abs_pair_pos_c243_p12_0_1: dist_diff_abs_pair_c243_p12_0_1
 - dist_diff_pair_c243_p12_0_1 >= 0
dist_diff_abs_pair_pos_c244_p13_0_1: dist_diff_abs_pair_c244_p13_0_1
 - dist_diff_pair_c244_p13_0_1 >= 0
dist_diff_abs_pair_pos_c245_p14_0_1: dist_diff_abs_pair_c245_p14_0_1
 - dist_diff_pair_c245_p14_0_1 >= 0
dist_diff_abs_pair_pos_c246_p15_0_1: dist_diff_abs_pair_c246_p15_0_1
 - dist_diff_pair_c246_p15_0_1 >= 0
dist_diff_abs_pair_pos_c247_p16_0_1: dist_diff_abs_pair_c247_p16_0_1
 - dist_diff_pair_c247_p16_0_1 >= 0
dist_diff_abs_pair_pos_c248_p17_0_1: dist_diff_abs_pair_c248_p17_0_1
 - dist_diff_pair_c248_p17_0_1 >= 0
dist_diff_abs_pair_pos_c249_p18_0_1: dist_diff_abs_pair_c249_p18_0_1
 - dist_diff_pair_c249_p18_0_1 >= 0
dist_diff_abs_pair_pos_c250_p19_0_1: dist_diff_abs_pair_c250_p19_0_1
 - dist_diff_pair_c250_p19_0_1 >= 0
dist_diff_abs_pair_pos_c251_p20_0_1: dist_diff_abs_pair_c251_p20_0_1
 - dist_diff_pair_c251_p20_0_1 >= 0
dist_diff_abs_pair_pos_c252_p21_0_1: dist_diff_abs_pair_c252_p21_0_1
 - dist_diff_pair_c252_p21_0_1 >= 0
dist_diff_abs_pair_upper_c231_p0_0_1: - 60 diff_dist_pair_c231_p0_0_1
 + dist_diff_abs_pair_c231_p0_0_1 <= 2.97
dist_diff_abs_pair_upper_c232_p1_0_1: - 60 diff_dist_pair_c232_p1_0_1
 + dist_diff_abs_pair_c232_p1_0_1 <= 2.97
dist_diff_abs_pair_upper_c233_p2_0_1: - 60 diff_dist_pair_c233_p2_0_1
 + dist_diff_abs_pair_c233_p2_0_1 <= 2.97
dist_diff_abs_pair_upper_c234_p3_0_1: - 60 diff_dist_pair_c234_p3_0_1
 + dist_diff_abs_pair_c234_p3_0_1 <= 2.97
dist_diff_abs_pair_upper_c235_p4_0_1: - 60 diff_dist_pair_c235_p4_0_1
 + dist_diff_abs_pair_c235_p4_0_1 <= 2.97
dist_diff_abs_pair_upper_c236_p5_0_1: - 60 diff_dist_pair_c236_p5_0_1
 + dist_diff_abs_pair_c236_p5_0_1 <= 2.97
dist_diff_abs_pair_upper_c237_p6_0_1: - 60 diff_dist_pair_c237_p6_0_1
 + dist_diff_abs_pair_c237_p6_0_1 <= 2.97
dist_diff_abs_pair_upper_c238_p7_0_1: - 60 diff_dist_pair_c238_p7_0_1
 + dist_diff_abs_pair_c238_p7_0_1 <= 2.97
dist_diff_abs_pair_upper_c239_p8_0_1: - 60 diff_dist_pair_c239_p8_0_1
 + dist_diff_abs_pair_c239_p8_0_1 <= 2.97
dist_diff_abs_pair_upper_c240_p9_0_1: - 60 diff_dist_pair_c240_p9_0_1
 + dist_diff_abs_pair_c240_p9_0_1 <= 2.97
dist_diff_abs_pair_upper_c241_p10_0_1: - 60 diff_dist_pair_c241_p10_0_1
 + dist_diff_abs_pair_c241_p10_0_1 <= 2.97
dist_diff_abs_pair_upper_c242_p11_0_1: - 60 diff_dist_pair_c242_p11_0_1
 + dist_diff_abs_pair_c242_p11_0_1 <= 2.97
dist_diff_abs_pair_upper_c243_p12_0_1: - 60 diff_dist_pair_c243_p12_0_1
 + dist_diff_abs_pair_c243_p12_0_1 <= 2.97
dist_diff_abs_pair_upper_c244_p13_0_1: - 60 diff_dist_pair_c244_p13_0_1
 + dist_diff_abs_pair_c244_p13_0_1 <= 2.97
dist_diff_abs_pair_upper_c245_p14_0_1: - 60 diff_dist_pair_c245_p14_0_1
 + dist_diff_abs_pair_c245_p14_0_1 <= 2.97
dist_diff_abs_pair_upper_c246_p15_0_1: - 60 diff_dist_pair_c246_p15_0_1
 + dist_diff_abs_pair_c246_p15_0_1 <= 2.97
dist_diff_abs_pair_upper_c247_p16_0_1: - 60 diff_dist_pair_c247_p16_0_1
 + dist_diff_abs_pair_c247_p16_0_1 <= 2.97
dist_diff_abs_pair_upper_c248_p17_0_1: - 60 diff_dist_pair_c248_p17_0_1
 + dist_diff_abs_pair_c248_p17_0_1 <= 2.97
dist_diff_abs_pair_upper_c249_p18_0_1: - 60 diff_dist_pair_c249_p18_0_1
 + dist_diff_abs_pair_c249_p18_0_1 <= 2.97
dist_diff_abs_pair_upper_c250_p19_0_1: - 60 diff_dist_pair_c250_p19_0_1
 + dist_diff_abs_pair_c250_p19_0_1 <= 2.97
dist_diff_abs_pair_upper_c251_p20_0_1: - 60 diff_dist_pair_c251_p20_0_1
 + dist_diff_abs_pair_c251_p20_0_1 <= 2.97
dist_diff_abs_pair_upper_c252_p21_0_1: - 60 diff_dist_pair_c252_p21_0_1
 + dist_diff_abs_pair_c252_p21_0_1 <= 2.97
dist_diff_pair_def_c231_p0_0_1: dist_diff_pair_c231_p0_0_1
 - dx_abs_pair_c231_p0_0_1 - dy_abs_pair_c231_p0_0_1 = -10
dist_diff_pair_def_c232_p1_0_1: dist_diff_pair_c232_p1_0_1
 - dx_abs_pair_c232_p1_0_1 - dy_abs_pair_c232_p1_0_1 = -10
dist_diff_pair_def_c233_p2_0_1: dist_diff_pair_c233_p2_0_1
 - dx_abs_pair_c233_p2_0_1 - dy_abs_pair_c233_p2_0_1 = -10
dist_diff_pair_def_c234_p3_0_1: dist_diff_pair_c234_p3_0_1
 - dx_abs_pair_c234_p3_0_1 - dy_abs_pair_c234_p3_0_1 = -10
dist_diff_pair_def_c235_p4_0_1: dist_diff_pair_c235_p4_0_1
 - dx_abs_pair_c235_p4_0_1 - dy_abs_pair_c235_p4_0_1 = -13
dist_diff_pair_def_c236_p5_0_1: dist_diff_pair_c236_p5_0_1
 - dx_abs_pair_c236_p5_0_1 - dy_abs_pair_c236_p5_0_1 = -13
dist_diff_pair_def_c237_p6_0_1: dist_diff_pair_c237_p6_0_1
 - dx_abs_pair_c237_p6_0_1 - dy_abs_pair_c237_p6_0_1 = -13
dist_diff_pair_def_c238_p7_0_1: dist_diff_pair_c238_p7_0_1
 - dx_abs_pair_c238_p7_0_1 - dy_abs_pair_c238_p7_0_1 = -13
dist_diff_pair_def_c239_p8_0_1: dist_diff_pair_c239_p8_0_1
 - dx_abs_pair_c239_p8_0_1 - dy_abs_pair_c239_p8_0_1 = -13
dist_diff_pair_def_c240_p9_0_1: dist_diff_pair_c240_p9_0_1
 - dx_abs_pair_c240_p9_0_1 - dy_abs_pair_c240_p9_0_1 = -13
dist_diff_pair_def_c241_p10_0_1: dist_diff_pair_c241_p10_0_1
 - dx_abs_pair_c241_p10_0_1 - dy_abs_pair_c241_p10_0_1 = -13
dist_diff_pair_def_c242_p11_0_1: dist_diff_pair_c242_p11_0_1
 - dx_abs_pair_c242_p11_0_1 - dy_abs_pair_c242_p11_0_1 = -13
dist_diff_pair_def_c243_p12_0_1: dist_diff_pair_c243_p12_0_1
 - dx_abs_pair_c243_p12_0_1 - dy_abs_pair_c243_p12_0_1 = -16
dist_diff_pair_def_c244_p13_0_1: dist_diff_pair_c244_p13_0_1
 - dx_abs_pair_c244_p13_0_1 - dy_abs_pair_c244_p13_0_1 = -16
dist_diff_pair_def_c245_p14_0_1: dist_diff_pair_c245_p14_0_1
 - dx_abs_pair_c245_p14_0_1 - dy_abs_pair_c245_p14_0_1 = -16
dist_diff_pair_def_c246_p15_0_1: dist_diff_pair_c246_p15_0_1
 - dx_abs_pair_c246_p15_0_1 - dy_abs_pair_c246_p15_0_1 = -16
dist_diff_pair_def_c247_p16_0_1: dist_diff_pair_c247_p16_0_1
 - dx_abs_pair_c247_p16_0_1 - dy_abs_pair_c247_p16_0_1 = -16
dist_diff_pair_def_c248_p17_0_1: dist_diff_pair_c248_p17_0_1
 - dx_abs_pair_c248_p17_0_1 - dy_abs_pair_c248_p17_0_1 = -16
dist_diff_pair_def_c249_p18_0_1: dist_diff_pair_c249_p18_0_1
 - dx_abs_pair_c249_p18_0_1 - dy_abs_pair_c249_p18_0_1 = -16
dist_diff_pair_def_c250_p19_0_1: dist_diff_pair_c250_p19_0_1
 - dx_abs_pair_c250_p19_0_1 - dy_abs_pair_c250_p19_0_1 = -16
dist_diff_pair_def_c251_p20_0_1: dist_diff_pair_c251_p20_0_1
 - dx_abs_pair_c251_p20_0_1 - dy_abs_pair_c251_p20_0_1 = -19
dist_diff_pair_def_c252_p21_0_1: dist_diff_pair_c252_p21_0_1
 - dx_abs_pair_c252_p21_0_1 - dy_abs_pair_c252_p21_0_1 = -19
dx_abs1_0_1: - cx_0 + cx_1 + dx_abs_0_1 >= 0
dx_abs2_0_1: cx_0 - cx_1 + dx_abs_0_1 >= 0
dx_abs_pair_neg_c231_p0_0_1: cx_0 - cx_1 + dx_abs_pair_c231_p0_0_1 >= 0
dx_abs_pair_neg_c232_p1_0_1: cx_0 - cx_1 + dx_abs_pair_c232_p1_0_1 >= 0
dx_abs_pair_neg_c233_p2_0_1: cx_0 - cx_1 + dx_abs_pair_c233_p2_0_1 >= 0
dx_abs_pair_neg_c234_p3_0_1: cx_0 - cx_1 + dx_abs_pair_c234_p3_0_1 >= 0
dx_abs_pair_neg_c235_p4_0_1: cx_0 - cx_1 + dx_abs_pair_c235_p4_0_1 >= 0
dx_abs_pair_neg_c236_p5_0_1: cx_0 - cx_1 + dx_abs_pair_c236_p5_0_1 >= 0
dx_abs_pair_neg_c237_p6_0_1: cx_0 - cx_1 + dx_abs_pair_c237_p6_0_1 >= 0
dx_abs_pair_neg_c238_p7_0_1: cx_0 - cx_1 + dx_abs_pair_c238_p7_0_1 >= 0
dx_abs_pair_neg_c239_p8_0_1: cx_0 - cx_1 + dx_abs_pair_c239_p8_0_1 >= 0
dx_abs_pair_neg_c240_p9_0_1: cx_0 - cx_1 + dx_abs_pair_c240_p9_0_1 >= 0
dx_abs_pair_neg_c241_p10_0_1: cx_0 - cx_1 + dx_abs_pair_c241_p10_0_1 >= 0
dx_abs_pair_neg_c242_p11_0_1: cx_0 - cx_1 + dx_abs_pair_c242_p11_0_1 >= 0
dx_abs_pair_neg_c243_p12_0_1: cx_0 - cx_1 + dx_abs_pair_c243_p12_0_1 >= 0
dx_abs_pair_neg_c244_p13_0_1: cx_0 - cx_1 + dx_abs_pair_c244_p13_0_1 >= 0
dx_abs_pair_neg_c245_p14_0_1: cx_0 - cx_1 + dx_abs_pair_c245_p14_0_1 >= 0
dx_abs_pair_neg_c246_p15_0_1: cx_0 - cx_1 + dx_abs_pair_c246_p15_0_1 >= 0
dx_abs_pair_neg_c247_p16_0_1: cx_0 - cx_1 + dx_abs_pair_c247_p16_0_1 >= 0
dx_abs_pair_neg_c248_p17_0_1: cx_0 - cx_1 + dx_abs_pair_c248_p17_0_1 >= 0
dx_abs_pair_neg_c249_p18_0_1: cx_0 - cx_1 + dx_abs_pair_c249_p18_0_1 >= 0
dx_abs_pair_neg_c250_p19_0_1: cx_0 - cx_1 + dx_abs_pair_c250_p19_0_1 >= 0
dx_abs_pair_neg_c251_p20_0_1: cx_0 - cx_1 + dx_abs_pair_c251_p20_0_1 >= 0
dx_abs_pair_neg_c252_p21_0_1: cx_0 - cx_1 + dx_abs_pair_c252_p21_0_1 >= 0
dx_abs_pair_pos_c231_p0_0_1: - cx_0 + cx_1 + dx_abs_pair_c231_p0_0_1 >= 0
dx_abs_pair_pos_c232_p1_0_1: - cx_0 + cx_1 + dx_abs_pair_c232_p1_0_1 >= 0
dx_abs_pair_pos_c233_p2_0_1: - cx_0 + cx_1 + dx_abs_pair_c233_p2_0_1 >= 0
dx_abs_pair_pos_c234_p3_0_1: - cx_0 + cx_1 + dx_abs_pair_c234_p3_0_1 >= 0
dx_abs_pair_pos_c235_p4_0_1: - cx_0 + cx_1 + dx_abs_pair_c235_p4_0_1 >= 0
dx_abs_pair_pos_c236_p5_0_1: - cx_0 + cx_1 + dx_abs_pair_c236_p5_0_1 >= 0
dx_abs_pair_pos_c237_p6_0_1: - cx_0 + cx_1 + dx_abs_pair_c237_p6_0_1 >= 0
dx_abs_pair_pos_c238_p7_0_1: - cx_0 + cx_1 + dx_abs_pair_c238_p7_0_1 >= 0
dx_abs_pair_pos_c239_p8_0_1: - cx_0 + cx_1 + dx_abs_pair_c239_p8_0_1 >= 0
dx_abs_pair_pos_c240_p9_0_1: - cx_0 + cx_1 + dx_abs_pair_c240_p9_0_1 >= 0
dx_abs_pair_pos_c241_p10_0_1: - cx_0 + cx_1 + dx_abs_pair_c241_p10_0_1 >= 0
dx_abs_pair_pos_c242_p11_0_1: - cx_0 + cx_1 + dx_abs_pair_c242_p11_0_1 >= 0
dx_abs_pair_pos_c243_p12_0_1: - cx_0 + cx_1 + dx_abs_pair_c243_p12_0_1 >= 0
dx_abs_pair_pos_c244_p13_0_1: - cx_0 + cx_1 + dx_abs_pair_c244_p13_0_1 >= 0
dx_abs_pair_pos_c245_p14_0_1: - cx_0 + cx_1 + dx_abs_pair_c245_p14_0_1 >= 0
dx_abs_pair_pos_c246_p15_0_1: - cx_0 + cx_1 + dx_abs_pair_c246_p15_0_1 >= 0
dx_abs_pair_pos_c247_p16_0_1: - cx_0 + cx_1 + dx_abs_pair_c247_p16_0_1 >= 0
dx_abs_pair_pos_c248_p17_0_1: - cx_0 + cx_1 + dx_abs_pair_c248_p17_0_1 >= 0
dx_abs_pair_pos_c249_p18_0_1: - cx_0 + cx_1 + dx_abs_pair_c249_p18_0_1 >= 0
dx_abs_pair_pos_c250_p19_0_1: - cx_0 + cx_1 + dx_abs_pair_c250_p19_0_1 >= 0
dx_abs_pair_pos_c251_p20_0_1: - cx_0 + cx_1 + dx_abs_pair_c251_p20_0_1 >= 0
dx_abs_pair_pos_c252_p21_0_1: - cx_0 + cx_1 + dx_abs_pair_c252_p21_0_1 >= 0
dy_abs1_0_1: - cy_0 + cy_1 + dy_abs_0_1 >= 0
dy_abs2_0_1: cy_0 - cy_1 + dy_abs_0_1 >= 0
dy_abs_pair_neg_c231_p0_0_1: cy_0 - cy_1 + dy_abs_pair_c231_p0_0_1 >= 0
dy_abs_pair_neg_c232_p1_0_1: cy_0 - cy_1 + dy_abs_pair_c232_p1_0_1 >= 0
dy_abs_pair_neg_c233_p2_0_1: cy_0 - cy_1 + dy_abs_pair_c233_p2_0_1 >= 0
dy_abs_pair_neg_c234_p3_0_1: cy_0 - cy_1 + dy_abs_pair_c234_p3_0_1 >= 0
dy_abs_pair_neg_c235_p4_0_1: cy_0 - cy_1 + dy_abs_pair_c235_p4_0_1 >= 0
dy_abs_pair_neg_c236_p5_0_1: cy_0 - cy_1 + dy_abs_pair_c236_p5_0_1 >= 0
dy_abs_pair_neg_c237_p6_0_1: cy_0 - cy_1 + dy_abs_pair_c237_p6_0_1 >= 0
dy_abs_pair_neg_c238_p7_0_1: cy_0 - cy_1 + dy_abs_pair_c238_p7_0_1 >= 0
dy_abs_pair_neg_c239_p8_0_1: cy_0 - cy_1 + dy_abs_pair_c239_p8_0_1 >= 0
dy_abs_pair_neg_c240_p9_0_1: cy_0 - cy_1 + dy_abs_pair_c240_p9_0_1 >= 0
dy_abs_pair_neg_c241_p10_0_1: cy_0 - cy_1 + dy_abs_pair_c241_p10_0_1 >= 0
dy_abs_pair_neg_c242_p11_0_1: cy_0 - cy_1 + dy_abs_pair_c242_p11_0_1 >= 0
dy_abs_pair_neg_c243_p12_0_1: cy_0 - cy_1 + dy_abs_pair_c243_p12_0_1 >= 0
dy_abs_pair_neg_c244_p13_0_1: cy_0 - cy_1 + dy_abs_pair_c244_p13_0_1 >= 0
dy_abs_pair_neg_c245_p14_0_1: cy_0 - cy_1 + dy_abs_pair_c245_p14_0_1 >= 0
dy_abs_pair_neg_c246_p15_0_1: cy_0 - cy_1 + dy_abs_pair_c246_p15_0_1 >= 0
dy_abs_pair_neg_c247_p16_0_1: cy_0 - cy_1 + dy_abs_pair_c247_p16_0_1 >= 0
dy_abs_pair_neg_c248_p17_0_1: cy_0 - cy_1 + dy_abs_pair_c248_p17_0_1 >= 0
dy_abs_pair_neg_c249_p18_0_1: cy_0 - cy_1 + dy_abs_pair_c249_p18_0_1 >= 0
dy_abs_pair_neg_c250_p19_0_1: cy_0 - cy_1 + dy_abs_pair_c250_p19_0_1 >= 0
dy_abs_pair_neg_c251_p20_0_1: cy_0 - cy_1 + dy_abs_pair_c251_p20_0_1 >= 0
dy_abs_pair_neg_c252_p21_0_1: cy_0 - cy_1 + dy_abs_pair_c252_p21_0_1 >= 0
dy_abs_pair_pos_c231_p0_0_1: - cy_0 + cy_1 + dy_abs_pair_c231_p0_0_1 >= 0
dy_abs_pair_pos_c232_p1_0_1: - cy_0 + cy_1 + dy_abs_pair_c232_p1_0_1 >= 0
dy_abs_pair_pos_c233_p2_0_1: - cy_0 + cy_1 + dy_abs_pair_c233_p2_0_1 >= 0
dy_abs_pair_pos_c234_p3_0_1: - cy_0 + cy_1 + dy_abs_pair_c234_p3_0_1 >= 0
dy_abs_pair_pos_c235_p4_0_1: - cy_0 + cy_1 + dy_abs_pair_c235_p4_0_1 >= 0
dy_abs_pair_pos_c236_p5_0_1: - cy_0 + cy_1 + dy_abs_pair_c236_p5_0_1 >= 0
dy_abs_pair_pos_c237_p6_0_1: - cy_0 + cy_1 + dy_abs_pair_c237_p6_0_1 >= 0
dy_abs_pair_pos_c238_p7_0_1: - cy_0 + cy_1 + dy_abs_pair_c238_p7_0_1 >= 0
dy_abs_pair_pos_c239_p8_0_1: - cy_0 + cy_1 + dy_abs_pair_c239_p8_0_1 >= 0
dy_abs_pair_pos_c240_p9_0_1: - cy_0 + cy_1 + dy_abs_pair_c240_p9_0_1 >= 0
dy_abs_pair_pos_c241_p10_0_1: - cy_0 + cy_1 + dy_abs_pair_c241_p10_0_1 >= 0
dy_abs_pair_pos_c242_p11_0_1: - cy_0 + cy_1 + dy_abs_pair_c242_p11_0_1 >= 0
dy_abs_pair_pos_c243_p12_0_1: - cy_0 + cy_1 + dy_abs_pair_c243_p12_0_1 >= 0
dy_abs_pair_pos_c244_p13_0_1: - cy_0 + cy_1 + dy_abs_pair_c244_p13_0_1 >= 0
dy_abs_pair_pos_c245_p14_0_1: - cy_0 + cy_1 + dy_abs_pair_c245_p14_0_1 >= 0
dy_abs_pair_pos_c246_p15_0_1: - cy_0 + cy_1 + dy_abs_pair_c246_p15_0_1 >= 0
dy_abs_pair_pos_c247_p16_0_1: - cy_0 + cy_1 + dy_abs_pair_c247_p16_0_1 >= 0
dy_abs_pair_pos_c248_p17_0_1: - cy_0 + cy_1 + dy_abs_pair_c248_p17_0_1 >= 0
dy_abs_pair_pos_c249_p18_0_1: - cy_0 + cy_1 + dy_abs_pair_c249_p18_0_1 >= 0
dy_abs_pair_pos_c250_p19_0_1: - cy_0 + cy_1 + dy_abs_pair_c250_p19_0_1 >= 0
dy_abs_pair_pos_c251_p20_0_1: - cy_0 + cy_1 + dy_abs_pair_c251_p20_0_1 >= 0
dy_abs_pair_pos_c252_p21_0_1: - cy_0 + cy_1 + dy_abs_pair_c252_p21_0_1 >= 0
exclude_solution_dist_pair_c231_p0: diff_dist_pair_c231_p0_0_1 >= 1
exclude_solution_dist_pair_c232_p1: diff_dist_pair_c232_p1_0_1 >= 1
exclude_solution_dist_pair_c233_p2: diff_dist_pair_c233_p2_0_1 >= 1
exclude_solution_dist_pair_c234_p3: diff_dist_pair_c234_p3_0_1 >= 1
exclude_solution_dist_pair_c235_p4: diff_dist_pair_c235_p4_0_1 >= 1
exclude_solution_dist_pair_c236_p5: diff_dist_pair_c236_p5_0_1 >= 1
exclude_solution_dist_pair_c237_p6: diff_dist_pair_c237_p6_0_1 >= 1
exclude_solution_dist_pair_c238_p7: diff_dist_pair_c238_p7_0_1 >= 1
exclude_solution_dist_pair_c239_p8: diff_dist_pair_c239_p8_0_1 >= 1
exclude_solution_dist_pair_c240_p9: diff_dist_pair_c240_p9_0_1 >= 1
exclude_solution_dist_pair_c241_p10: diff_dist_pair_c241_p10_0_1 >= 1
exclude_solution_dist_pair_c242_p11: diff_dist_pair_c242_p11_0_1 >= 1
exclude_solution_dist_pair_c243_p12: diff_dist_pair_c243_p12_0_1 >= 1
exclude_solution_dist_pair_c244_p13: diff_dist_pair_c244_p13_0_1 >= 1
exclude_solution_dist_pair_c245_p14: diff_dist_pair_c245_p14_0_1 >= 1
exclude_solution_dist_pair_c246_p15: diff_dist_pair_c246_p15_0_1 >= 1
exclude_solution_dist_pair_c247_p16: diff_dist_pair_c247_p16_0_1 >= 1
exclude_solution_dist_pair_c248_p17: diff_dist_pair_c248_p17_0_1 >= 1
exclude_solution_dist_pair_c249_p18: diff_dist_pair_c249_p18_0_1 >= 1
exclude_solution_dist_pair_c250_p19: diff_dist_pair_c250_p19_0_1 >= 1
exclude_solution_dist_pair_c251_p20: diff_dist_pair_c251_p20_0_1 >= 1
exclude_solution_dist_pair_c252_p21: diff_dist_pair_c252_p21_0_1 >= 1
exclude_solution_pos_c231_p0: diff_pos_c231_p0_1 >= 1
exclude_solution_pos_c232_p1: diff_pos_c232_p1_1 >= 1
exclude_solution_pos_c233_p2: diff_pos_c233_p2_1 >= 1
exclude_solution_pos_c234_p3: diff_pos_c234_p3_1 >= 1
exclude_solution_pos_c235_p4: diff_pos_c235_p4_1 >= 1
exclude_solution_pos_c236_p5: diff_pos_c236_p5_1 >= 1
exclude_solution_pos_c237_p6: diff_pos_c237_p6_1 >= 1
exclude_solution_pos_c238_p7: diff_pos_c238_p7_1 >= 1
exclude_solution_pos_c239_p8: diff_pos_c239_p8_1 >= 1
exclude_solution_pos_c240_p9: diff_pos_c240_p9_1 >= 1
exclude_solution_pos_c241_p10: diff_pos_c241_p10_1 >= 1
exclude_solution_pos_c242_p11: diff_pos_c242_p11_1 >= 1
exclude_solution_pos_c243_p12: diff_pos_c243_p12_1 >= 1
exclude_solution_pos_c244_p13: diff_pos_c244_p13_1 >= 1
exclude_solution_pos_c245_p14: diff_pos_c245_p14_1 >= 1
exclude_solution_pos_c246_p15: diff_pos_c246_p15_1 >= 1
exclude_solution_pos_c247_p16: diff_pos_c247_p16_1 >= 1
exclude_solution_pos_c248_p17: diff_pos_c248_p17_1 >= 1
exclude_solution_pos_c249_p18: diff_pos_c249_p18_1 >= 1
exclude_solution_pos_c250_p19: diff_pos_c250_p19_1 >= 1
exclude_solution_pos_c251_p20: diff_pos_c251_p20_1 >= 1
exclude_solution_pos_c252_p21: diff_pos_c252_p21_1 >= 1
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
x_diff_from_minus_c231_p0_1: x_diff_c231_p0_1 - x_diff_minus_c231_p0_1 >= 0
x_diff_from_minus_c232_p1_1: x_diff_c232_p1_1 - x_diff_minus_c232_p1_1 >= 0
x_diff_from_minus_c233_p2_1: x_diff_c233_p2_1 - x_diff_minus_c233_p2_1 >= 0
x_diff_from_minus_c234_p3_1: x_diff_c234_p3_1 - x_diff_minus_c234_p3_1 >= 0
x_diff_from_minus_c235_p4_1: x_diff_c235_p4_1 - x_diff_minus_c235_p4_1 >= 0
x_diff_from_minus_c236_p5_1: x_diff_c236_p5_1 - x_diff_minus_c236_p5_1 >= 0
x_diff_from_minus_c237_p6_1: x_diff_c237_p6_1 - x_diff_minus_c237_p6_1 >= 0
x_diff_from_minus_c238_p7_1: x_diff_c238_p7_1 - x_diff_minus_c238_p7_1 >= 0
x_diff_from_minus_c239_p8_1: x_diff_c239_p8_1 - x_diff_minus_c239_p8_1 >= 0
x_diff_from_minus_c240_p9_1: x_diff_c240_p9_1 - x_diff_minus_c240_p9_1 >= 0
x_diff_from_minus_c241_p10_1: x_diff_c241_p10_1 - x_diff_minus_c241_p10_1 >= 0
x_diff_from_minus_c242_p11_1: x_diff_c242_p11_1 - x_diff_minus_c242_p11_1 >= 0
x_diff_from_minus_c243_p12_1: x_diff_c243_p12_1 - x_diff_minus_c243_p12_1 >= 0
x_diff_from_minus_c244_p13_1: x_diff_c244_p13_1 - x_diff_minus_c244_p13_1 >= 0
x_diff_from_minus_c245_p14_1: x_diff_c245_p14_1 - x_diff_minus_c245_p14_1 >= 0
x_diff_from_minus_c246_p15_1: x_diff_c246_p15_1 - x_diff_minus_c246_p15_1 >= 0
x_diff_from_minus_c247_p16_1: x_diff_c247_p16_1 - x_diff_minus_c247_p16_1 >= 0
x_diff_from_minus_c248_p17_1: x_diff_c248_p17_1 - x_diff_minus_c248_p17_1 >= 0
x_diff_from_minus_c249_p18_1: x_diff_c249_p18_1 - x_diff_minus_c249_p18_1 >= 0
x_diff_from_minus_c250_p19_1: x_diff_c250_p19_1 - x_diff_minus_c250_p19_1 >= 0
x_diff_from_minus_c251_p20_1: x_diff_c251_p20_1 - x_diff_minus_c251_p20_1 >= 0
x_diff_from_minus_c252_p21_1: x_diff_c252_p21_1 - x_diff_minus_c252_p21_1 >= 0
x_diff_from_plus_c231_p0_1: x_diff_c231_p0_1 - x_diff_plus_c231_p0_1 >= 0
x_diff_from_plus_c232_p1_1: x_diff_c232_p1_1 - x_diff_plus_c232_p1_1 >= 0
x_diff_from_plus_c233_p2_1: x_diff_c233_p2_1 - x_diff_plus_c233_p2_1 >= 0
x_diff_from_plus_c234_p3_1: x_diff_c234_p3_1 - x_diff_plus_c234_p3_1 >= 0
x_diff_from_plus_c235_p4_1: x_diff_c235_p4_1 - x_diff_plus_c235_p4_1 >= 0
x_diff_from_plus_c236_p5_1: x_diff_c236_p5_1 - x_diff_plus_c236_p5_1 >= 0
x_diff_from_plus_c237_p6_1: x_diff_c237_p6_1 - x_diff_plus_c237_p6_1 >= 0
x_diff_from_plus_c238_p7_1: x_diff_c238_p7_1 - x_diff_plus_c238_p7_1 >= 0
x_diff_from_plus_c239_p8_1: x_diff_c239_p8_1 - x_diff_plus_c239_p8_1 >= 0
x_diff_from_plus_c240_p9_1: x_diff_c240_p9_1 - x_diff_plus_c240_p9_1 >= 0
x_diff_from_plus_c241_p10_1: x_diff_c241_p10_1 - x_diff_plus_c241_p10_1 >= 0
x_diff_from_plus_c242_p11_1: x_diff_c242_p11_1 - x_diff_plus_c242_p11_1 >= 0
x_diff_from_plus_c243_p12_1: x_diff_c243_p12_1 - x_diff_plus_c243_p12_1 >= 0
x_diff_from_plus_c244_p13_1: x_diff_c244_p13_1 - x_diff_plus_c244_p13_1 >= 0
x_diff_from_plus_c245_p14_1: x_diff_c245_p14_1 - x_diff_plus_c245_p14_1 >= 0
x_diff_from_plus_c246_p15_1: x_diff_c246_p15_1 - x_diff_plus_c246_p15_1 >= 0
x_diff_from_plus_c247_p16_1: x_diff_c247_p16_1 - x_diff_plus_c247_p16_1 >= 0
x_diff_from_plus_c248_p17_1: x_diff_c248_p17_1 - x_diff_plus_c248_p17_1 >= 0
x_diff_from_plus_c249_p18_1: x_diff_c249_p18_1 - x_diff_plus_c249_p18_1 >= 0
x_diff_from_plus_c250_p19_1: x_diff_c250_p19_1 - x_diff_plus_c250_p19_1 >= 0
x_diff_from_plus_c251_p20_1: x_diff_c251_p20_1 - x_diff_plus_c251_p20_1 >= 0
x_diff_from_plus_c252_p21_1: x_diff_c252_p21_1 - x_diff_plus_c252_p21_1 >= 0
x_diff_minus_lower_c231_p0_1: x_1 + 60 x_diff_minus_c231_p0_1 <= 67
x_diff_minus_lower_c232_p1_1: x_1 + 60 x_diff_minus_c232_p1_1 <= 57
x_diff_minus_lower_c233_p2_1: x_1 + 60 x_diff_minus_c233_p2_1 <= 67
x_diff_minus_lower_c234_p3_1: x_1 + 60 x_diff_minus_c234_p3_1 <= 77
x_diff_minus_lower_c235_p4_1: x_1 + 60 x_diff_minus_c235_p4_1 <= 70
x_diff_minus_lower_c236_p5_1: x_1 + 60 x_diff_minus_c236_p5_1 <= 64
x_diff_minus_lower_c237_p6_1: x_1 + 60 x_diff_minus_c237_p6_1 <= 70
x_diff_minus_lower_c238_p7_1: x_1 + 60 x_diff_minus_c238_p7_1 <= 77
x_diff_minus_lower_c239_p8_1: x_1 + 60 x_diff_minus_c239_p8_1 <= 77
x_diff_minus_lower_c240_p9_1: x_1 + 60 x_diff_minus_c240_p9_1 <= 57
x_diff_minus_lower_c241_p10_1: x_1 + 60 x_diff_minus_c241_p10_1 <= 57
x_diff_minus_lower_c242_p11_1: x_1 + 60 x_diff_minus_c242_p11_1 <= 64
x_diff_minus_lower_c243_p12_1: x_1 + 60 x_diff_minus_c243_p12_1 <= 77
x_diff_minus_lower_c244_p13_1: x_1 + 60 x_diff_minus_c244_p13_1 <= 57
x_diff_minus_lower_c245_p14_1: x_1 + 60 x_diff_minus_c245_p14_1 <= 73
x_diff_minus_lower_c246_p15_1: x_1 + 60 x_diff_minus_c246_p15_1 <= 73
x_diff_minus_lower_c247_p16_1: x_1 + 60 x_diff_minus_c247_p16_1 <= 77
x_diff_minus_lower_c248_p17_1: x_1 + 60 x_diff_minus_c248_p17_1 <= 61
x_diff_minus_lower_c249_p18_1: x_1 + 60 x_diff_minus_c249_p18_1 <= 61
x_diff_minus_lower_c250_p19_1: x_1 + 60 x_diff_minus_c250_p19_1 <= 57
x_diff_minus_lower_c251_p20_1: x_1 + 60 x_diff_minus_c251_p20_1 <= 57
x_diff_minus_lower_c252_p21_1: x_1 + 60 x_diff_minus_c252_p21_1 <= 76
x_diff_minus_upper_c231_p0_1: x_1 + 60 x_diff_minus_c231_p0_1 >= 7.001
x_diff_minus_upper_c232_p1_1: x_1 + 60 x_diff_minus_c232_p1_1 >= -2.999
x_diff_minus_upper_c233_p2_1: x_1 + 60 x_diff_minus_c233_p2_1 >= 7.001
x_diff_minus_upper_c234_p3_1: x_1 + 60 x_diff_minus_c234_p3_1 >= 17.001
x_diff_minus_upper_c235_p4_1: x_1 + 60 x_diff_minus_c235_p4_1 >= 10.001
x_diff_minus_upper_c236_p5_1: x_1 + 60 x_diff_minus_c236_p5_1 >= 4.001
x_diff_minus_upper_c237_p6_1: x_1 + 60 x_diff_minus_c237_p6_1 >= 10.001
x_diff_minus_upper_c238_p7_1: x_1 + 60 x_diff_minus_c238_p7_1 >= 17.001
x_diff_minus_upper_c239_p8_1: x_1 + 60 x_diff_minus_c239_p8_1 >= 17.001
x_diff_minus_upper_c240_p9_1: x_1 + 60 x_diff_minus_c240_p9_1 >= -2.999
x_diff_minus_upper_c241_p10_1: x_1 + 60 x_diff_minus_c241_p10_1 >= -2.999
x_diff_minus_upper_c242_p11_1: x_1 + 60 x_diff_minus_c242_p11_1 >= 4.001
x_diff_minus_upper_c243_p12_1: x_1 + 60 x_diff_minus_c243_p12_1 >= 17.001
x_diff_minus_upper_c244_p13_1: x_1 + 60 x_diff_minus_c244_p13_1 >= -2.999
x_diff_minus_upper_c245_p14_1: x_1 + 60 x_diff_minus_c245_p14_1 >= 13.001
x_diff_minus_upper_c246_p15_1: x_1 + 60 x_diff_minus_c246_p15_1 >= 13.001
x_diff_minus_upper_c247_p16_1: x_1 + 60 x_diff_minus_c247_p16_1 >= 17.001
x_diff_minus_upper_c248_p17_1: x_1 + 60 x_diff_minus_c248_p17_1 >= 1.001
x_diff_minus_upper_c249_p18_1: x_1 + 60 x_diff_minus_c249_p18_1 >= 1.001
x_diff_minus_upper_c250_p19_1: x_1 + 60 x_diff_minus_c250_p19_1 >= -2.999
x_diff_minus_upper_c251_p20_1: x_1 + 60 x_diff_minus_c251_p20_1 >= -2.999
x_diff_minus_upper_c252_p21_1: x_1 + 60 x_diff_minus_c252_p21_1 >= 16.001
x_diff_plus_lower_c231_p0_1: x_1 - 60 x_diff_plus_c231_p0_1 >= -47
x_diff_plus_lower_c232_p1_1: x_1 - 60 x_diff_plus_c232_p1_1 >= -57
x_diff_plus_lower_c233_p2_1: x_1 - 60 x_diff_plus_c233_p2_1 >= -47
x_diff_plus_lower_c234_p3_1: x_1 - 60 x_diff_plus_c234_p3_1 >= -37
x_diff_plus_lower_c235_p4_1: x_1 - 60 x_diff_plus_c235_p4_1 >= -44
x_diff_plus_lower_c236_p5_1: x_1 - 60 x_diff_plus_c236_p5_1 >= -50
x_diff_plus_lower_c237_p6_1: x_1 - 60 x_diff_plus_c237_p6_1 >= -44
x_diff_plus_lower_c238_p7_1: x_1 - 60 x_diff_plus_c238_p7_1 >= -37
x_diff_plus_lower_c239_p8_1: x_1 - 60 x_diff_plus_c239_p8_1 >= -37
x_diff_plus_lower_c240_p9_1: x_1 - 60 x_diff_plus_c240_p9_1 >= -57
x_diff_plus_lower_c241_p10_1: x_1 - 60 x_diff_plus_c241_p10_1 >= -57
x_diff_plus_lower_c242_p11_1: x_1 - 60 x_diff_plus_c242_p11_1 >= -50
x_diff_plus_lower_c243_p12_1: x_1 - 60 x_diff_plus_c243_p12_1 >= -37
x_diff_plus_lower_c244_p13_1: x_1 - 60 x_diff_plus_c244_p13_1 >= -57
x_diff_plus_lower_c245_p14_1: x_1 - 60 x_diff_plus_c245_p14_1 >= -41
x_diff_plus_lower_c246_p15_1: x_1 - 60 x_diff_plus_c246_p15_1 >= -41
x_diff_plus_lower_c247_p16_1: x_1 - 60 x_diff_plus_c247_p16_1 >= -37
x_diff_plus_lower_c248_p17_1: x_1 - 60 x_diff_plus_c248_p17_1 >= -53
x_diff_plus_lower_c249_p18_1: x_1 - 60 x_diff_plus_c249_p18_1 >= -53
x_diff_plus_lower_c250_p19_1: x_1 - 60 x_diff_plus_c250_p19_1 >= -57
x_diff_plus_lower_c251_p20_1: x_1 - 60 x_diff_plus_c251_p20_1 >= -57
x_diff_plus_lower_c252_p21_1: x_1 - 60 x_diff_plus_c252_p21_1 >= -38
x_diff_plus_upper_c231_p0_1: x_1 - 60 x_diff_plus_c231_p0_1 <= 12.999
x_diff_plus_upper_c232_p1_1: x_1 - 60 x_diff_plus_c232_p1_1 <= 2.999
x_diff_plus_upper_c233_p2_1: x_1 - 60 x_diff_plus_c233_p2_1 <= 12.999
x_diff_plus_upper_c234_p3_1: x_1 - 60 x_diff_plus_c234_p3_1 <= 22.999
x_diff_plus_upper_c235_p4_1: x_1 - 60 x_diff_plus_c235_p4_1 <= 15.999
x_diff_plus_upper_c236_p5_1: x_1 - 60 x_diff_plus_c236_p5_1 <= 9.999
x_diff_plus_upper_c237_p6_1: x_1 - 60 x_diff_plus_c237_p6_1 <= 15.999
x_diff_plus_upper_c238_p7_1: x_1 - 60 x_diff_plus_c238_p7_1 <= 22.999
x_diff_plus_upper_c239_p8_1: x_1 - 60 x_diff_plus_c239_p8_1 <= 22.999
x_diff_plus_upper_c240_p9_1: x_1 - 60 x_diff_plus_c240_p9_1 <= 2.999
x_diff_plus_upper_c241_p10_1: x_1 - 60 x_diff_plus_c241_p10_1 <= 2.999
x_diff_plus_upper_c242_p11_1: x_1 - 60 x_diff_plus_c242_p11_1 <= 9.999
x_diff_plus_upper_c243_p12_1: x_1 - 60 x_diff_plus_c243_p12_1 <= 22.999
x_diff_plus_upper_c244_p13_1: x_1 - 60 x_diff_plus_c244_p13_1 <= 2.999
x_diff_plus_upper_c245_p14_1: x_1 - 60 x_diff_plus_c245_p14_1 <= 18.999
x_diff_plus_upper_c246_p15_1: x_1 - 60 x_diff_plus_c246_p15_1 <= 18.999
x_diff_plus_upper_c247_p16_1: x_1 - 60 x_diff_plus_c247_p16_1 <= 22.999
x_diff_plus_upper_c248_p17_1: x_1 - 60 x_diff_plus_c248_p17_1 <= 6.999
x_diff_plus_upper_c249_p18_1: x_1 - 60 x_diff_plus_c249_p18_1 <= 6.999
x_diff_plus_upper_c250_p19_1: x_1 - 60 x_diff_plus_c250_p19_1 <= 2.999
x_diff_plus_upper_c251_p20_1: x_1 - 60 x_diff_plus_c251_p20_1 <= 2.999
x_diff_plus_upper_c252_p21_1: x_1 - 60 x_diff_plus_c252_p21_1 <= 21.999
x_diff_upper_c231_p0_1: x_diff_c231_p0_1 - x_diff_minus_c231_p0_1
 - x_diff_plus_c231_p0_1 <= 0
x_diff_upper_c232_p1_1: x_diff_c232_p1_1 - x_diff_minus_c232_p1_1
 - x_diff_plus_c232_p1_1 <= 0
x_diff_upper_c233_p2_1: x_diff_c233_p2_1 - x_diff_minus_c233_p2_1
 - x_diff_plus_c233_p2_1 <= 0
x_diff_upper_c234_p3_1: x_diff_c234_p3_1 - x_diff_minus_c234_p3_1
 - x_diff_plus_c234_p3_1 <= 0
x_diff_upper_c235_p4_1: x_diff_c235_p4_1 - x_diff_minus_c235_p4_1
 - x_diff_plus_c235_p4_1 <= 0
x_diff_upper_c236_p5_1: x_diff_c236_p5_1 - x_diff_minus_c236_p5_1
 - x_diff_plus_c236_p5_1 <= 0
x_diff_upper_c237_p6_1: x_diff_c237_p6_1 - x_diff_minus_c237_p6_1
 - x_diff_plus_c237_p6_1 <= 0
x_diff_upper_c238_p7_1: x_diff_c238_p7_1 - x_diff_minus_c238_p7_1
 - x_diff_plus_c238_p7_1 <= 0
x_diff_upper_c239_p8_1: x_diff_c239_p8_1 - x_diff_minus_c239_p8_1
 - x_diff_plus_c239_p8_1 <= 0
x_diff_upper_c240_p9_1: x_diff_c240_p9_1 - x_diff_minus_c240_p9_1
 - x_diff_plus_c240_p9_1 <= 0
x_diff_upper_c241_p10_1: x_diff_c241_p10_1 - x_diff_minus_c241_p10_1
 - x_diff_plus_c241_p10_1 <= 0
x_diff_upper_c242_p11_1: x_diff_c242_p11_1 - x_diff_minus_c242_p11_1
 - x_diff_plus_c242_p11_1 <= 0
x_diff_upper_c243_p12_1: x_diff_c243_p12_1 - x_diff_minus_c243_p12_1
 - x_diff_plus_c243_p12_1 <= 0
x_diff_upper_c244_p13_1: x_diff_c244_p13_1 - x_diff_minus_c244_p13_1
 - x_diff_plus_c244_p13_1 <= 0
x_diff_upper_c245_p14_1: x_diff_c245_p14_1 - x_diff_minus_c245_p14_1
 - x_diff_plus_c245_p14_1 <= 0
x_diff_upper_c246_p15_1: x_diff_c246_p15_1 - x_diff_minus_c246_p15_1
 - x_diff_plus_c246_p15_1 <= 0
x_diff_upper_c247_p16_1: x_diff_c247_p16_1 - x_diff_minus_c247_p16_1
 - x_diff_plus_c247_p16_1 <= 0
x_diff_upper_c248_p17_1: x_diff_c248_p17_1 - x_diff_minus_c248_p17_1
 - x_diff_plus_c248_p17_1 <= 0
x_diff_upper_c249_p18_1: x_diff_c249_p18_1 - x_diff_minus_c249_p18_1
 - x_diff_plus_c249_p18_1 <= 0
x_diff_upper_c250_p19_1: x_diff_c250_p19_1 - x_diff_minus_c250_p19_1
 - x_diff_plus_c250_p19_1 <= 0
x_diff_upper_c251_p20_1: x_diff_c251_p20_1 - x_diff_minus_c251_p20_1
 - x_diff_plus_c251_p20_1 <= 0
x_diff_upper_c252_p21_1: x_diff_c252_p21_1 - x_diff_minus_c252_p21_1
 - x_diff_plus_c252_p21_1 <= 0
x_grid_to_real_0: x_0 - 0.5 x_grid_0 = 0
x_grid_to_real_1: x_1 - 0.5 x_grid_1 = 0
y_diff_from_minus_c231_p0_1: y_diff_c231_p0_1 - y_diff_minus_c231_p0_1 >= 0
y_diff_from_minus_c232_p1_1: y_diff_c232_p1_1 - y_diff_minus_c232_p1_1 >= 0
y_diff_from_minus_c233_p2_1: y_diff_c233_p2_1 - y_diff_minus_c233_p2_1 >= 0
y_diff_from_minus_c234_p3_1: y_diff_c234_p3_1 - y_diff_minus_c234_p3_1 >= 0
y_diff_from_minus_c235_p4_1: y_diff_c235_p4_1 - y_diff_minus_c235_p4_1 >= 0
y_diff_from_minus_c236_p5_1: y_diff_c236_p5_1 - y_diff_minus_c236_p5_1 >= 0
y_diff_from_minus_c237_p6_1: y_diff_c237_p6_1 - y_diff_minus_c237_p6_1 >= 0
y_diff_from_minus_c238_p7_1: y_diff_c238_p7_1 - y_diff_minus_c238_p7_1 >= 0
y_diff_from_minus_c239_p8_1: y_diff_c239_p8_1 - y_diff_minus_c239_p8_1 >= 0
y_diff_from_minus_c240_p9_1: y_diff_c240_p9_1 - y_diff_minus_c240_p9_1 >= 0
y_diff_from_minus_c241_p10_1: y_diff_c241_p10_1 - y_diff_minus_c241_p10_1 >= 0
y_diff_from_minus_c242_p11_1: y_diff_c242_p11_1 - y_diff_minus_c242_p11_1 >= 0
y_diff_from_minus_c243_p12_1: y_diff_c243_p12_1 - y_diff_minus_c243_p12_1 >= 0
y_diff_from_minus_c244_p13_1: y_diff_c244_p13_1 - y_diff_minus_c244_p13_1 >= 0
y_diff_from_minus_c245_p14_1: y_diff_c245_p14_1 - y_diff_minus_c245_p14_1 >= 0
y_diff_from_minus_c246_p15_1: y_diff_c246_p15_1 - y_diff_minus_c246_p15_1 >= 0
y_diff_from_minus_c247_p16_1: y_diff_c247_p16_1 - y_diff_minus_c247_p16_1 >= 0
y_diff_from_minus_c248_p17_1: y_diff_c248_p17_1 - y_diff_minus_c248_p17_1 >= 0
y_diff_from_minus_c249_p18_1: y_diff_c249_p18_1 - y_diff_minus_c249_p18_1 >= 0
y_diff_from_minus_c250_p19_1: y_diff_c250_p19_1 - y_diff_minus_c250_p19_1 >= 0
y_diff_from_minus_c251_p20_1: y_diff_c251_p20_1 - y_diff_minus_c251_p20_1 >= 0
y_diff_from_minus_c252_p21_1: y_diff_c252_p21_1 - y_diff_minus_c252_p21_1 >= 0
y_diff_from_plus_c231_p0_1: y_diff_c231_p0_1 - y_diff_plus_c231_p0_1 >= 0
y_diff_from_plus_c232_p1_1: y_diff_c232_p1_1 - y_diff_plus_c232_p1_1 >= 0
y_diff_from_plus_c233_p2_1: y_diff_c233_p2_1 - y_diff_plus_c233_p2_1 >= 0
y_diff_from_plus_c234_p3_1: y_diff_c234_p3_1 - y_diff_plus_c234_p3_1 >= 0
y_diff_from_plus_c235_p4_1: y_diff_c235_p4_1 - y_diff_plus_c235_p4_1 >= 0
y_diff_from_plus_c236_p5_1: y_diff_c236_p5_1 - y_diff_plus_c236_p5_1 >= 0
y_diff_from_plus_c237_p6_1: y_diff_c237_p6_1 - y_diff_plus_c237_p6_1 >= 0
y_diff_from_plus_c238_p7_1: y_diff_c238_p7_1 - y_diff_plus_c238_p7_1 >= 0
y_diff_from_plus_c239_p8_1: y_diff_c239_p8_1 - y_diff_plus_c239_p8_1 >= 0
y_diff_from_plus_c240_p9_1: y_diff_c240_p9_1 - y_diff_plus_c240_p9_1 >= 0
y_diff_from_plus_c241_p10_1: y_diff_c241_p10_1 - y_diff_plus_c241_p10_1 >= 0
y_diff_from_plus_c242_p11_1: y_diff_c242_p11_1 - y_diff_plus_c242_p11_1 >= 0
y_diff_from_plus_c243_p12_1: y_diff_c243_p12_1 - y_diff_plus_c243_p12_1 >= 0
y_diff_from_plus_c244_p13_1: y_diff_c244_p13_1 - y_diff_plus_c244_p13_1 >= 0
y_diff_from_plus_c245_p14_1: y_diff_c245_p14_1 - y_diff_plus_c245_p14_1 >= 0
y_diff_from_plus_c246_p15_1: y_diff_c246_p15_1 - y_diff_plus_c246_p15_1 >= 0
y_diff_from_plus_c247_p16_1: y_diff_c247_p16_1 - y_diff_plus_c247_p16_1 >= 0
y_diff_from_plus_c248_p17_1: y_diff_c248_p17_1 - y_diff_plus_c248_p17_1 >= 0
y_diff_from_plus_c249_p18_1: y_diff_c249_p18_1 - y_diff_plus_c249_p18_1 >= 0
y_diff_from_plus_c250_p19_1: y_diff_c250_p19_1 - y_diff_plus_c250_p19_1 >= 0
y_diff_from_plus_c251_p20_1: y_diff_c251_p20_1 - y_diff_plus_c251_p20_1 >= 0
y_diff_from_plus_c252_p21_1: y_diff_c252_p21_1 - y_diff_plus_c252_p21_1 >= 0
y_diff_minus_lower_c231_p0_1: y_1 + 60 y_diff_minus_c231_p0_1 <= 77
y_diff_minus_lower_c232_p1_1: y_1 + 60 y_diff_minus_c232_p1_1 <= 67
y_diff_minus_lower_c233_p2_1: y_1 + 60 y_diff_minus_c233_p2_1 <= 57
y_diff_minus_lower_c234_p3_1: y_1 + 60 y_diff_minus_c234_p3_1 <= 67
y_diff_minus_lower_c235_p4_1: y_1 + 60 y_diff_minus_c235_p4_1 <= 77
y_diff_minus_lower_c236_p5_1: y_1 + 60 y_diff_minus_c236_p5_1 <= 77
y_diff_minus_lower_c237_p6_1: y_1 + 60 y_diff_minus_c237_p6_1 <= 57
y_diff_minus_lower_c238_p7_1: y_1 + 60 y_diff_minus_c238_p7_1 <= 64
y_diff_minus_lower_c239_p8_1: y_1 + 60 y_diff_minus_c239_p8_1 <= 70
y_diff_minus_lower_c240_p9_1: y_1 + 60 y_diff_minus_c240_p9_1 <= 70
y_diff_minus_lower_c241_p10_1: y_1 + 60 y_diff_minus_c241_p10_1 <= 64
y_diff_minus_lower_c242_p11_1: y_1 + 60 y_diff_minus_c242_p11_1 <= 57
y_diff_minus_lower_c243_p12_1: y_1 + 60 y_diff_minus_c243_p12_1 <= 61
y_diff_minus_lower_c244_p13_1: y_1 + 60 y_diff_minus_c244_p13_1 <= 73
y_diff_minus_lower_c245_p14_1: y_1 + 60 y_diff_minus_c245_p14_1 <= 57
y_diff_minus_lower_c246_p15_1: y_1 + 60 y_diff_minus_c246_p15_1 <= 77
y_diff_minus_lower_c247_p16_1: y_1 + 60 y_diff_minus_c247_p16_1 <= 73
y_diff_minus_lower_c248_p17_1: y_1 + 60 y_diff_minus_c248_p17_1 <= 77
y_diff_minus_lower_c249_p18_1: y_1 + 60 y_diff_minus_c249_p18_1 <= 57
y_diff_minus_lower_c250_p19_1: y_1 + 60 y_diff_minus_c250_p19_1 <= 61
y_diff_minus_lower_c251_p20_1: y_1 + 60 y_diff_minus_c251_p20_1 <= 58
y_diff_minus_lower_c252_p21_1: y_1 + 60 y_diff_minus_c252_p21_1 <= 57
y_diff_minus_upper_c231_p0_1: y_1 + 60 y_diff_minus_c231_p0_1 >= 17.001
y_diff_minus_upper_c232_p1_1: y_1 + 60 y_diff_minus_c232_p1_1 >= 7.001
y_diff_minus_upper_c233_p2_1: y_1 + 60 y_diff_minus_c233_p2_1 >= -2.999
y_diff_minus_upper_c234_p3_1: y_1 + 60 y_diff_minus_c234_p3_1 >= 7.001
y_diff_minus_upper_c235_p4_1: y_1 + 60 y_diff_minus_c235_p4_1 >= 17.001
y_diff_minus_upper_c236_p5_1: y_1 + 60 y_diff_minus_c236_p5_1 >= 17.001
y_diff_minus_upper_c237_p6_1: y_1 + 60 y_diff_minus_c237_p6_1 >= -2.999
y_diff_minus_upper_c238_p7_1: y_1 + 60 y_diff_minus_c238_p7_1 >= 4.001
y_diff_minus_upper_c239_p8_1: y_1 + 60 y_diff_minus_c239_p8_1 >= 10.001
y_diff_minus_upper_c240_p9_1: y_1 + 60 y_diff_minus_c240_p9_1 >= 10.001
y_diff_minus_upper_c241_p10_1: y_1 + 60 y_diff_minus_c241_p10_1 >= 4.001
y_diff_minus_upper_c242_p11_1: y_1 + 60 y_diff_minus_c242_p11_1 >= -2.999
y_diff_minus_upper_c243_p12_1: y_1 + 60 y_diff_minus_c243_p12_1 >= 1.001
y_diff_minus_upper_c244_p13_1: y_1 + 60 y_diff_minus_c244_p13_1 >= 13.001
y_diff_minus_upper_c245_p14_1: y_1 + 60 y_diff_minus_c245_p14_1 >= -2.999
y_diff_minus_upper_c246_p15_1: y_1 + 60 y_diff_minus_c246_p15_1 >= 17.001
y_diff_minus_upper_c247_p16_1: y_1 + 60 y_diff_minus_c247_p16_1 >= 13.001
y_diff_minus_upper_c248_p17_1: y_1 + 60 y_diff_minus_c248_p17_1 >= 17.001
y_diff_minus_upper_c249_p18_1: y_1 + 60 y_diff_minus_c249_p18_1 >= -2.999
y_diff_minus_upper_c250_p19_1: y_1 + 60 y_diff_minus_c250_p19_1 >= 1.001
y_diff_minus_upper_c251_p20_1: y_1 + 60 y_diff_minus_c251_p20_1 >= -1.999
y_diff_minus_upper_c252_p21_1: y_1 + 60 y_diff_minus_c252_p21_1 >= -2.999
y_diff_plus_lower_c231_p0_1: y_1 - 60 y_diff_plus_c231_p0_1 >= -37
y_diff_plus_lower_c232_p1_1: y_1 - 60 y_diff_plus_c232_p1_1 >= -47
y_diff_plus_lower_c233_p2_1: y_1 - 60 y_diff_plus_c233_p2_1 >= -57
y_diff_plus_lower_c234_p3_1: y_1 - 60 y_diff_plus_c234_p3_1 >= -47
y_diff_plus_lower_c235_p4_1: y_1 - 60 y_diff_plus_c235_p4_1 >= -37
y_diff_plus_lower_c236_p5_1: y_1 - 60 y_diff_plus_c236_p5_1 >= -37
y_diff_plus_lower_c237_p6_1: y_1 - 60 y_diff_plus_c237_p6_1 >= -57
y_diff_plus_lower_c238_p7_1: y_1 - 60 y_diff_plus_c238_p7_1 >= -50
y_diff_plus_lower_c239_p8_1: y_1 - 60 y_diff_plus_c239_p8_1 >= -44
y_diff_plus_lower_c240_p9_1: y_1 - 60 y_diff_plus_c240_p9_1 >= -44
y_diff_plus_lower_c241_p10_1: y_1 - 60 y_diff_plus_c241_p10_1 >= -50
y_diff_plus_lower_c242_p11_1: y_1 - 60 y_diff_plus_c242_p11_1 >= -57
y_diff_plus_lower_c243_p12_1: y_1 - 60 y_diff_plus_c243_p12_1 >= -53
y_diff_plus_lower_c244_p13_1: y_1 - 60 y_diff_plus_c244_p13_1 >= -41
y_diff_plus_lower_c245_p14_1: y_1 - 60 y_diff_plus_c245_p14_1 >= -57
y_diff_plus_lower_c246_p15_1: y_1 - 60 y_diff_plus_c246_p15_1 >= -37
y_diff_plus_lower_c247_p16_1: y_1 - 60 y_diff_plus_c247_p16_1 >= -41
y_diff_plus_lower_c248_p17_1: y_1 - 60 y_diff_plus_c248_p17_1 >= -37
y_diff_plus_lower_c249_p18_1: y_1 - 60 y_diff_plus_c249_p18_1 >= -57
y_diff_plus_lower_c250_p19_1: y_1 - 60 y_diff_plus_c250_p19_1 >= -53
y_diff_plus_lower_c251_p20_1: y_1 - 60 y_diff_plus_c251_p20_1 >= -56
y_diff_plus_lower_c252_p21_1: y_1 - 60 y_diff_plus_c252_p21_1 >= -57
y_diff_plus_upper_c231_p0_1: y_1 - 60 y_diff_plus_c231_p0_1 <= 22.999
y_diff_plus_upper_c232_p1_1: y_1 - 60 y_diff_plus_c232_p1_1 <= 12.999
y_diff_plus_upper_c233_p2_1: y_1 - 60 y_diff_plus_c233_p2_1 <= 2.999
y_diff_plus_upper_c234_p3_1: y_1 - 60 y_diff_plus_c234_p3_1 <= 12.999
y_diff_plus_upper_c235_p4_1: y_1 - 60 y_diff_plus_c235_p4_1 <= 22.999
y_diff_plus_upper_c236_p5_1: y_1 - 60 y_diff_plus_c236_p5_1 <= 22.999
y_diff_plus_upper_c237_p6_1: y_1 - 60 y_diff_plus_c237_p6_1 <= 2.999
y_diff_plus_upper_c238_p7_1: y_1 - 60 y_diff_plus_c238_p7_1 <= 9.999
y_diff_plus_upper_c239_p8_1: y_1 - 60 y_diff_plus_c239_p8_1 <= 15.999
y_diff_plus_upper_c240_p9_1: y_1 - 60 y_diff_plus_c240_p9_1 <= 15.999
y_diff_plus_upper_c241_p10_1: y_1 - 60 y_diff_plus_c241_p10_1 <= 9.999
y_diff_plus_upper_c242_p11_1: y_1 - 60 y_diff_plus_c242_p11_1 <= 2.999
y_diff_plus_upper_c243_p12_1: y_1 - 60 y_diff_plus_c243_p12_1 <= 6.999
y_diff_plus_upper_c244_p13_1: y_1 - 60 y_diff_plus_c244_p13_1 <= 18.999
y_diff_plus_upper_c245_p14_1: y_1 - 60 y_diff_plus_c245_p14_1 <= 2.999
y_diff_plus_upper_c246_p15_1: y_1 - 60 y_diff_plus_c246_p15_1 <= 22.999
y_diff_plus_upper_c247_p16_1: y_1 - 60 y_diff_plus_c247_p16_1 <= 18.999
y_diff_plus_upper_c248_p17_1: y_1 - 60 y_diff_plus_c248_p17_1 <= 22.999
y_diff_plus_upper_c249_p18_1: y_1 - 60 y_diff_plus_c249_p18_1 <= 2.999
y_diff_plus_upper_c250_p19_1: y_1 - 60 y_diff_plus_c250_p19_1 <= 6.999
y_diff_plus_upper_c251_p20_1: y_1 - 60 y_diff_plus_c251_p20_1 <= 3.999
y_diff_plus_upper_c252_p21_1: y_1 - 60 y_diff_plus_c252_p21_1 <= 2.999
y_diff_upper_c231_p0_1: y_diff_c231_p0_1 - y_diff_minus_c231_p0_1
 - y_diff_plus_c231_p0_1 <= 0
y_diff_upper_c232_p1_1: y_diff_c232_p1_1 - y_diff_minus_c232_p1_1
 - y_diff_plus_c232_p1_1 <= 0
y_diff_upper_c233_p2_1: y_diff_c233_p2_1 - y_diff_minus_c233_p2_1
 - y_diff_plus_c233_p2_1 <= 0
y_diff_upper_c234_p3_1: y_diff_c234_p3_1 - y_diff_minus_c234_p3_1
 - y_diff_plus_c234_p3_1 <= 0
y_diff_upper_c235_p4_1: y_diff_c235_p4_1 - y_diff_minus_c235_p4_1
 - y_diff_plus_c235_p4_1 <= 0
y_diff_upper_c236_p5_1: y_diff_c236_p5_1 - y_diff_minus_c236_p5_1
 - y_diff_plus_c236_p5_1 <= 0
y_diff_upper_c237_p6_1: y_diff_c237_p6_1 - y_diff_minus_c237_p6_1
 - y_diff_plus_c237_p6_1 <= 0
y_diff_upper_c238_p7_1: y_diff_c238_p7_1 - y_diff_minus_c238_p7_1
 - y_diff_plus_c238_p7_1 <= 0
y_diff_upper_c239_p8_1: y_diff_c239_p8_1 - y_diff_minus_c239_p8_1
 - y_diff_plus_c239_p8_1 <= 0
y_diff_upper_c240_p9_1: y_diff_c240_p9_1 - y_diff_minus_c240_p9_1
 - y_diff_plus_c240_p9_1 <= 0
y_diff_upper_c241_p10_1: y_diff_c241_p10_1 - y_diff_minus_c241_p10_1
 - y_diff_plus_c241_p10_1 <= 0
y_diff_upper_c242_p11_1: y_diff_c242_p11_1 - y_diff_minus_c242_p11_1
 - y_diff_plus_c242_p11_1 <= 0
y_diff_upper_c243_p12_1: y_diff_c243_p12_1 - y_diff_minus_c243_p12_1
 - y_diff_plus_c243_p12_1 <= 0
y_diff_upper_c244_p13_1: y_diff_c244_p13_1 - y_diff_minus_c244_p13_1
 - y_diff_plus_c244_p13_1 <= 0
y_diff_upper_c245_p14_1: y_diff_c245_p14_1 - y_diff_minus_c245_p14_1
 - y_diff_plus_c245_p14_1 <= 0
y_diff_upper_c246_p15_1: y_diff_c246_p15_1 - y_diff_minus_c246_p15_1
 - y_diff_plus_c246_p15_1 <= 0
y_diff_upper_c247_p16_1: y_diff_c247_p16_1 - y_diff_minus_c247_p16_1
 - y_diff_plus_c247_p16_1 <= 0
y_diff_upper_c248_p17_1: y_diff_c248_p17_1 - y_diff_minus_c248_p17_1
 - y_diff_plus_c248_p17_1 <= 0
y_diff_upper_c249_p18_1: y_diff_c249_p18_1 - y_diff_minus_c249_p18_1
 - y_diff_plus_c249_p18_1 <= 0
y_diff_upper_c250_p19_1: y_diff_c250_p19_1 - y_diff_minus_c250_p19_1
 - y_diff_plus_c250_p19_1 <= 0
y_diff_upper_c251_p20_1: y_diff_c251_p20_1 - y_diff_minus_c251_p20_1
 - y_diff_plus_c251_p20_1 <= 0
y_diff_upper_c252_p21_1: y_diff_c252_p21_1 - y_diff_minus_c252_p21_1
 - y_diff_plus_c252_p21_1 <= 0
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
 dist_diff_abs_pair_c231_p0_0_1 <= 60
 dist_diff_abs_pair_c232_p1_0_1 <= 60
 dist_diff_abs_pair_c233_p2_0_1 <= 60
 dist_diff_abs_pair_c234_p3_0_1 <= 60
 dist_diff_abs_pair_c235_p4_0_1 <= 60
 dist_diff_abs_pair_c236_p5_0_1 <= 60
 dist_diff_abs_pair_c237_p6_0_1 <= 60
 dist_diff_abs_pair_c238_p7_0_1 <= 60
 dist_diff_abs_pair_c239_p8_0_1 <= 60
 dist_diff_abs_pair_c240_p9_0_1 <= 60
 dist_diff_abs_pair_c241_p10_0_1 <= 60
 dist_diff_abs_pair_c242_p11_0_1 <= 60
 dist_diff_abs_pair_c243_p12_0_1 <= 60
 dist_diff_abs_pair_c244_p13_0_1 <= 60
 dist_diff_abs_pair_c245_p14_0_1 <= 60
 dist_diff_abs_pair_c246_p15_0_1 <= 60
 dist_diff_abs_pair_c247_p16_0_1 <= 60
 dist_diff_abs_pair_c248_p17_0_1 <= 60
 dist_diff_abs_pair_c249_p18_0_1 <= 60
 dist_diff_abs_pair_c250_p19_0_1 <= 60
 dist_diff_abs_pair_c251_p20_0_1 <= 60
 dist_diff_abs_pair_c252_p21_0_1 <= 60
 -60 <= dist_diff_pair_c231_p0_0_1 <= 60
 -60 <= dist_diff_pair_c232_p1_0_1 <= 60
 -60 <= dist_diff_pair_c233_p2_0_1 <= 60
 -60 <= dist_diff_pair_c234_p3_0_1 <= 60
 -60 <= dist_diff_pair_c235_p4_0_1 <= 60
 -60 <= dist_diff_pair_c236_p5_0_1 <= 60
 -60 <= dist_diff_pair_c237_p6_0_1 <= 60
 -60 <= dist_diff_pair_c238_p7_0_1 <= 60
 -60 <= dist_diff_pair_c239_p8_0_1 <= 60
 -60 <= dist_diff_pair_c240_p9_0_1 <= 60
 -60 <= dist_diff_pair_c241_p10_0_1 <= 60
 -60 <= dist_diff_pair_c242_p11_0_1 <= 60
 -60 <= dist_diff_pair_c243_p12_0_1 <= 60
 -60 <= dist_diff_pair_c244_p13_0_1 <= 60
 -60 <= dist_diff_pair_c245_p14_0_1 <= 60
 -60 <= dist_diff_pair_c246_p15_0_1 <= 60
 -60 <= dist_diff_pair_c247_p16_0_1 <= 60
 -60 <= dist_diff_pair_c248_p17_0_1 <= 60
 -60 <= dist_diff_pair_c249_p18_0_1 <= 60
 -60 <= dist_diff_pair_c250_p19_0_1 <= 60
 -60 <= dist_diff_pair_c251_p20_0_1 <= 60
 -60 <= dist_diff_pair_c252_p21_0_1 <= 60
 dx_abs_pair_c231_p0_0_1 <= 30
 dx_abs_pair_c232_p1_0_1 <= 30
 dx_abs_pair_c233_p2_0_1 <= 30
 dx_abs_pair_c234_p3_0_1 <= 30
 dx_abs_pair_c235_p4_0_1 <= 30
 dx_abs_pair_c236_p5_0_1 <= 30
 dx_abs_pair_c237_p6_0_1 <= 30
 dx_abs_pair_c238_p7_0_1 <= 30
 dx_abs_pair_c239_p8_0_1 <= 30
 dx_abs_pair_c240_p9_0_1 <= 30
 dx_abs_pair_c241_p10_0_1 <= 30
 dx_abs_pair_c242_p11_0_1 <= 30
 dx_abs_pair_c243_p12_0_1 <= 30
 dx_abs_pair_c244_p13_0_1 <= 30
 dx_abs_pair_c245_p14_0_1 <= 30
 dx_abs_pair_c246_p15_0_1 <= 30
 dx_abs_pair_c247_p16_0_1 <= 30
 dx_abs_pair_c248_p17_0_1 <= 30
 dx_abs_pair_c249_p18_0_1 <= 30
 dx_abs_pair_c250_p19_0_1 <= 30
 dx_abs_pair_c251_p20_0_1 <= 30
 dx_abs_pair_c252_p21_0_1 <= 30
 dy_abs_pair_c231_p0_0_1 <= 30
 dy_abs_pair_c232_p1_0_1 <= 30
 dy_abs_pair_c233_p2_0_1 <= 30
 dy_abs_pair_c234_p3_0_1 <= 30
 dy_abs_pair_c235_p4_0_1 <= 30
 dy_abs_pair_c236_p5_0_1 <= 30
 dy_abs_pair_c237_p6_0_1 <= 30
 dy_abs_pair_c238_p7_0_1 <= 30
 dy_abs_pair_c239_p8_0_1 <= 30
 dy_abs_pair_c240_p9_0_1 <= 30
 dy_abs_pair_c241_p10_0_1 <= 30
 dy_abs_pair_c242_p11_0_1 <= 30
 dy_abs_pair_c243_p12_0_1 <= 30
 dy_abs_pair_c244_p13_0_1 <= 30
 dy_abs_pair_c245_p14_0_1 <= 30
 dy_abs_pair_c246_p15_0_1 <= 30
 dy_abs_pair_c247_p16_0_1 <= 30
 dy_abs_pair_c248_p17_0_1 <= 30
 dy_abs_pair_c249_p18_0_1 <= 30
 dy_abs_pair_c250_p19_0_1 <= 30
 dy_abs_pair_c251_p20_0_1 <= 30
 dy_abs_pair_c252_p21_0_1 <= 30
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
diff_dist_pair_c231_p0_0_1
diff_dist_pair_c232_p1_0_1
diff_dist_pair_c233_p2_0_1
diff_dist_pair_c234_p3_0_1
diff_dist_pair_c235_p4_0_1
diff_dist_pair_c236_p5_0_1
diff_dist_pair_c237_p6_0_1
diff_dist_pair_c238_p7_0_1
diff_dist_pair_c239_p8_0_1
diff_dist_pair_c240_p9_0_1
diff_dist_pair_c241_p10_0_1
diff_dist_pair_c242_p11_0_1
diff_dist_pair_c243_p12_0_1
diff_dist_pair_c244_p13_0_1
diff_dist_pair_c245_p14_0_1
diff_dist_pair_c246_p15_0_1
diff_dist_pair_c247_p16_0_1
diff_dist_pair_c248_p17_0_1
diff_dist_pair_c249_p18_0_1
diff_dist_pair_c250_p19_0_1
diff_dist_pair_c251_p20_0_1
diff_dist_pair_c252_p21_0_1
diff_pos_c231_p0_0
diff_pos_c231_p0_1
diff_pos_c232_p1_0
diff_pos_c232_p1_1
diff_pos_c233_p2_0
diff_pos_c233_p2_1
diff_pos_c234_p3_0
diff_pos_c234_p3_1
diff_pos_c235_p4_0
diff_pos_c235_p4_1
diff_pos_c236_p5_0
diff_pos_c236_p5_1
diff_pos_c237_p6_0
diff_pos_c237_p6_1
diff_pos_c238_p7_0
diff_pos_c238_p7_1
diff_pos_c239_p8_0
diff_pos_c239_p8_1
diff_pos_c240_p9_0
diff_pos_c240_p9_1
diff_pos_c241_p10_0
diff_pos_c241_p10_1
diff_pos_c242_p11_0
diff_pos_c242_p11_1
diff_pos_c243_p12_0
diff_pos_c243_p12_1
diff_pos_c244_p13_0
diff_pos_c244_p13_1
diff_pos_c245_p14_0
diff_pos_c245_p14_1
diff_pos_c246_p15_0
diff_pos_c246_p15_1
diff_pos_c247_p16_0
diff_pos_c247_p16_1
diff_pos_c248_p17_0
diff_pos_c248_p17_1
diff_pos_c249_p18_0
diff_pos_c249_p18_1
diff_pos_c250_p19_0
diff_pos_c250_p19_1
diff_pos_c251_p20_0
diff_pos_c251_p20_1
diff_pos_c252_p21_0
diff_pos_c252_p21_1
p_down_0_1
p_left_0_1
p_right_0_1
p_up_0_1
x_diff_c231_p0_1
x_diff_c232_p1_1
x_diff_c233_p2_1
x_diff_c234_p3_1
x_diff_c235_p4_1
x_diff_c236_p5_1
x_diff_c237_p6_1
x_diff_c238_p7_1
x_diff_c239_p8_1
x_diff_c240_p9_1
x_diff_c241_p10_1
x_diff_c242_p11_1
x_diff_c243_p12_1
x_diff_c244_p13_1
x_diff_c245_p14_1
x_diff_c246_p15_1
x_diff_c247_p16_1
x_diff_c248_p17_1
x_diff_c249_p18_1
x_diff_c250_p19_1
x_diff_c251_p20_1
x_diff_c252_p21_1
x_diff_minus_c231_p0_1
x_diff_minus_c232_p1_1
x_diff_minus_c233_p2_1
x_diff_minus_c234_p3_1
x_diff_minus_c235_p4_1
x_diff_minus_c236_p5_1
x_diff_minus_c237_p6_1
x_diff_minus_c238_p7_1
x_diff_minus_c239_p8_1
x_diff_minus_c240_p9_1
x_diff_minus_c241_p10_1
x_diff_minus_c242_p11_1
x_diff_minus_c243_p12_1
x_diff_minus_c244_p13_1
x_diff_minus_c245_p14_1
x_diff_minus_c246_p15_1
x_diff_minus_c247_p16_1
x_diff_minus_c248_p17_1
x_diff_minus_c249_p18_1
x_diff_minus_c250_p19_1
x_diff_minus_c251_p20_1
x_diff_minus_c252_p21_1
x_diff_plus_c231_p0_1
x_diff_plus_c232_p1_1
x_diff_plus_c233_p2_1
x_diff_plus_c234_p3_1
x_diff_plus_c235_p4_1
x_diff_plus_c236_p5_1
x_diff_plus_c237_p6_1
x_diff_plus_c238_p7_1
x_diff_plus_c239_p8_1
x_diff_plus_c240_p9_1
x_diff_plus_c241_p10_1
x_diff_plus_c242_p11_1
x_diff_plus_c243_p12_1
x_diff_plus_c244_p13_1
x_diff_plus_c245_p14_1
x_diff_plus_c246_p15_1
x_diff_plus_c247_p16_1
x_diff_plus_c248_p17_1
x_diff_plus_c249_p18_1
x_diff_plus_c250_p19_1
x_diff_plus_c251_p20_1
x_diff_plus_c252_p21_1
y_diff_c231_p0_1
y_diff_c232_p1_1
y_diff_c233_p2_1
y_diff_c234_p3_1
y_diff_c235_p4_1
y_diff_c236_p5_1
y_diff_c237_p6_1
y_diff_c238_p7_1
y_diff_c239_p8_1
y_diff_c240_p9_1
y_diff_c241_p10_1
y_diff_c242_p11_1
y_diff_c243_p12_1
y_diff_c244_p13_1
y_diff_c245_p14_1
y_diff_c246_p15_1
y_diff_c247_p16_1
y_diff_c248_p17_1
y_diff_c249_p18_1
y_diff_c250_p19_1
y_diff_c251_p20_1
y_diff_c252_p21_1
y_diff_minus_c231_p0_1
y_diff_minus_c232_p1_1
y_diff_minus_c233_p2_1
y_diff_minus_c234_p3_1
y_diff_minus_c235_p4_1
y_diff_minus_c236_p5_1
y_diff_minus_c237_p6_1
y_diff_minus_c238_p7_1
y_diff_minus_c239_p8_1
y_diff_minus_c240_p9_1
y_diff_minus_c241_p10_1
y_diff_minus_c242_p11_1
y_diff_minus_c243_p12_1
y_diff_minus_c244_p13_1
y_diff_minus_c245_p14_1
y_diff_minus_c246_p15_1
y_diff_minus_c247_p16_1
y_diff_minus_c248_p17_1
y_diff_minus_c249_p18_1
y_diff_minus_c250_p19_1
y_diff_minus_c251_p20_1
y_diff_minus_c252_p21_1
y_diff_plus_c231_p0_1
y_diff_plus_c232_p1_1
y_diff_plus_c233_p2_1
y_diff_plus_c234_p3_1
y_diff_plus_c235_p4_1
y_diff_plus_c236_p5_1
y_diff_plus_c237_p6_1
y_diff_plus_c238_p7_1
y_diff_plus_c239_p8_1
y_diff_plus_c240_p9_1
y_diff_plus_c241_p10_1
y_diff_plus_c242_p11_1
y_diff_plus_c243_p12_1
y_diff_plus_c244_p13_1
y_diff_plus_c245_p14_1
y_diff_plus_c246_p15_1
y_diff_plus_c247_p16_1
y_diff_plus_c248_p17_1
y_diff_plus_c249_p18_1
y_diff_plus_c250_p19_1
y_diff_plus_c251_p20_1
y_diff_plus_c252_p21_1
z1L_0_1
z1R_0_1
z1_0_1
z2D_0_1
z2U_0_1
z2_0_1
End
