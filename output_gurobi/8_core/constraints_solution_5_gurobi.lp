\ Model ChipletPlacementGrid
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x403ec7078863d695
Minimize
  0 r_0 + 0 r_1 + 0 r_2 + 0 r_3 + 0 r_4 + 0 r_5 + 0 r_6 + 0 r_7 + dx_abs_0_1
   + dy_abs_0_1 + dx_abs_2_4 + dy_abs_2_4 + dx_abs_0_3 + dy_abs_0_3
   + dx_abs_2_3 + dy_abs_2_3 + dx_abs_6_7 + dy_abs_6_7 + dx_abs_2_6
   + dy_abs_2_6 + dx_abs_2_5 + dy_abs_2_5 + 0.1 bbox_area_proxy_t
Subject To
 x_grid_to_real_0: - 0.5 x_grid_0 + x_0 = 0
 y_grid_to_real_0: - 0.5 y_grid_0 + y_0 = 0
 x_grid_to_real_1: - 0.5 x_grid_1 + x_1 = 0
 y_grid_to_real_1: - 0.5 y_grid_1 + y_1 = 0
 x_grid_to_real_2: - 0.5 x_grid_2 + x_2 = 0
 y_grid_to_real_2: - 0.5 y_grid_2 + y_2 = 0
 x_grid_to_real_3: - 0.5 x_grid_3 + x_3 = 0
 y_grid_to_real_3: - 0.5 y_grid_3 + y_3 = 0
 x_grid_to_real_4: - 0.5 x_grid_4 + x_4 = 0
 y_grid_to_real_4: - 0.5 y_grid_4 + y_4 = 0
 x_grid_to_real_5: - 0.5 x_grid_5 + x_5 = 0
 y_grid_to_real_5: - 0.5 y_grid_5 + y_5 = 0
 x_grid_to_real_6: - 0.5 x_grid_6 + x_6 = 0
 y_grid_to_real_6: - 0.5 y_grid_6 + y_6 = 0
 x_grid_to_real_7: - 0.5 x_grid_7 + x_7 = 0
 y_grid_to_real_7: - 0.5 y_grid_7 + y_7 = 0
 width_rotation_0: w_0 = 10
 height_rotation_0: h_0 = 10
 width_rotation_1: w_1 = 5
 height_rotation_1: h_1 = 5
 width_rotation_2: w_2 = 10
 height_rotation_2: h_2 = 10
 width_rotation_3: w_3 = 5
 height_rotation_3: h_3 = 5
 width_rotation_4: w_4 = 5
 height_rotation_4: h_4 = 5
 width_rotation_5: w_5 = 5
 height_rotation_5: h_5 = 5
 width_rotation_6: w_6 = 8
 height_rotation_6: h_6 = 8
 width_rotation_7: w_7 = 6
 height_rotation_7: h_7 = 6
 cx_def_0: - x_0 - 0.5 w_0 + cx_0 = 0
 cy_def_0: - y_0 - 0.5 h_0 + cy_0 = 0
 cx_def_1: - x_1 - 0.5 w_1 + cx_1 = 0
 cy_def_1: - y_1 - 0.5 h_1 + cy_1 = 0
 cx_def_2: - x_2 - 0.5 w_2 + cx_2 = 0
 cy_def_2: - y_2 - 0.5 h_2 + cy_2 = 0
 cx_def_3: - x_3 - 0.5 w_3 + cx_3 = 0
 cy_def_3: - y_3 - 0.5 h_3 + cy_3 = 0
 cx_def_4: - x_4 - 0.5 w_4 + cx_4 = 0
 cy_def_4: - y_4 - 0.5 h_4 + cy_4 = 0
 cx_def_5: - x_5 - 0.5 w_5 + cx_5 = 0
 cy_def_5: - y_5 - 0.5 h_5 + cy_5 = 0
 cx_def_6: - x_6 - 0.5 w_6 + cx_6 = 0
 cy_def_6: - y_6 - 0.5 h_6 + cy_6 = 0
 cx_def_7: - x_7 - 0.5 w_7 + cx_7 = 0
 cy_def_7: - y_7 - 0.5 h_7 + cy_7 = 0
 w_align_grid_0: w_0 = 10
 h_align_grid_0: h_0 = 10
 w_max_0: w_0 <= 30
 h_max_0: h_0 <= 30
 w_align_grid_1: w_1 = 5
 h_align_grid_1: h_1 = 5
 w_max_1: w_1 <= 30
 h_max_1: h_1 <= 30
 w_align_grid_2: w_2 = 10
 h_align_grid_2: h_2 = 10
 w_max_2: w_2 <= 30
 h_max_2: h_2 <= 30
 w_align_grid_3: w_3 = 5
 h_align_grid_3: h_3 = 5
 w_max_3: w_3 <= 30
 h_max_3: h_3 <= 30
 w_align_grid_4: w_4 = 5
 h_align_grid_4: h_4 = 5
 w_max_4: w_4 <= 30
 h_max_4: h_4 <= 30
 w_align_grid_5: w_5 = 5
 h_align_grid_5: h_5 = 5
 w_max_5: w_5 <= 30
 h_max_5: h_5 <= 30
 w_align_grid_6: w_6 = 8
 h_align_grid_6: h_6 = 8
 w_max_6: w_6 <= 30
 h_max_6: h_6 <= 30
 w_align_grid_7: w_7 = 6
 h_align_grid_7: h_7 = 6
 w_max_7: w_7 <= 30
 h_max_7: h_7 <= 30
 must_adjacent_0_1: z1_0_1 + z2_0_1 = 1
 horizontal_direction_0_1: - z1_0_1 + z1L_0_1 + z1R_0_1 = 0
 vertical_direction_0_1: - z2_0_1 + z2D_0_1 + z2U_0_1 = 0
 horizontal_left_dist_0_1: - x_0 + x_1 - w_0 + 200 z1L_0_1 <= 200.5
 horizontal_left_dist_lb_0_1: - x_0 + x_1 - w_0 - 200 z1L_0_1 >= -200
 horizontal_right_dist_0_1: x_0 - x_1 - w_1 + 200 z1R_0_1 <= 200.5
 horizontal_right_dist_lb_0_1: x_0 - x_1 - w_1 - 200 z1R_0_1 >= -200
 horizontal_overlap_y1_0_1: y_0 - y_1 - h_1 + 200 z1_0_1 <= 200
 horizontal_overlap_y2_0_1: - y_0 + y_1 - h_0 + 200 z1_0_1 <= 200
 shared_y_ub1_0_1: - y_0 + y_1 - h_0 + 200 z1_0_1 + shared_y_0_1 <= 200
 shared_y_ub2_0_1: y_0 - y_1 - h_1 + 200 z1_0_1 + shared_y_0_1 <= 200
 shared_y_min_0_1: - 200 z1_0_1 + shared_y_0_1 >= -199.9
 shared_y_zero_0_1: - 200 z1_0_1 + shared_y_0_1 <= 0
 vertical_down_dist_0_1: - y_0 + y_1 - h_0 + 200 z2D_0_1 <= 200.5
 vertical_down_dist_lb_0_1: - y_0 + y_1 - h_0 - 200 z2D_0_1 >= -200
 vertical_up_dist_0_1: y_0 - y_1 - h_1 + 200 z2U_0_1 <= 200.5
 vertical_up_dist_lb_0_1: y_0 - y_1 - h_1 - 200 z2U_0_1 >= -200
 vertical_overlap_x1_0_1: x_0 - x_1 - w_1 + 200 z2_0_1 <= 200
 vertical_overlap_x2_0_1: - x_0 + x_1 - w_0 + 200 z2_0_1 <= 200
 shared_x_ub1_0_1: - x_0 + x_1 - w_0 + 200 z2_0_1 + shared_x_0_1 <= 200
 shared_x_ub2_0_1: x_0 - x_1 - w_1 + 200 z2_0_1 + shared_x_0_1 <= 200
 shared_x_min_0_1: - 200 z2_0_1 + shared_x_0_1 >= -199.9
 shared_x_zero_0_1: - 200 z2_0_1 + shared_x_0_1 <= 0
 must_adjacent_2_4: z1_2_4 + z2_2_4 = 1
 horizontal_direction_2_4: - z1_2_4 + z1L_2_4 + z1R_2_4 = 0
 vertical_direction_2_4: - z2_2_4 + z2D_2_4 + z2U_2_4 = 0
 horizontal_left_dist_2_4: - x_2 + x_4 - w_2 + 200 z1L_2_4 <= 200.5
 horizontal_left_dist_lb_2_4: - x_2 + x_4 - w_2 - 200 z1L_2_4 >= -200
 horizontal_right_dist_2_4: x_2 - x_4 - w_4 + 200 z1R_2_4 <= 200.5
 horizontal_right_dist_lb_2_4: x_2 - x_4 - w_4 - 200 z1R_2_4 >= -200
 horizontal_overlap_y1_2_4: y_2 - y_4 - h_4 + 200 z1_2_4 <= 200
 horizontal_overlap_y2_2_4: - y_2 + y_4 - h_2 + 200 z1_2_4 <= 200
 shared_y_ub1_2_4: - y_2 + y_4 - h_2 + 200 z1_2_4 + shared_y_2_4 <= 200
 shared_y_ub2_2_4: y_2 - y_4 - h_4 + 200 z1_2_4 + shared_y_2_4 <= 200
 shared_y_min_2_4: - 200 z1_2_4 + shared_y_2_4 >= -199.9
 shared_y_zero_2_4: - 200 z1_2_4 + shared_y_2_4 <= 0
 vertical_down_dist_2_4: - y_2 + y_4 - h_2 + 200 z2D_2_4 <= 200.5
 vertical_down_dist_lb_2_4: - y_2 + y_4 - h_2 - 200 z2D_2_4 >= -200
 vertical_up_dist_2_4: y_2 - y_4 - h_4 + 200 z2U_2_4 <= 200.5
 vertical_up_dist_lb_2_4: y_2 - y_4 - h_4 - 200 z2U_2_4 >= -200
 vertical_overlap_x1_2_4: x_2 - x_4 - w_4 + 200 z2_2_4 <= 200
 vertical_overlap_x2_2_4: - x_2 + x_4 - w_2 + 200 z2_2_4 <= 200
 shared_x_ub1_2_4: - x_2 + x_4 - w_2 + 200 z2_2_4 + shared_x_2_4 <= 200
 shared_x_ub2_2_4: x_2 - x_4 - w_4 + 200 z2_2_4 + shared_x_2_4 <= 200
 shared_x_min_2_4: - 200 z2_2_4 + shared_x_2_4 >= -199.9
 shared_x_zero_2_4: - 200 z2_2_4 + shared_x_2_4 <= 0
 must_adjacent_0_3: z1_0_3 + z2_0_3 = 1
 horizontal_direction_0_3: - z1_0_3 + z1L_0_3 + z1R_0_3 = 0
 vertical_direction_0_3: - z2_0_3 + z2D_0_3 + z2U_0_3 = 0
 horizontal_left_dist_0_3: - x_0 + x_3 - w_0 + 200 z1L_0_3 <= 200.5
 horizontal_left_dist_lb_0_3: - x_0 + x_3 - w_0 - 200 z1L_0_3 >= -200
 horizontal_right_dist_0_3: x_0 - x_3 - w_3 + 200 z1R_0_3 <= 200.5
 horizontal_right_dist_lb_0_3: x_0 - x_3 - w_3 - 200 z1R_0_3 >= -200
 horizontal_overlap_y1_0_3: y_0 - y_3 - h_3 + 200 z1_0_3 <= 200
 horizontal_overlap_y2_0_3: - y_0 + y_3 - h_0 + 200 z1_0_3 <= 200
 shared_y_ub1_0_3: - y_0 + y_3 - h_0 + 200 z1_0_3 + shared_y_0_3 <= 200
 shared_y_ub2_0_3: y_0 - y_3 - h_3 + 200 z1_0_3 + shared_y_0_3 <= 200
 shared_y_min_0_3: - 200 z1_0_3 + shared_y_0_3 >= -199.9
 shared_y_zero_0_3: - 200 z1_0_3 + shared_y_0_3 <= 0
 vertical_down_dist_0_3: - y_0 + y_3 - h_0 + 200 z2D_0_3 <= 200.5
 vertical_down_dist_lb_0_3: - y_0 + y_3 - h_0 - 200 z2D_0_3 >= -200
 vertical_up_dist_0_3: y_0 - y_3 - h_3 + 200 z2U_0_3 <= 200.5
 vertical_up_dist_lb_0_3: y_0 - y_3 - h_3 - 200 z2U_0_3 >= -200
 vertical_overlap_x1_0_3: x_0 - x_3 - w_3 + 200 z2_0_3 <= 200
 vertical_overlap_x2_0_3: - x_0 + x_3 - w_0 + 200 z2_0_3 <= 200
 shared_x_ub1_0_3: - x_0 + x_3 - w_0 + 200 z2_0_3 + shared_x_0_3 <= 200
 shared_x_ub2_0_3: x_0 - x_3 - w_3 + 200 z2_0_3 + shared_x_0_3 <= 200
 shared_x_min_0_3: - 200 z2_0_3 + shared_x_0_3 >= -199.9
 shared_x_zero_0_3: - 200 z2_0_3 + shared_x_0_3 <= 0
 must_adjacent_2_3: z1_2_3 + z2_2_3 = 1
 horizontal_direction_2_3: - z1_2_3 + z1L_2_3 + z1R_2_3 = 0
 vertical_direction_2_3: - z2_2_3 + z2D_2_3 + z2U_2_3 = 0
 horizontal_left_dist_2_3: - x_2 + x_3 - w_2 + 200 z1L_2_3 <= 200.5
 horizontal_left_dist_lb_2_3: - x_2 + x_3 - w_2 - 200 z1L_2_3 >= -200
 horizontal_right_dist_2_3: x_2 - x_3 - w_3 + 200 z1R_2_3 <= 200.5
 horizontal_right_dist_lb_2_3: x_2 - x_3 - w_3 - 200 z1R_2_3 >= -200
 horizontal_overlap_y1_2_3: y_2 - y_3 - h_3 + 200 z1_2_3 <= 200
 horizontal_overlap_y2_2_3: - y_2 + y_3 - h_2 + 200 z1_2_3 <= 200
 shared_y_ub1_2_3: - y_2 + y_3 - h_2 + 200 z1_2_3 + shared_y_2_3 <= 200
 shared_y_ub2_2_3: y_2 - y_3 - h_3 + 200 z1_2_3 + shared_y_2_3 <= 200
 shared_y_min_2_3: - 200 z1_2_3 + shared_y_2_3 >= -199.9
 shared_y_zero_2_3: - 200 z1_2_3 + shared_y_2_3 <= 0
 vertical_down_dist_2_3: - y_2 + y_3 - h_2 + 200 z2D_2_3 <= 200.5
 vertical_down_dist_lb_2_3: - y_2 + y_3 - h_2 - 200 z2D_2_3 >= -200
 vertical_up_dist_2_3: y_2 - y_3 - h_3 + 200 z2U_2_3 <= 200.5
 vertical_up_dist_lb_2_3: y_2 - y_3 - h_3 - 200 z2U_2_3 >= -200
 vertical_overlap_x1_2_3: x_2 - x_3 - w_3 + 200 z2_2_3 <= 200
 vertical_overlap_x2_2_3: - x_2 + x_3 - w_2 + 200 z2_2_3 <= 200
 shared_x_ub1_2_3: - x_2 + x_3 - w_2 + 200 z2_2_3 + shared_x_2_3 <= 200
 shared_x_ub2_2_3: x_2 - x_3 - w_3 + 200 z2_2_3 + shared_x_2_3 <= 200
 shared_x_min_2_3: - 200 z2_2_3 + shared_x_2_3 >= -199.9
 shared_x_zero_2_3: - 200 z2_2_3 + shared_x_2_3 <= 0
 must_adjacent_6_7: z1_6_7 + z2_6_7 = 1
 horizontal_direction_6_7: - z1_6_7 + z1L_6_7 + z1R_6_7 = 0
 vertical_direction_6_7: - z2_6_7 + z2D_6_7 + z2U_6_7 = 0
 horizontal_left_dist_6_7: - x_6 + x_7 - w_6 + 200 z1L_6_7 <= 200.5
 horizontal_left_dist_lb_6_7: - x_6 + x_7 - w_6 - 200 z1L_6_7 >= -200
 horizontal_right_dist_6_7: x_6 - x_7 - w_7 + 200 z1R_6_7 <= 200.5
 horizontal_right_dist_lb_6_7: x_6 - x_7 - w_7 - 200 z1R_6_7 >= -200
 horizontal_overlap_y1_6_7: y_6 - y_7 - h_7 + 200 z1_6_7 <= 200
 horizontal_overlap_y2_6_7: - y_6 + y_7 - h_6 + 200 z1_6_7 <= 200
 shared_y_ub1_6_7: - y_6 + y_7 - h_6 + 200 z1_6_7 + shared_y_6_7 <= 200
 shared_y_ub2_6_7: y_6 - y_7 - h_7 + 200 z1_6_7 + shared_y_6_7 <= 200
 shared_y_min_6_7: - 200 z1_6_7 + shared_y_6_7 >= -199.9
 shared_y_zero_6_7: - 200 z1_6_7 + shared_y_6_7 <= 0
 vertical_down_dist_6_7: - y_6 + y_7 - h_6 + 200 z2D_6_7 <= 200.5
 vertical_down_dist_lb_6_7: - y_6 + y_7 - h_6 - 200 z2D_6_7 >= -200
 vertical_up_dist_6_7: y_6 - y_7 - h_7 + 200 z2U_6_7 <= 200.5
 vertical_up_dist_lb_6_7: y_6 - y_7 - h_7 - 200 z2U_6_7 >= -200
 vertical_overlap_x1_6_7: x_6 - x_7 - w_7 + 200 z2_6_7 <= 200
 vertical_overlap_x2_6_7: - x_6 + x_7 - w_6 + 200 z2_6_7 <= 200
 shared_x_ub1_6_7: - x_6 + x_7 - w_6 + 200 z2_6_7 + shared_x_6_7 <= 200
 shared_x_ub2_6_7: x_6 - x_7 - w_7 + 200 z2_6_7 + shared_x_6_7 <= 200
 shared_x_min_6_7: - 200 z2_6_7 + shared_x_6_7 >= -199.9
 shared_x_zero_6_7: - 200 z2_6_7 + shared_x_6_7 <= 0
 must_adjacent_2_6: z1_2_6 + z2_2_6 = 1
 horizontal_direction_2_6: - z1_2_6 + z1L_2_6 + z1R_2_6 = 0
 vertical_direction_2_6: - z2_2_6 + z2D_2_6 + z2U_2_6 = 0
 horizontal_left_dist_2_6: - x_2 + x_6 - w_2 + 200 z1L_2_6 <= 200.5
 horizontal_left_dist_lb_2_6: - x_2 + x_6 - w_2 - 200 z1L_2_6 >= -200
 horizontal_right_dist_2_6: x_2 - x_6 - w_6 + 200 z1R_2_6 <= 200.5
 horizontal_right_dist_lb_2_6: x_2 - x_6 - w_6 - 200 z1R_2_6 >= -200
 horizontal_overlap_y1_2_6: y_2 - y_6 - h_6 + 200 z1_2_6 <= 200
 horizontal_overlap_y2_2_6: - y_2 + y_6 - h_2 + 200 z1_2_6 <= 200
 shared_y_ub1_2_6: - y_2 + y_6 - h_2 + 200 z1_2_6 + shared_y_2_6 <= 200
 shared_y_ub2_2_6: y_2 - y_6 - h_6 + 200 z1_2_6 + shared_y_2_6 <= 200
 shared_y_min_2_6: - 200 z1_2_6 + shared_y_2_6 >= -199.9
 shared_y_zero_2_6: - 200 z1_2_6 + shared_y_2_6 <= 0
 vertical_down_dist_2_6: - y_2 + y_6 - h_2 + 200 z2D_2_6 <= 200.5
 vertical_down_dist_lb_2_6: - y_2 + y_6 - h_2 - 200 z2D_2_6 >= -200
 vertical_up_dist_2_6: y_2 - y_6 - h_6 + 200 z2U_2_6 <= 200.5
 vertical_up_dist_lb_2_6: y_2 - y_6 - h_6 - 200 z2U_2_6 >= -200
 vertical_overlap_x1_2_6: x_2 - x_6 - w_6 + 200 z2_2_6 <= 200
 vertical_overlap_x2_2_6: - x_2 + x_6 - w_2 + 200 z2_2_6 <= 200
 shared_x_ub1_2_6: - x_2 + x_6 - w_2 + 200 z2_2_6 + shared_x_2_6 <= 200
 shared_x_ub2_2_6: x_2 - x_6 - w_6 + 200 z2_2_6 + shared_x_2_6 <= 200
 shared_x_min_2_6: - 200 z2_2_6 + shared_x_2_6 >= -199.9
 shared_x_zero_2_6: - 200 z2_2_6 + shared_x_2_6 <= 0
 must_adjacent_2_5: z1_2_5 + z2_2_5 = 1
 horizontal_direction_2_5: - z1_2_5 + z1L_2_5 + z1R_2_5 = 0
 vertical_direction_2_5: - z2_2_5 + z2D_2_5 + z2U_2_5 = 0
 horizontal_left_dist_2_5: - x_2 + x_5 - w_2 + 200 z1L_2_5 <= 200.5
 horizontal_left_dist_lb_2_5: - x_2 + x_5 - w_2 - 200 z1L_2_5 >= -200
 horizontal_right_dist_2_5: x_2 - x_5 - w_5 + 200 z1R_2_5 <= 200.5
 horizontal_right_dist_lb_2_5: x_2 - x_5 - w_5 - 200 z1R_2_5 >= -200
 horizontal_overlap_y1_2_5: y_2 - y_5 - h_5 + 200 z1_2_5 <= 200
 horizontal_overlap_y2_2_5: - y_2 + y_5 - h_2 + 200 z1_2_5 <= 200
 shared_y_ub1_2_5: - y_2 + y_5 - h_2 + 200 z1_2_5 + shared_y_2_5 <= 200
 shared_y_ub2_2_5: y_2 - y_5 - h_5 + 200 z1_2_5 + shared_y_2_5 <= 200
 shared_y_min_2_5: - 200 z1_2_5 + shared_y_2_5 >= -199.9
 shared_y_zero_2_5: - 200 z1_2_5 + shared_y_2_5 <= 0
 vertical_down_dist_2_5: - y_2 + y_5 - h_2 + 200 z2D_2_5 <= 200.5
 vertical_down_dist_lb_2_5: - y_2 + y_5 - h_2 - 200 z2D_2_5 >= -200
 vertical_up_dist_2_5: y_2 - y_5 - h_5 + 200 z2U_2_5 <= 200.5
 vertical_up_dist_lb_2_5: y_2 - y_5 - h_5 - 200 z2U_2_5 >= -200
 vertical_overlap_x1_2_5: x_2 - x_5 - w_5 + 200 z2_2_5 <= 200
 vertical_overlap_x2_2_5: - x_2 + x_5 - w_2 + 200 z2_2_5 <= 200
 shared_x_ub1_2_5: - x_2 + x_5 - w_2 + 200 z2_2_5 + shared_x_2_5 <= 200
 shared_x_ub2_2_5: x_2 - x_5 - w_5 + 200 z2_2_5 + shared_x_2_5 <= 200
 shared_x_min_2_5: - 200 z2_2_5 + shared_x_2_5 >= -199.9
 shared_x_zero_2_5: - 200 z2_2_5 + shared_x_2_5 <= 0
 boundary_x_0: x_0 + w_0 <= 30
 boundary_y_0: y_0 + h_0 <= 30
 boundary_x_1: x_1 + w_1 <= 30
 boundary_y_1: y_1 + h_1 <= 30
 boundary_x_2: x_2 + w_2 <= 30
 boundary_y_2: y_2 + h_2 <= 30
 boundary_x_3: x_3 + w_3 <= 30
 boundary_y_3: y_3 + h_3 <= 30
 boundary_x_4: x_4 + w_4 <= 30
 boundary_y_4: y_4 + h_4 <= 30
 boundary_x_5: x_5 + w_5 <= 30
 boundary_y_5: y_5 + h_5 <= 30
 boundary_x_6: x_6 + w_6 <= 30
 boundary_y_6: y_6 + h_6 <= 30
 boundary_x_7: x_7 + w_7 <= 30
 boundary_y_7: y_7 + h_7 <= 30
 non_overlap_any_0_1: p_left_0_1 + p_right_0_1 + p_down_0_1 + p_up_0_1 = 1
 non_overlap_left_0_1: x_0 - x_1 + w_0 + 200 p_left_0_1 <= 200
 non_overlap_left_rev_0_1: - x_0 + x_1 - w_0 - 200 p_left_0_1 <= 0
 non_overlap_right_0_1: - x_0 + x_1 + w_1 + 200 p_right_0_1 <= 200
 non_overlap_right_rev_0_1: x_0 - x_1 - w_1 - 200 p_right_0_1 <= 0
 non_overlap_down_0_1: y_0 - y_1 + h_0 + 200 p_down_0_1 <= 200
 non_overlap_down_rev_0_1: - y_0 + y_1 - h_0 - 200 p_down_0_1 <= 0
 non_overlap_up_0_1: - y_0 + y_1 + h_1 + 200 p_up_0_1 <= 200
 non_overlap_up_rev_0_1: y_0 - y_1 - h_1 - 200 p_up_0_1 <= 0
 non_overlap_any_0_2: p_left_0_2 + p_right_0_2 + p_down_0_2 + p_up_0_2 = 1
 non_overlap_left_0_2: x_0 - x_2 + w_0 + 200 p_left_0_2 <= 200
 non_overlap_left_rev_0_2: - x_0 + x_2 - w_0 - 200 p_left_0_2 <= 0
 non_overlap_right_0_2: - x_0 + x_2 + w_2 + 200 p_right_0_2 <= 200
 non_overlap_right_rev_0_2: x_0 - x_2 - w_2 - 200 p_right_0_2 <= 0
 non_overlap_down_0_2: y_0 - y_2 + h_0 + 200 p_down_0_2 <= 200
 non_overlap_down_rev_0_2: - y_0 + y_2 - h_0 - 200 p_down_0_2 <= 0
 non_overlap_up_0_2: - y_0 + y_2 + h_2 + 200 p_up_0_2 <= 200
 non_overlap_up_rev_0_2: y_0 - y_2 - h_2 - 200 p_up_0_2 <= 0
 non_overlap_any_0_3: p_left_0_3 + p_right_0_3 + p_down_0_3 + p_up_0_3 = 1
 non_overlap_left_0_3: x_0 - x_3 + w_0 + 200 p_left_0_3 <= 200
 non_overlap_left_rev_0_3: - x_0 + x_3 - w_0 - 200 p_left_0_3 <= 0
 non_overlap_right_0_3: - x_0 + x_3 + w_3 + 200 p_right_0_3 <= 200
 non_overlap_right_rev_0_3: x_0 - x_3 - w_3 - 200 p_right_0_3 <= 0
 non_overlap_down_0_3: y_0 - y_3 + h_0 + 200 p_down_0_3 <= 200
 non_overlap_down_rev_0_3: - y_0 + y_3 - h_0 - 200 p_down_0_3 <= 0
 non_overlap_up_0_3: - y_0 + y_3 + h_3 + 200 p_up_0_3 <= 200
 non_overlap_up_rev_0_3: y_0 - y_3 - h_3 - 200 p_up_0_3 <= 0
 non_overlap_any_0_4: p_left_0_4 + p_right_0_4 + p_down_0_4 + p_up_0_4 = 1
 non_overlap_left_0_4: x_0 - x_4 + w_0 + 200 p_left_0_4 <= 200
 non_overlap_left_rev_0_4: - x_0 + x_4 - w_0 - 200 p_left_0_4 <= 0
 non_overlap_right_0_4: - x_0 + x_4 + w_4 + 200 p_right_0_4 <= 200
 non_overlap_right_rev_0_4: x_0 - x_4 - w_4 - 200 p_right_0_4 <= 0
 non_overlap_down_0_4: y_0 - y_4 + h_0 + 200 p_down_0_4 <= 200
 non_overlap_down_rev_0_4: - y_0 + y_4 - h_0 - 200 p_down_0_4 <= 0
 non_overlap_up_0_4: - y_0 + y_4 + h_4 + 200 p_up_0_4 <= 200
 non_overlap_up_rev_0_4: y_0 - y_4 - h_4 - 200 p_up_0_4 <= 0
 non_overlap_any_0_5: p_left_0_5 + p_right_0_5 + p_down_0_5 + p_up_0_5 = 1
 non_overlap_left_0_5: x_0 - x_5 + w_0 + 200 p_left_0_5 <= 200
 non_overlap_left_rev_0_5: - x_0 + x_5 - w_0 - 200 p_left_0_5 <= 0
 non_overlap_right_0_5: - x_0 + x_5 + w_5 + 200 p_right_0_5 <= 200
 non_overlap_right_rev_0_5: x_0 - x_5 - w_5 - 200 p_right_0_5 <= 0
 non_overlap_down_0_5: y_0 - y_5 + h_0 + 200 p_down_0_5 <= 200
 non_overlap_down_rev_0_5: - y_0 + y_5 - h_0 - 200 p_down_0_5 <= 0
 non_overlap_up_0_5: - y_0 + y_5 + h_5 + 200 p_up_0_5 <= 200
 non_overlap_up_rev_0_5: y_0 - y_5 - h_5 - 200 p_up_0_5 <= 0
 non_overlap_any_0_6: p_left_0_6 + p_right_0_6 + p_down_0_6 + p_up_0_6 = 1
 non_overlap_left_0_6: x_0 - x_6 + w_0 + 200 p_left_0_6 <= 200
 non_overlap_left_rev_0_6: - x_0 + x_6 - w_0 - 200 p_left_0_6 <= 0
 non_overlap_right_0_6: - x_0 + x_6 + w_6 + 200 p_right_0_6 <= 200
 non_overlap_right_rev_0_6: x_0 - x_6 - w_6 - 200 p_right_0_6 <= 0
 non_overlap_down_0_6: y_0 - y_6 + h_0 + 200 p_down_0_6 <= 200
 non_overlap_down_rev_0_6: - y_0 + y_6 - h_0 - 200 p_down_0_6 <= 0
 non_overlap_up_0_6: - y_0 + y_6 + h_6 + 200 p_up_0_6 <= 200
 non_overlap_up_rev_0_6: y_0 - y_6 - h_6 - 200 p_up_0_6 <= 0
 non_overlap_any_0_7: p_left_0_7 + p_right_0_7 + p_down_0_7 + p_up_0_7 = 1
 non_overlap_left_0_7: x_0 - x_7 + w_0 + 200 p_left_0_7 <= 200
 non_overlap_left_rev_0_7: - x_0 + x_7 - w_0 - 200 p_left_0_7 <= 0
 non_overlap_right_0_7: - x_0 + x_7 + w_7 + 200 p_right_0_7 <= 200
 non_overlap_right_rev_0_7: x_0 - x_7 - w_7 - 200 p_right_0_7 <= 0
 non_overlap_down_0_7: y_0 - y_7 + h_0 + 200 p_down_0_7 <= 200
 non_overlap_down_rev_0_7: - y_0 + y_7 - h_0 - 200 p_down_0_7 <= 0
 non_overlap_up_0_7: - y_0 + y_7 + h_7 + 200 p_up_0_7 <= 200
 non_overlap_up_rev_0_7: y_0 - y_7 - h_7 - 200 p_up_0_7 <= 0
 non_overlap_any_1_2: p_left_1_2 + p_right_1_2 + p_down_1_2 + p_up_1_2 = 1
 non_overlap_left_1_2: x_1 - x_2 + w_1 + 200 p_left_1_2 <= 200
 non_overlap_left_rev_1_2: - x_1 + x_2 - w_1 - 200 p_left_1_2 <= 0
 non_overlap_right_1_2: - x_1 + x_2 + w_2 + 200 p_right_1_2 <= 200
 non_overlap_right_rev_1_2: x_1 - x_2 - w_2 - 200 p_right_1_2 <= 0
 non_overlap_down_1_2: y_1 - y_2 + h_1 + 200 p_down_1_2 <= 200
 non_overlap_down_rev_1_2: - y_1 + y_2 - h_1 - 200 p_down_1_2 <= 0
 non_overlap_up_1_2: - y_1 + y_2 + h_2 + 200 p_up_1_2 <= 200
 non_overlap_up_rev_1_2: y_1 - y_2 - h_2 - 200 p_up_1_2 <= 0
 non_overlap_any_1_3: p_left_1_3 + p_right_1_3 + p_down_1_3 + p_up_1_3 = 1
 non_overlap_left_1_3: x_1 - x_3 + w_1 + 200 p_left_1_3 <= 200
 non_overlap_left_rev_1_3: - x_1 + x_3 - w_1 - 200 p_left_1_3 <= 0
 non_overlap_right_1_3: - x_1 + x_3 + w_3 + 200 p_right_1_3 <= 200
 non_overlap_right_rev_1_3: x_1 - x_3 - w_3 - 200 p_right_1_3 <= 0
 non_overlap_down_1_3: y_1 - y_3 + h_1 + 200 p_down_1_3 <= 200
 non_overlap_down_rev_1_3: - y_1 + y_3 - h_1 - 200 p_down_1_3 <= 0
 non_overlap_up_1_3: - y_1 + y_3 + h_3 + 200 p_up_1_3 <= 200
 non_overlap_up_rev_1_3: y_1 - y_3 - h_3 - 200 p_up_1_3 <= 0
 non_overlap_any_1_4: p_left_1_4 + p_right_1_4 + p_down_1_4 + p_up_1_4 = 1
 non_overlap_left_1_4: x_1 - x_4 + w_1 + 200 p_left_1_4 <= 200
 non_overlap_left_rev_1_4: - x_1 + x_4 - w_1 - 200 p_left_1_4 <= 0
 non_overlap_right_1_4: - x_1 + x_4 + w_4 + 200 p_right_1_4 <= 200
 non_overlap_right_rev_1_4: x_1 - x_4 - w_4 - 200 p_right_1_4 <= 0
 non_overlap_down_1_4: y_1 - y_4 + h_1 + 200 p_down_1_4 <= 200
 non_overlap_down_rev_1_4: - y_1 + y_4 - h_1 - 200 p_down_1_4 <= 0
 non_overlap_up_1_4: - y_1 + y_4 + h_4 + 200 p_up_1_4 <= 200
 non_overlap_up_rev_1_4: y_1 - y_4 - h_4 - 200 p_up_1_4 <= 0
 non_overlap_any_1_5: p_left_1_5 + p_right_1_5 + p_down_1_5 + p_up_1_5 = 1
 non_overlap_left_1_5: x_1 - x_5 + w_1 + 200 p_left_1_5 <= 200
 non_overlap_left_rev_1_5: - x_1 + x_5 - w_1 - 200 p_left_1_5 <= 0
 non_overlap_right_1_5: - x_1 + x_5 + w_5 + 200 p_right_1_5 <= 200
 non_overlap_right_rev_1_5: x_1 - x_5 - w_5 - 200 p_right_1_5 <= 0
 non_overlap_down_1_5: y_1 - y_5 + h_1 + 200 p_down_1_5 <= 200
 non_overlap_down_rev_1_5: - y_1 + y_5 - h_1 - 200 p_down_1_5 <= 0
 non_overlap_up_1_5: - y_1 + y_5 + h_5 + 200 p_up_1_5 <= 200
 non_overlap_up_rev_1_5: y_1 - y_5 - h_5 - 200 p_up_1_5 <= 0
 non_overlap_any_1_6: p_left_1_6 + p_right_1_6 + p_down_1_6 + p_up_1_6 = 1
 non_overlap_left_1_6: x_1 - x_6 + w_1 + 200 p_left_1_6 <= 200
 non_overlap_left_rev_1_6: - x_1 + x_6 - w_1 - 200 p_left_1_6 <= 0
 non_overlap_right_1_6: - x_1 + x_6 + w_6 + 200 p_right_1_6 <= 200
 non_overlap_right_rev_1_6: x_1 - x_6 - w_6 - 200 p_right_1_6 <= 0
 non_overlap_down_1_6: y_1 - y_6 + h_1 + 200 p_down_1_6 <= 200
 non_overlap_down_rev_1_6: - y_1 + y_6 - h_1 - 200 p_down_1_6 <= 0
 non_overlap_up_1_6: - y_1 + y_6 + h_6 + 200 p_up_1_6 <= 200
 non_overlap_up_rev_1_6: y_1 - y_6 - h_6 - 200 p_up_1_6 <= 0
 non_overlap_any_1_7: p_left_1_7 + p_right_1_7 + p_down_1_7 + p_up_1_7 = 1
 non_overlap_left_1_7: x_1 - x_7 + w_1 + 200 p_left_1_7 <= 200
 non_overlap_left_rev_1_7: - x_1 + x_7 - w_1 - 200 p_left_1_7 <= 0
 non_overlap_right_1_7: - x_1 + x_7 + w_7 + 200 p_right_1_7 <= 200
 non_overlap_right_rev_1_7: x_1 - x_7 - w_7 - 200 p_right_1_7 <= 0
 non_overlap_down_1_7: y_1 - y_7 + h_1 + 200 p_down_1_7 <= 200
 non_overlap_down_rev_1_7: - y_1 + y_7 - h_1 - 200 p_down_1_7 <= 0
 non_overlap_up_1_7: - y_1 + y_7 + h_7 + 200 p_up_1_7 <= 200
 non_overlap_up_rev_1_7: y_1 - y_7 - h_7 - 200 p_up_1_7 <= 0
 non_overlap_any_2_3: p_left_2_3 + p_right_2_3 + p_down_2_3 + p_up_2_3 = 1
 non_overlap_left_2_3: x_2 - x_3 + w_2 + 200 p_left_2_3 <= 200
 non_overlap_left_rev_2_3: - x_2 + x_3 - w_2 - 200 p_left_2_3 <= 0
 non_overlap_right_2_3: - x_2 + x_3 + w_3 + 200 p_right_2_3 <= 200
 non_overlap_right_rev_2_3: x_2 - x_3 - w_3 - 200 p_right_2_3 <= 0
 non_overlap_down_2_3: y_2 - y_3 + h_2 + 200 p_down_2_3 <= 200
 non_overlap_down_rev_2_3: - y_2 + y_3 - h_2 - 200 p_down_2_3 <= 0
 non_overlap_up_2_3: - y_2 + y_3 + h_3 + 200 p_up_2_3 <= 200
 non_overlap_up_rev_2_3: y_2 - y_3 - h_3 - 200 p_up_2_3 <= 0
 non_overlap_any_2_4: p_left_2_4 + p_right_2_4 + p_down_2_4 + p_up_2_4 = 1
 non_overlap_left_2_4: x_2 - x_4 + w_2 + 200 p_left_2_4 <= 200
 non_overlap_left_rev_2_4: - x_2 + x_4 - w_2 - 200 p_left_2_4 <= 0
 non_overlap_right_2_4: - x_2 + x_4 + w_4 + 200 p_right_2_4 <= 200
 non_overlap_right_rev_2_4: x_2 - x_4 - w_4 - 200 p_right_2_4 <= 0
 non_overlap_down_2_4: y_2 - y_4 + h_2 + 200 p_down_2_4 <= 200
 non_overlap_down_rev_2_4: - y_2 + y_4 - h_2 - 200 p_down_2_4 <= 0
 non_overlap_up_2_4: - y_2 + y_4 + h_4 + 200 p_up_2_4 <= 200
 non_overlap_up_rev_2_4: y_2 - y_4 - h_4 - 200 p_up_2_4 <= 0
 non_overlap_any_2_5: p_left_2_5 + p_right_2_5 + p_down_2_5 + p_up_2_5 = 1
 non_overlap_left_2_5: x_2 - x_5 + w_2 + 200 p_left_2_5 <= 200
 non_overlap_left_rev_2_5: - x_2 + x_5 - w_2 - 200 p_left_2_5 <= 0
 non_overlap_right_2_5: - x_2 + x_5 + w_5 + 200 p_right_2_5 <= 200
 non_overlap_right_rev_2_5: x_2 - x_5 - w_5 - 200 p_right_2_5 <= 0
 non_overlap_down_2_5: y_2 - y_5 + h_2 + 200 p_down_2_5 <= 200
 non_overlap_down_rev_2_5: - y_2 + y_5 - h_2 - 200 p_down_2_5 <= 0
 non_overlap_up_2_5: - y_2 + y_5 + h_5 + 200 p_up_2_5 <= 200
 non_overlap_up_rev_2_5: y_2 - y_5 - h_5 - 200 p_up_2_5 <= 0
 non_overlap_any_2_6: p_left_2_6 + p_right_2_6 + p_down_2_6 + p_up_2_6 = 1
 non_overlap_left_2_6: x_2 - x_6 + w_2 + 200 p_left_2_6 <= 200
 non_overlap_left_rev_2_6: - x_2 + x_6 - w_2 - 200 p_left_2_6 <= 0
 non_overlap_right_2_6: - x_2 + x_6 + w_6 + 200 p_right_2_6 <= 200
 non_overlap_right_rev_2_6: x_2 - x_6 - w_6 - 200 p_right_2_6 <= 0
 non_overlap_down_2_6: y_2 - y_6 + h_2 + 200 p_down_2_6 <= 200
 non_overlap_down_rev_2_6: - y_2 + y_6 - h_2 - 200 p_down_2_6 <= 0
 non_overlap_up_2_6: - y_2 + y_6 + h_6 + 200 p_up_2_6 <= 200
 non_overlap_up_rev_2_6: y_2 - y_6 - h_6 - 200 p_up_2_6 <= 0
 non_overlap_any_2_7: p_left_2_7 + p_right_2_7 + p_down_2_7 + p_up_2_7 = 1
 non_overlap_left_2_7: x_2 - x_7 + w_2 + 200 p_left_2_7 <= 200
 non_overlap_left_rev_2_7: - x_2 + x_7 - w_2 - 200 p_left_2_7 <= 0
 non_overlap_right_2_7: - x_2 + x_7 + w_7 + 200 p_right_2_7 <= 200
 non_overlap_right_rev_2_7: x_2 - x_7 - w_7 - 200 p_right_2_7 <= 0
 non_overlap_down_2_7: y_2 - y_7 + h_2 + 200 p_down_2_7 <= 200
 non_overlap_down_rev_2_7: - y_2 + y_7 - h_2 - 200 p_down_2_7 <= 0
 non_overlap_up_2_7: - y_2 + y_7 + h_7 + 200 p_up_2_7 <= 200
 non_overlap_up_rev_2_7: y_2 - y_7 - h_7 - 200 p_up_2_7 <= 0
 non_overlap_any_3_4: p_left_3_4 + p_right_3_4 + p_down_3_4 + p_up_3_4 = 1
 non_overlap_left_3_4: x_3 - x_4 + w_3 + 200 p_left_3_4 <= 200
 non_overlap_left_rev_3_4: - x_3 + x_4 - w_3 - 200 p_left_3_4 <= 0
 non_overlap_right_3_4: - x_3 + x_4 + w_4 + 200 p_right_3_4 <= 200
 non_overlap_right_rev_3_4: x_3 - x_4 - w_4 - 200 p_right_3_4 <= 0
 non_overlap_down_3_4: y_3 - y_4 + h_3 + 200 p_down_3_4 <= 200
 non_overlap_down_rev_3_4: - y_3 + y_4 - h_3 - 200 p_down_3_4 <= 0
 non_overlap_up_3_4: - y_3 + y_4 + h_4 + 200 p_up_3_4 <= 200
 non_overlap_up_rev_3_4: y_3 - y_4 - h_4 - 200 p_up_3_4 <= 0
 non_overlap_any_3_5: p_left_3_5 + p_right_3_5 + p_down_3_5 + p_up_3_5 = 1
 non_overlap_left_3_5: x_3 - x_5 + w_3 + 200 p_left_3_5 <= 200
 non_overlap_left_rev_3_5: - x_3 + x_5 - w_3 - 200 p_left_3_5 <= 0
 non_overlap_right_3_5: - x_3 + x_5 + w_5 + 200 p_right_3_5 <= 200
 non_overlap_right_rev_3_5: x_3 - x_5 - w_5 - 200 p_right_3_5 <= 0
 non_overlap_down_3_5: y_3 - y_5 + h_3 + 200 p_down_3_5 <= 200
 non_overlap_down_rev_3_5: - y_3 + y_5 - h_3 - 200 p_down_3_5 <= 0
 non_overlap_up_3_5: - y_3 + y_5 + h_5 + 200 p_up_3_5 <= 200
 non_overlap_up_rev_3_5: y_3 - y_5 - h_5 - 200 p_up_3_5 <= 0
 non_overlap_any_3_6: p_left_3_6 + p_right_3_6 + p_down_3_6 + p_up_3_6 = 1
 non_overlap_left_3_6: x_3 - x_6 + w_3 + 200 p_left_3_6 <= 200
 non_overlap_left_rev_3_6: - x_3 + x_6 - w_3 - 200 p_left_3_6 <= 0
 non_overlap_right_3_6: - x_3 + x_6 + w_6 + 200 p_right_3_6 <= 200
 non_overlap_right_rev_3_6: x_3 - x_6 - w_6 - 200 p_right_3_6 <= 0
 non_overlap_down_3_6: y_3 - y_6 + h_3 + 200 p_down_3_6 <= 200
 non_overlap_down_rev_3_6: - y_3 + y_6 - h_3 - 200 p_down_3_6 <= 0
 non_overlap_up_3_6: - y_3 + y_6 + h_6 + 200 p_up_3_6 <= 200
 non_overlap_up_rev_3_6: y_3 - y_6 - h_6 - 200 p_up_3_6 <= 0
 non_overlap_any_3_7: p_left_3_7 + p_right_3_7 + p_down_3_7 + p_up_3_7 = 1
 non_overlap_left_3_7: x_3 - x_7 + w_3 + 200 p_left_3_7 <= 200
 non_overlap_left_rev_3_7: - x_3 + x_7 - w_3 - 200 p_left_3_7 <= 0
 non_overlap_right_3_7: - x_3 + x_7 + w_7 + 200 p_right_3_7 <= 200
 non_overlap_right_rev_3_7: x_3 - x_7 - w_7 - 200 p_right_3_7 <= 0
 non_overlap_down_3_7: y_3 - y_7 + h_3 + 200 p_down_3_7 <= 200
 non_overlap_down_rev_3_7: - y_3 + y_7 - h_3 - 200 p_down_3_7 <= 0
 non_overlap_up_3_7: - y_3 + y_7 + h_7 + 200 p_up_3_7 <= 200
 non_overlap_up_rev_3_7: y_3 - y_7 - h_7 - 200 p_up_3_7 <= 0
 non_overlap_any_4_5: p_left_4_5 + p_right_4_5 + p_down_4_5 + p_up_4_5 = 1
 non_overlap_left_4_5: x_4 - x_5 + w_4 + 200 p_left_4_5 <= 200
 non_overlap_left_rev_4_5: - x_4 + x_5 - w_4 - 200 p_left_4_5 <= 0
 non_overlap_right_4_5: - x_4 + x_5 + w_5 + 200 p_right_4_5 <= 200
 non_overlap_right_rev_4_5: x_4 - x_5 - w_5 - 200 p_right_4_5 <= 0
 non_overlap_down_4_5: y_4 - y_5 + h_4 + 200 p_down_4_5 <= 200
 non_overlap_down_rev_4_5: - y_4 + y_5 - h_4 - 200 p_down_4_5 <= 0
 non_overlap_up_4_5: - y_4 + y_5 + h_5 + 200 p_up_4_5 <= 200
 non_overlap_up_rev_4_5: y_4 - y_5 - h_5 - 200 p_up_4_5 <= 0
 non_overlap_any_4_6: p_left_4_6 + p_right_4_6 + p_down_4_6 + p_up_4_6 = 1
 non_overlap_left_4_6: x_4 - x_6 + w_4 + 200 p_left_4_6 <= 200
 non_overlap_left_rev_4_6: - x_4 + x_6 - w_4 - 200 p_left_4_6 <= 0
 non_overlap_right_4_6: - x_4 + x_6 + w_6 + 200 p_right_4_6 <= 200
 non_overlap_right_rev_4_6: x_4 - x_6 - w_6 - 200 p_right_4_6 <= 0
 non_overlap_down_4_6: y_4 - y_6 + h_4 + 200 p_down_4_6 <= 200
 non_overlap_down_rev_4_6: - y_4 + y_6 - h_4 - 200 p_down_4_6 <= 0
 non_overlap_up_4_6: - y_4 + y_6 + h_6 + 200 p_up_4_6 <= 200
 non_overlap_up_rev_4_6: y_4 - y_6 - h_6 - 200 p_up_4_6 <= 0
 non_overlap_any_4_7: p_left_4_7 + p_right_4_7 + p_down_4_7 + p_up_4_7 = 1
 non_overlap_left_4_7: x_4 - x_7 + w_4 + 200 p_left_4_7 <= 200
 non_overlap_left_rev_4_7: - x_4 + x_7 - w_4 - 200 p_left_4_7 <= 0
 non_overlap_right_4_7: - x_4 + x_7 + w_7 + 200 p_right_4_7 <= 200
 non_overlap_right_rev_4_7: x_4 - x_7 - w_7 - 200 p_right_4_7 <= 0
 non_overlap_down_4_7: y_4 - y_7 + h_4 + 200 p_down_4_7 <= 200
 non_overlap_down_rev_4_7: - y_4 + y_7 - h_4 - 200 p_down_4_7 <= 0
 non_overlap_up_4_7: - y_4 + y_7 + h_7 + 200 p_up_4_7 <= 200
 non_overlap_up_rev_4_7: y_4 - y_7 - h_7 - 200 p_up_4_7 <= 0
 non_overlap_any_5_6: p_left_5_6 + p_right_5_6 + p_down_5_6 + p_up_5_6 = 1
 non_overlap_left_5_6: x_5 - x_6 + w_5 + 200 p_left_5_6 <= 200
 non_overlap_left_rev_5_6: - x_5 + x_6 - w_5 - 200 p_left_5_6 <= 0
 non_overlap_right_5_6: - x_5 + x_6 + w_6 + 200 p_right_5_6 <= 200
 non_overlap_right_rev_5_6: x_5 - x_6 - w_6 - 200 p_right_5_6 <= 0
 non_overlap_down_5_6: y_5 - y_6 + h_5 + 200 p_down_5_6 <= 200
 non_overlap_down_rev_5_6: - y_5 + y_6 - h_5 - 200 p_down_5_6 <= 0
 non_overlap_up_5_6: - y_5 + y_6 + h_6 + 200 p_up_5_6 <= 200
 non_overlap_up_rev_5_6: y_5 - y_6 - h_6 - 200 p_up_5_6 <= 0
 non_overlap_any_5_7: p_left_5_7 + p_right_5_7 + p_down_5_7 + p_up_5_7 = 1
 non_overlap_left_5_7: x_5 - x_7 + w_5 + 200 p_left_5_7 <= 200
 non_overlap_left_rev_5_7: - x_5 + x_7 - w_5 - 200 p_left_5_7 <= 0
 non_overlap_right_5_7: - x_5 + x_7 + w_7 + 200 p_right_5_7 <= 200
 non_overlap_right_rev_5_7: x_5 - x_7 - w_7 - 200 p_right_5_7 <= 0
 non_overlap_down_5_7: y_5 - y_7 + h_5 + 200 p_down_5_7 <= 200
 non_overlap_down_rev_5_7: - y_5 + y_7 - h_5 - 200 p_down_5_7 <= 0
 non_overlap_up_5_7: - y_5 + y_7 + h_7 + 200 p_up_5_7 <= 200
 non_overlap_up_rev_5_7: y_5 - y_7 - h_7 - 200 p_up_5_7 <= 0
 non_overlap_any_6_7: p_left_6_7 + p_right_6_7 + p_down_6_7 + p_up_6_7 = 1
 non_overlap_left_6_7: x_6 - x_7 + w_6 + 200 p_left_6_7 <= 200
 non_overlap_left_rev_6_7: - x_6 + x_7 - w_6 - 200 p_left_6_7 <= 0
 non_overlap_right_6_7: - x_6 + x_7 + w_7 + 200 p_right_6_7 <= 200
 non_overlap_right_rev_6_7: x_6 - x_7 - w_7 - 200 p_right_6_7 <= 0
 non_overlap_down_6_7: y_6 - y_7 + h_6 + 200 p_down_6_7 <= 200
 non_overlap_down_rev_6_7: - y_6 + y_7 - h_6 - 200 p_down_6_7 <= 0
 non_overlap_up_6_7: - y_6 + y_7 + h_7 + 200 p_up_6_7 <= 200
 non_overlap_up_rev_6_7: y_6 - y_7 - h_7 - 200 p_up_6_7 <= 0
 bbox_min_x_0: - x_0 + bbox_min_x <= 0
 bbox_max_x_0: - x_0 - w_0 + bbox_max_x >= 0
 bbox_min_y_0: - y_0 + bbox_min_y <= 0
 bbox_max_y_0: - y_0 - h_0 + bbox_max_y >= 0
 bbox_min_x_1: - x_1 + bbox_min_x <= 0
 bbox_max_x_1: - x_1 - w_1 + bbox_max_x >= 0
 bbox_min_y_1: - y_1 + bbox_min_y <= 0
 bbox_max_y_1: - y_1 - h_1 + bbox_max_y >= 0
 bbox_min_x_2: - x_2 + bbox_min_x <= 0
 bbox_max_x_2: - x_2 - w_2 + bbox_max_x >= 0
 bbox_min_y_2: - y_2 + bbox_min_y <= 0
 bbox_max_y_2: - y_2 - h_2 + bbox_max_y >= 0
 bbox_min_x_3: - x_3 + bbox_min_x <= 0
 bbox_max_x_3: - x_3 - w_3 + bbox_max_x >= 0
 bbox_min_y_3: - y_3 + bbox_min_y <= 0
 bbox_max_y_3: - y_3 - h_3 + bbox_max_y >= 0
 bbox_min_x_4: - x_4 + bbox_min_x <= 0
 bbox_max_x_4: - x_4 - w_4 + bbox_max_x >= 0
 bbox_min_y_4: - y_4 + bbox_min_y <= 0
 bbox_max_y_4: - y_4 - h_4 + bbox_max_y >= 0
 bbox_min_x_5: - x_5 + bbox_min_x <= 0
 bbox_max_x_5: - x_5 - w_5 + bbox_max_x >= 0
 bbox_min_y_5: - y_5 + bbox_min_y <= 0
 bbox_max_y_5: - y_5 - h_5 + bbox_max_y >= 0
 bbox_min_x_6: - x_6 + bbox_min_x <= 0
 bbox_max_x_6: - x_6 - w_6 + bbox_max_x >= 0
 bbox_min_y_6: - y_6 + bbox_min_y <= 0
 bbox_max_y_6: - y_6 - h_6 + bbox_max_y >= 0
 bbox_min_x_7: - x_7 + bbox_min_x <= 0
 bbox_max_x_7: - x_7 - w_7 + bbox_max_x >= 0
 bbox_min_y_7: - y_7 + bbox_min_y <= 0
 bbox_max_y_7: - y_7 - h_7 + bbox_max_y >= 0
 bbox_w_def: bbox_min_x - bbox_max_x + bbox_w = 0
 bbox_h_def: bbox_min_y - bbox_max_y + bbox_h = 0
 dx_diff_def_0_1: - cx_0 + cx_1 + dx_diff_0_1 = 0
 dy_diff_def_0_1: - cy_0 + cy_1 + dy_diff_0_1 = 0
 dx_abs_0_1_abs_ge_orig: dx_abs_0_1 - dx_diff_0_1
   - 30 dx_abs_0_1_is_positive >= -30
 dx_abs_0_1_abs_le_orig: dx_abs_0_1 - dx_diff_0_1
   + 30 dx_abs_0_1_is_positive <= 30
 dx_abs_0_1_abs_ge_neg_orig: dx_abs_0_1 + dx_diff_0_1
   + 30 dx_abs_0_1_is_positive >= 0
 dx_abs_0_1_abs_le_neg_orig: dx_abs_0_1 + dx_diff_0_1
   - 30 dx_abs_0_1_is_positive <= 0
 dx_abs_0_1_force_positive: dx_diff_0_1 - 30 dx_abs_0_1_is_positive >= -30
 dx_abs_0_1_force_negative: dx_diff_0_1 - 30 dx_abs_0_1_is_positive
   <= -0.001
 dy_abs_0_1_abs_ge_orig: dy_abs_0_1 - dy_diff_0_1
   - 30 dy_abs_0_1_is_positive >= -30
 dy_abs_0_1_abs_le_orig: dy_abs_0_1 - dy_diff_0_1
   + 30 dy_abs_0_1_is_positive <= 30
 dy_abs_0_1_abs_ge_neg_orig: dy_abs_0_1 + dy_diff_0_1
   + 30 dy_abs_0_1_is_positive >= 0
 dy_abs_0_1_abs_le_neg_orig: dy_abs_0_1 + dy_diff_0_1
   - 30 dy_abs_0_1_is_positive <= 0
 dy_abs_0_1_force_positive: dy_diff_0_1 - 30 dy_abs_0_1_is_positive >= -30
 dy_abs_0_1_force_negative: dy_diff_0_1 - 30 dy_abs_0_1_is_positive
   <= -0.001
 dx_diff_def_2_4: - cx_2 + cx_4 + dx_diff_2_4 = 0
 dy_diff_def_2_4: - cy_2 + cy_4 + dy_diff_2_4 = 0
 dx_abs_2_4_abs_ge_orig: dx_abs_2_4 - dx_diff_2_4
   - 30 dx_abs_2_4_is_positive >= -30
 dx_abs_2_4_abs_le_orig: dx_abs_2_4 - dx_diff_2_4
   + 30 dx_abs_2_4_is_positive <= 30
 dx_abs_2_4_abs_ge_neg_orig: dx_abs_2_4 + dx_diff_2_4
   + 30 dx_abs_2_4_is_positive >= 0
 dx_abs_2_4_abs_le_neg_orig: dx_abs_2_4 + dx_diff_2_4
   - 30 dx_abs_2_4_is_positive <= 0
 dx_abs_2_4_force_positive: dx_diff_2_4 - 30 dx_abs_2_4_is_positive >= -30
 dx_abs_2_4_force_negative: dx_diff_2_4 - 30 dx_abs_2_4_is_positive
   <= -0.001
 dy_abs_2_4_abs_ge_orig: dy_abs_2_4 - dy_diff_2_4
   - 30 dy_abs_2_4_is_positive >= -30
 dy_abs_2_4_abs_le_orig: dy_abs_2_4 - dy_diff_2_4
   + 30 dy_abs_2_4_is_positive <= 30
 dy_abs_2_4_abs_ge_neg_orig: dy_abs_2_4 + dy_diff_2_4
   + 30 dy_abs_2_4_is_positive >= 0
 dy_abs_2_4_abs_le_neg_orig: dy_abs_2_4 + dy_diff_2_4
   - 30 dy_abs_2_4_is_positive <= 0
 dy_abs_2_4_force_positive: dy_diff_2_4 - 30 dy_abs_2_4_is_positive >= -30
 dy_abs_2_4_force_negative: dy_diff_2_4 - 30 dy_abs_2_4_is_positive
   <= -0.001
 dx_diff_def_0_3: - cx_0 + cx_3 + dx_diff_0_3 = 0
 dy_diff_def_0_3: - cy_0 + cy_3 + dy_diff_0_3 = 0
 dx_abs_0_3_abs_ge_orig: dx_abs_0_3 - dx_diff_0_3
   - 30 dx_abs_0_3_is_positive >= -30
 dx_abs_0_3_abs_le_orig: dx_abs_0_3 - dx_diff_0_3
   + 30 dx_abs_0_3_is_positive <= 30
 dx_abs_0_3_abs_ge_neg_orig: dx_abs_0_3 + dx_diff_0_3
   + 30 dx_abs_0_3_is_positive >= 0
 dx_abs_0_3_abs_le_neg_orig: dx_abs_0_3 + dx_diff_0_3
   - 30 dx_abs_0_3_is_positive <= 0
 dx_abs_0_3_force_positive: dx_diff_0_3 - 30 dx_abs_0_3_is_positive >= -30
 dx_abs_0_3_force_negative: dx_diff_0_3 - 30 dx_abs_0_3_is_positive
   <= -0.001
 dy_abs_0_3_abs_ge_orig: dy_abs_0_3 - dy_diff_0_3
   - 30 dy_abs_0_3_is_positive >= -30
 dy_abs_0_3_abs_le_orig: dy_abs_0_3 - dy_diff_0_3
   + 30 dy_abs_0_3_is_positive <= 30
 dy_abs_0_3_abs_ge_neg_orig: dy_abs_0_3 + dy_diff_0_3
   + 30 dy_abs_0_3_is_positive >= 0
 dy_abs_0_3_abs_le_neg_orig: dy_abs_0_3 + dy_diff_0_3
   - 30 dy_abs_0_3_is_positive <= 0
 dy_abs_0_3_force_positive: dy_diff_0_3 - 30 dy_abs_0_3_is_positive >= -30
 dy_abs_0_3_force_negative: dy_diff_0_3 - 30 dy_abs_0_3_is_positive
   <= -0.001
 dx_diff_def_2_3: - cx_2 + cx_3 + dx_diff_2_3 = 0
 dy_diff_def_2_3: - cy_2 + cy_3 + dy_diff_2_3 = 0
 dx_abs_2_3_abs_ge_orig: dx_abs_2_3 - dx_diff_2_3
   - 30 dx_abs_2_3_is_positive >= -30
 dx_abs_2_3_abs_le_orig: dx_abs_2_3 - dx_diff_2_3
   + 30 dx_abs_2_3_is_positive <= 30
 dx_abs_2_3_abs_ge_neg_orig: dx_abs_2_3 + dx_diff_2_3
   + 30 dx_abs_2_3_is_positive >= 0
 dx_abs_2_3_abs_le_neg_orig: dx_abs_2_3 + dx_diff_2_3
   - 30 dx_abs_2_3_is_positive <= 0
 dx_abs_2_3_force_positive: dx_diff_2_3 - 30 dx_abs_2_3_is_positive >= -30
 dx_abs_2_3_force_negative: dx_diff_2_3 - 30 dx_abs_2_3_is_positive
   <= -0.001
 dy_abs_2_3_abs_ge_orig: dy_abs_2_3 - dy_diff_2_3
   - 30 dy_abs_2_3_is_positive >= -30
 dy_abs_2_3_abs_le_orig: dy_abs_2_3 - dy_diff_2_3
   + 30 dy_abs_2_3_is_positive <= 30
 dy_abs_2_3_abs_ge_neg_orig: dy_abs_2_3 + dy_diff_2_3
   + 30 dy_abs_2_3_is_positive >= 0
 dy_abs_2_3_abs_le_neg_orig: dy_abs_2_3 + dy_diff_2_3
   - 30 dy_abs_2_3_is_positive <= 0
 dy_abs_2_3_force_positive: dy_diff_2_3 - 30 dy_abs_2_3_is_positive >= -30
 dy_abs_2_3_force_negative: dy_diff_2_3 - 30 dy_abs_2_3_is_positive
   <= -0.001
 dx_diff_def_6_7: - cx_6 + cx_7 + dx_diff_6_7 = 0
 dy_diff_def_6_7: - cy_6 + cy_7 + dy_diff_6_7 = 0
 dx_abs_6_7_abs_ge_orig: dx_abs_6_7 - dx_diff_6_7
   - 30 dx_abs_6_7_is_positive >= -30
 dx_abs_6_7_abs_le_orig: dx_abs_6_7 - dx_diff_6_7
   + 30 dx_abs_6_7_is_positive <= 30
 dx_abs_6_7_abs_ge_neg_orig: dx_abs_6_7 + dx_diff_6_7
   + 30 dx_abs_6_7_is_positive >= 0
 dx_abs_6_7_abs_le_neg_orig: dx_abs_6_7 + dx_diff_6_7
   - 30 dx_abs_6_7_is_positive <= 0
 dx_abs_6_7_force_positive: dx_diff_6_7 - 30 dx_abs_6_7_is_positive >= -30
 dx_abs_6_7_force_negative: dx_diff_6_7 - 30 dx_abs_6_7_is_positive
   <= -0.001
 dy_abs_6_7_abs_ge_orig: dy_abs_6_7 - dy_diff_6_7
   - 30 dy_abs_6_7_is_positive >= -30
 dy_abs_6_7_abs_le_orig: dy_abs_6_7 - dy_diff_6_7
   + 30 dy_abs_6_7_is_positive <= 30
 dy_abs_6_7_abs_ge_neg_orig: dy_abs_6_7 + dy_diff_6_7
   + 30 dy_abs_6_7_is_positive >= 0
 dy_abs_6_7_abs_le_neg_orig: dy_abs_6_7 + dy_diff_6_7
   - 30 dy_abs_6_7_is_positive <= 0
 dy_abs_6_7_force_positive: dy_diff_6_7 - 30 dy_abs_6_7_is_positive >= -30
 dy_abs_6_7_force_negative: dy_diff_6_7 - 30 dy_abs_6_7_is_positive
   <= -0.001
 dx_diff_def_2_6: - cx_2 + cx_6 + dx_diff_2_6 = 0
 dy_diff_def_2_6: - cy_2 + cy_6 + dy_diff_2_6 = 0
 dx_abs_2_6_abs_ge_orig: dx_abs_2_6 - dx_diff_2_6
   - 30 dx_abs_2_6_is_positive >= -30
 dx_abs_2_6_abs_le_orig: dx_abs_2_6 - dx_diff_2_6
   + 30 dx_abs_2_6_is_positive <= 30
 dx_abs_2_6_abs_ge_neg_orig: dx_abs_2_6 + dx_diff_2_6
   + 30 dx_abs_2_6_is_positive >= 0
 dx_abs_2_6_abs_le_neg_orig: dx_abs_2_6 + dx_diff_2_6
   - 30 dx_abs_2_6_is_positive <= 0
 dx_abs_2_6_force_positive: dx_diff_2_6 - 30 dx_abs_2_6_is_positive >= -30
 dx_abs_2_6_force_negative: dx_diff_2_6 - 30 dx_abs_2_6_is_positive
   <= -0.001
 dy_abs_2_6_abs_ge_orig: dy_abs_2_6 - dy_diff_2_6
   - 30 dy_abs_2_6_is_positive >= -30
 dy_abs_2_6_abs_le_orig: dy_abs_2_6 - dy_diff_2_6
   + 30 dy_abs_2_6_is_positive <= 30
 dy_abs_2_6_abs_ge_neg_orig: dy_abs_2_6 + dy_diff_2_6
   + 30 dy_abs_2_6_is_positive >= 0
 dy_abs_2_6_abs_le_neg_orig: dy_abs_2_6 + dy_diff_2_6
   - 30 dy_abs_2_6_is_positive <= 0
 dy_abs_2_6_force_positive: dy_diff_2_6 - 30 dy_abs_2_6_is_positive >= -30
 dy_abs_2_6_force_negative: dy_diff_2_6 - 30 dy_abs_2_6_is_positive
   <= -0.001
 dx_diff_def_2_5: - cx_2 + cx_5 + dx_diff_2_5 = 0
 dy_diff_def_2_5: - cy_2 + cy_5 + dy_diff_2_5 = 0
 dx_abs_2_5_abs_ge_orig: dx_abs_2_5 - dx_diff_2_5
   - 30 dx_abs_2_5_is_positive >= -30
 dx_abs_2_5_abs_le_orig: dx_abs_2_5 - dx_diff_2_5
   + 30 dx_abs_2_5_is_positive <= 30
 dx_abs_2_5_abs_ge_neg_orig: dx_abs_2_5 + dx_diff_2_5
   + 30 dx_abs_2_5_is_positive >= 0
 dx_abs_2_5_abs_le_neg_orig: dx_abs_2_5 + dx_diff_2_5
   - 30 dx_abs_2_5_is_positive <= 0
 dx_abs_2_5_force_positive: dx_diff_2_5 - 30 dx_abs_2_5_is_positive >= -30
 dx_abs_2_5_force_negative: dx_diff_2_5 - 30 dx_abs_2_5_is_positive
   <= -0.001
 dy_abs_2_5_abs_ge_orig: dy_abs_2_5 - dy_diff_2_5
   - 30 dy_abs_2_5_is_positive >= -30
 dy_abs_2_5_abs_le_orig: dy_abs_2_5 - dy_diff_2_5
   + 30 dy_abs_2_5_is_positive <= 30
 dy_abs_2_5_abs_ge_neg_orig: dy_abs_2_5 + dy_diff_2_5
   + 30 dy_abs_2_5_is_positive >= 0
 dy_abs_2_5_abs_le_neg_orig: dy_abs_2_5 + dy_diff_2_5
   - 30 dy_abs_2_5_is_positive <= 0
 dy_abs_2_5_force_positive: dy_diff_2_5 - 30 dy_abs_2_5_is_positive >= -30
 dy_abs_2_5_force_negative: dy_diff_2_5 - 30 dy_abs_2_5_is_positive
   <= -0.001
 t_ge_width: - bbox_w + bbox_area_proxy_t >= 0
 t_ge_height: - bbox_h + bbox_area_proxy_t >= 0
 t_ge_scaled_mean: - 0.8 bbox_w - 0.8 bbox_h + bbox_area_proxy_t >= 0
 dx_grid_diff_def_c3_0_1: - x_grid_0 + x_grid_1 + dx_grid_diff_c3_0_1 = 0
 dx_grid_abs_pair_c3_0_1_abs_ge_orig: dx_grid_abs_pair_c3_0_1
   - dx_grid_diff_c3_0_1 - 60 dx_grid_abs_pair_c3_0_1_is_positive >= -60
 dx_grid_abs_pair_c3_0_1_abs_le_orig: dx_grid_abs_pair_c3_0_1
   - dx_grid_diff_c3_0_1 + 60 dx_grid_abs_pair_c3_0_1_is_positive <= 60
 dx_grid_abs_pair_c3_0_1_abs_ge_neg_orig: dx_grid_abs_pair_c3_0_1
   + dx_grid_diff_c3_0_1 + 60 dx_grid_abs_pair_c3_0_1_is_positive >= 0
 dx_grid_abs_pair_c3_0_1_abs_le_neg_orig: dx_grid_abs_pair_c3_0_1
   + dx_grid_diff_c3_0_1 - 60 dx_grid_abs_pair_c3_0_1_is_positive <= 0
 dx_grid_abs_pair_c3_0_1_force_positive: dx_grid_diff_c3_0_1
   - 60 dx_grid_abs_pair_c3_0_1_is_positive >= -60
 dx_grid_abs_pair_c3_0_1_force_negative: dx_grid_diff_c3_0_1
   - 60 dx_grid_abs_pair_c3_0_1_is_positive <= -0.001
 dy_grid_diff_def_c3_0_1: - y_grid_0 + y_grid_1 + dy_grid_diff_c3_0_1 = 0
 dy_grid_abs_pair_c3_0_1_abs_ge_orig: dy_grid_abs_pair_c3_0_1
   - dy_grid_diff_c3_0_1 - 60 dy_grid_abs_pair_c3_0_1_is_positive >= -60
 dy_grid_abs_pair_c3_0_1_abs_le_orig: dy_grid_abs_pair_c3_0_1
   - dy_grid_diff_c3_0_1 + 60 dy_grid_abs_pair_c3_0_1_is_positive <= 60
 dy_grid_abs_pair_c3_0_1_abs_ge_neg_orig: dy_grid_abs_pair_c3_0_1
   + dy_grid_diff_c3_0_1 + 60 dy_grid_abs_pair_c3_0_1_is_positive >= 0
 dy_grid_abs_pair_c3_0_1_abs_le_neg_orig: dy_grid_abs_pair_c3_0_1
   + dy_grid_diff_c3_0_1 - 60 dy_grid_abs_pair_c3_0_1_is_positive <= 0
 dy_grid_abs_pair_c3_0_1_force_positive: dy_grid_diff_c3_0_1
   - 60 dy_grid_abs_pair_c3_0_1_is_positive >= -60
 dy_grid_abs_pair_c3_0_1_force_negative: dy_grid_diff_c3_0_1
   - 60 dy_grid_abs_pair_c3_0_1_is_positive <= -0.001
 dist_curr_pair_def_c3_0_1: - dx_grid_abs_pair_c3_0_1
   - dy_grid_abs_pair_c3_0_1 + dist_curr_pair_c3_0_1 = 0
 dist_diff_pair_def_c3_0_1_prev0: - dist_curr_pair_c3_0_1
   + dist_diff_pair_c3_0_1_prev0 = -16
 dist_diff_abs_pair_c3_0_1_prev0_abs_ge_orig: - dist_diff_pair_c3_0_1_prev0
   + dist_diff_abs_pair_c3_0_1_prev0
   - 120 dist_diff_abs_pair_c3_0_1_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_1_prev0_abs_le_orig: - dist_diff_pair_c3_0_1_prev0
   + dist_diff_abs_pair_c3_0_1_prev0
   + 120 dist_diff_abs_pair_c3_0_1_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_0_1_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_0_1_prev0 + dist_diff_abs_pair_c3_0_1_prev0
   + 120 dist_diff_abs_pair_c3_0_1_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_0_1_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_0_1_prev0 + dist_diff_abs_pair_c3_0_1_prev0
   - 120 dist_diff_abs_pair_c3_0_1_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_0_1_prev0_force_positive:
   dist_diff_pair_c3_0_1_prev0
   - 120 dist_diff_abs_pair_c3_0_1_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_1_prev0_force_negative:
   dist_diff_pair_c3_0_1_prev0
   - 120 dist_diff_abs_pair_c3_0_1_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_0_1_prev0: 120 same_dist_pair_c3_0_1_prev0
   + dist_diff_abs_pair_c3_0_1_prev0 <= 129.8
 same_dist_pair_lower_c3_0_1_prev0: 120 same_dist_pair_c3_0_1_prev0
   + dist_diff_abs_pair_c3_0_1_prev0 >= 10
 dist_diff_pair_def_c3_0_1_prev1: - dist_curr_pair_c3_0_1
   + dist_diff_pair_c3_0_1_prev1 = -29
 dist_diff_abs_pair_c3_0_1_prev1_abs_ge_orig: - dist_diff_pair_c3_0_1_prev1
   + dist_diff_abs_pair_c3_0_1_prev1
   - 120 dist_diff_abs_pair_c3_0_1_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_1_prev1_abs_le_orig: - dist_diff_pair_c3_0_1_prev1
   + dist_diff_abs_pair_c3_0_1_prev1
   + 120 dist_diff_abs_pair_c3_0_1_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_0_1_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_0_1_prev1 + dist_diff_abs_pair_c3_0_1_prev1
   + 120 dist_diff_abs_pair_c3_0_1_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_0_1_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_0_1_prev1 + dist_diff_abs_pair_c3_0_1_prev1
   - 120 dist_diff_abs_pair_c3_0_1_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_0_1_prev1_force_positive:
   dist_diff_pair_c3_0_1_prev1
   - 120 dist_diff_abs_pair_c3_0_1_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_1_prev1_force_negative:
   dist_diff_pair_c3_0_1_prev1
   - 120 dist_diff_abs_pair_c3_0_1_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_0_1_prev1: 120 same_dist_pair_c3_0_1_prev1
   + dist_diff_abs_pair_c3_0_1_prev1 <= 129.8
 same_dist_pair_lower_c3_0_1_prev1: 120 same_dist_pair_c3_0_1_prev1
   + dist_diff_abs_pair_c3_0_1_prev1 >= 10
 dist_diff_pair_def_c3_0_1_prev2: - dist_curr_pair_c3_0_1
   + dist_diff_pair_c3_0_1_prev2 = -19
 dist_diff_abs_pair_c3_0_1_prev2_abs_ge_orig: - dist_diff_pair_c3_0_1_prev2
   + dist_diff_abs_pair_c3_0_1_prev2
   - 120 dist_diff_abs_pair_c3_0_1_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_1_prev2_abs_le_orig: - dist_diff_pair_c3_0_1_prev2
   + dist_diff_abs_pair_c3_0_1_prev2
   + 120 dist_diff_abs_pair_c3_0_1_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_0_1_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_0_1_prev2 + dist_diff_abs_pair_c3_0_1_prev2
   + 120 dist_diff_abs_pair_c3_0_1_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_0_1_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_0_1_prev2 + dist_diff_abs_pair_c3_0_1_prev2
   - 120 dist_diff_abs_pair_c3_0_1_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_0_1_prev2_force_positive:
   dist_diff_pair_c3_0_1_prev2
   - 120 dist_diff_abs_pair_c3_0_1_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_1_prev2_force_negative:
   dist_diff_pair_c3_0_1_prev2
   - 120 dist_diff_abs_pair_c3_0_1_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_0_1_prev2: 120 same_dist_pair_c3_0_1_prev2
   + dist_diff_abs_pair_c3_0_1_prev2 <= 129.8
 same_dist_pair_lower_c3_0_1_prev2: 120 same_dist_pair_c3_0_1_prev2
   + dist_diff_abs_pair_c3_0_1_prev2 >= 10
 dist_diff_pair_def_c3_0_1_prev3: - dist_curr_pair_c3_0_1
   + dist_diff_pair_c3_0_1_prev3 = -29
 dist_diff_abs_pair_c3_0_1_prev3_abs_ge_orig: - dist_diff_pair_c3_0_1_prev3
   + dist_diff_abs_pair_c3_0_1_prev3
   - 120 dist_diff_abs_pair_c3_0_1_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_1_prev3_abs_le_orig: - dist_diff_pair_c3_0_1_prev3
   + dist_diff_abs_pair_c3_0_1_prev3
   + 120 dist_diff_abs_pair_c3_0_1_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_0_1_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_0_1_prev3 + dist_diff_abs_pair_c3_0_1_prev3
   + 120 dist_diff_abs_pair_c3_0_1_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_0_1_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_0_1_prev3 + dist_diff_abs_pair_c3_0_1_prev3
   - 120 dist_diff_abs_pair_c3_0_1_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_0_1_prev3_force_positive:
   dist_diff_pair_c3_0_1_prev3
   - 120 dist_diff_abs_pair_c3_0_1_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_1_prev3_force_negative:
   dist_diff_pair_c3_0_1_prev3
   - 120 dist_diff_abs_pair_c3_0_1_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_0_1_prev3: 120 same_dist_pair_c3_0_1_prev3
   + dist_diff_abs_pair_c3_0_1_prev3 <= 129.8
 same_dist_pair_lower_c3_0_1_prev3: 120 same_dist_pair_c3_0_1_prev3
   + dist_diff_abs_pair_c3_0_1_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_0_1_prev0: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_0_1_prev0: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_0_1_prev1: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_0_1_prev1: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_0_1_prev2: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_0_1_prev2: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_0_1_prev3: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_0_1_prev3: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_0_1: diff_dist_pair_c3_0_1
   + same_dist_pair_c3_0_1_prev0 + same_dist_pair_c3_0_1_prev1
   + same_dist_pair_c3_0_1_prev2 + same_dist_pair_c3_0_1_prev3 >= 1
 dx_grid_diff_def_c3_0_2: - x_grid_0 + x_grid_2 + dx_grid_diff_c3_0_2 = 0
 dx_grid_abs_pair_c3_0_2_abs_ge_orig: dx_grid_abs_pair_c3_0_2
   - dx_grid_diff_c3_0_2 - 60 dx_grid_abs_pair_c3_0_2_is_positive >= -60
 dx_grid_abs_pair_c3_0_2_abs_le_orig: dx_grid_abs_pair_c3_0_2
   - dx_grid_diff_c3_0_2 + 60 dx_grid_abs_pair_c3_0_2_is_positive <= 60
 dx_grid_abs_pair_c3_0_2_abs_ge_neg_orig: dx_grid_abs_pair_c3_0_2
   + dx_grid_diff_c3_0_2 + 60 dx_grid_abs_pair_c3_0_2_is_positive >= 0
 dx_grid_abs_pair_c3_0_2_abs_le_neg_orig: dx_grid_abs_pair_c3_0_2
   + dx_grid_diff_c3_0_2 - 60 dx_grid_abs_pair_c3_0_2_is_positive <= 0
 dx_grid_abs_pair_c3_0_2_force_positive: dx_grid_diff_c3_0_2
   - 60 dx_grid_abs_pair_c3_0_2_is_positive >= -60
 dx_grid_abs_pair_c3_0_2_force_negative: dx_grid_diff_c3_0_2
   - 60 dx_grid_abs_pair_c3_0_2_is_positive <= -0.001
 dy_grid_diff_def_c3_0_2: - y_grid_0 + y_grid_2 + dy_grid_diff_c3_0_2 = 0
 dy_grid_abs_pair_c3_0_2_abs_ge_orig: dy_grid_abs_pair_c3_0_2
   - dy_grid_diff_c3_0_2 - 60 dy_grid_abs_pair_c3_0_2_is_positive >= -60
 dy_grid_abs_pair_c3_0_2_abs_le_orig: dy_grid_abs_pair_c3_0_2
   - dy_grid_diff_c3_0_2 + 60 dy_grid_abs_pair_c3_0_2_is_positive <= 60
 dy_grid_abs_pair_c3_0_2_abs_ge_neg_orig: dy_grid_abs_pair_c3_0_2
   + dy_grid_diff_c3_0_2 + 60 dy_grid_abs_pair_c3_0_2_is_positive >= 0
 dy_grid_abs_pair_c3_0_2_abs_le_neg_orig: dy_grid_abs_pair_c3_0_2
   + dy_grid_diff_c3_0_2 - 60 dy_grid_abs_pair_c3_0_2_is_positive <= 0
 dy_grid_abs_pair_c3_0_2_force_positive: dy_grid_diff_c3_0_2
   - 60 dy_grid_abs_pair_c3_0_2_is_positive >= -60
 dy_grid_abs_pair_c3_0_2_force_negative: dy_grid_diff_c3_0_2
   - 60 dy_grid_abs_pair_c3_0_2_is_positive <= -0.001
 dist_curr_pair_def_c3_0_2: - dx_grid_abs_pair_c3_0_2
   - dy_grid_abs_pair_c3_0_2 + dist_curr_pair_c3_0_2 = 0
 dist_diff_pair_def_c3_0_2_prev0: - dist_curr_pair_c3_0_2
   + dist_diff_pair_c3_0_2_prev0 = -44
 dist_diff_abs_pair_c3_0_2_prev0_abs_ge_orig: - dist_diff_pair_c3_0_2_prev0
   + dist_diff_abs_pair_c3_0_2_prev0
   - 120 dist_diff_abs_pair_c3_0_2_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_2_prev0_abs_le_orig: - dist_diff_pair_c3_0_2_prev0
   + dist_diff_abs_pair_c3_0_2_prev0
   + 120 dist_diff_abs_pair_c3_0_2_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_0_2_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_0_2_prev0 + dist_diff_abs_pair_c3_0_2_prev0
   + 120 dist_diff_abs_pair_c3_0_2_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_0_2_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_0_2_prev0 + dist_diff_abs_pair_c3_0_2_prev0
   - 120 dist_diff_abs_pair_c3_0_2_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_0_2_prev0_force_positive:
   dist_diff_pair_c3_0_2_prev0
   - 120 dist_diff_abs_pair_c3_0_2_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_2_prev0_force_negative:
   dist_diff_pair_c3_0_2_prev0
   - 120 dist_diff_abs_pair_c3_0_2_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_0_2_prev0: 120 same_dist_pair_c3_0_2_prev0
   + dist_diff_abs_pair_c3_0_2_prev0 <= 129.8
 same_dist_pair_lower_c3_0_2_prev0: 120 same_dist_pair_c3_0_2_prev0
   + dist_diff_abs_pair_c3_0_2_prev0 >= 10
 dist_diff_pair_def_c3_0_2_prev1: - dist_curr_pair_c3_0_2
   + dist_diff_pair_c3_0_2_prev1 = -30
 dist_diff_abs_pair_c3_0_2_prev1_abs_ge_orig: - dist_diff_pair_c3_0_2_prev1
   + dist_diff_abs_pair_c3_0_2_prev1
   - 120 dist_diff_abs_pair_c3_0_2_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_2_prev1_abs_le_orig: - dist_diff_pair_c3_0_2_prev1
   + dist_diff_abs_pair_c3_0_2_prev1
   + 120 dist_diff_abs_pair_c3_0_2_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_0_2_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_0_2_prev1 + dist_diff_abs_pair_c3_0_2_prev1
   + 120 dist_diff_abs_pair_c3_0_2_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_0_2_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_0_2_prev1 + dist_diff_abs_pair_c3_0_2_prev1
   - 120 dist_diff_abs_pair_c3_0_2_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_0_2_prev1_force_positive:
   dist_diff_pair_c3_0_2_prev1
   - 120 dist_diff_abs_pair_c3_0_2_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_2_prev1_force_negative:
   dist_diff_pair_c3_0_2_prev1
   - 120 dist_diff_abs_pair_c3_0_2_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_0_2_prev1: 120 same_dist_pair_c3_0_2_prev1
   + dist_diff_abs_pair_c3_0_2_prev1 <= 129.8
 same_dist_pair_lower_c3_0_2_prev1: 120 same_dist_pair_c3_0_2_prev1
   + dist_diff_abs_pair_c3_0_2_prev1 >= 10
 dist_diff_pair_def_c3_0_2_prev2: - dist_curr_pair_c3_0_2
   + dist_diff_pair_c3_0_2_prev2 = -30
 dist_diff_abs_pair_c3_0_2_prev2_abs_ge_orig: - dist_diff_pair_c3_0_2_prev2
   + dist_diff_abs_pair_c3_0_2_prev2
   - 120 dist_diff_abs_pair_c3_0_2_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_2_prev2_abs_le_orig: - dist_diff_pair_c3_0_2_prev2
   + dist_diff_abs_pair_c3_0_2_prev2
   + 120 dist_diff_abs_pair_c3_0_2_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_0_2_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_0_2_prev2 + dist_diff_abs_pair_c3_0_2_prev2
   + 120 dist_diff_abs_pair_c3_0_2_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_0_2_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_0_2_prev2 + dist_diff_abs_pair_c3_0_2_prev2
   - 120 dist_diff_abs_pair_c3_0_2_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_0_2_prev2_force_positive:
   dist_diff_pair_c3_0_2_prev2
   - 120 dist_diff_abs_pair_c3_0_2_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_2_prev2_force_negative:
   dist_diff_pair_c3_0_2_prev2
   - 120 dist_diff_abs_pair_c3_0_2_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_0_2_prev2: 120 same_dist_pair_c3_0_2_prev2
   + dist_diff_abs_pair_c3_0_2_prev2 <= 129.8
 same_dist_pair_lower_c3_0_2_prev2: 120 same_dist_pair_c3_0_2_prev2
   + dist_diff_abs_pair_c3_0_2_prev2 >= 10
 dist_diff_pair_def_c3_0_2_prev3: - dist_curr_pair_c3_0_2
   + dist_diff_pair_c3_0_2_prev3 = -30
 dist_diff_abs_pair_c3_0_2_prev3_abs_ge_orig: - dist_diff_pair_c3_0_2_prev3
   + dist_diff_abs_pair_c3_0_2_prev3
   - 120 dist_diff_abs_pair_c3_0_2_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_2_prev3_abs_le_orig: - dist_diff_pair_c3_0_2_prev3
   + dist_diff_abs_pair_c3_0_2_prev3
   + 120 dist_diff_abs_pair_c3_0_2_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_0_2_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_0_2_prev3 + dist_diff_abs_pair_c3_0_2_prev3
   + 120 dist_diff_abs_pair_c3_0_2_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_0_2_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_0_2_prev3 + dist_diff_abs_pair_c3_0_2_prev3
   - 120 dist_diff_abs_pair_c3_0_2_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_0_2_prev3_force_positive:
   dist_diff_pair_c3_0_2_prev3
   - 120 dist_diff_abs_pair_c3_0_2_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_2_prev3_force_negative:
   dist_diff_pair_c3_0_2_prev3
   - 120 dist_diff_abs_pair_c3_0_2_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_0_2_prev3: 120 same_dist_pair_c3_0_2_prev3
   + dist_diff_abs_pair_c3_0_2_prev3 <= 129.8
 same_dist_pair_lower_c3_0_2_prev3: 120 same_dist_pair_c3_0_2_prev3
   + dist_diff_abs_pair_c3_0_2_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_0_2_prev0: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_0_2_prev0: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_0_2_prev1: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_0_2_prev1: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_0_2_prev2: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_0_2_prev2: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_0_2_prev3: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_0_2_prev3: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_0_2: diff_dist_pair_c3_0_2
   + same_dist_pair_c3_0_2_prev0 + same_dist_pair_c3_0_2_prev1
   + same_dist_pair_c3_0_2_prev2 + same_dist_pair_c3_0_2_prev3 >= 1
 dx_grid_diff_def_c3_0_3: - x_grid_0 + x_grid_3 + dx_grid_diff_c3_0_3 = 0
 dx_grid_abs_pair_c3_0_3_abs_ge_orig: dx_grid_abs_pair_c3_0_3
   - dx_grid_diff_c3_0_3 - 60 dx_grid_abs_pair_c3_0_3_is_positive >= -60
 dx_grid_abs_pair_c3_0_3_abs_le_orig: dx_grid_abs_pair_c3_0_3
   - dx_grid_diff_c3_0_3 + 60 dx_grid_abs_pair_c3_0_3_is_positive <= 60
 dx_grid_abs_pair_c3_0_3_abs_ge_neg_orig: dx_grid_abs_pair_c3_0_3
   + dx_grid_diff_c3_0_3 + 60 dx_grid_abs_pair_c3_0_3_is_positive >= 0
 dx_grid_abs_pair_c3_0_3_abs_le_neg_orig: dx_grid_abs_pair_c3_0_3
   + dx_grid_diff_c3_0_3 - 60 dx_grid_abs_pair_c3_0_3_is_positive <= 0
 dx_grid_abs_pair_c3_0_3_force_positive: dx_grid_diff_c3_0_3
   - 60 dx_grid_abs_pair_c3_0_3_is_positive >= -60
 dx_grid_abs_pair_c3_0_3_force_negative: dx_grid_diff_c3_0_3
   - 60 dx_grid_abs_pair_c3_0_3_is_positive <= -0.001
 dy_grid_diff_def_c3_0_3: - y_grid_0 + y_grid_3 + dy_grid_diff_c3_0_3 = 0
 dy_grid_abs_pair_c3_0_3_abs_ge_orig: dy_grid_abs_pair_c3_0_3
   - dy_grid_diff_c3_0_3 - 60 dy_grid_abs_pair_c3_0_3_is_positive >= -60
 dy_grid_abs_pair_c3_0_3_abs_le_orig: dy_grid_abs_pair_c3_0_3
   - dy_grid_diff_c3_0_3 + 60 dy_grid_abs_pair_c3_0_3_is_positive <= 60
 dy_grid_abs_pair_c3_0_3_abs_ge_neg_orig: dy_grid_abs_pair_c3_0_3
   + dy_grid_diff_c3_0_3 + 60 dy_grid_abs_pair_c3_0_3_is_positive >= 0
 dy_grid_abs_pair_c3_0_3_abs_le_neg_orig: dy_grid_abs_pair_c3_0_3
   + dy_grid_diff_c3_0_3 - 60 dy_grid_abs_pair_c3_0_3_is_positive <= 0
 dy_grid_abs_pair_c3_0_3_force_positive: dy_grid_diff_c3_0_3
   - 60 dy_grid_abs_pair_c3_0_3_is_positive >= -60
 dy_grid_abs_pair_c3_0_3_force_negative: dy_grid_diff_c3_0_3
   - 60 dy_grid_abs_pair_c3_0_3_is_positive <= -0.001
 dist_curr_pair_def_c3_0_3: - dx_grid_abs_pair_c3_0_3
   - dy_grid_abs_pair_c3_0_3 + dist_curr_pair_c3_0_3 = 0
 dist_diff_pair_def_c3_0_3_prev0: - dist_curr_pair_c3_0_3
   + dist_diff_pair_c3_0_3_prev0 = -10
 dist_diff_abs_pair_c3_0_3_prev0_abs_ge_orig: - dist_diff_pair_c3_0_3_prev0
   + dist_diff_abs_pair_c3_0_3_prev0
   - 120 dist_diff_abs_pair_c3_0_3_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_3_prev0_abs_le_orig: - dist_diff_pair_c3_0_3_prev0
   + dist_diff_abs_pair_c3_0_3_prev0
   + 120 dist_diff_abs_pair_c3_0_3_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_0_3_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_0_3_prev0 + dist_diff_abs_pair_c3_0_3_prev0
   + 120 dist_diff_abs_pair_c3_0_3_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_0_3_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_0_3_prev0 + dist_diff_abs_pair_c3_0_3_prev0
   - 120 dist_diff_abs_pair_c3_0_3_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_0_3_prev0_force_positive:
   dist_diff_pair_c3_0_3_prev0
   - 120 dist_diff_abs_pair_c3_0_3_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_3_prev0_force_negative:
   dist_diff_pair_c3_0_3_prev0
   - 120 dist_diff_abs_pair_c3_0_3_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_0_3_prev0: 120 same_dist_pair_c3_0_3_prev0
   + dist_diff_abs_pair_c3_0_3_prev0 <= 129.8
 same_dist_pair_lower_c3_0_3_prev0: 120 same_dist_pair_c3_0_3_prev0
   + dist_diff_abs_pair_c3_0_3_prev0 >= 10
 dist_diff_pair_def_c3_0_3_prev1: - dist_curr_pair_c3_0_3
   + dist_diff_pair_c3_0_3_prev1 = -30
 dist_diff_abs_pair_c3_0_3_prev1_abs_ge_orig: - dist_diff_pair_c3_0_3_prev1
   + dist_diff_abs_pair_c3_0_3_prev1
   - 120 dist_diff_abs_pair_c3_0_3_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_3_prev1_abs_le_orig: - dist_diff_pair_c3_0_3_prev1
   + dist_diff_abs_pair_c3_0_3_prev1
   + 120 dist_diff_abs_pair_c3_0_3_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_0_3_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_0_3_prev1 + dist_diff_abs_pair_c3_0_3_prev1
   + 120 dist_diff_abs_pair_c3_0_3_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_0_3_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_0_3_prev1 + dist_diff_abs_pair_c3_0_3_prev1
   - 120 dist_diff_abs_pair_c3_0_3_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_0_3_prev1_force_positive:
   dist_diff_pair_c3_0_3_prev1
   - 120 dist_diff_abs_pair_c3_0_3_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_3_prev1_force_negative:
   dist_diff_pair_c3_0_3_prev1
   - 120 dist_diff_abs_pair_c3_0_3_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_0_3_prev1: 120 same_dist_pair_c3_0_3_prev1
   + dist_diff_abs_pair_c3_0_3_prev1 <= 129.8
 same_dist_pair_lower_c3_0_3_prev1: 120 same_dist_pair_c3_0_3_prev1
   + dist_diff_abs_pair_c3_0_3_prev1 >= 10
 dist_diff_pair_def_c3_0_3_prev2: - dist_curr_pair_c3_0_3
   + dist_diff_pair_c3_0_3_prev2 = -10
 dist_diff_abs_pair_c3_0_3_prev2_abs_ge_orig: - dist_diff_pair_c3_0_3_prev2
   + dist_diff_abs_pair_c3_0_3_prev2
   - 120 dist_diff_abs_pair_c3_0_3_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_3_prev2_abs_le_orig: - dist_diff_pair_c3_0_3_prev2
   + dist_diff_abs_pair_c3_0_3_prev2
   + 120 dist_diff_abs_pair_c3_0_3_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_0_3_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_0_3_prev2 + dist_diff_abs_pair_c3_0_3_prev2
   + 120 dist_diff_abs_pair_c3_0_3_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_0_3_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_0_3_prev2 + dist_diff_abs_pair_c3_0_3_prev2
   - 120 dist_diff_abs_pair_c3_0_3_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_0_3_prev2_force_positive:
   dist_diff_pair_c3_0_3_prev2
   - 120 dist_diff_abs_pair_c3_0_3_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_3_prev2_force_negative:
   dist_diff_pair_c3_0_3_prev2
   - 120 dist_diff_abs_pair_c3_0_3_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_0_3_prev2: 120 same_dist_pair_c3_0_3_prev2
   + dist_diff_abs_pair_c3_0_3_prev2 <= 129.8
 same_dist_pair_lower_c3_0_3_prev2: 120 same_dist_pair_c3_0_3_prev2
   + dist_diff_abs_pair_c3_0_3_prev2 >= 10
 dist_diff_pair_def_c3_0_3_prev3: - dist_curr_pair_c3_0_3
   + dist_diff_pair_c3_0_3_prev3 = -20
 dist_diff_abs_pair_c3_0_3_prev3_abs_ge_orig: - dist_diff_pair_c3_0_3_prev3
   + dist_diff_abs_pair_c3_0_3_prev3
   - 120 dist_diff_abs_pair_c3_0_3_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_3_prev3_abs_le_orig: - dist_diff_pair_c3_0_3_prev3
   + dist_diff_abs_pair_c3_0_3_prev3
   + 120 dist_diff_abs_pair_c3_0_3_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_0_3_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_0_3_prev3 + dist_diff_abs_pair_c3_0_3_prev3
   + 120 dist_diff_abs_pair_c3_0_3_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_0_3_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_0_3_prev3 + dist_diff_abs_pair_c3_0_3_prev3
   - 120 dist_diff_abs_pair_c3_0_3_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_0_3_prev3_force_positive:
   dist_diff_pair_c3_0_3_prev3
   - 120 dist_diff_abs_pair_c3_0_3_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_3_prev3_force_negative:
   dist_diff_pair_c3_0_3_prev3
   - 120 dist_diff_abs_pair_c3_0_3_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_0_3_prev3: 120 same_dist_pair_c3_0_3_prev3
   + dist_diff_abs_pair_c3_0_3_prev3 <= 129.8
 same_dist_pair_lower_c3_0_3_prev3: 120 same_dist_pair_c3_0_3_prev3
   + dist_diff_abs_pair_c3_0_3_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_0_3_prev0: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_0_3_prev0: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_0_3_prev1: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_0_3_prev1: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_0_3_prev2: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_0_3_prev2: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_0_3_prev3: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_0_3_prev3: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_0_3: diff_dist_pair_c3_0_3
   + same_dist_pair_c3_0_3_prev0 + same_dist_pair_c3_0_3_prev1
   + same_dist_pair_c3_0_3_prev2 + same_dist_pair_c3_0_3_prev3 >= 1
 dx_grid_diff_def_c3_0_4: - x_grid_0 + x_grid_4 + dx_grid_diff_c3_0_4 = 0
 dx_grid_abs_pair_c3_0_4_abs_ge_orig: dx_grid_abs_pair_c3_0_4
   - dx_grid_diff_c3_0_4 - 60 dx_grid_abs_pair_c3_0_4_is_positive >= -60
 dx_grid_abs_pair_c3_0_4_abs_le_orig: dx_grid_abs_pair_c3_0_4
   - dx_grid_diff_c3_0_4 + 60 dx_grid_abs_pair_c3_0_4_is_positive <= 60
 dx_grid_abs_pair_c3_0_4_abs_ge_neg_orig: dx_grid_abs_pair_c3_0_4
   + dx_grid_diff_c3_0_4 + 60 dx_grid_abs_pair_c3_0_4_is_positive >= 0
 dx_grid_abs_pair_c3_0_4_abs_le_neg_orig: dx_grid_abs_pair_c3_0_4
   + dx_grid_diff_c3_0_4 - 60 dx_grid_abs_pair_c3_0_4_is_positive <= 0
 dx_grid_abs_pair_c3_0_4_force_positive: dx_grid_diff_c3_0_4
   - 60 dx_grid_abs_pair_c3_0_4_is_positive >= -60
 dx_grid_abs_pair_c3_0_4_force_negative: dx_grid_diff_c3_0_4
   - 60 dx_grid_abs_pair_c3_0_4_is_positive <= -0.001
 dy_grid_diff_def_c3_0_4: - y_grid_0 + y_grid_4 + dy_grid_diff_c3_0_4 = 0
 dy_grid_abs_pair_c3_0_4_abs_ge_orig: dy_grid_abs_pair_c3_0_4
   - dy_grid_diff_c3_0_4 - 60 dy_grid_abs_pair_c3_0_4_is_positive >= -60
 dy_grid_abs_pair_c3_0_4_abs_le_orig: dy_grid_abs_pair_c3_0_4
   - dy_grid_diff_c3_0_4 + 60 dy_grid_abs_pair_c3_0_4_is_positive <= 60
 dy_grid_abs_pair_c3_0_4_abs_ge_neg_orig: dy_grid_abs_pair_c3_0_4
   + dy_grid_diff_c3_0_4 + 60 dy_grid_abs_pair_c3_0_4_is_positive >= 0
 dy_grid_abs_pair_c3_0_4_abs_le_neg_orig: dy_grid_abs_pair_c3_0_4
   + dy_grid_diff_c3_0_4 - 60 dy_grid_abs_pair_c3_0_4_is_positive <= 0
 dy_grid_abs_pair_c3_0_4_force_positive: dy_grid_diff_c3_0_4
   - 60 dy_grid_abs_pair_c3_0_4_is_positive >= -60
 dy_grid_abs_pair_c3_0_4_force_negative: dy_grid_diff_c3_0_4
   - 60 dy_grid_abs_pair_c3_0_4_is_positive <= -0.001
 dist_curr_pair_def_c3_0_4: - dx_grid_abs_pair_c3_0_4
   - dy_grid_abs_pair_c3_0_4 + dist_curr_pair_c3_0_4 = 0
 dist_diff_pair_def_c3_0_4_prev0: - dist_curr_pair_c3_0_4
   + dist_diff_pair_c3_0_4_prev0 = -14
 dist_diff_abs_pair_c3_0_4_prev0_abs_ge_orig: - dist_diff_pair_c3_0_4_prev0
   + dist_diff_abs_pair_c3_0_4_prev0
   - 120 dist_diff_abs_pair_c3_0_4_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_4_prev0_abs_le_orig: - dist_diff_pair_c3_0_4_prev0
   + dist_diff_abs_pair_c3_0_4_prev0
   + 120 dist_diff_abs_pair_c3_0_4_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_0_4_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_0_4_prev0 + dist_diff_abs_pair_c3_0_4_prev0
   + 120 dist_diff_abs_pair_c3_0_4_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_0_4_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_0_4_prev0 + dist_diff_abs_pair_c3_0_4_prev0
   - 120 dist_diff_abs_pair_c3_0_4_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_0_4_prev0_force_positive:
   dist_diff_pair_c3_0_4_prev0
   - 120 dist_diff_abs_pair_c3_0_4_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_4_prev0_force_negative:
   dist_diff_pair_c3_0_4_prev0
   - 120 dist_diff_abs_pair_c3_0_4_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_0_4_prev0: 120 same_dist_pair_c3_0_4_prev0
   + dist_diff_abs_pair_c3_0_4_prev0 <= 129.8
 same_dist_pair_lower_c3_0_4_prev0: 120 same_dist_pair_c3_0_4_prev0
   + dist_diff_abs_pair_c3_0_4_prev0 >= 10
 dist_diff_pair_def_c3_0_4_prev1: - dist_curr_pair_c3_0_4
   + dist_diff_pair_c3_0_4_prev1 = -20
 dist_diff_abs_pair_c3_0_4_prev1_abs_ge_orig: - dist_diff_pair_c3_0_4_prev1
   + dist_diff_abs_pair_c3_0_4_prev1
   - 120 dist_diff_abs_pair_c3_0_4_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_4_prev1_abs_le_orig: - dist_diff_pair_c3_0_4_prev1
   + dist_diff_abs_pair_c3_0_4_prev1
   + 120 dist_diff_abs_pair_c3_0_4_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_0_4_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_0_4_prev1 + dist_diff_abs_pair_c3_0_4_prev1
   + 120 dist_diff_abs_pair_c3_0_4_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_0_4_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_0_4_prev1 + dist_diff_abs_pair_c3_0_4_prev1
   - 120 dist_diff_abs_pair_c3_0_4_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_0_4_prev1_force_positive:
   dist_diff_pair_c3_0_4_prev1
   - 120 dist_diff_abs_pair_c3_0_4_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_4_prev1_force_negative:
   dist_diff_pair_c3_0_4_prev1
   - 120 dist_diff_abs_pair_c3_0_4_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_0_4_prev1: 120 same_dist_pair_c3_0_4_prev1
   + dist_diff_abs_pair_c3_0_4_prev1 <= 129.8
 same_dist_pair_lower_c3_0_4_prev1: 120 same_dist_pair_c3_0_4_prev1
   + dist_diff_abs_pair_c3_0_4_prev1 >= 10
 dist_diff_pair_def_c3_0_4_prev2: - dist_curr_pair_c3_0_4
   + dist_diff_pair_c3_0_4_prev2 = -29
 dist_diff_abs_pair_c3_0_4_prev2_abs_ge_orig: - dist_diff_pair_c3_0_4_prev2
   + dist_diff_abs_pair_c3_0_4_prev2
   - 120 dist_diff_abs_pair_c3_0_4_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_4_prev2_abs_le_orig: - dist_diff_pair_c3_0_4_prev2
   + dist_diff_abs_pair_c3_0_4_prev2
   + 120 dist_diff_abs_pair_c3_0_4_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_0_4_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_0_4_prev2 + dist_diff_abs_pair_c3_0_4_prev2
   + 120 dist_diff_abs_pair_c3_0_4_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_0_4_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_0_4_prev2 + dist_diff_abs_pair_c3_0_4_prev2
   - 120 dist_diff_abs_pair_c3_0_4_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_0_4_prev2_force_positive:
   dist_diff_pair_c3_0_4_prev2
   - 120 dist_diff_abs_pair_c3_0_4_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_4_prev2_force_negative:
   dist_diff_pair_c3_0_4_prev2
   - 120 dist_diff_abs_pair_c3_0_4_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_0_4_prev2: 120 same_dist_pair_c3_0_4_prev2
   + dist_diff_abs_pair_c3_0_4_prev2 <= 129.8
 same_dist_pair_lower_c3_0_4_prev2: 120 same_dist_pair_c3_0_4_prev2
   + dist_diff_abs_pair_c3_0_4_prev2 >= 10
 dist_diff_pair_def_c3_0_4_prev3: - dist_curr_pair_c3_0_4
   + dist_diff_pair_c3_0_4_prev3 = -30
 dist_diff_abs_pair_c3_0_4_prev3_abs_ge_orig: - dist_diff_pair_c3_0_4_prev3
   + dist_diff_abs_pair_c3_0_4_prev3
   - 120 dist_diff_abs_pair_c3_0_4_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_4_prev3_abs_le_orig: - dist_diff_pair_c3_0_4_prev3
   + dist_diff_abs_pair_c3_0_4_prev3
   + 120 dist_diff_abs_pair_c3_0_4_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_0_4_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_0_4_prev3 + dist_diff_abs_pair_c3_0_4_prev3
   + 120 dist_diff_abs_pair_c3_0_4_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_0_4_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_0_4_prev3 + dist_diff_abs_pair_c3_0_4_prev3
   - 120 dist_diff_abs_pair_c3_0_4_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_0_4_prev3_force_positive:
   dist_diff_pair_c3_0_4_prev3
   - 120 dist_diff_abs_pair_c3_0_4_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_4_prev3_force_negative:
   dist_diff_pair_c3_0_4_prev3
   - 120 dist_diff_abs_pair_c3_0_4_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_0_4_prev3: 120 same_dist_pair_c3_0_4_prev3
   + dist_diff_abs_pair_c3_0_4_prev3 <= 129.8
 same_dist_pair_lower_c3_0_4_prev3: 120 same_dist_pair_c3_0_4_prev3
   + dist_diff_abs_pair_c3_0_4_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_0_4_prev0: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_0_4_prev0: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_0_4_prev1: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_0_4_prev1: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_0_4_prev2: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_0_4_prev2: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_0_4_prev3: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_0_4_prev3: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_0_4: diff_dist_pair_c3_0_4
   + same_dist_pair_c3_0_4_prev0 + same_dist_pair_c3_0_4_prev1
   + same_dist_pair_c3_0_4_prev2 + same_dist_pair_c3_0_4_prev3 >= 1
 dx_grid_diff_def_c3_0_5: - x_grid_0 + x_grid_5 + dx_grid_diff_c3_0_5 = 0
 dx_grid_abs_pair_c3_0_5_abs_ge_orig: dx_grid_abs_pair_c3_0_5
   - dx_grid_diff_c3_0_5 - 60 dx_grid_abs_pair_c3_0_5_is_positive >= -60
 dx_grid_abs_pair_c3_0_5_abs_le_orig: dx_grid_abs_pair_c3_0_5
   - dx_grid_diff_c3_0_5 + 60 dx_grid_abs_pair_c3_0_5_is_positive <= 60
 dx_grid_abs_pair_c3_0_5_abs_ge_neg_orig: dx_grid_abs_pair_c3_0_5
   + dx_grid_diff_c3_0_5 + 60 dx_grid_abs_pair_c3_0_5_is_positive >= 0
 dx_grid_abs_pair_c3_0_5_abs_le_neg_orig: dx_grid_abs_pair_c3_0_5
   + dx_grid_diff_c3_0_5 - 60 dx_grid_abs_pair_c3_0_5_is_positive <= 0
 dx_grid_abs_pair_c3_0_5_force_positive: dx_grid_diff_c3_0_5
   - 60 dx_grid_abs_pair_c3_0_5_is_positive >= -60
 dx_grid_abs_pair_c3_0_5_force_negative: dx_grid_diff_c3_0_5
   - 60 dx_grid_abs_pair_c3_0_5_is_positive <= -0.001
 dy_grid_diff_def_c3_0_5: - y_grid_0 + y_grid_5 + dy_grid_diff_c3_0_5 = 0
 dy_grid_abs_pair_c3_0_5_abs_ge_orig: dy_grid_abs_pair_c3_0_5
   - dy_grid_diff_c3_0_5 - 60 dy_grid_abs_pair_c3_0_5_is_positive >= -60
 dy_grid_abs_pair_c3_0_5_abs_le_orig: dy_grid_abs_pair_c3_0_5
   - dy_grid_diff_c3_0_5 + 60 dy_grid_abs_pair_c3_0_5_is_positive <= 60
 dy_grid_abs_pair_c3_0_5_abs_ge_neg_orig: dy_grid_abs_pair_c3_0_5
   + dy_grid_diff_c3_0_5 + 60 dy_grid_abs_pair_c3_0_5_is_positive >= 0
 dy_grid_abs_pair_c3_0_5_abs_le_neg_orig: dy_grid_abs_pair_c3_0_5
   + dy_grid_diff_c3_0_5 - 60 dy_grid_abs_pair_c3_0_5_is_positive <= 0
 dy_grid_abs_pair_c3_0_5_force_positive: dy_grid_diff_c3_0_5
   - 60 dy_grid_abs_pair_c3_0_5_is_positive >= -60
 dy_grid_abs_pair_c3_0_5_force_negative: dy_grid_diff_c3_0_5
   - 60 dy_grid_abs_pair_c3_0_5_is_positive <= -0.001
 dist_curr_pair_def_c3_0_5: - dx_grid_abs_pair_c3_0_5
   - dy_grid_abs_pair_c3_0_5 + dist_curr_pair_c3_0_5 = 0
 dist_diff_pair_def_c3_0_5_prev0: - dist_curr_pair_c3_0_5
   + dist_diff_pair_c3_0_5_prev0 = -20
 dist_diff_abs_pair_c3_0_5_prev0_abs_ge_orig: - dist_diff_pair_c3_0_5_prev0
   + dist_diff_abs_pair_c3_0_5_prev0
   - 120 dist_diff_abs_pair_c3_0_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_5_prev0_abs_le_orig: - dist_diff_pair_c3_0_5_prev0
   + dist_diff_abs_pair_c3_0_5_prev0
   + 120 dist_diff_abs_pair_c3_0_5_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_0_5_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_0_5_prev0 + dist_diff_abs_pair_c3_0_5_prev0
   + 120 dist_diff_abs_pair_c3_0_5_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_0_5_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_0_5_prev0 + dist_diff_abs_pair_c3_0_5_prev0
   - 120 dist_diff_abs_pair_c3_0_5_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_0_5_prev0_force_positive:
   dist_diff_pair_c3_0_5_prev0
   - 120 dist_diff_abs_pair_c3_0_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_5_prev0_force_negative:
   dist_diff_pair_c3_0_5_prev0
   - 120 dist_diff_abs_pair_c3_0_5_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_0_5_prev0: 120 same_dist_pair_c3_0_5_prev0
   + dist_diff_abs_pair_c3_0_5_prev0 <= 129.8
 same_dist_pair_lower_c3_0_5_prev0: 120 same_dist_pair_c3_0_5_prev0
   + dist_diff_abs_pair_c3_0_5_prev0 >= 10
 dist_diff_pair_def_c3_0_5_prev1: - dist_curr_pair_c3_0_5
   + dist_diff_pair_c3_0_5_prev1 = -39
 dist_diff_abs_pair_c3_0_5_prev1_abs_ge_orig: - dist_diff_pair_c3_0_5_prev1
   + dist_diff_abs_pair_c3_0_5_prev1
   - 120 dist_diff_abs_pair_c3_0_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_5_prev1_abs_le_orig: - dist_diff_pair_c3_0_5_prev1
   + dist_diff_abs_pair_c3_0_5_prev1
   + 120 dist_diff_abs_pair_c3_0_5_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_0_5_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_0_5_prev1 + dist_diff_abs_pair_c3_0_5_prev1
   + 120 dist_diff_abs_pair_c3_0_5_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_0_5_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_0_5_prev1 + dist_diff_abs_pair_c3_0_5_prev1
   - 120 dist_diff_abs_pair_c3_0_5_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_0_5_prev1_force_positive:
   dist_diff_pair_c3_0_5_prev1
   - 120 dist_diff_abs_pair_c3_0_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_5_prev1_force_negative:
   dist_diff_pair_c3_0_5_prev1
   - 120 dist_diff_abs_pair_c3_0_5_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_0_5_prev1: 120 same_dist_pair_c3_0_5_prev1
   + dist_diff_abs_pair_c3_0_5_prev1 <= 129.8
 same_dist_pair_lower_c3_0_5_prev1: 120 same_dist_pair_c3_0_5_prev1
   + dist_diff_abs_pair_c3_0_5_prev1 >= 10
 dist_diff_pair_def_c3_0_5_prev2: - dist_curr_pair_c3_0_5
   + dist_diff_pair_c3_0_5_prev2 = -20
 dist_diff_abs_pair_c3_0_5_prev2_abs_ge_orig: - dist_diff_pair_c3_0_5_prev2
   + dist_diff_abs_pair_c3_0_5_prev2
   - 120 dist_diff_abs_pair_c3_0_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_5_prev2_abs_le_orig: - dist_diff_pair_c3_0_5_prev2
   + dist_diff_abs_pair_c3_0_5_prev2
   + 120 dist_diff_abs_pair_c3_0_5_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_0_5_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_0_5_prev2 + dist_diff_abs_pair_c3_0_5_prev2
   + 120 dist_diff_abs_pair_c3_0_5_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_0_5_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_0_5_prev2 + dist_diff_abs_pair_c3_0_5_prev2
   - 120 dist_diff_abs_pair_c3_0_5_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_0_5_prev2_force_positive:
   dist_diff_pair_c3_0_5_prev2
   - 120 dist_diff_abs_pair_c3_0_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_5_prev2_force_negative:
   dist_diff_pair_c3_0_5_prev2
   - 120 dist_diff_abs_pair_c3_0_5_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_0_5_prev2: 120 same_dist_pair_c3_0_5_prev2
   + dist_diff_abs_pair_c3_0_5_prev2 <= 129.8
 same_dist_pair_lower_c3_0_5_prev2: 120 same_dist_pair_c3_0_5_prev2
   + dist_diff_abs_pair_c3_0_5_prev2 >= 10
 dist_diff_pair_def_c3_0_5_prev3: - dist_curr_pair_c3_0_5
   + dist_diff_pair_c3_0_5_prev3 = -39
 dist_diff_abs_pair_c3_0_5_prev3_abs_ge_orig: - dist_diff_pair_c3_0_5_prev3
   + dist_diff_abs_pair_c3_0_5_prev3
   - 120 dist_diff_abs_pair_c3_0_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_5_prev3_abs_le_orig: - dist_diff_pair_c3_0_5_prev3
   + dist_diff_abs_pair_c3_0_5_prev3
   + 120 dist_diff_abs_pair_c3_0_5_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_0_5_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_0_5_prev3 + dist_diff_abs_pair_c3_0_5_prev3
   + 120 dist_diff_abs_pair_c3_0_5_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_0_5_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_0_5_prev3 + dist_diff_abs_pair_c3_0_5_prev3
   - 120 dist_diff_abs_pair_c3_0_5_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_0_5_prev3_force_positive:
   dist_diff_pair_c3_0_5_prev3
   - 120 dist_diff_abs_pair_c3_0_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_5_prev3_force_negative:
   dist_diff_pair_c3_0_5_prev3
   - 120 dist_diff_abs_pair_c3_0_5_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_0_5_prev3: 120 same_dist_pair_c3_0_5_prev3
   + dist_diff_abs_pair_c3_0_5_prev3 <= 129.8
 same_dist_pair_lower_c3_0_5_prev3: 120 same_dist_pair_c3_0_5_prev3
   + dist_diff_abs_pair_c3_0_5_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_0_5_prev0: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_0_5_prev0: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_0_5_prev1: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_0_5_prev1: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_0_5_prev2: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_0_5_prev2: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_0_5_prev3: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_0_5_prev3: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_0_5: diff_dist_pair_c3_0_5
   + same_dist_pair_c3_0_5_prev0 + same_dist_pair_c3_0_5_prev1
   + same_dist_pair_c3_0_5_prev2 + same_dist_pair_c3_0_5_prev3 >= 1
 dx_grid_diff_def_c3_0_6: - x_grid_0 + x_grid_6 + dx_grid_diff_c3_0_6 = 0
 dx_grid_abs_pair_c3_0_6_abs_ge_orig: dx_grid_abs_pair_c3_0_6
   - dx_grid_diff_c3_0_6 - 60 dx_grid_abs_pair_c3_0_6_is_positive >= -60
 dx_grid_abs_pair_c3_0_6_abs_le_orig: dx_grid_abs_pair_c3_0_6
   - dx_grid_diff_c3_0_6 + 60 dx_grid_abs_pair_c3_0_6_is_positive <= 60
 dx_grid_abs_pair_c3_0_6_abs_ge_neg_orig: dx_grid_abs_pair_c3_0_6
   + dx_grid_diff_c3_0_6 + 60 dx_grid_abs_pair_c3_0_6_is_positive >= 0
 dx_grid_abs_pair_c3_0_6_abs_le_neg_orig: dx_grid_abs_pair_c3_0_6
   + dx_grid_diff_c3_0_6 - 60 dx_grid_abs_pair_c3_0_6_is_positive <= 0
 dx_grid_abs_pair_c3_0_6_force_positive: dx_grid_diff_c3_0_6
   - 60 dx_grid_abs_pair_c3_0_6_is_positive >= -60
 dx_grid_abs_pair_c3_0_6_force_negative: dx_grid_diff_c3_0_6
   - 60 dx_grid_abs_pair_c3_0_6_is_positive <= -0.001
 dy_grid_diff_def_c3_0_6: - y_grid_0 + y_grid_6 + dy_grid_diff_c3_0_6 = 0
 dy_grid_abs_pair_c3_0_6_abs_ge_orig: dy_grid_abs_pair_c3_0_6
   - dy_grid_diff_c3_0_6 - 60 dy_grid_abs_pair_c3_0_6_is_positive >= -60
 dy_grid_abs_pair_c3_0_6_abs_le_orig: dy_grid_abs_pair_c3_0_6
   - dy_grid_diff_c3_0_6 + 60 dy_grid_abs_pair_c3_0_6_is_positive <= 60
 dy_grid_abs_pair_c3_0_6_abs_ge_neg_orig: dy_grid_abs_pair_c3_0_6
   + dy_grid_diff_c3_0_6 + 60 dy_grid_abs_pair_c3_0_6_is_positive >= 0
 dy_grid_abs_pair_c3_0_6_abs_le_neg_orig: dy_grid_abs_pair_c3_0_6
   + dy_grid_diff_c3_0_6 - 60 dy_grid_abs_pair_c3_0_6_is_positive <= 0
 dy_grid_abs_pair_c3_0_6_force_positive: dy_grid_diff_c3_0_6
   - 60 dy_grid_abs_pair_c3_0_6_is_positive >= -60
 dy_grid_abs_pair_c3_0_6_force_negative: dy_grid_diff_c3_0_6
   - 60 dy_grid_abs_pair_c3_0_6_is_positive <= -0.001
 dist_curr_pair_def_c3_0_6: - dx_grid_abs_pair_c3_0_6
   - dy_grid_abs_pair_c3_0_6 + dist_curr_pair_c3_0_6 = 0
 dist_diff_pair_def_c3_0_6_prev0: - dist_curr_pair_c3_0_6
   + dist_diff_pair_c3_0_6_prev0 = -56
 dist_diff_abs_pair_c3_0_6_prev0_abs_ge_orig: - dist_diff_pair_c3_0_6_prev0
   + dist_diff_abs_pair_c3_0_6_prev0
   - 120 dist_diff_abs_pair_c3_0_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_6_prev0_abs_le_orig: - dist_diff_pair_c3_0_6_prev0
   + dist_diff_abs_pair_c3_0_6_prev0
   + 120 dist_diff_abs_pair_c3_0_6_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_0_6_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_0_6_prev0 + dist_diff_abs_pair_c3_0_6_prev0
   + 120 dist_diff_abs_pair_c3_0_6_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_0_6_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_0_6_prev0 + dist_diff_abs_pair_c3_0_6_prev0
   - 120 dist_diff_abs_pair_c3_0_6_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_0_6_prev0_force_positive:
   dist_diff_pair_c3_0_6_prev0
   - 120 dist_diff_abs_pair_c3_0_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_6_prev0_force_negative:
   dist_diff_pair_c3_0_6_prev0
   - 120 dist_diff_abs_pair_c3_0_6_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_0_6_prev0: 120 same_dist_pair_c3_0_6_prev0
   + dist_diff_abs_pair_c3_0_6_prev0 <= 129.8
 same_dist_pair_lower_c3_0_6_prev0: 120 same_dist_pair_c3_0_6_prev0
   + dist_diff_abs_pair_c3_0_6_prev0 >= 10
 dist_diff_pair_def_c3_0_6_prev1: - dist_curr_pair_c3_0_6
   + dist_diff_pair_c3_0_6_prev1 = -49
 dist_diff_abs_pair_c3_0_6_prev1_abs_ge_orig: - dist_diff_pair_c3_0_6_prev1
   + dist_diff_abs_pair_c3_0_6_prev1
   - 120 dist_diff_abs_pair_c3_0_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_6_prev1_abs_le_orig: - dist_diff_pair_c3_0_6_prev1
   + dist_diff_abs_pair_c3_0_6_prev1
   + 120 dist_diff_abs_pair_c3_0_6_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_0_6_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_0_6_prev1 + dist_diff_abs_pair_c3_0_6_prev1
   + 120 dist_diff_abs_pair_c3_0_6_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_0_6_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_0_6_prev1 + dist_diff_abs_pair_c3_0_6_prev1
   - 120 dist_diff_abs_pair_c3_0_6_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_0_6_prev1_force_positive:
   dist_diff_pair_c3_0_6_prev1
   - 120 dist_diff_abs_pair_c3_0_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_6_prev1_force_negative:
   dist_diff_pair_c3_0_6_prev1
   - 120 dist_diff_abs_pair_c3_0_6_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_0_6_prev1: 120 same_dist_pair_c3_0_6_prev1
   + dist_diff_abs_pair_c3_0_6_prev1 <= 129.8
 same_dist_pair_lower_c3_0_6_prev1: 120 same_dist_pair_c3_0_6_prev1
   + dist_diff_abs_pair_c3_0_6_prev1 >= 10
 dist_diff_pair_def_c3_0_6_prev2: - dist_curr_pair_c3_0_6
   + dist_diff_pair_c3_0_6_prev2 = -45
 dist_diff_abs_pair_c3_0_6_prev2_abs_ge_orig: - dist_diff_pair_c3_0_6_prev2
   + dist_diff_abs_pair_c3_0_6_prev2
   - 120 dist_diff_abs_pair_c3_0_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_6_prev2_abs_le_orig: - dist_diff_pair_c3_0_6_prev2
   + dist_diff_abs_pair_c3_0_6_prev2
   + 120 dist_diff_abs_pair_c3_0_6_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_0_6_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_0_6_prev2 + dist_diff_abs_pair_c3_0_6_prev2
   + 120 dist_diff_abs_pair_c3_0_6_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_0_6_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_0_6_prev2 + dist_diff_abs_pair_c3_0_6_prev2
   - 120 dist_diff_abs_pair_c3_0_6_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_0_6_prev2_force_positive:
   dist_diff_pair_c3_0_6_prev2
   - 120 dist_diff_abs_pair_c3_0_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_6_prev2_force_negative:
   dist_diff_pair_c3_0_6_prev2
   - 120 dist_diff_abs_pair_c3_0_6_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_0_6_prev2: 120 same_dist_pair_c3_0_6_prev2
   + dist_diff_abs_pair_c3_0_6_prev2 <= 129.8
 same_dist_pair_lower_c3_0_6_prev2: 120 same_dist_pair_c3_0_6_prev2
   + dist_diff_abs_pair_c3_0_6_prev2 >= 10
 dist_diff_pair_def_c3_0_6_prev3: - dist_curr_pair_c3_0_6
   + dist_diff_pair_c3_0_6_prev3 = -49
 dist_diff_abs_pair_c3_0_6_prev3_abs_ge_orig: - dist_diff_pair_c3_0_6_prev3
   + dist_diff_abs_pair_c3_0_6_prev3
   - 120 dist_diff_abs_pair_c3_0_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_6_prev3_abs_le_orig: - dist_diff_pair_c3_0_6_prev3
   + dist_diff_abs_pair_c3_0_6_prev3
   + 120 dist_diff_abs_pair_c3_0_6_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_0_6_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_0_6_prev3 + dist_diff_abs_pair_c3_0_6_prev3
   + 120 dist_diff_abs_pair_c3_0_6_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_0_6_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_0_6_prev3 + dist_diff_abs_pair_c3_0_6_prev3
   - 120 dist_diff_abs_pair_c3_0_6_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_0_6_prev3_force_positive:
   dist_diff_pair_c3_0_6_prev3
   - 120 dist_diff_abs_pair_c3_0_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_6_prev3_force_negative:
   dist_diff_pair_c3_0_6_prev3
   - 120 dist_diff_abs_pair_c3_0_6_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_0_6_prev3: 120 same_dist_pair_c3_0_6_prev3
   + dist_diff_abs_pair_c3_0_6_prev3 <= 129.8
 same_dist_pair_lower_c3_0_6_prev3: 120 same_dist_pair_c3_0_6_prev3
   + dist_diff_abs_pair_c3_0_6_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_0_6_prev0: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_0_6_prev0: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_0_6_prev1: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_0_6_prev1: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_0_6_prev2: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_0_6_prev2: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_0_6_prev3: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_0_6_prev3: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_0_6: diff_dist_pair_c3_0_6
   + same_dist_pair_c3_0_6_prev0 + same_dist_pair_c3_0_6_prev1
   + same_dist_pair_c3_0_6_prev2 + same_dist_pair_c3_0_6_prev3 >= 1
 dx_grid_diff_def_c3_0_7: - x_grid_0 + x_grid_7 + dx_grid_diff_c3_0_7 = 0
 dx_grid_abs_pair_c3_0_7_abs_ge_orig: dx_grid_abs_pair_c3_0_7
   - dx_grid_diff_c3_0_7 - 60 dx_grid_abs_pair_c3_0_7_is_positive >= -60
 dx_grid_abs_pair_c3_0_7_abs_le_orig: dx_grid_abs_pair_c3_0_7
   - dx_grid_diff_c3_0_7 + 60 dx_grid_abs_pair_c3_0_7_is_positive <= 60
 dx_grid_abs_pair_c3_0_7_abs_ge_neg_orig: dx_grid_abs_pair_c3_0_7
   + dx_grid_diff_c3_0_7 + 60 dx_grid_abs_pair_c3_0_7_is_positive >= 0
 dx_grid_abs_pair_c3_0_7_abs_le_neg_orig: dx_grid_abs_pair_c3_0_7
   + dx_grid_diff_c3_0_7 - 60 dx_grid_abs_pair_c3_0_7_is_positive <= 0
 dx_grid_abs_pair_c3_0_7_force_positive: dx_grid_diff_c3_0_7
   - 60 dx_grid_abs_pair_c3_0_7_is_positive >= -60
 dx_grid_abs_pair_c3_0_7_force_negative: dx_grid_diff_c3_0_7
   - 60 dx_grid_abs_pair_c3_0_7_is_positive <= -0.001
 dy_grid_diff_def_c3_0_7: - y_grid_0 + y_grid_7 + dy_grid_diff_c3_0_7 = 0
 dy_grid_abs_pair_c3_0_7_abs_ge_orig: dy_grid_abs_pair_c3_0_7
   - dy_grid_diff_c3_0_7 - 60 dy_grid_abs_pair_c3_0_7_is_positive >= -60
 dy_grid_abs_pair_c3_0_7_abs_le_orig: dy_grid_abs_pair_c3_0_7
   - dy_grid_diff_c3_0_7 + 60 dy_grid_abs_pair_c3_0_7_is_positive <= 60
 dy_grid_abs_pair_c3_0_7_abs_ge_neg_orig: dy_grid_abs_pair_c3_0_7
   + dy_grid_diff_c3_0_7 + 60 dy_grid_abs_pair_c3_0_7_is_positive >= 0
 dy_grid_abs_pair_c3_0_7_abs_le_neg_orig: dy_grid_abs_pair_c3_0_7
   + dy_grid_diff_c3_0_7 - 60 dy_grid_abs_pair_c3_0_7_is_positive <= 0
 dy_grid_abs_pair_c3_0_7_force_positive: dy_grid_diff_c3_0_7
   - 60 dy_grid_abs_pair_c3_0_7_is_positive >= -60
 dy_grid_abs_pair_c3_0_7_force_negative: dy_grid_diff_c3_0_7
   - 60 dy_grid_abs_pair_c3_0_7_is_positive <= -0.001
 dist_curr_pair_def_c3_0_7: - dx_grid_abs_pair_c3_0_7
   - dy_grid_abs_pair_c3_0_7 + dist_curr_pair_c3_0_7 = 0
 dist_diff_pair_def_c3_0_7_prev0: - dist_curr_pair_c3_0_7
   + dist_diff_pair_c3_0_7_prev0 = -38
 dist_diff_abs_pair_c3_0_7_prev0_abs_ge_orig: - dist_diff_pair_c3_0_7_prev0
   + dist_diff_abs_pair_c3_0_7_prev0
   - 120 dist_diff_abs_pair_c3_0_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_7_prev0_abs_le_orig: - dist_diff_pair_c3_0_7_prev0
   + dist_diff_abs_pair_c3_0_7_prev0
   + 120 dist_diff_abs_pair_c3_0_7_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_0_7_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_0_7_prev0 + dist_diff_abs_pair_c3_0_7_prev0
   + 120 dist_diff_abs_pair_c3_0_7_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_0_7_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_0_7_prev0 + dist_diff_abs_pair_c3_0_7_prev0
   - 120 dist_diff_abs_pair_c3_0_7_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_0_7_prev0_force_positive:
   dist_diff_pair_c3_0_7_prev0
   - 120 dist_diff_abs_pair_c3_0_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_0_7_prev0_force_negative:
   dist_diff_pair_c3_0_7_prev0
   - 120 dist_diff_abs_pair_c3_0_7_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_0_7_prev0: 120 same_dist_pair_c3_0_7_prev0
   + dist_diff_abs_pair_c3_0_7_prev0 <= 129.8
 same_dist_pair_lower_c3_0_7_prev0: 120 same_dist_pair_c3_0_7_prev0
   + dist_diff_abs_pair_c3_0_7_prev0 >= 10
 dist_diff_pair_def_c3_0_7_prev1: - dist_curr_pair_c3_0_7
   + dist_diff_pair_c3_0_7_prev1 = -37
 dist_diff_abs_pair_c3_0_7_prev1_abs_ge_orig: - dist_diff_pair_c3_0_7_prev1
   + dist_diff_abs_pair_c3_0_7_prev1
   - 120 dist_diff_abs_pair_c3_0_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_7_prev1_abs_le_orig: - dist_diff_pair_c3_0_7_prev1
   + dist_diff_abs_pair_c3_0_7_prev1
   + 120 dist_diff_abs_pair_c3_0_7_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_0_7_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_0_7_prev1 + dist_diff_abs_pair_c3_0_7_prev1
   + 120 dist_diff_abs_pair_c3_0_7_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_0_7_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_0_7_prev1 + dist_diff_abs_pair_c3_0_7_prev1
   - 120 dist_diff_abs_pair_c3_0_7_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_0_7_prev1_force_positive:
   dist_diff_pair_c3_0_7_prev1
   - 120 dist_diff_abs_pair_c3_0_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_0_7_prev1_force_negative:
   dist_diff_pair_c3_0_7_prev1
   - 120 dist_diff_abs_pair_c3_0_7_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_0_7_prev1: 120 same_dist_pair_c3_0_7_prev1
   + dist_diff_abs_pair_c3_0_7_prev1 <= 129.8
 same_dist_pair_lower_c3_0_7_prev1: 120 same_dist_pair_c3_0_7_prev1
   + dist_diff_abs_pair_c3_0_7_prev1 >= 10
 dist_diff_pair_def_c3_0_7_prev2: - dist_curr_pair_c3_0_7
   + dist_diff_pair_c3_0_7_prev2 = -29
 dist_diff_abs_pair_c3_0_7_prev2_abs_ge_orig: - dist_diff_pair_c3_0_7_prev2
   + dist_diff_abs_pair_c3_0_7_prev2
   - 120 dist_diff_abs_pair_c3_0_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_7_prev2_abs_le_orig: - dist_diff_pair_c3_0_7_prev2
   + dist_diff_abs_pair_c3_0_7_prev2
   + 120 dist_diff_abs_pair_c3_0_7_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_0_7_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_0_7_prev2 + dist_diff_abs_pair_c3_0_7_prev2
   + 120 dist_diff_abs_pair_c3_0_7_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_0_7_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_0_7_prev2 + dist_diff_abs_pair_c3_0_7_prev2
   - 120 dist_diff_abs_pair_c3_0_7_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_0_7_prev2_force_positive:
   dist_diff_pair_c3_0_7_prev2
   - 120 dist_diff_abs_pair_c3_0_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_0_7_prev2_force_negative:
   dist_diff_pair_c3_0_7_prev2
   - 120 dist_diff_abs_pair_c3_0_7_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_0_7_prev2: 120 same_dist_pair_c3_0_7_prev2
   + dist_diff_abs_pair_c3_0_7_prev2 <= 129.8
 same_dist_pair_lower_c3_0_7_prev2: 120 same_dist_pair_c3_0_7_prev2
   + dist_diff_abs_pair_c3_0_7_prev2 >= 10
 dist_diff_pair_def_c3_0_7_prev3: - dist_curr_pair_c3_0_7
   + dist_diff_pair_c3_0_7_prev3 = -37
 dist_diff_abs_pair_c3_0_7_prev3_abs_ge_orig: - dist_diff_pair_c3_0_7_prev3
   + dist_diff_abs_pair_c3_0_7_prev3
   - 120 dist_diff_abs_pair_c3_0_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_7_prev3_abs_le_orig: - dist_diff_pair_c3_0_7_prev3
   + dist_diff_abs_pair_c3_0_7_prev3
   + 120 dist_diff_abs_pair_c3_0_7_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_0_7_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_0_7_prev3 + dist_diff_abs_pair_c3_0_7_prev3
   + 120 dist_diff_abs_pair_c3_0_7_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_0_7_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_0_7_prev3 + dist_diff_abs_pair_c3_0_7_prev3
   - 120 dist_diff_abs_pair_c3_0_7_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_0_7_prev3_force_positive:
   dist_diff_pair_c3_0_7_prev3
   - 120 dist_diff_abs_pair_c3_0_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_0_7_prev3_force_negative:
   dist_diff_pair_c3_0_7_prev3
   - 120 dist_diff_abs_pair_c3_0_7_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_0_7_prev3: 120 same_dist_pair_c3_0_7_prev3
   + dist_diff_abs_pair_c3_0_7_prev3 <= 129.8
 same_dist_pair_lower_c3_0_7_prev3: 120 same_dist_pair_c3_0_7_prev3
   + dist_diff_abs_pair_c3_0_7_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_0_7_prev0: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_0_7_prev0: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_0_7_prev1: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_0_7_prev1: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_0_7_prev2: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_0_7_prev2: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_0_7_prev3: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_0_7_prev3: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_0_7: diff_dist_pair_c3_0_7
   + same_dist_pair_c3_0_7_prev0 + same_dist_pair_c3_0_7_prev1
   + same_dist_pair_c3_0_7_prev2 + same_dist_pair_c3_0_7_prev3 >= 1
 dx_grid_diff_def_c3_1_2: - x_grid_1 + x_grid_2 + dx_grid_diff_c3_1_2 = 0
 dx_grid_abs_pair_c3_1_2_abs_ge_orig: dx_grid_abs_pair_c3_1_2
   - dx_grid_diff_c3_1_2 - 60 dx_grid_abs_pair_c3_1_2_is_positive >= -60
 dx_grid_abs_pair_c3_1_2_abs_le_orig: dx_grid_abs_pair_c3_1_2
   - dx_grid_diff_c3_1_2 + 60 dx_grid_abs_pair_c3_1_2_is_positive <= 60
 dx_grid_abs_pair_c3_1_2_abs_ge_neg_orig: dx_grid_abs_pair_c3_1_2
   + dx_grid_diff_c3_1_2 + 60 dx_grid_abs_pair_c3_1_2_is_positive >= 0
 dx_grid_abs_pair_c3_1_2_abs_le_neg_orig: dx_grid_abs_pair_c3_1_2
   + dx_grid_diff_c3_1_2 - 60 dx_grid_abs_pair_c3_1_2_is_positive <= 0
 dx_grid_abs_pair_c3_1_2_force_positive: dx_grid_diff_c3_1_2
   - 60 dx_grid_abs_pair_c3_1_2_is_positive >= -60
 dx_grid_abs_pair_c3_1_2_force_negative: dx_grid_diff_c3_1_2
   - 60 dx_grid_abs_pair_c3_1_2_is_positive <= -0.001
 dy_grid_diff_def_c3_1_2: - y_grid_1 + y_grid_2 + dy_grid_diff_c3_1_2 = 0
 dy_grid_abs_pair_c3_1_2_abs_ge_orig: dy_grid_abs_pair_c3_1_2
   - dy_grid_diff_c3_1_2 - 60 dy_grid_abs_pair_c3_1_2_is_positive >= -60
 dy_grid_abs_pair_c3_1_2_abs_le_orig: dy_grid_abs_pair_c3_1_2
   - dy_grid_diff_c3_1_2 + 60 dy_grid_abs_pair_c3_1_2_is_positive <= 60
 dy_grid_abs_pair_c3_1_2_abs_ge_neg_orig: dy_grid_abs_pair_c3_1_2
   + dy_grid_diff_c3_1_2 + 60 dy_grid_abs_pair_c3_1_2_is_positive >= 0
 dy_grid_abs_pair_c3_1_2_abs_le_neg_orig: dy_grid_abs_pair_c3_1_2
   + dy_grid_diff_c3_1_2 - 60 dy_grid_abs_pair_c3_1_2_is_positive <= 0
 dy_grid_abs_pair_c3_1_2_force_positive: dy_grid_diff_c3_1_2
   - 60 dy_grid_abs_pair_c3_1_2_is_positive >= -60
 dy_grid_abs_pair_c3_1_2_force_negative: dy_grid_diff_c3_1_2
   - 60 dy_grid_abs_pair_c3_1_2_is_positive <= -0.001
 dist_curr_pair_def_c3_1_2: - dx_grid_abs_pair_c3_1_2
   - dy_grid_abs_pair_c3_1_2 + dist_curr_pair_c3_1_2 = 0
 dist_diff_pair_def_c3_1_2_prev0: - dist_curr_pair_c3_1_2
   + dist_diff_pair_c3_1_2_prev0 = -40
 dist_diff_abs_pair_c3_1_2_prev0_abs_ge_orig: - dist_diff_pair_c3_1_2_prev0
   + dist_diff_abs_pair_c3_1_2_prev0
   - 120 dist_diff_abs_pair_c3_1_2_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_2_prev0_abs_le_orig: - dist_diff_pair_c3_1_2_prev0
   + dist_diff_abs_pair_c3_1_2_prev0
   + 120 dist_diff_abs_pair_c3_1_2_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_1_2_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_1_2_prev0 + dist_diff_abs_pair_c3_1_2_prev0
   + 120 dist_diff_abs_pair_c3_1_2_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_1_2_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_1_2_prev0 + dist_diff_abs_pair_c3_1_2_prev0
   - 120 dist_diff_abs_pair_c3_1_2_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_1_2_prev0_force_positive:
   dist_diff_pair_c3_1_2_prev0
   - 120 dist_diff_abs_pair_c3_1_2_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_2_prev0_force_negative:
   dist_diff_pair_c3_1_2_prev0
   - 120 dist_diff_abs_pair_c3_1_2_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_1_2_prev0: 120 same_dist_pair_c3_1_2_prev0
   + dist_diff_abs_pair_c3_1_2_prev0 <= 129.8
 same_dist_pair_lower_c3_1_2_prev0: 120 same_dist_pair_c3_1_2_prev0
   + dist_diff_abs_pair_c3_1_2_prev0 >= 10
 dist_diff_pair_def_c3_1_2_prev1: - dist_curr_pair_c3_1_2
   + dist_diff_pair_c3_1_2_prev1 = -21
 dist_diff_abs_pair_c3_1_2_prev1_abs_ge_orig: - dist_diff_pair_c3_1_2_prev1
   + dist_diff_abs_pair_c3_1_2_prev1
   - 120 dist_diff_abs_pair_c3_1_2_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_2_prev1_abs_le_orig: - dist_diff_pair_c3_1_2_prev1
   + dist_diff_abs_pair_c3_1_2_prev1
   + 120 dist_diff_abs_pair_c3_1_2_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_1_2_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_1_2_prev1 + dist_diff_abs_pair_c3_1_2_prev1
   + 120 dist_diff_abs_pair_c3_1_2_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_1_2_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_1_2_prev1 + dist_diff_abs_pair_c3_1_2_prev1
   - 120 dist_diff_abs_pair_c3_1_2_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_1_2_prev1_force_positive:
   dist_diff_pair_c3_1_2_prev1
   - 120 dist_diff_abs_pair_c3_1_2_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_2_prev1_force_negative:
   dist_diff_pair_c3_1_2_prev1
   - 120 dist_diff_abs_pair_c3_1_2_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_1_2_prev1: 120 same_dist_pair_c3_1_2_prev1
   + dist_diff_abs_pair_c3_1_2_prev1 <= 129.8
 same_dist_pair_lower_c3_1_2_prev1: 120 same_dist_pair_c3_1_2_prev1
   + dist_diff_abs_pair_c3_1_2_prev1 >= 10
 dist_diff_pair_def_c3_1_2_prev2: - dist_curr_pair_c3_1_2
   + dist_diff_pair_c3_1_2_prev2 = -31
 dist_diff_abs_pair_c3_1_2_prev2_abs_ge_orig: - dist_diff_pair_c3_1_2_prev2
   + dist_diff_abs_pair_c3_1_2_prev2
   - 120 dist_diff_abs_pair_c3_1_2_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_2_prev2_abs_le_orig: - dist_diff_pair_c3_1_2_prev2
   + dist_diff_abs_pair_c3_1_2_prev2
   + 120 dist_diff_abs_pair_c3_1_2_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_1_2_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_1_2_prev2 + dist_diff_abs_pair_c3_1_2_prev2
   + 120 dist_diff_abs_pair_c3_1_2_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_1_2_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_1_2_prev2 + dist_diff_abs_pair_c3_1_2_prev2
   - 120 dist_diff_abs_pair_c3_1_2_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_1_2_prev2_force_positive:
   dist_diff_pair_c3_1_2_prev2
   - 120 dist_diff_abs_pair_c3_1_2_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_2_prev2_force_negative:
   dist_diff_pair_c3_1_2_prev2
   - 120 dist_diff_abs_pair_c3_1_2_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_1_2_prev2: 120 same_dist_pair_c3_1_2_prev2
   + dist_diff_abs_pair_c3_1_2_prev2 <= 129.8
 same_dist_pair_lower_c3_1_2_prev2: 120 same_dist_pair_c3_1_2_prev2
   + dist_diff_abs_pair_c3_1_2_prev2 >= 10
 dist_diff_pair_def_c3_1_2_prev3: - dist_curr_pair_c3_1_2
   + dist_diff_pair_c3_1_2_prev3 = -21
 dist_diff_abs_pair_c3_1_2_prev3_abs_ge_orig: - dist_diff_pair_c3_1_2_prev3
   + dist_diff_abs_pair_c3_1_2_prev3
   - 120 dist_diff_abs_pair_c3_1_2_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_2_prev3_abs_le_orig: - dist_diff_pair_c3_1_2_prev3
   + dist_diff_abs_pair_c3_1_2_prev3
   + 120 dist_diff_abs_pair_c3_1_2_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_1_2_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_1_2_prev3 + dist_diff_abs_pair_c3_1_2_prev3
   + 120 dist_diff_abs_pair_c3_1_2_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_1_2_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_1_2_prev3 + dist_diff_abs_pair_c3_1_2_prev3
   - 120 dist_diff_abs_pair_c3_1_2_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_1_2_prev3_force_positive:
   dist_diff_pair_c3_1_2_prev3
   - 120 dist_diff_abs_pair_c3_1_2_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_2_prev3_force_negative:
   dist_diff_pair_c3_1_2_prev3
   - 120 dist_diff_abs_pair_c3_1_2_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_1_2_prev3: 120 same_dist_pair_c3_1_2_prev3
   + dist_diff_abs_pair_c3_1_2_prev3 <= 129.8
 same_dist_pair_lower_c3_1_2_prev3: 120 same_dist_pair_c3_1_2_prev3
   + dist_diff_abs_pair_c3_1_2_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_1_2_prev0: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_1_2_prev0: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_1_2_prev1: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_1_2_prev1: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_1_2_prev2: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_1_2_prev2: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_1_2_prev3: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_1_2_prev3: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_1_2: diff_dist_pair_c3_1_2
   + same_dist_pair_c3_1_2_prev0 + same_dist_pair_c3_1_2_prev1
   + same_dist_pair_c3_1_2_prev2 + same_dist_pair_c3_1_2_prev3 >= 1
 dx_grid_diff_def_c3_1_3: - x_grid_1 + x_grid_3 + dx_grid_diff_c3_1_3 = 0
 dx_grid_abs_pair_c3_1_3_abs_ge_orig: dx_grid_abs_pair_c3_1_3
   - dx_grid_diff_c3_1_3 - 60 dx_grid_abs_pair_c3_1_3_is_positive >= -60
 dx_grid_abs_pair_c3_1_3_abs_le_orig: dx_grid_abs_pair_c3_1_3
   - dx_grid_diff_c3_1_3 + 60 dx_grid_abs_pair_c3_1_3_is_positive <= 60
 dx_grid_abs_pair_c3_1_3_abs_ge_neg_orig: dx_grid_abs_pair_c3_1_3
   + dx_grid_diff_c3_1_3 + 60 dx_grid_abs_pair_c3_1_3_is_positive >= 0
 dx_grid_abs_pair_c3_1_3_abs_le_neg_orig: dx_grid_abs_pair_c3_1_3
   + dx_grid_diff_c3_1_3 - 60 dx_grid_abs_pair_c3_1_3_is_positive <= 0
 dx_grid_abs_pair_c3_1_3_force_positive: dx_grid_diff_c3_1_3
   - 60 dx_grid_abs_pair_c3_1_3_is_positive >= -60
 dx_grid_abs_pair_c3_1_3_force_negative: dx_grid_diff_c3_1_3
   - 60 dx_grid_abs_pair_c3_1_3_is_positive <= -0.001
 dy_grid_diff_def_c3_1_3: - y_grid_1 + y_grid_3 + dy_grid_diff_c3_1_3 = 0
 dy_grid_abs_pair_c3_1_3_abs_ge_orig: dy_grid_abs_pair_c3_1_3
   - dy_grid_diff_c3_1_3 - 60 dy_grid_abs_pair_c3_1_3_is_positive >= -60
 dy_grid_abs_pair_c3_1_3_abs_le_orig: dy_grid_abs_pair_c3_1_3
   - dy_grid_diff_c3_1_3 + 60 dy_grid_abs_pair_c3_1_3_is_positive <= 60
 dy_grid_abs_pair_c3_1_3_abs_ge_neg_orig: dy_grid_abs_pair_c3_1_3
   + dy_grid_diff_c3_1_3 + 60 dy_grid_abs_pair_c3_1_3_is_positive >= 0
 dy_grid_abs_pair_c3_1_3_abs_le_neg_orig: dy_grid_abs_pair_c3_1_3
   + dy_grid_diff_c3_1_3 - 60 dy_grid_abs_pair_c3_1_3_is_positive <= 0
 dy_grid_abs_pair_c3_1_3_force_positive: dy_grid_diff_c3_1_3
   - 60 dy_grid_abs_pair_c3_1_3_is_positive >= -60
 dy_grid_abs_pair_c3_1_3_force_negative: dy_grid_diff_c3_1_3
   - 60 dy_grid_abs_pair_c3_1_3_is_positive <= -0.001
 dist_curr_pair_def_c3_1_3: - dx_grid_abs_pair_c3_1_3
   - dy_grid_abs_pair_c3_1_3 + dist_curr_pair_c3_1_3 = 0
 dist_diff_pair_def_c3_1_3_prev0: - dist_curr_pair_c3_1_3
   + dist_diff_pair_c3_1_3_prev0 = -26
 dist_diff_abs_pair_c3_1_3_prev0_abs_ge_orig: - dist_diff_pair_c3_1_3_prev0
   + dist_diff_abs_pair_c3_1_3_prev0
   - 120 dist_diff_abs_pair_c3_1_3_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_3_prev0_abs_le_orig: - dist_diff_pair_c3_1_3_prev0
   + dist_diff_abs_pair_c3_1_3_prev0
   + 120 dist_diff_abs_pair_c3_1_3_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_1_3_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_1_3_prev0 + dist_diff_abs_pair_c3_1_3_prev0
   + 120 dist_diff_abs_pair_c3_1_3_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_1_3_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_1_3_prev0 + dist_diff_abs_pair_c3_1_3_prev0
   - 120 dist_diff_abs_pair_c3_1_3_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_1_3_prev0_force_positive:
   dist_diff_pair_c3_1_3_prev0
   - 120 dist_diff_abs_pair_c3_1_3_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_3_prev0_force_negative:
   dist_diff_pair_c3_1_3_prev0
   - 120 dist_diff_abs_pair_c3_1_3_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_1_3_prev0: 120 same_dist_pair_c3_1_3_prev0
   + dist_diff_abs_pair_c3_1_3_prev0 <= 129.8
 same_dist_pair_lower_c3_1_3_prev0: 120 same_dist_pair_c3_1_3_prev0
   + dist_diff_abs_pair_c3_1_3_prev0 >= 10
 dist_diff_pair_def_c3_1_3_prev1: - dist_curr_pair_c3_1_3
   + dist_diff_pair_c3_1_3_prev1 = -21
 dist_diff_abs_pair_c3_1_3_prev1_abs_ge_orig: - dist_diff_pair_c3_1_3_prev1
   + dist_diff_abs_pair_c3_1_3_prev1
   - 120 dist_diff_abs_pair_c3_1_3_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_3_prev1_abs_le_orig: - dist_diff_pair_c3_1_3_prev1
   + dist_diff_abs_pair_c3_1_3_prev1
   + 120 dist_diff_abs_pair_c3_1_3_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_1_3_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_1_3_prev1 + dist_diff_abs_pair_c3_1_3_prev1
   + 120 dist_diff_abs_pair_c3_1_3_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_1_3_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_1_3_prev1 + dist_diff_abs_pair_c3_1_3_prev1
   - 120 dist_diff_abs_pair_c3_1_3_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_1_3_prev1_force_positive:
   dist_diff_pair_c3_1_3_prev1
   - 120 dist_diff_abs_pair_c3_1_3_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_3_prev1_force_negative:
   dist_diff_pair_c3_1_3_prev1
   - 120 dist_diff_abs_pair_c3_1_3_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_1_3_prev1: 120 same_dist_pair_c3_1_3_prev1
   + dist_diff_abs_pair_c3_1_3_prev1 <= 129.8
 same_dist_pair_lower_c3_1_3_prev1: 120 same_dist_pair_c3_1_3_prev1
   + dist_diff_abs_pair_c3_1_3_prev1 >= 10
 dist_diff_pair_def_c3_1_3_prev2: - dist_curr_pair_c3_1_3
   + dist_diff_pair_c3_1_3_prev2 = -11
 dist_diff_abs_pair_c3_1_3_prev2_abs_ge_orig: - dist_diff_pair_c3_1_3_prev2
   + dist_diff_abs_pair_c3_1_3_prev2
   - 120 dist_diff_abs_pair_c3_1_3_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_3_prev2_abs_le_orig: - dist_diff_pair_c3_1_3_prev2
   + dist_diff_abs_pair_c3_1_3_prev2
   + 120 dist_diff_abs_pair_c3_1_3_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_1_3_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_1_3_prev2 + dist_diff_abs_pair_c3_1_3_prev2
   + 120 dist_diff_abs_pair_c3_1_3_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_1_3_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_1_3_prev2 + dist_diff_abs_pair_c3_1_3_prev2
   - 120 dist_diff_abs_pair_c3_1_3_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_1_3_prev2_force_positive:
   dist_diff_pair_c3_1_3_prev2
   - 120 dist_diff_abs_pair_c3_1_3_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_3_prev2_force_negative:
   dist_diff_pair_c3_1_3_prev2
   - 120 dist_diff_abs_pair_c3_1_3_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_1_3_prev2: 120 same_dist_pair_c3_1_3_prev2
   + dist_diff_abs_pair_c3_1_3_prev2 <= 129.8
 same_dist_pair_lower_c3_1_3_prev2: 120 same_dist_pair_c3_1_3_prev2
   + dist_diff_abs_pair_c3_1_3_prev2 >= 10
 dist_diff_pair_def_c3_1_3_prev3: - dist_curr_pair_c3_1_3
   + dist_diff_pair_c3_1_3_prev3 = -11
 dist_diff_abs_pair_c3_1_3_prev3_abs_ge_orig: - dist_diff_pair_c3_1_3_prev3
   + dist_diff_abs_pair_c3_1_3_prev3
   - 120 dist_diff_abs_pair_c3_1_3_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_3_prev3_abs_le_orig: - dist_diff_pair_c3_1_3_prev3
   + dist_diff_abs_pair_c3_1_3_prev3
   + 120 dist_diff_abs_pair_c3_1_3_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_1_3_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_1_3_prev3 + dist_diff_abs_pair_c3_1_3_prev3
   + 120 dist_diff_abs_pair_c3_1_3_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_1_3_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_1_3_prev3 + dist_diff_abs_pair_c3_1_3_prev3
   - 120 dist_diff_abs_pair_c3_1_3_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_1_3_prev3_force_positive:
   dist_diff_pair_c3_1_3_prev3
   - 120 dist_diff_abs_pair_c3_1_3_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_3_prev3_force_negative:
   dist_diff_pair_c3_1_3_prev3
   - 120 dist_diff_abs_pair_c3_1_3_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_1_3_prev3: 120 same_dist_pair_c3_1_3_prev3
   + dist_diff_abs_pair_c3_1_3_prev3 <= 129.8
 same_dist_pair_lower_c3_1_3_prev3: 120 same_dist_pair_c3_1_3_prev3
   + dist_diff_abs_pair_c3_1_3_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_1_3_prev0: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_1_3_prev0: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_1_3_prev1: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_1_3_prev1: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_1_3_prev2: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_1_3_prev2: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_1_3_prev3: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_1_3_prev3: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_1_3: diff_dist_pair_c3_1_3
   + same_dist_pair_c3_1_3_prev0 + same_dist_pair_c3_1_3_prev1
   + same_dist_pair_c3_1_3_prev2 + same_dist_pair_c3_1_3_prev3 >= 1
 dx_grid_diff_def_c3_1_4: - x_grid_1 + x_grid_4 + dx_grid_diff_c3_1_4 = 0
 dx_grid_abs_pair_c3_1_4_abs_ge_orig: dx_grid_abs_pair_c3_1_4
   - dx_grid_diff_c3_1_4 - 60 dx_grid_abs_pair_c3_1_4_is_positive >= -60
 dx_grid_abs_pair_c3_1_4_abs_le_orig: dx_grid_abs_pair_c3_1_4
   - dx_grid_diff_c3_1_4 + 60 dx_grid_abs_pair_c3_1_4_is_positive <= 60
 dx_grid_abs_pair_c3_1_4_abs_ge_neg_orig: dx_grid_abs_pair_c3_1_4
   + dx_grid_diff_c3_1_4 + 60 dx_grid_abs_pair_c3_1_4_is_positive >= 0
 dx_grid_abs_pair_c3_1_4_abs_le_neg_orig: dx_grid_abs_pair_c3_1_4
   + dx_grid_diff_c3_1_4 - 60 dx_grid_abs_pair_c3_1_4_is_positive <= 0
 dx_grid_abs_pair_c3_1_4_force_positive: dx_grid_diff_c3_1_4
   - 60 dx_grid_abs_pair_c3_1_4_is_positive >= -60
 dx_grid_abs_pair_c3_1_4_force_negative: dx_grid_diff_c3_1_4
   - 60 dx_grid_abs_pair_c3_1_4_is_positive <= -0.001
 dy_grid_diff_def_c3_1_4: - y_grid_1 + y_grid_4 + dy_grid_diff_c3_1_4 = 0
 dy_grid_abs_pair_c3_1_4_abs_ge_orig: dy_grid_abs_pair_c3_1_4
   - dy_grid_diff_c3_1_4 - 60 dy_grid_abs_pair_c3_1_4_is_positive >= -60
 dy_grid_abs_pair_c3_1_4_abs_le_orig: dy_grid_abs_pair_c3_1_4
   - dy_grid_diff_c3_1_4 + 60 dy_grid_abs_pair_c3_1_4_is_positive <= 60
 dy_grid_abs_pair_c3_1_4_abs_ge_neg_orig: dy_grid_abs_pair_c3_1_4
   + dy_grid_diff_c3_1_4 + 60 dy_grid_abs_pair_c3_1_4_is_positive >= 0
 dy_grid_abs_pair_c3_1_4_abs_le_neg_orig: dy_grid_abs_pair_c3_1_4
   + dy_grid_diff_c3_1_4 - 60 dy_grid_abs_pair_c3_1_4_is_positive <= 0
 dy_grid_abs_pair_c3_1_4_force_positive: dy_grid_diff_c3_1_4
   - 60 dy_grid_abs_pair_c3_1_4_is_positive >= -60
 dy_grid_abs_pair_c3_1_4_force_negative: dy_grid_diff_c3_1_4
   - 60 dy_grid_abs_pair_c3_1_4_is_positive <= -0.001
 dist_curr_pair_def_c3_1_4: - dx_grid_abs_pair_c3_1_4
   - dy_grid_abs_pair_c3_1_4 + dist_curr_pair_c3_1_4 = 0
 dist_diff_pair_def_c3_1_4_prev0: - dist_curr_pair_c3_1_4
   + dist_diff_pair_c3_1_4_prev0 = -10
 dist_diff_abs_pair_c3_1_4_prev0_abs_ge_orig: - dist_diff_pair_c3_1_4_prev0
   + dist_diff_abs_pair_c3_1_4_prev0
   - 120 dist_diff_abs_pair_c3_1_4_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_4_prev0_abs_le_orig: - dist_diff_pair_c3_1_4_prev0
   + dist_diff_abs_pair_c3_1_4_prev0
   + 120 dist_diff_abs_pair_c3_1_4_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_1_4_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_1_4_prev0 + dist_diff_abs_pair_c3_1_4_prev0
   + 120 dist_diff_abs_pair_c3_1_4_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_1_4_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_1_4_prev0 + dist_diff_abs_pair_c3_1_4_prev0
   - 120 dist_diff_abs_pair_c3_1_4_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_1_4_prev0_force_positive:
   dist_diff_pair_c3_1_4_prev0
   - 120 dist_diff_abs_pair_c3_1_4_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_4_prev0_force_negative:
   dist_diff_pair_c3_1_4_prev0
   - 120 dist_diff_abs_pair_c3_1_4_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_1_4_prev0: 120 same_dist_pair_c3_1_4_prev0
   + dist_diff_abs_pair_c3_1_4_prev0 <= 129.8
 same_dist_pair_lower_c3_1_4_prev0: 120 same_dist_pair_c3_1_4_prev0
   + dist_diff_abs_pair_c3_1_4_prev0 >= 10
 dist_diff_pair_def_c3_1_4_prev1: - dist_curr_pair_c3_1_4
   + dist_diff_pair_c3_1_4_prev1 = -11
 dist_diff_abs_pair_c3_1_4_prev1_abs_ge_orig: - dist_diff_pair_c3_1_4_prev1
   + dist_diff_abs_pair_c3_1_4_prev1
   - 120 dist_diff_abs_pair_c3_1_4_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_4_prev1_abs_le_orig: - dist_diff_pair_c3_1_4_prev1
   + dist_diff_abs_pair_c3_1_4_prev1
   + 120 dist_diff_abs_pair_c3_1_4_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_1_4_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_1_4_prev1 + dist_diff_abs_pair_c3_1_4_prev1
   + 120 dist_diff_abs_pair_c3_1_4_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_1_4_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_1_4_prev1 + dist_diff_abs_pair_c3_1_4_prev1
   - 120 dist_diff_abs_pair_c3_1_4_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_1_4_prev1_force_positive:
   dist_diff_pair_c3_1_4_prev1
   - 120 dist_diff_abs_pair_c3_1_4_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_4_prev1_force_negative:
   dist_diff_pair_c3_1_4_prev1
   - 120 dist_diff_abs_pair_c3_1_4_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_1_4_prev1: 120 same_dist_pair_c3_1_4_prev1
   + dist_diff_abs_pair_c3_1_4_prev1 <= 129.8
 same_dist_pair_lower_c3_1_4_prev1: 120 same_dist_pair_c3_1_4_prev1
   + dist_diff_abs_pair_c3_1_4_prev1 >= 10
 dist_diff_pair_def_c3_1_4_prev2: - dist_curr_pair_c3_1_4
   + dist_diff_pair_c3_1_4_prev2 = -10
 dist_diff_abs_pair_c3_1_4_prev2_abs_ge_orig: - dist_diff_pair_c3_1_4_prev2
   + dist_diff_abs_pair_c3_1_4_prev2
   - 120 dist_diff_abs_pair_c3_1_4_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_4_prev2_abs_le_orig: - dist_diff_pair_c3_1_4_prev2
   + dist_diff_abs_pair_c3_1_4_prev2
   + 120 dist_diff_abs_pair_c3_1_4_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_1_4_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_1_4_prev2 + dist_diff_abs_pair_c3_1_4_prev2
   + 120 dist_diff_abs_pair_c3_1_4_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_1_4_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_1_4_prev2 + dist_diff_abs_pair_c3_1_4_prev2
   - 120 dist_diff_abs_pair_c3_1_4_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_1_4_prev2_force_positive:
   dist_diff_pair_c3_1_4_prev2
   - 120 dist_diff_abs_pair_c3_1_4_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_4_prev2_force_negative:
   dist_diff_pair_c3_1_4_prev2
   - 120 dist_diff_abs_pair_c3_1_4_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_1_4_prev2: 120 same_dist_pair_c3_1_4_prev2
   + dist_diff_abs_pair_c3_1_4_prev2 <= 129.8
 same_dist_pair_lower_c3_1_4_prev2: 120 same_dist_pair_c3_1_4_prev2
   + dist_diff_abs_pair_c3_1_4_prev2 >= 10
 dist_diff_pair_def_c3_1_4_prev3: - dist_curr_pair_c3_1_4
   + dist_diff_pair_c3_1_4_prev3 = -21
 dist_diff_abs_pair_c3_1_4_prev3_abs_ge_orig: - dist_diff_pair_c3_1_4_prev3
   + dist_diff_abs_pair_c3_1_4_prev3
   - 120 dist_diff_abs_pair_c3_1_4_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_4_prev3_abs_le_orig: - dist_diff_pair_c3_1_4_prev3
   + dist_diff_abs_pair_c3_1_4_prev3
   + 120 dist_diff_abs_pair_c3_1_4_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_1_4_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_1_4_prev3 + dist_diff_abs_pair_c3_1_4_prev3
   + 120 dist_diff_abs_pair_c3_1_4_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_1_4_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_1_4_prev3 + dist_diff_abs_pair_c3_1_4_prev3
   - 120 dist_diff_abs_pair_c3_1_4_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_1_4_prev3_force_positive:
   dist_diff_pair_c3_1_4_prev3
   - 120 dist_diff_abs_pair_c3_1_4_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_4_prev3_force_negative:
   dist_diff_pair_c3_1_4_prev3
   - 120 dist_diff_abs_pair_c3_1_4_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_1_4_prev3: 120 same_dist_pair_c3_1_4_prev3
   + dist_diff_abs_pair_c3_1_4_prev3 <= 129.8
 same_dist_pair_lower_c3_1_4_prev3: 120 same_dist_pair_c3_1_4_prev3
   + dist_diff_abs_pair_c3_1_4_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_1_4_prev0: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_1_4_prev0: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_1_4_prev1: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_1_4_prev1: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_1_4_prev2: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_1_4_prev2: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_1_4_prev3: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_1_4_prev3: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_1_4: diff_dist_pair_c3_1_4
   + same_dist_pair_c3_1_4_prev0 + same_dist_pair_c3_1_4_prev1
   + same_dist_pair_c3_1_4_prev2 + same_dist_pair_c3_1_4_prev3 >= 1
 dx_grid_diff_def_c3_1_5: - x_grid_1 + x_grid_5 + dx_grid_diff_c3_1_5 = 0
 dx_grid_abs_pair_c3_1_5_abs_ge_orig: dx_grid_abs_pair_c3_1_5
   - dx_grid_diff_c3_1_5 - 60 dx_grid_abs_pair_c3_1_5_is_positive >= -60
 dx_grid_abs_pair_c3_1_5_abs_le_orig: dx_grid_abs_pair_c3_1_5
   - dx_grid_diff_c3_1_5 + 60 dx_grid_abs_pair_c3_1_5_is_positive <= 60
 dx_grid_abs_pair_c3_1_5_abs_ge_neg_orig: dx_grid_abs_pair_c3_1_5
   + dx_grid_diff_c3_1_5 + 60 dx_grid_abs_pair_c3_1_5_is_positive >= 0
 dx_grid_abs_pair_c3_1_5_abs_le_neg_orig: dx_grid_abs_pair_c3_1_5
   + dx_grid_diff_c3_1_5 - 60 dx_grid_abs_pair_c3_1_5_is_positive <= 0
 dx_grid_abs_pair_c3_1_5_force_positive: dx_grid_diff_c3_1_5
   - 60 dx_grid_abs_pair_c3_1_5_is_positive >= -60
 dx_grid_abs_pair_c3_1_5_force_negative: dx_grid_diff_c3_1_5
   - 60 dx_grid_abs_pair_c3_1_5_is_positive <= -0.001
 dy_grid_diff_def_c3_1_5: - y_grid_1 + y_grid_5 + dy_grid_diff_c3_1_5 = 0
 dy_grid_abs_pair_c3_1_5_abs_ge_orig: dy_grid_abs_pair_c3_1_5
   - dy_grid_diff_c3_1_5 - 60 dy_grid_abs_pair_c3_1_5_is_positive >= -60
 dy_grid_abs_pair_c3_1_5_abs_le_orig: dy_grid_abs_pair_c3_1_5
   - dy_grid_diff_c3_1_5 + 60 dy_grid_abs_pair_c3_1_5_is_positive <= 60
 dy_grid_abs_pair_c3_1_5_abs_ge_neg_orig: dy_grid_abs_pair_c3_1_5
   + dy_grid_diff_c3_1_5 + 60 dy_grid_abs_pair_c3_1_5_is_positive >= 0
 dy_grid_abs_pair_c3_1_5_abs_le_neg_orig: dy_grid_abs_pair_c3_1_5
   + dy_grid_diff_c3_1_5 - 60 dy_grid_abs_pair_c3_1_5_is_positive <= 0
 dy_grid_abs_pair_c3_1_5_force_positive: dy_grid_diff_c3_1_5
   - 60 dy_grid_abs_pair_c3_1_5_is_positive >= -60
 dy_grid_abs_pair_c3_1_5_force_negative: dy_grid_diff_c3_1_5
   - 60 dy_grid_abs_pair_c3_1_5_is_positive <= -0.001
 dist_curr_pair_def_c3_1_5: - dx_grid_abs_pair_c3_1_5
   - dy_grid_abs_pair_c3_1_5 + dist_curr_pair_c3_1_5 = 0
 dist_diff_pair_def_c3_1_5_prev0: - dist_curr_pair_c3_1_5
   + dist_diff_pair_c3_1_5_prev0 = -36
 dist_diff_abs_pair_c3_1_5_prev0_abs_ge_orig: - dist_diff_pair_c3_1_5_prev0
   + dist_diff_abs_pair_c3_1_5_prev0
   - 120 dist_diff_abs_pair_c3_1_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_5_prev0_abs_le_orig: - dist_diff_pair_c3_1_5_prev0
   + dist_diff_abs_pair_c3_1_5_prev0
   + 120 dist_diff_abs_pair_c3_1_5_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_1_5_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_1_5_prev0 + dist_diff_abs_pair_c3_1_5_prev0
   + 120 dist_diff_abs_pair_c3_1_5_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_1_5_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_1_5_prev0 + dist_diff_abs_pair_c3_1_5_prev0
   - 120 dist_diff_abs_pair_c3_1_5_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_1_5_prev0_force_positive:
   dist_diff_pair_c3_1_5_prev0
   - 120 dist_diff_abs_pair_c3_1_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_5_prev0_force_negative:
   dist_diff_pair_c3_1_5_prev0
   - 120 dist_diff_abs_pair_c3_1_5_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_1_5_prev0: 120 same_dist_pair_c3_1_5_prev0
   + dist_diff_abs_pair_c3_1_5_prev0 <= 129.8
 same_dist_pair_lower_c3_1_5_prev0: 120 same_dist_pair_c3_1_5_prev0
   + dist_diff_abs_pair_c3_1_5_prev0 >= 10
 dist_diff_pair_def_c3_1_5_prev1: - dist_curr_pair_c3_1_5
   + dist_diff_pair_c3_1_5_prev1 = -10
 dist_diff_abs_pair_c3_1_5_prev1_abs_ge_orig: - dist_diff_pair_c3_1_5_prev1
   + dist_diff_abs_pair_c3_1_5_prev1
   - 120 dist_diff_abs_pair_c3_1_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_5_prev1_abs_le_orig: - dist_diff_pair_c3_1_5_prev1
   + dist_diff_abs_pair_c3_1_5_prev1
   + 120 dist_diff_abs_pair_c3_1_5_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_1_5_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_1_5_prev1 + dist_diff_abs_pair_c3_1_5_prev1
   + 120 dist_diff_abs_pair_c3_1_5_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_1_5_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_1_5_prev1 + dist_diff_abs_pair_c3_1_5_prev1
   - 120 dist_diff_abs_pair_c3_1_5_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_1_5_prev1_force_positive:
   dist_diff_pair_c3_1_5_prev1
   - 120 dist_diff_abs_pair_c3_1_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_5_prev1_force_negative:
   dist_diff_pair_c3_1_5_prev1
   - 120 dist_diff_abs_pair_c3_1_5_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_1_5_prev1: 120 same_dist_pair_c3_1_5_prev1
   + dist_diff_abs_pair_c3_1_5_prev1 <= 129.8
 same_dist_pair_lower_c3_1_5_prev1: 120 same_dist_pair_c3_1_5_prev1
   + dist_diff_abs_pair_c3_1_5_prev1 >= 10
 dist_diff_pair_def_c3_1_5_prev2: - dist_curr_pair_c3_1_5
   + dist_diff_pair_c3_1_5_prev2 = -21
 dist_diff_abs_pair_c3_1_5_prev2_abs_ge_orig: - dist_diff_pair_c3_1_5_prev2
   + dist_diff_abs_pair_c3_1_5_prev2
   - 120 dist_diff_abs_pair_c3_1_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_5_prev2_abs_le_orig: - dist_diff_pair_c3_1_5_prev2
   + dist_diff_abs_pair_c3_1_5_prev2
   + 120 dist_diff_abs_pair_c3_1_5_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_1_5_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_1_5_prev2 + dist_diff_abs_pair_c3_1_5_prev2
   + 120 dist_diff_abs_pair_c3_1_5_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_1_5_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_1_5_prev2 + dist_diff_abs_pair_c3_1_5_prev2
   - 120 dist_diff_abs_pair_c3_1_5_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_1_5_prev2_force_positive:
   dist_diff_pair_c3_1_5_prev2
   - 120 dist_diff_abs_pair_c3_1_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_5_prev2_force_negative:
   dist_diff_pair_c3_1_5_prev2
   - 120 dist_diff_abs_pair_c3_1_5_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_1_5_prev2: 120 same_dist_pair_c3_1_5_prev2
   + dist_diff_abs_pair_c3_1_5_prev2 <= 129.8
 same_dist_pair_lower_c3_1_5_prev2: 120 same_dist_pair_c3_1_5_prev2
   + dist_diff_abs_pair_c3_1_5_prev2 >= 10
 dist_diff_pair_def_c3_1_5_prev3: - dist_curr_pair_c3_1_5
   + dist_diff_pair_c3_1_5_prev3 = -10
 dist_diff_abs_pair_c3_1_5_prev3_abs_ge_orig: - dist_diff_pair_c3_1_5_prev3
   + dist_diff_abs_pair_c3_1_5_prev3
   - 120 dist_diff_abs_pair_c3_1_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_5_prev3_abs_le_orig: - dist_diff_pair_c3_1_5_prev3
   + dist_diff_abs_pair_c3_1_5_prev3
   + 120 dist_diff_abs_pair_c3_1_5_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_1_5_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_1_5_prev3 + dist_diff_abs_pair_c3_1_5_prev3
   + 120 dist_diff_abs_pair_c3_1_5_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_1_5_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_1_5_prev3 + dist_diff_abs_pair_c3_1_5_prev3
   - 120 dist_diff_abs_pair_c3_1_5_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_1_5_prev3_force_positive:
   dist_diff_pair_c3_1_5_prev3
   - 120 dist_diff_abs_pair_c3_1_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_5_prev3_force_negative:
   dist_diff_pair_c3_1_5_prev3
   - 120 dist_diff_abs_pair_c3_1_5_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_1_5_prev3: 120 same_dist_pair_c3_1_5_prev3
   + dist_diff_abs_pair_c3_1_5_prev3 <= 129.8
 same_dist_pair_lower_c3_1_5_prev3: 120 same_dist_pair_c3_1_5_prev3
   + dist_diff_abs_pair_c3_1_5_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_1_5_prev0: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_1_5_prev0: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_1_5_prev1: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_1_5_prev1: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_1_5_prev2: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_1_5_prev2: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_1_5_prev3: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_1_5_prev3: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_1_5: diff_dist_pair_c3_1_5
   + same_dist_pair_c3_1_5_prev0 + same_dist_pair_c3_1_5_prev1
   + same_dist_pair_c3_1_5_prev2 + same_dist_pair_c3_1_5_prev3 >= 1
 dx_grid_diff_def_c3_1_6: - x_grid_1 + x_grid_6 + dx_grid_diff_c3_1_6 = 0
 dx_grid_abs_pair_c3_1_6_abs_ge_orig: dx_grid_abs_pair_c3_1_6
   - dx_grid_diff_c3_1_6 - 60 dx_grid_abs_pair_c3_1_6_is_positive >= -60
 dx_grid_abs_pair_c3_1_6_abs_le_orig: dx_grid_abs_pair_c3_1_6
   - dx_grid_diff_c3_1_6 + 60 dx_grid_abs_pair_c3_1_6_is_positive <= 60
 dx_grid_abs_pair_c3_1_6_abs_ge_neg_orig: dx_grid_abs_pair_c3_1_6
   + dx_grid_diff_c3_1_6 + 60 dx_grid_abs_pair_c3_1_6_is_positive >= 0
 dx_grid_abs_pair_c3_1_6_abs_le_neg_orig: dx_grid_abs_pair_c3_1_6
   + dx_grid_diff_c3_1_6 - 60 dx_grid_abs_pair_c3_1_6_is_positive <= 0
 dx_grid_abs_pair_c3_1_6_force_positive: dx_grid_diff_c3_1_6
   - 60 dx_grid_abs_pair_c3_1_6_is_positive >= -60
 dx_grid_abs_pair_c3_1_6_force_negative: dx_grid_diff_c3_1_6
   - 60 dx_grid_abs_pair_c3_1_6_is_positive <= -0.001
 dy_grid_diff_def_c3_1_6: - y_grid_1 + y_grid_6 + dy_grid_diff_c3_1_6 = 0
 dy_grid_abs_pair_c3_1_6_abs_ge_orig: dy_grid_abs_pair_c3_1_6
   - dy_grid_diff_c3_1_6 - 60 dy_grid_abs_pair_c3_1_6_is_positive >= -60
 dy_grid_abs_pair_c3_1_6_abs_le_orig: dy_grid_abs_pair_c3_1_6
   - dy_grid_diff_c3_1_6 + 60 dy_grid_abs_pair_c3_1_6_is_positive <= 60
 dy_grid_abs_pair_c3_1_6_abs_ge_neg_orig: dy_grid_abs_pair_c3_1_6
   + dy_grid_diff_c3_1_6 + 60 dy_grid_abs_pair_c3_1_6_is_positive >= 0
 dy_grid_abs_pair_c3_1_6_abs_le_neg_orig: dy_grid_abs_pair_c3_1_6
   + dy_grid_diff_c3_1_6 - 60 dy_grid_abs_pair_c3_1_6_is_positive <= 0
 dy_grid_abs_pair_c3_1_6_force_positive: dy_grid_diff_c3_1_6
   - 60 dy_grid_abs_pair_c3_1_6_is_positive >= -60
 dy_grid_abs_pair_c3_1_6_force_negative: dy_grid_diff_c3_1_6
   - 60 dy_grid_abs_pair_c3_1_6_is_positive <= -0.001
 dist_curr_pair_def_c3_1_6: - dx_grid_abs_pair_c3_1_6
   - dy_grid_abs_pair_c3_1_6 + dist_curr_pair_c3_1_6 = 0
 dist_diff_pair_def_c3_1_6_prev0: - dist_curr_pair_c3_1_6
   + dist_diff_pair_c3_1_6_prev0 = -52
 dist_diff_abs_pair_c3_1_6_prev0_abs_ge_orig: - dist_diff_pair_c3_1_6_prev0
   + dist_diff_abs_pair_c3_1_6_prev0
   - 120 dist_diff_abs_pair_c3_1_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_6_prev0_abs_le_orig: - dist_diff_pair_c3_1_6_prev0
   + dist_diff_abs_pair_c3_1_6_prev0
   + 120 dist_diff_abs_pair_c3_1_6_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_1_6_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_1_6_prev0 + dist_diff_abs_pair_c3_1_6_prev0
   + 120 dist_diff_abs_pair_c3_1_6_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_1_6_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_1_6_prev0 + dist_diff_abs_pair_c3_1_6_prev0
   - 120 dist_diff_abs_pair_c3_1_6_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_1_6_prev0_force_positive:
   dist_diff_pair_c3_1_6_prev0
   - 120 dist_diff_abs_pair_c3_1_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_6_prev0_force_negative:
   dist_diff_pair_c3_1_6_prev0
   - 120 dist_diff_abs_pair_c3_1_6_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_1_6_prev0: 120 same_dist_pair_c3_1_6_prev0
   + dist_diff_abs_pair_c3_1_6_prev0 <= 129.8
 same_dist_pair_lower_c3_1_6_prev0: 120 same_dist_pair_c3_1_6_prev0
   + dist_diff_abs_pair_c3_1_6_prev0 >= 10
 dist_diff_pair_def_c3_1_6_prev1: - dist_curr_pair_c3_1_6
   + dist_diff_pair_c3_1_6_prev1 = -40
 dist_diff_abs_pair_c3_1_6_prev1_abs_ge_orig: - dist_diff_pair_c3_1_6_prev1
   + dist_diff_abs_pair_c3_1_6_prev1
   - 120 dist_diff_abs_pair_c3_1_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_6_prev1_abs_le_orig: - dist_diff_pair_c3_1_6_prev1
   + dist_diff_abs_pair_c3_1_6_prev1
   + 120 dist_diff_abs_pair_c3_1_6_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_1_6_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_1_6_prev1 + dist_diff_abs_pair_c3_1_6_prev1
   + 120 dist_diff_abs_pair_c3_1_6_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_1_6_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_1_6_prev1 + dist_diff_abs_pair_c3_1_6_prev1
   - 120 dist_diff_abs_pair_c3_1_6_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_1_6_prev1_force_positive:
   dist_diff_pair_c3_1_6_prev1
   - 120 dist_diff_abs_pair_c3_1_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_6_prev1_force_negative:
   dist_diff_pair_c3_1_6_prev1
   - 120 dist_diff_abs_pair_c3_1_6_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_1_6_prev1: 120 same_dist_pair_c3_1_6_prev1
   + dist_diff_abs_pair_c3_1_6_prev1 <= 129.8
 same_dist_pair_lower_c3_1_6_prev1: 120 same_dist_pair_c3_1_6_prev1
   + dist_diff_abs_pair_c3_1_6_prev1 >= 10
 dist_diff_pair_def_c3_1_6_prev2: - dist_curr_pair_c3_1_6
   + dist_diff_pair_c3_1_6_prev2 = -46
 dist_diff_abs_pair_c3_1_6_prev2_abs_ge_orig: - dist_diff_pair_c3_1_6_prev2
   + dist_diff_abs_pair_c3_1_6_prev2
   - 120 dist_diff_abs_pair_c3_1_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_6_prev2_abs_le_orig: - dist_diff_pair_c3_1_6_prev2
   + dist_diff_abs_pair_c3_1_6_prev2
   + 120 dist_diff_abs_pair_c3_1_6_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_1_6_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_1_6_prev2 + dist_diff_abs_pair_c3_1_6_prev2
   + 120 dist_diff_abs_pair_c3_1_6_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_1_6_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_1_6_prev2 + dist_diff_abs_pair_c3_1_6_prev2
   - 120 dist_diff_abs_pair_c3_1_6_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_1_6_prev2_force_positive:
   dist_diff_pair_c3_1_6_prev2
   - 120 dist_diff_abs_pair_c3_1_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_6_prev2_force_negative:
   dist_diff_pair_c3_1_6_prev2
   - 120 dist_diff_abs_pair_c3_1_6_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_1_6_prev2: 120 same_dist_pair_c3_1_6_prev2
   + dist_diff_abs_pair_c3_1_6_prev2 <= 129.8
 same_dist_pair_lower_c3_1_6_prev2: 120 same_dist_pair_c3_1_6_prev2
   + dist_diff_abs_pair_c3_1_6_prev2 >= 10
 dist_diff_pair_def_c3_1_6_prev3: - dist_curr_pair_c3_1_6
   + dist_diff_pair_c3_1_6_prev3 = -40
 dist_diff_abs_pair_c3_1_6_prev3_abs_ge_orig: - dist_diff_pair_c3_1_6_prev3
   + dist_diff_abs_pair_c3_1_6_prev3
   - 120 dist_diff_abs_pair_c3_1_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_6_prev3_abs_le_orig: - dist_diff_pair_c3_1_6_prev3
   + dist_diff_abs_pair_c3_1_6_prev3
   + 120 dist_diff_abs_pair_c3_1_6_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_1_6_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_1_6_prev3 + dist_diff_abs_pair_c3_1_6_prev3
   + 120 dist_diff_abs_pair_c3_1_6_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_1_6_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_1_6_prev3 + dist_diff_abs_pair_c3_1_6_prev3
   - 120 dist_diff_abs_pair_c3_1_6_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_1_6_prev3_force_positive:
   dist_diff_pair_c3_1_6_prev3
   - 120 dist_diff_abs_pair_c3_1_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_6_prev3_force_negative:
   dist_diff_pair_c3_1_6_prev3
   - 120 dist_diff_abs_pair_c3_1_6_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_1_6_prev3: 120 same_dist_pair_c3_1_6_prev3
   + dist_diff_abs_pair_c3_1_6_prev3 <= 129.8
 same_dist_pair_lower_c3_1_6_prev3: 120 same_dist_pair_c3_1_6_prev3
   + dist_diff_abs_pair_c3_1_6_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_1_6_prev0: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_1_6_prev0: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_1_6_prev1: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_1_6_prev1: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_1_6_prev2: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_1_6_prev2: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_1_6_prev3: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_1_6_prev3: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_1_6: diff_dist_pair_c3_1_6
   + same_dist_pair_c3_1_6_prev0 + same_dist_pair_c3_1_6_prev1
   + same_dist_pair_c3_1_6_prev2 + same_dist_pair_c3_1_6_prev3 >= 1
 dx_grid_diff_def_c3_1_7: - x_grid_1 + x_grid_7 + dx_grid_diff_c3_1_7 = 0
 dx_grid_abs_pair_c3_1_7_abs_ge_orig: dx_grid_abs_pair_c3_1_7
   - dx_grid_diff_c3_1_7 - 60 dx_grid_abs_pair_c3_1_7_is_positive >= -60
 dx_grid_abs_pair_c3_1_7_abs_le_orig: dx_grid_abs_pair_c3_1_7
   - dx_grid_diff_c3_1_7 + 60 dx_grid_abs_pair_c3_1_7_is_positive <= 60
 dx_grid_abs_pair_c3_1_7_abs_ge_neg_orig: dx_grid_abs_pair_c3_1_7
   + dx_grid_diff_c3_1_7 + 60 dx_grid_abs_pair_c3_1_7_is_positive >= 0
 dx_grid_abs_pair_c3_1_7_abs_le_neg_orig: dx_grid_abs_pair_c3_1_7
   + dx_grid_diff_c3_1_7 - 60 dx_grid_abs_pair_c3_1_7_is_positive <= 0
 dx_grid_abs_pair_c3_1_7_force_positive: dx_grid_diff_c3_1_7
   - 60 dx_grid_abs_pair_c3_1_7_is_positive >= -60
 dx_grid_abs_pair_c3_1_7_force_negative: dx_grid_diff_c3_1_7
   - 60 dx_grid_abs_pair_c3_1_7_is_positive <= -0.001
 dy_grid_diff_def_c3_1_7: - y_grid_1 + y_grid_7 + dy_grid_diff_c3_1_7 = 0
 dy_grid_abs_pair_c3_1_7_abs_ge_orig: dy_grid_abs_pair_c3_1_7
   - dy_grid_diff_c3_1_7 - 60 dy_grid_abs_pair_c3_1_7_is_positive >= -60
 dy_grid_abs_pair_c3_1_7_abs_le_orig: dy_grid_abs_pair_c3_1_7
   - dy_grid_diff_c3_1_7 + 60 dy_grid_abs_pair_c3_1_7_is_positive <= 60
 dy_grid_abs_pair_c3_1_7_abs_ge_neg_orig: dy_grid_abs_pair_c3_1_7
   + dy_grid_diff_c3_1_7 + 60 dy_grid_abs_pair_c3_1_7_is_positive >= 0
 dy_grid_abs_pair_c3_1_7_abs_le_neg_orig: dy_grid_abs_pair_c3_1_7
   + dy_grid_diff_c3_1_7 - 60 dy_grid_abs_pair_c3_1_7_is_positive <= 0
 dy_grid_abs_pair_c3_1_7_force_positive: dy_grid_diff_c3_1_7
   - 60 dy_grid_abs_pair_c3_1_7_is_positive >= -60
 dy_grid_abs_pair_c3_1_7_force_negative: dy_grid_diff_c3_1_7
   - 60 dy_grid_abs_pair_c3_1_7_is_positive <= -0.001
 dist_curr_pair_def_c3_1_7: - dx_grid_abs_pair_c3_1_7
   - dy_grid_abs_pair_c3_1_7 + dist_curr_pair_c3_1_7 = 0
 dist_diff_pair_def_c3_1_7_prev0: - dist_curr_pair_c3_1_7
   + dist_diff_pair_c3_1_7_prev0 = -54
 dist_diff_abs_pair_c3_1_7_prev0_abs_ge_orig: - dist_diff_pair_c3_1_7_prev0
   + dist_diff_abs_pair_c3_1_7_prev0
   - 120 dist_diff_abs_pair_c3_1_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_7_prev0_abs_le_orig: - dist_diff_pair_c3_1_7_prev0
   + dist_diff_abs_pair_c3_1_7_prev0
   + 120 dist_diff_abs_pair_c3_1_7_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_1_7_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_1_7_prev0 + dist_diff_abs_pair_c3_1_7_prev0
   + 120 dist_diff_abs_pair_c3_1_7_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_1_7_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_1_7_prev0 + dist_diff_abs_pair_c3_1_7_prev0
   - 120 dist_diff_abs_pair_c3_1_7_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_1_7_prev0_force_positive:
   dist_diff_pair_c3_1_7_prev0
   - 120 dist_diff_abs_pair_c3_1_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_1_7_prev0_force_negative:
   dist_diff_pair_c3_1_7_prev0
   - 120 dist_diff_abs_pair_c3_1_7_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_1_7_prev0: 120 same_dist_pair_c3_1_7_prev0
   + dist_diff_abs_pair_c3_1_7_prev0 <= 129.8
 same_dist_pair_lower_c3_1_7_prev0: 120 same_dist_pair_c3_1_7_prev0
   + dist_diff_abs_pair_c3_1_7_prev0 >= 10
 dist_diff_pair_def_c3_1_7_prev1: - dist_curr_pair_c3_1_7
   + dist_diff_pair_c3_1_7_prev1 = -32
 dist_diff_abs_pair_c3_1_7_prev1_abs_ge_orig: - dist_diff_pair_c3_1_7_prev1
   + dist_diff_abs_pair_c3_1_7_prev1
   - 120 dist_diff_abs_pair_c3_1_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_7_prev1_abs_le_orig: - dist_diff_pair_c3_1_7_prev1
   + dist_diff_abs_pair_c3_1_7_prev1
   + 120 dist_diff_abs_pair_c3_1_7_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_1_7_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_1_7_prev1 + dist_diff_abs_pair_c3_1_7_prev1
   + 120 dist_diff_abs_pair_c3_1_7_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_1_7_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_1_7_prev1 + dist_diff_abs_pair_c3_1_7_prev1
   - 120 dist_diff_abs_pair_c3_1_7_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_1_7_prev1_force_positive:
   dist_diff_pair_c3_1_7_prev1
   - 120 dist_diff_abs_pair_c3_1_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_1_7_prev1_force_negative:
   dist_diff_pair_c3_1_7_prev1
   - 120 dist_diff_abs_pair_c3_1_7_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_1_7_prev1: 120 same_dist_pair_c3_1_7_prev1
   + dist_diff_abs_pair_c3_1_7_prev1 <= 129.8
 same_dist_pair_lower_c3_1_7_prev1: 120 same_dist_pair_c3_1_7_prev1
   + dist_diff_abs_pair_c3_1_7_prev1 >= 10
 dist_diff_pair_def_c3_1_7_prev2: - dist_curr_pair_c3_1_7
   + dist_diff_pair_c3_1_7_prev2 = -30
 dist_diff_abs_pair_c3_1_7_prev2_abs_ge_orig: - dist_diff_pair_c3_1_7_prev2
   + dist_diff_abs_pair_c3_1_7_prev2
   - 120 dist_diff_abs_pair_c3_1_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_7_prev2_abs_le_orig: - dist_diff_pair_c3_1_7_prev2
   + dist_diff_abs_pair_c3_1_7_prev2
   + 120 dist_diff_abs_pair_c3_1_7_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_1_7_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_1_7_prev2 + dist_diff_abs_pair_c3_1_7_prev2
   + 120 dist_diff_abs_pair_c3_1_7_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_1_7_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_1_7_prev2 + dist_diff_abs_pair_c3_1_7_prev2
   - 120 dist_diff_abs_pair_c3_1_7_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_1_7_prev2_force_positive:
   dist_diff_pair_c3_1_7_prev2
   - 120 dist_diff_abs_pair_c3_1_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_1_7_prev2_force_negative:
   dist_diff_pair_c3_1_7_prev2
   - 120 dist_diff_abs_pair_c3_1_7_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_1_7_prev2: 120 same_dist_pair_c3_1_7_prev2
   + dist_diff_abs_pair_c3_1_7_prev2 <= 129.8
 same_dist_pair_lower_c3_1_7_prev2: 120 same_dist_pair_c3_1_7_prev2
   + dist_diff_abs_pair_c3_1_7_prev2 >= 10
 dist_diff_pair_def_c3_1_7_prev3: - dist_curr_pair_c3_1_7
   + dist_diff_pair_c3_1_7_prev3 = -32
 dist_diff_abs_pair_c3_1_7_prev3_abs_ge_orig: - dist_diff_pair_c3_1_7_prev3
   + dist_diff_abs_pair_c3_1_7_prev3
   - 120 dist_diff_abs_pair_c3_1_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_7_prev3_abs_le_orig: - dist_diff_pair_c3_1_7_prev3
   + dist_diff_abs_pair_c3_1_7_prev3
   + 120 dist_diff_abs_pair_c3_1_7_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_1_7_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_1_7_prev3 + dist_diff_abs_pair_c3_1_7_prev3
   + 120 dist_diff_abs_pair_c3_1_7_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_1_7_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_1_7_prev3 + dist_diff_abs_pair_c3_1_7_prev3
   - 120 dist_diff_abs_pair_c3_1_7_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_1_7_prev3_force_positive:
   dist_diff_pair_c3_1_7_prev3
   - 120 dist_diff_abs_pair_c3_1_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_1_7_prev3_force_negative:
   dist_diff_pair_c3_1_7_prev3
   - 120 dist_diff_abs_pair_c3_1_7_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_1_7_prev3: 120 same_dist_pair_c3_1_7_prev3
   + dist_diff_abs_pair_c3_1_7_prev3 <= 129.8
 same_dist_pair_lower_c3_1_7_prev3: 120 same_dist_pair_c3_1_7_prev3
   + dist_diff_abs_pair_c3_1_7_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_1_7_prev0: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_1_7_prev0: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_1_7_prev1: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_1_7_prev1: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_1_7_prev2: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_1_7_prev2: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_1_7_prev3: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_1_7_prev3: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_1_7: diff_dist_pair_c3_1_7
   + same_dist_pair_c3_1_7_prev0 + same_dist_pair_c3_1_7_prev1
   + same_dist_pair_c3_1_7_prev2 + same_dist_pair_c3_1_7_prev3 >= 1
 dx_grid_diff_def_c3_2_3: - x_grid_2 + x_grid_3 + dx_grid_diff_c3_2_3 = 0
 dx_grid_abs_pair_c3_2_3_abs_ge_orig: dx_grid_abs_pair_c3_2_3
   - dx_grid_diff_c3_2_3 - 60 dx_grid_abs_pair_c3_2_3_is_positive >= -60
 dx_grid_abs_pair_c3_2_3_abs_le_orig: dx_grid_abs_pair_c3_2_3
   - dx_grid_diff_c3_2_3 + 60 dx_grid_abs_pair_c3_2_3_is_positive <= 60
 dx_grid_abs_pair_c3_2_3_abs_ge_neg_orig: dx_grid_abs_pair_c3_2_3
   + dx_grid_diff_c3_2_3 + 60 dx_grid_abs_pair_c3_2_3_is_positive >= 0
 dx_grid_abs_pair_c3_2_3_abs_le_neg_orig: dx_grid_abs_pair_c3_2_3
   + dx_grid_diff_c3_2_3 - 60 dx_grid_abs_pair_c3_2_3_is_positive <= 0
 dx_grid_abs_pair_c3_2_3_force_positive: dx_grid_diff_c3_2_3
   - 60 dx_grid_abs_pair_c3_2_3_is_positive >= -60
 dx_grid_abs_pair_c3_2_3_force_negative: dx_grid_diff_c3_2_3
   - 60 dx_grid_abs_pair_c3_2_3_is_positive <= -0.001
 dy_grid_diff_def_c3_2_3: - y_grid_2 + y_grid_3 + dy_grid_diff_c3_2_3 = 0
 dy_grid_abs_pair_c3_2_3_abs_ge_orig: dy_grid_abs_pair_c3_2_3
   - dy_grid_diff_c3_2_3 - 60 dy_grid_abs_pair_c3_2_3_is_positive >= -60
 dy_grid_abs_pair_c3_2_3_abs_le_orig: dy_grid_abs_pair_c3_2_3
   - dy_grid_diff_c3_2_3 + 60 dy_grid_abs_pair_c3_2_3_is_positive <= 60
 dy_grid_abs_pair_c3_2_3_abs_ge_neg_orig: dy_grid_abs_pair_c3_2_3
   + dy_grid_diff_c3_2_3 + 60 dy_grid_abs_pair_c3_2_3_is_positive >= 0
 dy_grid_abs_pair_c3_2_3_abs_le_neg_orig: dy_grid_abs_pair_c3_2_3
   + dy_grid_diff_c3_2_3 - 60 dy_grid_abs_pair_c3_2_3_is_positive <= 0
 dy_grid_abs_pair_c3_2_3_force_positive: dy_grid_diff_c3_2_3
   - 60 dy_grid_abs_pair_c3_2_3_is_positive >= -60
 dy_grid_abs_pair_c3_2_3_force_negative: dy_grid_diff_c3_2_3
   - 60 dy_grid_abs_pair_c3_2_3_is_positive <= -0.001
 dist_curr_pair_def_c3_2_3: - dx_grid_abs_pair_c3_2_3
   - dy_grid_abs_pair_c3_2_3 + dist_curr_pair_c3_2_3 = 0
 dist_diff_pair_def_c3_2_3_prev0: - dist_curr_pair_c3_2_3
   + dist_diff_pair_c3_2_3_prev0 = -34
 dist_diff_abs_pair_c3_2_3_prev0_abs_ge_orig: - dist_diff_pair_c3_2_3_prev0
   + dist_diff_abs_pair_c3_2_3_prev0
   - 120 dist_diff_abs_pair_c3_2_3_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_3_prev0_abs_le_orig: - dist_diff_pair_c3_2_3_prev0
   + dist_diff_abs_pair_c3_2_3_prev0
   + 120 dist_diff_abs_pair_c3_2_3_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_2_3_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_2_3_prev0 + dist_diff_abs_pair_c3_2_3_prev0
   + 120 dist_diff_abs_pair_c3_2_3_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_2_3_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_2_3_prev0 + dist_diff_abs_pair_c3_2_3_prev0
   - 120 dist_diff_abs_pair_c3_2_3_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_2_3_prev0_force_positive:
   dist_diff_pair_c3_2_3_prev0
   - 120 dist_diff_abs_pair_c3_2_3_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_3_prev0_force_negative:
   dist_diff_pair_c3_2_3_prev0
   - 120 dist_diff_abs_pair_c3_2_3_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_2_3_prev0: 120 same_dist_pair_c3_2_3_prev0
   + dist_diff_abs_pair_c3_2_3_prev0 <= 129.8
 same_dist_pair_lower_c3_2_3_prev0: 120 same_dist_pair_c3_2_3_prev0
   + dist_diff_abs_pair_c3_2_3_prev0 >= 10
 dist_diff_pair_def_c3_2_3_prev1: - dist_curr_pair_c3_2_3
   + dist_diff_pair_c3_2_3_prev1 = -20
 dist_diff_abs_pair_c3_2_3_prev1_abs_ge_orig: - dist_diff_pair_c3_2_3_prev1
   + dist_diff_abs_pair_c3_2_3_prev1
   - 120 dist_diff_abs_pair_c3_2_3_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_3_prev1_abs_le_orig: - dist_diff_pair_c3_2_3_prev1
   + dist_diff_abs_pair_c3_2_3_prev1
   + 120 dist_diff_abs_pair_c3_2_3_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_2_3_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_2_3_prev1 + dist_diff_abs_pair_c3_2_3_prev1
   + 120 dist_diff_abs_pair_c3_2_3_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_2_3_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_2_3_prev1 + dist_diff_abs_pair_c3_2_3_prev1
   - 120 dist_diff_abs_pair_c3_2_3_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_2_3_prev1_force_positive:
   dist_diff_pair_c3_2_3_prev1
   - 120 dist_diff_abs_pair_c3_2_3_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_3_prev1_force_negative:
   dist_diff_pair_c3_2_3_prev1
   - 120 dist_diff_abs_pair_c3_2_3_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_2_3_prev1: 120 same_dist_pair_c3_2_3_prev1
   + dist_diff_abs_pair_c3_2_3_prev1 <= 129.8
 same_dist_pair_lower_c3_2_3_prev1: 120 same_dist_pair_c3_2_3_prev1
   + dist_diff_abs_pair_c3_2_3_prev1 >= 10
 dist_diff_pair_def_c3_2_3_prev2: - dist_curr_pair_c3_2_3
   + dist_diff_pair_c3_2_3_prev2 = -20
 dist_diff_abs_pair_c3_2_3_prev2_abs_ge_orig: - dist_diff_pair_c3_2_3_prev2
   + dist_diff_abs_pair_c3_2_3_prev2
   - 120 dist_diff_abs_pair_c3_2_3_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_3_prev2_abs_le_orig: - dist_diff_pair_c3_2_3_prev2
   + dist_diff_abs_pair_c3_2_3_prev2
   + 120 dist_diff_abs_pair_c3_2_3_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_2_3_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_2_3_prev2 + dist_diff_abs_pair_c3_2_3_prev2
   + 120 dist_diff_abs_pair_c3_2_3_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_2_3_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_2_3_prev2 + dist_diff_abs_pair_c3_2_3_prev2
   - 120 dist_diff_abs_pair_c3_2_3_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_2_3_prev2_force_positive:
   dist_diff_pair_c3_2_3_prev2
   - 120 dist_diff_abs_pair_c3_2_3_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_3_prev2_force_negative:
   dist_diff_pair_c3_2_3_prev2
   - 120 dist_diff_abs_pair_c3_2_3_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_2_3_prev2: 120 same_dist_pair_c3_2_3_prev2
   + dist_diff_abs_pair_c3_2_3_prev2 <= 129.8
 same_dist_pair_lower_c3_2_3_prev2: 120 same_dist_pair_c3_2_3_prev2
   + dist_diff_abs_pair_c3_2_3_prev2 >= 10
 dist_diff_pair_def_c3_2_3_prev3: - dist_curr_pair_c3_2_3
   + dist_diff_pair_c3_2_3_prev3 = -10
 dist_diff_abs_pair_c3_2_3_prev3_abs_ge_orig: - dist_diff_pair_c3_2_3_prev3
   + dist_diff_abs_pair_c3_2_3_prev3
   - 120 dist_diff_abs_pair_c3_2_3_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_3_prev3_abs_le_orig: - dist_diff_pair_c3_2_3_prev3
   + dist_diff_abs_pair_c3_2_3_prev3
   + 120 dist_diff_abs_pair_c3_2_3_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_2_3_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_2_3_prev3 + dist_diff_abs_pair_c3_2_3_prev3
   + 120 dist_diff_abs_pair_c3_2_3_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_2_3_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_2_3_prev3 + dist_diff_abs_pair_c3_2_3_prev3
   - 120 dist_diff_abs_pair_c3_2_3_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_2_3_prev3_force_positive:
   dist_diff_pair_c3_2_3_prev3
   - 120 dist_diff_abs_pair_c3_2_3_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_3_prev3_force_negative:
   dist_diff_pair_c3_2_3_prev3
   - 120 dist_diff_abs_pair_c3_2_3_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_2_3_prev3: 120 same_dist_pair_c3_2_3_prev3
   + dist_diff_abs_pair_c3_2_3_prev3 <= 129.8
 same_dist_pair_lower_c3_2_3_prev3: 120 same_dist_pair_c3_2_3_prev3
   + dist_diff_abs_pair_c3_2_3_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_2_3_prev0: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_2_3_prev0: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_2_3_prev1: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_2_3_prev1: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_2_3_prev2: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_2_3_prev2: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_2_3_prev3: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_2_3_prev3: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_2_3: diff_dist_pair_c3_2_3
   + same_dist_pair_c3_2_3_prev0 + same_dist_pair_c3_2_3_prev1
   + same_dist_pair_c3_2_3_prev2 + same_dist_pair_c3_2_3_prev3 >= 1
 dx_grid_diff_def_c3_2_4: - x_grid_2 + x_grid_4 + dx_grid_diff_c3_2_4 = 0
 dx_grid_abs_pair_c3_2_4_abs_ge_orig: dx_grid_abs_pair_c3_2_4
   - dx_grid_diff_c3_2_4 - 60 dx_grid_abs_pair_c3_2_4_is_positive >= -60
 dx_grid_abs_pair_c3_2_4_abs_le_orig: dx_grid_abs_pair_c3_2_4
   - dx_grid_diff_c3_2_4 + 60 dx_grid_abs_pair_c3_2_4_is_positive <= 60
 dx_grid_abs_pair_c3_2_4_abs_ge_neg_orig: dx_grid_abs_pair_c3_2_4
   + dx_grid_diff_c3_2_4 + 60 dx_grid_abs_pair_c3_2_4_is_positive >= 0
 dx_grid_abs_pair_c3_2_4_abs_le_neg_orig: dx_grid_abs_pair_c3_2_4
   + dx_grid_diff_c3_2_4 - 60 dx_grid_abs_pair_c3_2_4_is_positive <= 0
 dx_grid_abs_pair_c3_2_4_force_positive: dx_grid_diff_c3_2_4
   - 60 dx_grid_abs_pair_c3_2_4_is_positive >= -60
 dx_grid_abs_pair_c3_2_4_force_negative: dx_grid_diff_c3_2_4
   - 60 dx_grid_abs_pair_c3_2_4_is_positive <= -0.001
 dy_grid_diff_def_c3_2_4: - y_grid_2 + y_grid_4 + dy_grid_diff_c3_2_4 = 0
 dy_grid_abs_pair_c3_2_4_abs_ge_orig: dy_grid_abs_pair_c3_2_4
   - dy_grid_diff_c3_2_4 - 60 dy_grid_abs_pair_c3_2_4_is_positive >= -60
 dy_grid_abs_pair_c3_2_4_abs_le_orig: dy_grid_abs_pair_c3_2_4
   - dy_grid_diff_c3_2_4 + 60 dy_grid_abs_pair_c3_2_4_is_positive <= 60
 dy_grid_abs_pair_c3_2_4_abs_ge_neg_orig: dy_grid_abs_pair_c3_2_4
   + dy_grid_diff_c3_2_4 + 60 dy_grid_abs_pair_c3_2_4_is_positive >= 0
 dy_grid_abs_pair_c3_2_4_abs_le_neg_orig: dy_grid_abs_pair_c3_2_4
   + dy_grid_diff_c3_2_4 - 60 dy_grid_abs_pair_c3_2_4_is_positive <= 0
 dy_grid_abs_pair_c3_2_4_force_positive: dy_grid_diff_c3_2_4
   - 60 dy_grid_abs_pair_c3_2_4_is_positive >= -60
 dy_grid_abs_pair_c3_2_4_force_negative: dy_grid_diff_c3_2_4
   - 60 dy_grid_abs_pair_c3_2_4_is_positive <= -0.001
 dist_curr_pair_def_c3_2_4: - dx_grid_abs_pair_c3_2_4
   - dy_grid_abs_pair_c3_2_4 + dist_curr_pair_c3_2_4 = 0
 dist_diff_pair_def_c3_2_4_prev0: - dist_curr_pair_c3_2_4
   + dist_diff_pair_c3_2_4_prev0 = -30
 dist_diff_abs_pair_c3_2_4_prev0_abs_ge_orig: - dist_diff_pair_c3_2_4_prev0
   + dist_diff_abs_pair_c3_2_4_prev0
   - 120 dist_diff_abs_pair_c3_2_4_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_4_prev0_abs_le_orig: - dist_diff_pair_c3_2_4_prev0
   + dist_diff_abs_pair_c3_2_4_prev0
   + 120 dist_diff_abs_pair_c3_2_4_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_2_4_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_2_4_prev0 + dist_diff_abs_pair_c3_2_4_prev0
   + 120 dist_diff_abs_pair_c3_2_4_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_2_4_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_2_4_prev0 + dist_diff_abs_pair_c3_2_4_prev0
   - 120 dist_diff_abs_pair_c3_2_4_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_2_4_prev0_force_positive:
   dist_diff_pair_c3_2_4_prev0
   - 120 dist_diff_abs_pair_c3_2_4_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_4_prev0_force_negative:
   dist_diff_pair_c3_2_4_prev0
   - 120 dist_diff_abs_pair_c3_2_4_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_2_4_prev0: 120 same_dist_pair_c3_2_4_prev0
   + dist_diff_abs_pair_c3_2_4_prev0 <= 129.8
 same_dist_pair_lower_c3_2_4_prev0: 120 same_dist_pair_c3_2_4_prev0
   + dist_diff_abs_pair_c3_2_4_prev0 >= 10
 dist_diff_pair_def_c3_2_4_prev1: - dist_curr_pair_c3_2_4
   + dist_diff_pair_c3_2_4_prev1 = -10
 dist_diff_abs_pair_c3_2_4_prev1_abs_ge_orig: - dist_diff_pair_c3_2_4_prev1
   + dist_diff_abs_pair_c3_2_4_prev1
   - 120 dist_diff_abs_pair_c3_2_4_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_4_prev1_abs_le_orig: - dist_diff_pair_c3_2_4_prev1
   + dist_diff_abs_pair_c3_2_4_prev1
   + 120 dist_diff_abs_pair_c3_2_4_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_2_4_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_2_4_prev1 + dist_diff_abs_pair_c3_2_4_prev1
   + 120 dist_diff_abs_pair_c3_2_4_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_2_4_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_2_4_prev1 + dist_diff_abs_pair_c3_2_4_prev1
   - 120 dist_diff_abs_pair_c3_2_4_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_2_4_prev1_force_positive:
   dist_diff_pair_c3_2_4_prev1
   - 120 dist_diff_abs_pair_c3_2_4_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_4_prev1_force_negative:
   dist_diff_pair_c3_2_4_prev1
   - 120 dist_diff_abs_pair_c3_2_4_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_2_4_prev1: 120 same_dist_pair_c3_2_4_prev1
   + dist_diff_abs_pair_c3_2_4_prev1 <= 129.8
 same_dist_pair_lower_c3_2_4_prev1: 120 same_dist_pair_c3_2_4_prev1
   + dist_diff_abs_pair_c3_2_4_prev1 >= 10
 dist_diff_pair_def_c3_2_4_prev2: - dist_curr_pair_c3_2_4
   + dist_diff_pair_c3_2_4_prev2 = -21
 dist_diff_abs_pair_c3_2_4_prev2_abs_ge_orig: - dist_diff_pair_c3_2_4_prev2
   + dist_diff_abs_pair_c3_2_4_prev2
   - 120 dist_diff_abs_pair_c3_2_4_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_4_prev2_abs_le_orig: - dist_diff_pair_c3_2_4_prev2
   + dist_diff_abs_pair_c3_2_4_prev2
   + 120 dist_diff_abs_pair_c3_2_4_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_2_4_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_2_4_prev2 + dist_diff_abs_pair_c3_2_4_prev2
   + 120 dist_diff_abs_pair_c3_2_4_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_2_4_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_2_4_prev2 + dist_diff_abs_pair_c3_2_4_prev2
   - 120 dist_diff_abs_pair_c3_2_4_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_2_4_prev2_force_positive:
   dist_diff_pair_c3_2_4_prev2
   - 120 dist_diff_abs_pair_c3_2_4_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_4_prev2_force_negative:
   dist_diff_pair_c3_2_4_prev2
   - 120 dist_diff_abs_pair_c3_2_4_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_2_4_prev2: 120 same_dist_pair_c3_2_4_prev2
   + dist_diff_abs_pair_c3_2_4_prev2 <= 129.8
 same_dist_pair_lower_c3_2_4_prev2: 120 same_dist_pair_c3_2_4_prev2
   + dist_diff_abs_pair_c3_2_4_prev2 >= 10
 dist_diff_pair_def_c3_2_4_prev3: - dist_curr_pair_c3_2_4
   + dist_diff_pair_c3_2_4_prev3 = -20
 dist_diff_abs_pair_c3_2_4_prev3_abs_ge_orig: - dist_diff_pair_c3_2_4_prev3
   + dist_diff_abs_pair_c3_2_4_prev3
   - 120 dist_diff_abs_pair_c3_2_4_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_4_prev3_abs_le_orig: - dist_diff_pair_c3_2_4_prev3
   + dist_diff_abs_pair_c3_2_4_prev3
   + 120 dist_diff_abs_pair_c3_2_4_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_2_4_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_2_4_prev3 + dist_diff_abs_pair_c3_2_4_prev3
   + 120 dist_diff_abs_pair_c3_2_4_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_2_4_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_2_4_prev3 + dist_diff_abs_pair_c3_2_4_prev3
   - 120 dist_diff_abs_pair_c3_2_4_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_2_4_prev3_force_positive:
   dist_diff_pair_c3_2_4_prev3
   - 120 dist_diff_abs_pair_c3_2_4_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_4_prev3_force_negative:
   dist_diff_pair_c3_2_4_prev3
   - 120 dist_diff_abs_pair_c3_2_4_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_2_4_prev3: 120 same_dist_pair_c3_2_4_prev3
   + dist_diff_abs_pair_c3_2_4_prev3 <= 129.8
 same_dist_pair_lower_c3_2_4_prev3: 120 same_dist_pair_c3_2_4_prev3
   + dist_diff_abs_pair_c3_2_4_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_2_4_prev0: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_2_4_prev0: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_2_4_prev1: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_2_4_prev1: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_2_4_prev2: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_2_4_prev2: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_2_4_prev3: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_2_4_prev3: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_2_4: diff_dist_pair_c3_2_4
   + same_dist_pair_c3_2_4_prev0 + same_dist_pair_c3_2_4_prev1
   + same_dist_pair_c3_2_4_prev2 + same_dist_pair_c3_2_4_prev3 >= 1
 dx_grid_diff_def_c3_2_5: - x_grid_2 + x_grid_5 + dx_grid_diff_c3_2_5 = 0
 dx_grid_abs_pair_c3_2_5_abs_ge_orig: dx_grid_abs_pair_c3_2_5
   - dx_grid_diff_c3_2_5 - 60 dx_grid_abs_pair_c3_2_5_is_positive >= -60
 dx_grid_abs_pair_c3_2_5_abs_le_orig: dx_grid_abs_pair_c3_2_5
   - dx_grid_diff_c3_2_5 + 60 dx_grid_abs_pair_c3_2_5_is_positive <= 60
 dx_grid_abs_pair_c3_2_5_abs_ge_neg_orig: dx_grid_abs_pair_c3_2_5
   + dx_grid_diff_c3_2_5 + 60 dx_grid_abs_pair_c3_2_5_is_positive >= 0
 dx_grid_abs_pair_c3_2_5_abs_le_neg_orig: dx_grid_abs_pair_c3_2_5
   + dx_grid_diff_c3_2_5 - 60 dx_grid_abs_pair_c3_2_5_is_positive <= 0
 dx_grid_abs_pair_c3_2_5_force_positive: dx_grid_diff_c3_2_5
   - 60 dx_grid_abs_pair_c3_2_5_is_positive >= -60
 dx_grid_abs_pair_c3_2_5_force_negative: dx_grid_diff_c3_2_5
   - 60 dx_grid_abs_pair_c3_2_5_is_positive <= -0.001
 dy_grid_diff_def_c3_2_5: - y_grid_2 + y_grid_5 + dy_grid_diff_c3_2_5 = 0
 dy_grid_abs_pair_c3_2_5_abs_ge_orig: dy_grid_abs_pair_c3_2_5
   - dy_grid_diff_c3_2_5 - 60 dy_grid_abs_pair_c3_2_5_is_positive >= -60
 dy_grid_abs_pair_c3_2_5_abs_le_orig: dy_grid_abs_pair_c3_2_5
   - dy_grid_diff_c3_2_5 + 60 dy_grid_abs_pair_c3_2_5_is_positive <= 60
 dy_grid_abs_pair_c3_2_5_abs_ge_neg_orig: dy_grid_abs_pair_c3_2_5
   + dy_grid_diff_c3_2_5 + 60 dy_grid_abs_pair_c3_2_5_is_positive >= 0
 dy_grid_abs_pair_c3_2_5_abs_le_neg_orig: dy_grid_abs_pair_c3_2_5
   + dy_grid_diff_c3_2_5 - 60 dy_grid_abs_pair_c3_2_5_is_positive <= 0
 dy_grid_abs_pair_c3_2_5_force_positive: dy_grid_diff_c3_2_5
   - 60 dy_grid_abs_pair_c3_2_5_is_positive >= -60
 dy_grid_abs_pair_c3_2_5_force_negative: dy_grid_diff_c3_2_5
   - 60 dy_grid_abs_pair_c3_2_5_is_positive <= -0.001
 dist_curr_pair_def_c3_2_5: - dx_grid_abs_pair_c3_2_5
   - dy_grid_abs_pair_c3_2_5 + dist_curr_pair_c3_2_5 = 0
 dist_diff_pair_def_c3_2_5_prev0: - dist_curr_pair_c3_2_5
   + dist_diff_pair_c3_2_5_prev0 = -24
 dist_diff_abs_pair_c3_2_5_prev0_abs_ge_orig: - dist_diff_pair_c3_2_5_prev0
   + dist_diff_abs_pair_c3_2_5_prev0
   - 120 dist_diff_abs_pair_c3_2_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_5_prev0_abs_le_orig: - dist_diff_pair_c3_2_5_prev0
   + dist_diff_abs_pair_c3_2_5_prev0
   + 120 dist_diff_abs_pair_c3_2_5_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_2_5_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_2_5_prev0 + dist_diff_abs_pair_c3_2_5_prev0
   + 120 dist_diff_abs_pair_c3_2_5_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_2_5_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_2_5_prev0 + dist_diff_abs_pair_c3_2_5_prev0
   - 120 dist_diff_abs_pair_c3_2_5_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_2_5_prev0_force_positive:
   dist_diff_pair_c3_2_5_prev0
   - 120 dist_diff_abs_pair_c3_2_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_5_prev0_force_negative:
   dist_diff_pair_c3_2_5_prev0
   - 120 dist_diff_abs_pair_c3_2_5_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_2_5_prev0: 120 same_dist_pair_c3_2_5_prev0
   + dist_diff_abs_pair_c3_2_5_prev0 <= 129.8
 same_dist_pair_lower_c3_2_5_prev0: 120 same_dist_pair_c3_2_5_prev0
   + dist_diff_abs_pair_c3_2_5_prev0 >= 10
 dist_diff_pair_def_c3_2_5_prev1: - dist_curr_pair_c3_2_5
   + dist_diff_pair_c3_2_5_prev1 = -11
 dist_diff_abs_pair_c3_2_5_prev1_abs_ge_orig: - dist_diff_pair_c3_2_5_prev1
   + dist_diff_abs_pair_c3_2_5_prev1
   - 120 dist_diff_abs_pair_c3_2_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_5_prev1_abs_le_orig: - dist_diff_pair_c3_2_5_prev1
   + dist_diff_abs_pair_c3_2_5_prev1
   + 120 dist_diff_abs_pair_c3_2_5_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_2_5_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_2_5_prev1 + dist_diff_abs_pair_c3_2_5_prev1
   + 120 dist_diff_abs_pair_c3_2_5_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_2_5_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_2_5_prev1 + dist_diff_abs_pair_c3_2_5_prev1
   - 120 dist_diff_abs_pair_c3_2_5_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_2_5_prev1_force_positive:
   dist_diff_pair_c3_2_5_prev1
   - 120 dist_diff_abs_pair_c3_2_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_5_prev1_force_negative:
   dist_diff_pair_c3_2_5_prev1
   - 120 dist_diff_abs_pair_c3_2_5_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_2_5_prev1: 120 same_dist_pair_c3_2_5_prev1
   + dist_diff_abs_pair_c3_2_5_prev1 <= 129.8
 same_dist_pair_lower_c3_2_5_prev1: 120 same_dist_pair_c3_2_5_prev1
   + dist_diff_abs_pair_c3_2_5_prev1 >= 10
 dist_diff_pair_def_c3_2_5_prev2: - dist_curr_pair_c3_2_5
   + dist_diff_pair_c3_2_5_prev2 = -30
 dist_diff_abs_pair_c3_2_5_prev2_abs_ge_orig: - dist_diff_pair_c3_2_5_prev2
   + dist_diff_abs_pair_c3_2_5_prev2
   - 120 dist_diff_abs_pair_c3_2_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_5_prev2_abs_le_orig: - dist_diff_pair_c3_2_5_prev2
   + dist_diff_abs_pair_c3_2_5_prev2
   + 120 dist_diff_abs_pair_c3_2_5_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_2_5_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_2_5_prev2 + dist_diff_abs_pair_c3_2_5_prev2
   + 120 dist_diff_abs_pair_c3_2_5_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_2_5_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_2_5_prev2 + dist_diff_abs_pair_c3_2_5_prev2
   - 120 dist_diff_abs_pair_c3_2_5_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_2_5_prev2_force_positive:
   dist_diff_pair_c3_2_5_prev2
   - 120 dist_diff_abs_pair_c3_2_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_5_prev2_force_negative:
   dist_diff_pair_c3_2_5_prev2
   - 120 dist_diff_abs_pair_c3_2_5_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_2_5_prev2: 120 same_dist_pair_c3_2_5_prev2
   + dist_diff_abs_pair_c3_2_5_prev2 <= 129.8
 same_dist_pair_lower_c3_2_5_prev2: 120 same_dist_pair_c3_2_5_prev2
   + dist_diff_abs_pair_c3_2_5_prev2 >= 10
 dist_diff_pair_def_c3_2_5_prev3: - dist_curr_pair_c3_2_5
   + dist_diff_pair_c3_2_5_prev3 = -11
 dist_diff_abs_pair_c3_2_5_prev3_abs_ge_orig: - dist_diff_pair_c3_2_5_prev3
   + dist_diff_abs_pair_c3_2_5_prev3
   - 120 dist_diff_abs_pair_c3_2_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_5_prev3_abs_le_orig: - dist_diff_pair_c3_2_5_prev3
   + dist_diff_abs_pair_c3_2_5_prev3
   + 120 dist_diff_abs_pair_c3_2_5_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_2_5_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_2_5_prev3 + dist_diff_abs_pair_c3_2_5_prev3
   + 120 dist_diff_abs_pair_c3_2_5_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_2_5_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_2_5_prev3 + dist_diff_abs_pair_c3_2_5_prev3
   - 120 dist_diff_abs_pair_c3_2_5_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_2_5_prev3_force_positive:
   dist_diff_pair_c3_2_5_prev3
   - 120 dist_diff_abs_pair_c3_2_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_5_prev3_force_negative:
   dist_diff_pair_c3_2_5_prev3
   - 120 dist_diff_abs_pair_c3_2_5_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_2_5_prev3: 120 same_dist_pair_c3_2_5_prev3
   + dist_diff_abs_pair_c3_2_5_prev3 <= 129.8
 same_dist_pair_lower_c3_2_5_prev3: 120 same_dist_pair_c3_2_5_prev3
   + dist_diff_abs_pair_c3_2_5_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_2_5_prev0: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_2_5_prev0: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_2_5_prev1: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_2_5_prev1: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_2_5_prev2: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_2_5_prev2: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_2_5_prev3: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_2_5_prev3: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_2_5: diff_dist_pair_c3_2_5
   + same_dist_pair_c3_2_5_prev0 + same_dist_pair_c3_2_5_prev1
   + same_dist_pair_c3_2_5_prev2 + same_dist_pair_c3_2_5_prev3 >= 1
 dx_grid_diff_def_c3_2_6: - x_grid_2 + x_grid_6 + dx_grid_diff_c3_2_6 = 0
 dx_grid_abs_pair_c3_2_6_abs_ge_orig: dx_grid_abs_pair_c3_2_6
   - dx_grid_diff_c3_2_6 - 60 dx_grid_abs_pair_c3_2_6_is_positive >= -60
 dx_grid_abs_pair_c3_2_6_abs_le_orig: dx_grid_abs_pair_c3_2_6
   - dx_grid_diff_c3_2_6 + 60 dx_grid_abs_pair_c3_2_6_is_positive <= 60
 dx_grid_abs_pair_c3_2_6_abs_ge_neg_orig: dx_grid_abs_pair_c3_2_6
   + dx_grid_diff_c3_2_6 + 60 dx_grid_abs_pair_c3_2_6_is_positive >= 0
 dx_grid_abs_pair_c3_2_6_abs_le_neg_orig: dx_grid_abs_pair_c3_2_6
   + dx_grid_diff_c3_2_6 - 60 dx_grid_abs_pair_c3_2_6_is_positive <= 0
 dx_grid_abs_pair_c3_2_6_force_positive: dx_grid_diff_c3_2_6
   - 60 dx_grid_abs_pair_c3_2_6_is_positive >= -60
 dx_grid_abs_pair_c3_2_6_force_negative: dx_grid_diff_c3_2_6
   - 60 dx_grid_abs_pair_c3_2_6_is_positive <= -0.001
 dy_grid_diff_def_c3_2_6: - y_grid_2 + y_grid_6 + dy_grid_diff_c3_2_6 = 0
 dy_grid_abs_pair_c3_2_6_abs_ge_orig: dy_grid_abs_pair_c3_2_6
   - dy_grid_diff_c3_2_6 - 60 dy_grid_abs_pair_c3_2_6_is_positive >= -60
 dy_grid_abs_pair_c3_2_6_abs_le_orig: dy_grid_abs_pair_c3_2_6
   - dy_grid_diff_c3_2_6 + 60 dy_grid_abs_pair_c3_2_6_is_positive <= 60
 dy_grid_abs_pair_c3_2_6_abs_ge_neg_orig: dy_grid_abs_pair_c3_2_6
   + dy_grid_diff_c3_2_6 + 60 dy_grid_abs_pair_c3_2_6_is_positive >= 0
 dy_grid_abs_pair_c3_2_6_abs_le_neg_orig: dy_grid_abs_pair_c3_2_6
   + dy_grid_diff_c3_2_6 - 60 dy_grid_abs_pair_c3_2_6_is_positive <= 0
 dy_grid_abs_pair_c3_2_6_force_positive: dy_grid_diff_c3_2_6
   - 60 dy_grid_abs_pair_c3_2_6_is_positive >= -60
 dy_grid_abs_pair_c3_2_6_force_negative: dy_grid_diff_c3_2_6
   - 60 dy_grid_abs_pair_c3_2_6_is_positive <= -0.001
 dist_curr_pair_def_c3_2_6: - dx_grid_abs_pair_c3_2_6
   - dy_grid_abs_pair_c3_2_6 + dist_curr_pair_c3_2_6 = 0
 dist_diff_pair_def_c3_2_6_prev0: - dist_curr_pair_c3_2_6
   + dist_diff_pair_c3_2_6_prev0 = -20
 dist_diff_abs_pair_c3_2_6_prev0_abs_ge_orig: - dist_diff_pair_c3_2_6_prev0
   + dist_diff_abs_pair_c3_2_6_prev0
   - 120 dist_diff_abs_pair_c3_2_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_6_prev0_abs_le_orig: - dist_diff_pair_c3_2_6_prev0
   + dist_diff_abs_pair_c3_2_6_prev0
   + 120 dist_diff_abs_pair_c3_2_6_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_2_6_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_2_6_prev0 + dist_diff_abs_pair_c3_2_6_prev0
   + 120 dist_diff_abs_pair_c3_2_6_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_2_6_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_2_6_prev0 + dist_diff_abs_pair_c3_2_6_prev0
   - 120 dist_diff_abs_pair_c3_2_6_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_2_6_prev0_force_positive:
   dist_diff_pair_c3_2_6_prev0
   - 120 dist_diff_abs_pair_c3_2_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_6_prev0_force_negative:
   dist_diff_pair_c3_2_6_prev0
   - 120 dist_diff_abs_pair_c3_2_6_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_2_6_prev0: 120 same_dist_pair_c3_2_6_prev0
   + dist_diff_abs_pair_c3_2_6_prev0 <= 129.8
 same_dist_pair_lower_c3_2_6_prev0: 120 same_dist_pair_c3_2_6_prev0
   + dist_diff_abs_pair_c3_2_6_prev0 >= 10
 dist_diff_pair_def_c3_2_6_prev1: - dist_curr_pair_c3_2_6
   + dist_diff_pair_c3_2_6_prev1 = -21
 dist_diff_abs_pair_c3_2_6_prev1_abs_ge_orig: - dist_diff_pair_c3_2_6_prev1
   + dist_diff_abs_pair_c3_2_6_prev1
   - 120 dist_diff_abs_pair_c3_2_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_6_prev1_abs_le_orig: - dist_diff_pair_c3_2_6_prev1
   + dist_diff_abs_pair_c3_2_6_prev1
   + 120 dist_diff_abs_pair_c3_2_6_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_2_6_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_2_6_prev1 + dist_diff_abs_pair_c3_2_6_prev1
   + 120 dist_diff_abs_pair_c3_2_6_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_2_6_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_2_6_prev1 + dist_diff_abs_pair_c3_2_6_prev1
   - 120 dist_diff_abs_pair_c3_2_6_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_2_6_prev1_force_positive:
   dist_diff_pair_c3_2_6_prev1
   - 120 dist_diff_abs_pair_c3_2_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_6_prev1_force_negative:
   dist_diff_pair_c3_2_6_prev1
   - 120 dist_diff_abs_pair_c3_2_6_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_2_6_prev1: 120 same_dist_pair_c3_2_6_prev1
   + dist_diff_abs_pair_c3_2_6_prev1 <= 129.8
 same_dist_pair_lower_c3_2_6_prev1: 120 same_dist_pair_c3_2_6_prev1
   + dist_diff_abs_pair_c3_2_6_prev1 >= 10
 dist_diff_pair_def_c3_2_6_prev2: - dist_curr_pair_c3_2_6
   + dist_diff_pair_c3_2_6_prev2 = -25
 dist_diff_abs_pair_c3_2_6_prev2_abs_ge_orig: - dist_diff_pair_c3_2_6_prev2
   + dist_diff_abs_pair_c3_2_6_prev2
   - 120 dist_diff_abs_pair_c3_2_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_6_prev2_abs_le_orig: - dist_diff_pair_c3_2_6_prev2
   + dist_diff_abs_pair_c3_2_6_prev2
   + 120 dist_diff_abs_pair_c3_2_6_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_2_6_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_2_6_prev2 + dist_diff_abs_pair_c3_2_6_prev2
   + 120 dist_diff_abs_pair_c3_2_6_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_2_6_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_2_6_prev2 + dist_diff_abs_pair_c3_2_6_prev2
   - 120 dist_diff_abs_pair_c3_2_6_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_2_6_prev2_force_positive:
   dist_diff_pair_c3_2_6_prev2
   - 120 dist_diff_abs_pair_c3_2_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_6_prev2_force_negative:
   dist_diff_pair_c3_2_6_prev2
   - 120 dist_diff_abs_pair_c3_2_6_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_2_6_prev2: 120 same_dist_pair_c3_2_6_prev2
   + dist_diff_abs_pair_c3_2_6_prev2 <= 129.8
 same_dist_pair_lower_c3_2_6_prev2: 120 same_dist_pair_c3_2_6_prev2
   + dist_diff_abs_pair_c3_2_6_prev2 >= 10
 dist_diff_pair_def_c3_2_6_prev3: - dist_curr_pair_c3_2_6
   + dist_diff_pair_c3_2_6_prev3 = -21
 dist_diff_abs_pair_c3_2_6_prev3_abs_ge_orig: - dist_diff_pair_c3_2_6_prev3
   + dist_diff_abs_pair_c3_2_6_prev3
   - 120 dist_diff_abs_pair_c3_2_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_6_prev3_abs_le_orig: - dist_diff_pair_c3_2_6_prev3
   + dist_diff_abs_pair_c3_2_6_prev3
   + 120 dist_diff_abs_pair_c3_2_6_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_2_6_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_2_6_prev3 + dist_diff_abs_pair_c3_2_6_prev3
   + 120 dist_diff_abs_pair_c3_2_6_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_2_6_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_2_6_prev3 + dist_diff_abs_pair_c3_2_6_prev3
   - 120 dist_diff_abs_pair_c3_2_6_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_2_6_prev3_force_positive:
   dist_diff_pair_c3_2_6_prev3
   - 120 dist_diff_abs_pair_c3_2_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_6_prev3_force_negative:
   dist_diff_pair_c3_2_6_prev3
   - 120 dist_diff_abs_pair_c3_2_6_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_2_6_prev3: 120 same_dist_pair_c3_2_6_prev3
   + dist_diff_abs_pair_c3_2_6_prev3 <= 129.8
 same_dist_pair_lower_c3_2_6_prev3: 120 same_dist_pair_c3_2_6_prev3
   + dist_diff_abs_pair_c3_2_6_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_2_6_prev0: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_2_6_prev0: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_2_6_prev1: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_2_6_prev1: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_2_6_prev2: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_2_6_prev2: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_2_6_prev3: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_2_6_prev3: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_2_6: diff_dist_pair_c3_2_6
   + same_dist_pair_c3_2_6_prev0 + same_dist_pair_c3_2_6_prev1
   + same_dist_pair_c3_2_6_prev2 + same_dist_pair_c3_2_6_prev3 >= 1
 dx_grid_diff_def_c3_2_7: - x_grid_2 + x_grid_7 + dx_grid_diff_c3_2_7 = 0
 dx_grid_abs_pair_c3_2_7_abs_ge_orig: dx_grid_abs_pair_c3_2_7
   - dx_grid_diff_c3_2_7 - 60 dx_grid_abs_pair_c3_2_7_is_positive >= -60
 dx_grid_abs_pair_c3_2_7_abs_le_orig: dx_grid_abs_pair_c3_2_7
   - dx_grid_diff_c3_2_7 + 60 dx_grid_abs_pair_c3_2_7_is_positive <= 60
 dx_grid_abs_pair_c3_2_7_abs_ge_neg_orig: dx_grid_abs_pair_c3_2_7
   + dx_grid_diff_c3_2_7 + 60 dx_grid_abs_pair_c3_2_7_is_positive >= 0
 dx_grid_abs_pair_c3_2_7_abs_le_neg_orig: dx_grid_abs_pair_c3_2_7
   + dx_grid_diff_c3_2_7 - 60 dx_grid_abs_pair_c3_2_7_is_positive <= 0
 dx_grid_abs_pair_c3_2_7_force_positive: dx_grid_diff_c3_2_7
   - 60 dx_grid_abs_pair_c3_2_7_is_positive >= -60
 dx_grid_abs_pair_c3_2_7_force_negative: dx_grid_diff_c3_2_7
   - 60 dx_grid_abs_pair_c3_2_7_is_positive <= -0.001
 dy_grid_diff_def_c3_2_7: - y_grid_2 + y_grid_7 + dy_grid_diff_c3_2_7 = 0
 dy_grid_abs_pair_c3_2_7_abs_ge_orig: dy_grid_abs_pair_c3_2_7
   - dy_grid_diff_c3_2_7 - 60 dy_grid_abs_pair_c3_2_7_is_positive >= -60
 dy_grid_abs_pair_c3_2_7_abs_le_orig: dy_grid_abs_pair_c3_2_7
   - dy_grid_diff_c3_2_7 + 60 dy_grid_abs_pair_c3_2_7_is_positive <= 60
 dy_grid_abs_pair_c3_2_7_abs_ge_neg_orig: dy_grid_abs_pair_c3_2_7
   + dy_grid_diff_c3_2_7 + 60 dy_grid_abs_pair_c3_2_7_is_positive >= 0
 dy_grid_abs_pair_c3_2_7_abs_le_neg_orig: dy_grid_abs_pair_c3_2_7
   + dy_grid_diff_c3_2_7 - 60 dy_grid_abs_pair_c3_2_7_is_positive <= 0
 dy_grid_abs_pair_c3_2_7_force_positive: dy_grid_diff_c3_2_7
   - 60 dy_grid_abs_pair_c3_2_7_is_positive >= -60
 dy_grid_abs_pair_c3_2_7_force_negative: dy_grid_diff_c3_2_7
   - 60 dy_grid_abs_pair_c3_2_7_is_positive <= -0.001
 dist_curr_pair_def_c3_2_7: - dx_grid_abs_pair_c3_2_7
   - dy_grid_abs_pair_c3_2_7 + dist_curr_pair_c3_2_7 = 0
 dist_diff_pair_def_c3_2_7_prev0: - dist_curr_pair_c3_2_7
   + dist_diff_pair_c3_2_7_prev0 = -34
 dist_diff_abs_pair_c3_2_7_prev0_abs_ge_orig: - dist_diff_pair_c3_2_7_prev0
   + dist_diff_abs_pair_c3_2_7_prev0
   - 120 dist_diff_abs_pair_c3_2_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_7_prev0_abs_le_orig: - dist_diff_pair_c3_2_7_prev0
   + dist_diff_abs_pair_c3_2_7_prev0
   + 120 dist_diff_abs_pair_c3_2_7_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_2_7_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_2_7_prev0 + dist_diff_abs_pair_c3_2_7_prev0
   + 120 dist_diff_abs_pair_c3_2_7_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_2_7_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_2_7_prev0 + dist_diff_abs_pair_c3_2_7_prev0
   - 120 dist_diff_abs_pair_c3_2_7_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_2_7_prev0_force_positive:
   dist_diff_pair_c3_2_7_prev0
   - 120 dist_diff_abs_pair_c3_2_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_2_7_prev0_force_negative:
   dist_diff_pair_c3_2_7_prev0
   - 120 dist_diff_abs_pair_c3_2_7_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_2_7_prev0: 120 same_dist_pair_c3_2_7_prev0
   + dist_diff_abs_pair_c3_2_7_prev0 <= 129.8
 same_dist_pair_lower_c3_2_7_prev0: 120 same_dist_pair_c3_2_7_prev0
   + dist_diff_abs_pair_c3_2_7_prev0 >= 10
 dist_diff_pair_def_c3_2_7_prev1: - dist_curr_pair_c3_2_7
   + dist_diff_pair_c3_2_7_prev1 = -33
 dist_diff_abs_pair_c3_2_7_prev1_abs_ge_orig: - dist_diff_pair_c3_2_7_prev1
   + dist_diff_abs_pair_c3_2_7_prev1
   - 120 dist_diff_abs_pair_c3_2_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_7_prev1_abs_le_orig: - dist_diff_pair_c3_2_7_prev1
   + dist_diff_abs_pair_c3_2_7_prev1
   + 120 dist_diff_abs_pair_c3_2_7_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_2_7_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_2_7_prev1 + dist_diff_abs_pair_c3_2_7_prev1
   + 120 dist_diff_abs_pair_c3_2_7_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_2_7_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_2_7_prev1 + dist_diff_abs_pair_c3_2_7_prev1
   - 120 dist_diff_abs_pair_c3_2_7_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_2_7_prev1_force_positive:
   dist_diff_pair_c3_2_7_prev1
   - 120 dist_diff_abs_pair_c3_2_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_2_7_prev1_force_negative:
   dist_diff_pair_c3_2_7_prev1
   - 120 dist_diff_abs_pair_c3_2_7_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_2_7_prev1: 120 same_dist_pair_c3_2_7_prev1
   + dist_diff_abs_pair_c3_2_7_prev1 <= 129.8
 same_dist_pair_lower_c3_2_7_prev1: 120 same_dist_pair_c3_2_7_prev1
   + dist_diff_abs_pair_c3_2_7_prev1 >= 10
 dist_diff_pair_def_c3_2_7_prev2: - dist_curr_pair_c3_2_7
   + dist_diff_pair_c3_2_7_prev2 = -41
 dist_diff_abs_pair_c3_2_7_prev2_abs_ge_orig: - dist_diff_pair_c3_2_7_prev2
   + dist_diff_abs_pair_c3_2_7_prev2
   - 120 dist_diff_abs_pair_c3_2_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_7_prev2_abs_le_orig: - dist_diff_pair_c3_2_7_prev2
   + dist_diff_abs_pair_c3_2_7_prev2
   + 120 dist_diff_abs_pair_c3_2_7_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_2_7_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_2_7_prev2 + dist_diff_abs_pair_c3_2_7_prev2
   + 120 dist_diff_abs_pair_c3_2_7_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_2_7_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_2_7_prev2 + dist_diff_abs_pair_c3_2_7_prev2
   - 120 dist_diff_abs_pair_c3_2_7_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_2_7_prev2_force_positive:
   dist_diff_pair_c3_2_7_prev2
   - 120 dist_diff_abs_pair_c3_2_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_2_7_prev2_force_negative:
   dist_diff_pair_c3_2_7_prev2
   - 120 dist_diff_abs_pair_c3_2_7_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_2_7_prev2: 120 same_dist_pair_c3_2_7_prev2
   + dist_diff_abs_pair_c3_2_7_prev2 <= 129.8
 same_dist_pair_lower_c3_2_7_prev2: 120 same_dist_pair_c3_2_7_prev2
   + dist_diff_abs_pair_c3_2_7_prev2 >= 10
 dist_diff_pair_def_c3_2_7_prev3: - dist_curr_pair_c3_2_7
   + dist_diff_pair_c3_2_7_prev3 = -33
 dist_diff_abs_pair_c3_2_7_prev3_abs_ge_orig: - dist_diff_pair_c3_2_7_prev3
   + dist_diff_abs_pair_c3_2_7_prev3
   - 120 dist_diff_abs_pair_c3_2_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_7_prev3_abs_le_orig: - dist_diff_pair_c3_2_7_prev3
   + dist_diff_abs_pair_c3_2_7_prev3
   + 120 dist_diff_abs_pair_c3_2_7_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_2_7_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_2_7_prev3 + dist_diff_abs_pair_c3_2_7_prev3
   + 120 dist_diff_abs_pair_c3_2_7_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_2_7_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_2_7_prev3 + dist_diff_abs_pair_c3_2_7_prev3
   - 120 dist_diff_abs_pair_c3_2_7_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_2_7_prev3_force_positive:
   dist_diff_pair_c3_2_7_prev3
   - 120 dist_diff_abs_pair_c3_2_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_2_7_prev3_force_negative:
   dist_diff_pair_c3_2_7_prev3
   - 120 dist_diff_abs_pair_c3_2_7_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_2_7_prev3: 120 same_dist_pair_c3_2_7_prev3
   + dist_diff_abs_pair_c3_2_7_prev3 <= 129.8
 same_dist_pair_lower_c3_2_7_prev3: 120 same_dist_pair_c3_2_7_prev3
   + dist_diff_abs_pair_c3_2_7_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_2_7_prev0: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_2_7_prev0: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_2_7_prev1: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_2_7_prev1: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_2_7_prev2: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_2_7_prev2: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_2_7_prev3: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_2_7_prev3: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_2_7: diff_dist_pair_c3_2_7
   + same_dist_pair_c3_2_7_prev0 + same_dist_pair_c3_2_7_prev1
   + same_dist_pair_c3_2_7_prev2 + same_dist_pair_c3_2_7_prev3 >= 1
 dx_grid_diff_def_c3_3_4: - x_grid_3 + x_grid_4 + dx_grid_diff_c3_3_4 = 0
 dx_grid_abs_pair_c3_3_4_abs_ge_orig: dx_grid_abs_pair_c3_3_4
   - dx_grid_diff_c3_3_4 - 60 dx_grid_abs_pair_c3_3_4_is_positive >= -60
 dx_grid_abs_pair_c3_3_4_abs_le_orig: dx_grid_abs_pair_c3_3_4
   - dx_grid_diff_c3_3_4 + 60 dx_grid_abs_pair_c3_3_4_is_positive <= 60
 dx_grid_abs_pair_c3_3_4_abs_ge_neg_orig: dx_grid_abs_pair_c3_3_4
   + dx_grid_diff_c3_3_4 + 60 dx_grid_abs_pair_c3_3_4_is_positive >= 0
 dx_grid_abs_pair_c3_3_4_abs_le_neg_orig: dx_grid_abs_pair_c3_3_4
   + dx_grid_diff_c3_3_4 - 60 dx_grid_abs_pair_c3_3_4_is_positive <= 0
 dx_grid_abs_pair_c3_3_4_force_positive: dx_grid_diff_c3_3_4
   - 60 dx_grid_abs_pair_c3_3_4_is_positive >= -60
 dx_grid_abs_pair_c3_3_4_force_negative: dx_grid_diff_c3_3_4
   - 60 dx_grid_abs_pair_c3_3_4_is_positive <= -0.001
 dy_grid_diff_def_c3_3_4: - y_grid_3 + y_grid_4 + dy_grid_diff_c3_3_4 = 0
 dy_grid_abs_pair_c3_3_4_abs_ge_orig: dy_grid_abs_pair_c3_3_4
   - dy_grid_diff_c3_3_4 - 60 dy_grid_abs_pair_c3_3_4_is_positive >= -60
 dy_grid_abs_pair_c3_3_4_abs_le_orig: dy_grid_abs_pair_c3_3_4
   - dy_grid_diff_c3_3_4 + 60 dy_grid_abs_pair_c3_3_4_is_positive <= 60
 dy_grid_abs_pair_c3_3_4_abs_ge_neg_orig: dy_grid_abs_pair_c3_3_4
   + dy_grid_diff_c3_3_4 + 60 dy_grid_abs_pair_c3_3_4_is_positive >= 0
 dy_grid_abs_pair_c3_3_4_abs_le_neg_orig: dy_grid_abs_pair_c3_3_4
   + dy_grid_diff_c3_3_4 - 60 dy_grid_abs_pair_c3_3_4_is_positive <= 0
 dy_grid_abs_pair_c3_3_4_force_positive: dy_grid_diff_c3_3_4
   - 60 dy_grid_abs_pair_c3_3_4_is_positive >= -60
 dy_grid_abs_pair_c3_3_4_force_negative: dy_grid_diff_c3_3_4
   - 60 dy_grid_abs_pair_c3_3_4_is_positive <= -0.001
 dist_curr_pair_def_c3_3_4: - dx_grid_abs_pair_c3_3_4
   - dy_grid_abs_pair_c3_3_4 + dist_curr_pair_c3_3_4 = 0
 dist_diff_pair_def_c3_3_4_prev0: - dist_curr_pair_c3_3_4
   + dist_diff_pair_c3_3_4_prev0 = -16
 dist_diff_abs_pair_c3_3_4_prev0_abs_ge_orig: - dist_diff_pair_c3_3_4_prev0
   + dist_diff_abs_pair_c3_3_4_prev0
   - 120 dist_diff_abs_pair_c3_3_4_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_3_4_prev0_abs_le_orig: - dist_diff_pair_c3_3_4_prev0
   + dist_diff_abs_pair_c3_3_4_prev0
   + 120 dist_diff_abs_pair_c3_3_4_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_3_4_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_3_4_prev0 + dist_diff_abs_pair_c3_3_4_prev0
   + 120 dist_diff_abs_pair_c3_3_4_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_3_4_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_3_4_prev0 + dist_diff_abs_pair_c3_3_4_prev0
   - 120 dist_diff_abs_pair_c3_3_4_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_3_4_prev0_force_positive:
   dist_diff_pair_c3_3_4_prev0
   - 120 dist_diff_abs_pair_c3_3_4_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_3_4_prev0_force_negative:
   dist_diff_pair_c3_3_4_prev0
   - 120 dist_diff_abs_pair_c3_3_4_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_3_4_prev0: 120 same_dist_pair_c3_3_4_prev0
   + dist_diff_abs_pair_c3_3_4_prev0 <= 129.8
 same_dist_pair_lower_c3_3_4_prev0: 120 same_dist_pair_c3_3_4_prev0
   + dist_diff_abs_pair_c3_3_4_prev0 >= 10
 dist_diff_pair_def_c3_3_4_prev1: - dist_curr_pair_c3_3_4
   + dist_diff_pair_c3_3_4_prev1 = -10
 dist_diff_abs_pair_c3_3_4_prev1_abs_ge_orig: - dist_diff_pair_c3_3_4_prev1
   + dist_diff_abs_pair_c3_3_4_prev1
   - 120 dist_diff_abs_pair_c3_3_4_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_3_4_prev1_abs_le_orig: - dist_diff_pair_c3_3_4_prev1
   + dist_diff_abs_pair_c3_3_4_prev1
   + 120 dist_diff_abs_pair_c3_3_4_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_3_4_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_3_4_prev1 + dist_diff_abs_pair_c3_3_4_prev1
   + 120 dist_diff_abs_pair_c3_3_4_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_3_4_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_3_4_prev1 + dist_diff_abs_pair_c3_3_4_prev1
   - 120 dist_diff_abs_pair_c3_3_4_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_3_4_prev1_force_positive:
   dist_diff_pair_c3_3_4_prev1
   - 120 dist_diff_abs_pair_c3_3_4_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_3_4_prev1_force_negative:
   dist_diff_pair_c3_3_4_prev1
   - 120 dist_diff_abs_pair_c3_3_4_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_3_4_prev1: 120 same_dist_pair_c3_3_4_prev1
   + dist_diff_abs_pair_c3_3_4_prev1 <= 129.8
 same_dist_pair_lower_c3_3_4_prev1: 120 same_dist_pair_c3_3_4_prev1
   + dist_diff_abs_pair_c3_3_4_prev1 >= 10
 dist_diff_pair_def_c3_3_4_prev2: - dist_curr_pair_c3_3_4
   + dist_diff_pair_c3_3_4_prev2 = -19
 dist_diff_abs_pair_c3_3_4_prev2_abs_ge_orig: - dist_diff_pair_c3_3_4_prev2
   + dist_diff_abs_pair_c3_3_4_prev2
   - 120 dist_diff_abs_pair_c3_3_4_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_3_4_prev2_abs_le_orig: - dist_diff_pair_c3_3_4_prev2
   + dist_diff_abs_pair_c3_3_4_prev2
   + 120 dist_diff_abs_pair_c3_3_4_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_3_4_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_3_4_prev2 + dist_diff_abs_pair_c3_3_4_prev2
   + 120 dist_diff_abs_pair_c3_3_4_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_3_4_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_3_4_prev2 + dist_diff_abs_pair_c3_3_4_prev2
   - 120 dist_diff_abs_pair_c3_3_4_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_3_4_prev2_force_positive:
   dist_diff_pair_c3_3_4_prev2
   - 120 dist_diff_abs_pair_c3_3_4_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_3_4_prev2_force_negative:
   dist_diff_pair_c3_3_4_prev2
   - 120 dist_diff_abs_pair_c3_3_4_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_3_4_prev2: 120 same_dist_pair_c3_3_4_prev2
   + dist_diff_abs_pair_c3_3_4_prev2 <= 129.8
 same_dist_pair_lower_c3_3_4_prev2: 120 same_dist_pair_c3_3_4_prev2
   + dist_diff_abs_pair_c3_3_4_prev2 >= 10
 dist_diff_pair_def_c3_3_4_prev3: - dist_curr_pair_c3_3_4
   + dist_diff_pair_c3_3_4_prev3 = -10
 dist_diff_abs_pair_c3_3_4_prev3_abs_ge_orig: - dist_diff_pair_c3_3_4_prev3
   + dist_diff_abs_pair_c3_3_4_prev3
   - 120 dist_diff_abs_pair_c3_3_4_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_3_4_prev3_abs_le_orig: - dist_diff_pair_c3_3_4_prev3
   + dist_diff_abs_pair_c3_3_4_prev3
   + 120 dist_diff_abs_pair_c3_3_4_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_3_4_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_3_4_prev3 + dist_diff_abs_pair_c3_3_4_prev3
   + 120 dist_diff_abs_pair_c3_3_4_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_3_4_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_3_4_prev3 + dist_diff_abs_pair_c3_3_4_prev3
   - 120 dist_diff_abs_pair_c3_3_4_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_3_4_prev3_force_positive:
   dist_diff_pair_c3_3_4_prev3
   - 120 dist_diff_abs_pair_c3_3_4_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_3_4_prev3_force_negative:
   dist_diff_pair_c3_3_4_prev3
   - 120 dist_diff_abs_pair_c3_3_4_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_3_4_prev3: 120 same_dist_pair_c3_3_4_prev3
   + dist_diff_abs_pair_c3_3_4_prev3 <= 129.8
 same_dist_pair_lower_c3_3_4_prev3: 120 same_dist_pair_c3_3_4_prev3
   + dist_diff_abs_pair_c3_3_4_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_3_4_prev0: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_3_4_prev0: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_3_4_prev1: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_3_4_prev1: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_3_4_prev2: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_3_4_prev2: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_3_4_prev3: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_3_4_prev3: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_3_4: diff_dist_pair_c3_3_4
   + same_dist_pair_c3_3_4_prev0 + same_dist_pair_c3_3_4_prev1
   + same_dist_pair_c3_3_4_prev2 + same_dist_pair_c3_3_4_prev3 >= 1
 dx_grid_diff_def_c3_3_5: - x_grid_3 + x_grid_5 + dx_grid_diff_c3_3_5 = 0
 dx_grid_abs_pair_c3_3_5_abs_ge_orig: dx_grid_abs_pair_c3_3_5
   - dx_grid_diff_c3_3_5 - 60 dx_grid_abs_pair_c3_3_5_is_positive >= -60
 dx_grid_abs_pair_c3_3_5_abs_le_orig: dx_grid_abs_pair_c3_3_5
   - dx_grid_diff_c3_3_5 + 60 dx_grid_abs_pair_c3_3_5_is_positive <= 60
 dx_grid_abs_pair_c3_3_5_abs_ge_neg_orig: dx_grid_abs_pair_c3_3_5
   + dx_grid_diff_c3_3_5 + 60 dx_grid_abs_pair_c3_3_5_is_positive >= 0
 dx_grid_abs_pair_c3_3_5_abs_le_neg_orig: dx_grid_abs_pair_c3_3_5
   + dx_grid_diff_c3_3_5 - 60 dx_grid_abs_pair_c3_3_5_is_positive <= 0
 dx_grid_abs_pair_c3_3_5_force_positive: dx_grid_diff_c3_3_5
   - 60 dx_grid_abs_pair_c3_3_5_is_positive >= -60
 dx_grid_abs_pair_c3_3_5_force_negative: dx_grid_diff_c3_3_5
   - 60 dx_grid_abs_pair_c3_3_5_is_positive <= -0.001
 dy_grid_diff_def_c3_3_5: - y_grid_3 + y_grid_5 + dy_grid_diff_c3_3_5 = 0
 dy_grid_abs_pair_c3_3_5_abs_ge_orig: dy_grid_abs_pair_c3_3_5
   - dy_grid_diff_c3_3_5 - 60 dy_grid_abs_pair_c3_3_5_is_positive >= -60
 dy_grid_abs_pair_c3_3_5_abs_le_orig: dy_grid_abs_pair_c3_3_5
   - dy_grid_diff_c3_3_5 + 60 dy_grid_abs_pair_c3_3_5_is_positive <= 60
 dy_grid_abs_pair_c3_3_5_abs_ge_neg_orig: dy_grid_abs_pair_c3_3_5
   + dy_grid_diff_c3_3_5 + 60 dy_grid_abs_pair_c3_3_5_is_positive >= 0
 dy_grid_abs_pair_c3_3_5_abs_le_neg_orig: dy_grid_abs_pair_c3_3_5
   + dy_grid_diff_c3_3_5 - 60 dy_grid_abs_pair_c3_3_5_is_positive <= 0
 dy_grid_abs_pair_c3_3_5_force_positive: dy_grid_diff_c3_3_5
   - 60 dy_grid_abs_pair_c3_3_5_is_positive >= -60
 dy_grid_abs_pair_c3_3_5_force_negative: dy_grid_diff_c3_3_5
   - 60 dy_grid_abs_pair_c3_3_5_is_positive <= -0.001
 dist_curr_pair_def_c3_3_5: - dx_grid_abs_pair_c3_3_5
   - dy_grid_abs_pair_c3_3_5 + dist_curr_pair_c3_3_5 = 0
 dist_diff_pair_def_c3_3_5_prev0: - dist_curr_pair_c3_3_5
   + dist_diff_pair_c3_3_5_prev0 = -10
 dist_diff_abs_pair_c3_3_5_prev0_abs_ge_orig: - dist_diff_pair_c3_3_5_prev0
   + dist_diff_abs_pair_c3_3_5_prev0
   - 120 dist_diff_abs_pair_c3_3_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_3_5_prev0_abs_le_orig: - dist_diff_pair_c3_3_5_prev0
   + dist_diff_abs_pair_c3_3_5_prev0
   + 120 dist_diff_abs_pair_c3_3_5_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_3_5_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_3_5_prev0 + dist_diff_abs_pair_c3_3_5_prev0
   + 120 dist_diff_abs_pair_c3_3_5_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_3_5_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_3_5_prev0 + dist_diff_abs_pair_c3_3_5_prev0
   - 120 dist_diff_abs_pair_c3_3_5_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_3_5_prev0_force_positive:
   dist_diff_pair_c3_3_5_prev0
   - 120 dist_diff_abs_pair_c3_3_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_3_5_prev0_force_negative:
   dist_diff_pair_c3_3_5_prev0
   - 120 dist_diff_abs_pair_c3_3_5_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_3_5_prev0: 120 same_dist_pair_c3_3_5_prev0
   + dist_diff_abs_pair_c3_3_5_prev0 <= 129.8
 same_dist_pair_lower_c3_3_5_prev0: 120 same_dist_pair_c3_3_5_prev0
   + dist_diff_abs_pair_c3_3_5_prev0 >= 10
 dist_diff_pair_def_c3_3_5_prev1: - dist_curr_pair_c3_3_5
   + dist_diff_pair_c3_3_5_prev1 = -29
 dist_diff_abs_pair_c3_3_5_prev1_abs_ge_orig: - dist_diff_pair_c3_3_5_prev1
   + dist_diff_abs_pair_c3_3_5_prev1
   - 120 dist_diff_abs_pair_c3_3_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_3_5_prev1_abs_le_orig: - dist_diff_pair_c3_3_5_prev1
   + dist_diff_abs_pair_c3_3_5_prev1
   + 120 dist_diff_abs_pair_c3_3_5_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_3_5_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_3_5_prev1 + dist_diff_abs_pair_c3_3_5_prev1
   + 120 dist_diff_abs_pair_c3_3_5_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_3_5_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_3_5_prev1 + dist_diff_abs_pair_c3_3_5_prev1
   - 120 dist_diff_abs_pair_c3_3_5_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_3_5_prev1_force_positive:
   dist_diff_pair_c3_3_5_prev1
   - 120 dist_diff_abs_pair_c3_3_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_3_5_prev1_force_negative:
   dist_diff_pair_c3_3_5_prev1
   - 120 dist_diff_abs_pair_c3_3_5_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_3_5_prev1: 120 same_dist_pair_c3_3_5_prev1
   + dist_diff_abs_pair_c3_3_5_prev1 <= 129.8
 same_dist_pair_lower_c3_3_5_prev1: 120 same_dist_pair_c3_3_5_prev1
   + dist_diff_abs_pair_c3_3_5_prev1 >= 10
 dist_diff_pair_def_c3_3_5_prev2: - dist_curr_pair_c3_3_5
   + dist_diff_pair_c3_3_5_prev2 = -10
 dist_diff_abs_pair_c3_3_5_prev2_abs_ge_orig: - dist_diff_pair_c3_3_5_prev2
   + dist_diff_abs_pair_c3_3_5_prev2
   - 120 dist_diff_abs_pair_c3_3_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_3_5_prev2_abs_le_orig: - dist_diff_pair_c3_3_5_prev2
   + dist_diff_abs_pair_c3_3_5_prev2
   + 120 dist_diff_abs_pair_c3_3_5_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_3_5_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_3_5_prev2 + dist_diff_abs_pair_c3_3_5_prev2
   + 120 dist_diff_abs_pair_c3_3_5_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_3_5_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_3_5_prev2 + dist_diff_abs_pair_c3_3_5_prev2
   - 120 dist_diff_abs_pair_c3_3_5_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_3_5_prev2_force_positive:
   dist_diff_pair_c3_3_5_prev2
   - 120 dist_diff_abs_pair_c3_3_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_3_5_prev2_force_negative:
   dist_diff_pair_c3_3_5_prev2
   - 120 dist_diff_abs_pair_c3_3_5_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_3_5_prev2: 120 same_dist_pair_c3_3_5_prev2
   + dist_diff_abs_pair_c3_3_5_prev2 <= 129.8
 same_dist_pair_lower_c3_3_5_prev2: 120 same_dist_pair_c3_3_5_prev2
   + dist_diff_abs_pair_c3_3_5_prev2 >= 10
 dist_diff_pair_def_c3_3_5_prev3: - dist_curr_pair_c3_3_5
   + dist_diff_pair_c3_3_5_prev3 = -19
 dist_diff_abs_pair_c3_3_5_prev3_abs_ge_orig: - dist_diff_pair_c3_3_5_prev3
   + dist_diff_abs_pair_c3_3_5_prev3
   - 120 dist_diff_abs_pair_c3_3_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_3_5_prev3_abs_le_orig: - dist_diff_pair_c3_3_5_prev3
   + dist_diff_abs_pair_c3_3_5_prev3
   + 120 dist_diff_abs_pair_c3_3_5_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_3_5_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_3_5_prev3 + dist_diff_abs_pair_c3_3_5_prev3
   + 120 dist_diff_abs_pair_c3_3_5_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_3_5_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_3_5_prev3 + dist_diff_abs_pair_c3_3_5_prev3
   - 120 dist_diff_abs_pair_c3_3_5_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_3_5_prev3_force_positive:
   dist_diff_pair_c3_3_5_prev3
   - 120 dist_diff_abs_pair_c3_3_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_3_5_prev3_force_negative:
   dist_diff_pair_c3_3_5_prev3
   - 120 dist_diff_abs_pair_c3_3_5_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_3_5_prev3: 120 same_dist_pair_c3_3_5_prev3
   + dist_diff_abs_pair_c3_3_5_prev3 <= 129.8
 same_dist_pair_lower_c3_3_5_prev3: 120 same_dist_pair_c3_3_5_prev3
   + dist_diff_abs_pair_c3_3_5_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_3_5_prev0: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_3_5_prev0: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_3_5_prev1: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_3_5_prev1: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_3_5_prev2: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_3_5_prev2: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_3_5_prev3: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_3_5_prev3: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_3_5: diff_dist_pair_c3_3_5
   + same_dist_pair_c3_3_5_prev0 + same_dist_pair_c3_3_5_prev1
   + same_dist_pair_c3_3_5_prev2 + same_dist_pair_c3_3_5_prev3 >= 1
 dx_grid_diff_def_c3_3_6: - x_grid_3 + x_grid_6 + dx_grid_diff_c3_3_6 = 0
 dx_grid_abs_pair_c3_3_6_abs_ge_orig: dx_grid_abs_pair_c3_3_6
   - dx_grid_diff_c3_3_6 - 60 dx_grid_abs_pair_c3_3_6_is_positive >= -60
 dx_grid_abs_pair_c3_3_6_abs_le_orig: dx_grid_abs_pair_c3_3_6
   - dx_grid_diff_c3_3_6 + 60 dx_grid_abs_pair_c3_3_6_is_positive <= 60
 dx_grid_abs_pair_c3_3_6_abs_ge_neg_orig: dx_grid_abs_pair_c3_3_6
   + dx_grid_diff_c3_3_6 + 60 dx_grid_abs_pair_c3_3_6_is_positive >= 0
 dx_grid_abs_pair_c3_3_6_abs_le_neg_orig: dx_grid_abs_pair_c3_3_6
   + dx_grid_diff_c3_3_6 - 60 dx_grid_abs_pair_c3_3_6_is_positive <= 0
 dx_grid_abs_pair_c3_3_6_force_positive: dx_grid_diff_c3_3_6
   - 60 dx_grid_abs_pair_c3_3_6_is_positive >= -60
 dx_grid_abs_pair_c3_3_6_force_negative: dx_grid_diff_c3_3_6
   - 60 dx_grid_abs_pair_c3_3_6_is_positive <= -0.001
 dy_grid_diff_def_c3_3_6: - y_grid_3 + y_grid_6 + dy_grid_diff_c3_3_6 = 0
 dy_grid_abs_pair_c3_3_6_abs_ge_orig: dy_grid_abs_pair_c3_3_6
   - dy_grid_diff_c3_3_6 - 60 dy_grid_abs_pair_c3_3_6_is_positive >= -60
 dy_grid_abs_pair_c3_3_6_abs_le_orig: dy_grid_abs_pair_c3_3_6
   - dy_grid_diff_c3_3_6 + 60 dy_grid_abs_pair_c3_3_6_is_positive <= 60
 dy_grid_abs_pair_c3_3_6_abs_ge_neg_orig: dy_grid_abs_pair_c3_3_6
   + dy_grid_diff_c3_3_6 + 60 dy_grid_abs_pair_c3_3_6_is_positive >= 0
 dy_grid_abs_pair_c3_3_6_abs_le_neg_orig: dy_grid_abs_pair_c3_3_6
   + dy_grid_diff_c3_3_6 - 60 dy_grid_abs_pair_c3_3_6_is_positive <= 0
 dy_grid_abs_pair_c3_3_6_force_positive: dy_grid_diff_c3_3_6
   - 60 dy_grid_abs_pair_c3_3_6_is_positive >= -60
 dy_grid_abs_pair_c3_3_6_force_negative: dy_grid_diff_c3_3_6
   - 60 dy_grid_abs_pair_c3_3_6_is_positive <= -0.001
 dist_curr_pair_def_c3_3_6: - dx_grid_abs_pair_c3_3_6
   - dy_grid_abs_pair_c3_3_6 + dist_curr_pair_c3_3_6 = 0
 dist_diff_pair_def_c3_3_6_prev0: - dist_curr_pair_c3_3_6
   + dist_diff_pair_c3_3_6_prev0 = -46
 dist_diff_abs_pair_c3_3_6_prev0_abs_ge_orig: - dist_diff_pair_c3_3_6_prev0
   + dist_diff_abs_pair_c3_3_6_prev0
   - 120 dist_diff_abs_pair_c3_3_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_3_6_prev0_abs_le_orig: - dist_diff_pair_c3_3_6_prev0
   + dist_diff_abs_pair_c3_3_6_prev0
   + 120 dist_diff_abs_pair_c3_3_6_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_3_6_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_3_6_prev0 + dist_diff_abs_pair_c3_3_6_prev0
   + 120 dist_diff_abs_pair_c3_3_6_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_3_6_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_3_6_prev0 + dist_diff_abs_pair_c3_3_6_prev0
   - 120 dist_diff_abs_pair_c3_3_6_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_3_6_prev0_force_positive:
   dist_diff_pair_c3_3_6_prev0
   - 120 dist_diff_abs_pair_c3_3_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_3_6_prev0_force_negative:
   dist_diff_pair_c3_3_6_prev0
   - 120 dist_diff_abs_pair_c3_3_6_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_3_6_prev0: 120 same_dist_pair_c3_3_6_prev0
   + dist_diff_abs_pair_c3_3_6_prev0 <= 129.8
 same_dist_pair_lower_c3_3_6_prev0: 120 same_dist_pair_c3_3_6_prev0
   + dist_diff_abs_pair_c3_3_6_prev0 >= 10
 dist_diff_pair_def_c3_3_6_prev1: - dist_curr_pair_c3_3_6
   + dist_diff_pair_c3_3_6_prev1 = -19
 dist_diff_abs_pair_c3_3_6_prev1_abs_ge_orig: - dist_diff_pair_c3_3_6_prev1
   + dist_diff_abs_pair_c3_3_6_prev1
   - 120 dist_diff_abs_pair_c3_3_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_3_6_prev1_abs_le_orig: - dist_diff_pair_c3_3_6_prev1
   + dist_diff_abs_pair_c3_3_6_prev1
   + 120 dist_diff_abs_pair_c3_3_6_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_3_6_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_3_6_prev1 + dist_diff_abs_pair_c3_3_6_prev1
   + 120 dist_diff_abs_pair_c3_3_6_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_3_6_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_3_6_prev1 + dist_diff_abs_pair_c3_3_6_prev1
   - 120 dist_diff_abs_pair_c3_3_6_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_3_6_prev1_force_positive:
   dist_diff_pair_c3_3_6_prev1
   - 120 dist_diff_abs_pair_c3_3_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_3_6_prev1_force_negative:
   dist_diff_pair_c3_3_6_prev1
   - 120 dist_diff_abs_pair_c3_3_6_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_3_6_prev1: 120 same_dist_pair_c3_3_6_prev1
   + dist_diff_abs_pair_c3_3_6_prev1 <= 129.8
 same_dist_pair_lower_c3_3_6_prev1: 120 same_dist_pair_c3_3_6_prev1
   + dist_diff_abs_pair_c3_3_6_prev1 >= 10
 dist_diff_pair_def_c3_3_6_prev2: - dist_curr_pair_c3_3_6
   + dist_diff_pair_c3_3_6_prev2 = -35
 dist_diff_abs_pair_c3_3_6_prev2_abs_ge_orig: - dist_diff_pair_c3_3_6_prev2
   + dist_diff_abs_pair_c3_3_6_prev2
   - 120 dist_diff_abs_pair_c3_3_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_3_6_prev2_abs_le_orig: - dist_diff_pair_c3_3_6_prev2
   + dist_diff_abs_pair_c3_3_6_prev2
   + 120 dist_diff_abs_pair_c3_3_6_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_3_6_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_3_6_prev2 + dist_diff_abs_pair_c3_3_6_prev2
   + 120 dist_diff_abs_pair_c3_3_6_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_3_6_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_3_6_prev2 + dist_diff_abs_pair_c3_3_6_prev2
   - 120 dist_diff_abs_pair_c3_3_6_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_3_6_prev2_force_positive:
   dist_diff_pair_c3_3_6_prev2
   - 120 dist_diff_abs_pair_c3_3_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_3_6_prev2_force_negative:
   dist_diff_pair_c3_3_6_prev2
   - 120 dist_diff_abs_pair_c3_3_6_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_3_6_prev2: 120 same_dist_pair_c3_3_6_prev2
   + dist_diff_abs_pair_c3_3_6_prev2 <= 129.8
 same_dist_pair_lower_c3_3_6_prev2: 120 same_dist_pair_c3_3_6_prev2
   + dist_diff_abs_pair_c3_3_6_prev2 >= 10
 dist_diff_pair_def_c3_3_6_prev3: - dist_curr_pair_c3_3_6
   + dist_diff_pair_c3_3_6_prev3 = -29
 dist_diff_abs_pair_c3_3_6_prev3_abs_ge_orig: - dist_diff_pair_c3_3_6_prev3
   + dist_diff_abs_pair_c3_3_6_prev3
   - 120 dist_diff_abs_pair_c3_3_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_3_6_prev3_abs_le_orig: - dist_diff_pair_c3_3_6_prev3
   + dist_diff_abs_pair_c3_3_6_prev3
   + 120 dist_diff_abs_pair_c3_3_6_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_3_6_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_3_6_prev3 + dist_diff_abs_pair_c3_3_6_prev3
   + 120 dist_diff_abs_pair_c3_3_6_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_3_6_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_3_6_prev3 + dist_diff_abs_pair_c3_3_6_prev3
   - 120 dist_diff_abs_pair_c3_3_6_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_3_6_prev3_force_positive:
   dist_diff_pair_c3_3_6_prev3
   - 120 dist_diff_abs_pair_c3_3_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_3_6_prev3_force_negative:
   dist_diff_pair_c3_3_6_prev3
   - 120 dist_diff_abs_pair_c3_3_6_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_3_6_prev3: 120 same_dist_pair_c3_3_6_prev3
   + dist_diff_abs_pair_c3_3_6_prev3 <= 129.8
 same_dist_pair_lower_c3_3_6_prev3: 120 same_dist_pair_c3_3_6_prev3
   + dist_diff_abs_pair_c3_3_6_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_3_6_prev0: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_3_6_prev0: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_3_6_prev1: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_3_6_prev1: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_3_6_prev2: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_3_6_prev2: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_3_6_prev3: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_3_6_prev3: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_3_6: diff_dist_pair_c3_3_6
   + same_dist_pair_c3_3_6_prev0 + same_dist_pair_c3_3_6_prev1
   + same_dist_pair_c3_3_6_prev2 + same_dist_pair_c3_3_6_prev3 >= 1
 dx_grid_diff_def_c3_3_7: - x_grid_3 + x_grid_7 + dx_grid_diff_c3_3_7 = 0
 dx_grid_abs_pair_c3_3_7_abs_ge_orig: dx_grid_abs_pair_c3_3_7
   - dx_grid_diff_c3_3_7 - 60 dx_grid_abs_pair_c3_3_7_is_positive >= -60
 dx_grid_abs_pair_c3_3_7_abs_le_orig: dx_grid_abs_pair_c3_3_7
   - dx_grid_diff_c3_3_7 + 60 dx_grid_abs_pair_c3_3_7_is_positive <= 60
 dx_grid_abs_pair_c3_3_7_abs_ge_neg_orig: dx_grid_abs_pair_c3_3_7
   + dx_grid_diff_c3_3_7 + 60 dx_grid_abs_pair_c3_3_7_is_positive >= 0
 dx_grid_abs_pair_c3_3_7_abs_le_neg_orig: dx_grid_abs_pair_c3_3_7
   + dx_grid_diff_c3_3_7 - 60 dx_grid_abs_pair_c3_3_7_is_positive <= 0
 dx_grid_abs_pair_c3_3_7_force_positive: dx_grid_diff_c3_3_7
   - 60 dx_grid_abs_pair_c3_3_7_is_positive >= -60
 dx_grid_abs_pair_c3_3_7_force_negative: dx_grid_diff_c3_3_7
   - 60 dx_grid_abs_pair_c3_3_7_is_positive <= -0.001
 dy_grid_diff_def_c3_3_7: - y_grid_3 + y_grid_7 + dy_grid_diff_c3_3_7 = 0
 dy_grid_abs_pair_c3_3_7_abs_ge_orig: dy_grid_abs_pair_c3_3_7
   - dy_grid_diff_c3_3_7 - 60 dy_grid_abs_pair_c3_3_7_is_positive >= -60
 dy_grid_abs_pair_c3_3_7_abs_le_orig: dy_grid_abs_pair_c3_3_7
   - dy_grid_diff_c3_3_7 + 60 dy_grid_abs_pair_c3_3_7_is_positive <= 60
 dy_grid_abs_pair_c3_3_7_abs_ge_neg_orig: dy_grid_abs_pair_c3_3_7
   + dy_grid_diff_c3_3_7 + 60 dy_grid_abs_pair_c3_3_7_is_positive >= 0
 dy_grid_abs_pair_c3_3_7_abs_le_neg_orig: dy_grid_abs_pair_c3_3_7
   + dy_grid_diff_c3_3_7 - 60 dy_grid_abs_pair_c3_3_7_is_positive <= 0
 dy_grid_abs_pair_c3_3_7_force_positive: dy_grid_diff_c3_3_7
   - 60 dy_grid_abs_pair_c3_3_7_is_positive >= -60
 dy_grid_abs_pair_c3_3_7_force_negative: dy_grid_diff_c3_3_7
   - 60 dy_grid_abs_pair_c3_3_7_is_positive <= -0.001
 dist_curr_pair_def_c3_3_7: - dx_grid_abs_pair_c3_3_7
   - dy_grid_abs_pair_c3_3_7 + dist_curr_pair_c3_3_7 = 0
 dist_diff_pair_def_c3_3_7_prev0: - dist_curr_pair_c3_3_7
   + dist_diff_pair_c3_3_7_prev0 = -28
 dist_diff_abs_pair_c3_3_7_prev0_abs_ge_orig: - dist_diff_pair_c3_3_7_prev0
   + dist_diff_abs_pair_c3_3_7_prev0
   - 120 dist_diff_abs_pair_c3_3_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_3_7_prev0_abs_le_orig: - dist_diff_pair_c3_3_7_prev0
   + dist_diff_abs_pair_c3_3_7_prev0
   + 120 dist_diff_abs_pair_c3_3_7_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_3_7_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_3_7_prev0 + dist_diff_abs_pair_c3_3_7_prev0
   + 120 dist_diff_abs_pair_c3_3_7_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_3_7_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_3_7_prev0 + dist_diff_abs_pair_c3_3_7_prev0
   - 120 dist_diff_abs_pair_c3_3_7_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_3_7_prev0_force_positive:
   dist_diff_pair_c3_3_7_prev0
   - 120 dist_diff_abs_pair_c3_3_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_3_7_prev0_force_negative:
   dist_diff_pair_c3_3_7_prev0
   - 120 dist_diff_abs_pair_c3_3_7_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_3_7_prev0: 120 same_dist_pair_c3_3_7_prev0
   + dist_diff_abs_pair_c3_3_7_prev0 <= 129.8
 same_dist_pair_lower_c3_3_7_prev0: 120 same_dist_pair_c3_3_7_prev0
   + dist_diff_abs_pair_c3_3_7_prev0 >= 10
 dist_diff_pair_def_c3_3_7_prev1: - dist_curr_pair_c3_3_7
   + dist_diff_pair_c3_3_7_prev1 = -13
 dist_diff_abs_pair_c3_3_7_prev1_abs_ge_orig: - dist_diff_pair_c3_3_7_prev1
   + dist_diff_abs_pair_c3_3_7_prev1
   - 120 dist_diff_abs_pair_c3_3_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_3_7_prev1_abs_le_orig: - dist_diff_pair_c3_3_7_prev1
   + dist_diff_abs_pair_c3_3_7_prev1
   + 120 dist_diff_abs_pair_c3_3_7_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_3_7_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_3_7_prev1 + dist_diff_abs_pair_c3_3_7_prev1
   + 120 dist_diff_abs_pair_c3_3_7_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_3_7_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_3_7_prev1 + dist_diff_abs_pair_c3_3_7_prev1
   - 120 dist_diff_abs_pair_c3_3_7_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_3_7_prev1_force_positive:
   dist_diff_pair_c3_3_7_prev1
   - 120 dist_diff_abs_pair_c3_3_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_3_7_prev1_force_negative:
   dist_diff_pair_c3_3_7_prev1
   - 120 dist_diff_abs_pair_c3_3_7_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_3_7_prev1: 120 same_dist_pair_c3_3_7_prev1
   + dist_diff_abs_pair_c3_3_7_prev1 <= 129.8
 same_dist_pair_lower_c3_3_7_prev1: 120 same_dist_pair_c3_3_7_prev1
   + dist_diff_abs_pair_c3_3_7_prev1 >= 10
 dist_diff_pair_def_c3_3_7_prev2: - dist_curr_pair_c3_3_7
   + dist_diff_pair_c3_3_7_prev2 = -21
 dist_diff_abs_pair_c3_3_7_prev2_abs_ge_orig: - dist_diff_pair_c3_3_7_prev2
   + dist_diff_abs_pair_c3_3_7_prev2
   - 120 dist_diff_abs_pair_c3_3_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_3_7_prev2_abs_le_orig: - dist_diff_pair_c3_3_7_prev2
   + dist_diff_abs_pair_c3_3_7_prev2
   + 120 dist_diff_abs_pair_c3_3_7_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_3_7_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_3_7_prev2 + dist_diff_abs_pair_c3_3_7_prev2
   + 120 dist_diff_abs_pair_c3_3_7_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_3_7_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_3_7_prev2 + dist_diff_abs_pair_c3_3_7_prev2
   - 120 dist_diff_abs_pair_c3_3_7_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_3_7_prev2_force_positive:
   dist_diff_pair_c3_3_7_prev2
   - 120 dist_diff_abs_pair_c3_3_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_3_7_prev2_force_negative:
   dist_diff_pair_c3_3_7_prev2
   - 120 dist_diff_abs_pair_c3_3_7_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_3_7_prev2: 120 same_dist_pair_c3_3_7_prev2
   + dist_diff_abs_pair_c3_3_7_prev2 <= 129.8
 same_dist_pair_lower_c3_3_7_prev2: 120 same_dist_pair_c3_3_7_prev2
   + dist_diff_abs_pair_c3_3_7_prev2 >= 10
 dist_diff_pair_def_c3_3_7_prev3: - dist_curr_pair_c3_3_7
   + dist_diff_pair_c3_3_7_prev3 = -23
 dist_diff_abs_pair_c3_3_7_prev3_abs_ge_orig: - dist_diff_pair_c3_3_7_prev3
   + dist_diff_abs_pair_c3_3_7_prev3
   - 120 dist_diff_abs_pair_c3_3_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_3_7_prev3_abs_le_orig: - dist_diff_pair_c3_3_7_prev3
   + dist_diff_abs_pair_c3_3_7_prev3
   + 120 dist_diff_abs_pair_c3_3_7_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_3_7_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_3_7_prev3 + dist_diff_abs_pair_c3_3_7_prev3
   + 120 dist_diff_abs_pair_c3_3_7_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_3_7_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_3_7_prev3 + dist_diff_abs_pair_c3_3_7_prev3
   - 120 dist_diff_abs_pair_c3_3_7_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_3_7_prev3_force_positive:
   dist_diff_pair_c3_3_7_prev3
   - 120 dist_diff_abs_pair_c3_3_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_3_7_prev3_force_negative:
   dist_diff_pair_c3_3_7_prev3
   - 120 dist_diff_abs_pair_c3_3_7_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_3_7_prev3: 120 same_dist_pair_c3_3_7_prev3
   + dist_diff_abs_pair_c3_3_7_prev3 <= 129.8
 same_dist_pair_lower_c3_3_7_prev3: 120 same_dist_pair_c3_3_7_prev3
   + dist_diff_abs_pair_c3_3_7_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_3_7_prev0: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_3_7_prev0: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_3_7_prev1: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_3_7_prev1: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_3_7_prev2: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_3_7_prev2: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_3_7_prev3: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_3_7_prev3: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_3_7: diff_dist_pair_c3_3_7
   + same_dist_pair_c3_3_7_prev0 + same_dist_pair_c3_3_7_prev1
   + same_dist_pair_c3_3_7_prev2 + same_dist_pair_c3_3_7_prev3 >= 1
 dx_grid_diff_def_c3_4_5: - x_grid_4 + x_grid_5 + dx_grid_diff_c3_4_5 = 0
 dx_grid_abs_pair_c3_4_5_abs_ge_orig: dx_grid_abs_pair_c3_4_5
   - dx_grid_diff_c3_4_5 - 60 dx_grid_abs_pair_c3_4_5_is_positive >= -60
 dx_grid_abs_pair_c3_4_5_abs_le_orig: dx_grid_abs_pair_c3_4_5
   - dx_grid_diff_c3_4_5 + 60 dx_grid_abs_pair_c3_4_5_is_positive <= 60
 dx_grid_abs_pair_c3_4_5_abs_ge_neg_orig: dx_grid_abs_pair_c3_4_5
   + dx_grid_diff_c3_4_5 + 60 dx_grid_abs_pair_c3_4_5_is_positive >= 0
 dx_grid_abs_pair_c3_4_5_abs_le_neg_orig: dx_grid_abs_pair_c3_4_5
   + dx_grid_diff_c3_4_5 - 60 dx_grid_abs_pair_c3_4_5_is_positive <= 0
 dx_grid_abs_pair_c3_4_5_force_positive: dx_grid_diff_c3_4_5
   - 60 dx_grid_abs_pair_c3_4_5_is_positive >= -60
 dx_grid_abs_pair_c3_4_5_force_negative: dx_grid_diff_c3_4_5
   - 60 dx_grid_abs_pair_c3_4_5_is_positive <= -0.001
 dy_grid_diff_def_c3_4_5: - y_grid_4 + y_grid_5 + dy_grid_diff_c3_4_5 = 0
 dy_grid_abs_pair_c3_4_5_abs_ge_orig: dy_grid_abs_pair_c3_4_5
   - dy_grid_diff_c3_4_5 - 60 dy_grid_abs_pair_c3_4_5_is_positive >= -60
 dy_grid_abs_pair_c3_4_5_abs_le_orig: dy_grid_abs_pair_c3_4_5
   - dy_grid_diff_c3_4_5 + 60 dy_grid_abs_pair_c3_4_5_is_positive <= 60
 dy_grid_abs_pair_c3_4_5_abs_ge_neg_orig: dy_grid_abs_pair_c3_4_5
   + dy_grid_diff_c3_4_5 + 60 dy_grid_abs_pair_c3_4_5_is_positive >= 0
 dy_grid_abs_pair_c3_4_5_abs_le_neg_orig: dy_grid_abs_pair_c3_4_5
   + dy_grid_diff_c3_4_5 - 60 dy_grid_abs_pair_c3_4_5_is_positive <= 0
 dy_grid_abs_pair_c3_4_5_force_positive: dy_grid_diff_c3_4_5
   - 60 dy_grid_abs_pair_c3_4_5_is_positive >= -60
 dy_grid_abs_pair_c3_4_5_force_negative: dy_grid_diff_c3_4_5
   - 60 dy_grid_abs_pair_c3_4_5_is_positive <= -0.001
 dist_curr_pair_def_c3_4_5: - dx_grid_abs_pair_c3_4_5
   - dy_grid_abs_pair_c3_4_5 + dist_curr_pair_c3_4_5 = 0
 dist_diff_pair_def_c3_4_5_prev0: - dist_curr_pair_c3_4_5
   + dist_diff_pair_c3_4_5_prev0 = -26
 dist_diff_abs_pair_c3_4_5_prev0_abs_ge_orig: - dist_diff_pair_c3_4_5_prev0
   + dist_diff_abs_pair_c3_4_5_prev0
   - 120 dist_diff_abs_pair_c3_4_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_4_5_prev0_abs_le_orig: - dist_diff_pair_c3_4_5_prev0
   + dist_diff_abs_pair_c3_4_5_prev0
   + 120 dist_diff_abs_pair_c3_4_5_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_4_5_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_4_5_prev0 + dist_diff_abs_pair_c3_4_5_prev0
   + 120 dist_diff_abs_pair_c3_4_5_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_4_5_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_4_5_prev0 + dist_diff_abs_pair_c3_4_5_prev0
   - 120 dist_diff_abs_pair_c3_4_5_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_4_5_prev0_force_positive:
   dist_diff_pair_c3_4_5_prev0
   - 120 dist_diff_abs_pair_c3_4_5_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_4_5_prev0_force_negative:
   dist_diff_pair_c3_4_5_prev0
   - 120 dist_diff_abs_pair_c3_4_5_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_4_5_prev0: 120 same_dist_pair_c3_4_5_prev0
   + dist_diff_abs_pair_c3_4_5_prev0 <= 129.8
 same_dist_pair_lower_c3_4_5_prev0: 120 same_dist_pair_c3_4_5_prev0
   + dist_diff_abs_pair_c3_4_5_prev0 >= 10
 dist_diff_pair_def_c3_4_5_prev1: - dist_curr_pair_c3_4_5
   + dist_diff_pair_c3_4_5_prev1 = -19
 dist_diff_abs_pair_c3_4_5_prev1_abs_ge_orig: - dist_diff_pair_c3_4_5_prev1
   + dist_diff_abs_pair_c3_4_5_prev1
   - 120 dist_diff_abs_pair_c3_4_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_4_5_prev1_abs_le_orig: - dist_diff_pair_c3_4_5_prev1
   + dist_diff_abs_pair_c3_4_5_prev1
   + 120 dist_diff_abs_pair_c3_4_5_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_4_5_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_4_5_prev1 + dist_diff_abs_pair_c3_4_5_prev1
   + 120 dist_diff_abs_pair_c3_4_5_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_4_5_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_4_5_prev1 + dist_diff_abs_pair_c3_4_5_prev1
   - 120 dist_diff_abs_pair_c3_4_5_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_4_5_prev1_force_positive:
   dist_diff_pair_c3_4_5_prev1
   - 120 dist_diff_abs_pair_c3_4_5_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_4_5_prev1_force_negative:
   dist_diff_pair_c3_4_5_prev1
   - 120 dist_diff_abs_pair_c3_4_5_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_4_5_prev1: 120 same_dist_pair_c3_4_5_prev1
   + dist_diff_abs_pair_c3_4_5_prev1 <= 129.8
 same_dist_pair_lower_c3_4_5_prev1: 120 same_dist_pair_c3_4_5_prev1
   + dist_diff_abs_pair_c3_4_5_prev1 >= 10
 dist_diff_pair_def_c3_4_5_prev2: - dist_curr_pair_c3_4_5
   + dist_diff_pair_c3_4_5_prev2 = -29
 dist_diff_abs_pair_c3_4_5_prev2_abs_ge_orig: - dist_diff_pair_c3_4_5_prev2
   + dist_diff_abs_pair_c3_4_5_prev2
   - 120 dist_diff_abs_pair_c3_4_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_4_5_prev2_abs_le_orig: - dist_diff_pair_c3_4_5_prev2
   + dist_diff_abs_pair_c3_4_5_prev2
   + 120 dist_diff_abs_pair_c3_4_5_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_4_5_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_4_5_prev2 + dist_diff_abs_pair_c3_4_5_prev2
   + 120 dist_diff_abs_pair_c3_4_5_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_4_5_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_4_5_prev2 + dist_diff_abs_pair_c3_4_5_prev2
   - 120 dist_diff_abs_pair_c3_4_5_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_4_5_prev2_force_positive:
   dist_diff_pair_c3_4_5_prev2
   - 120 dist_diff_abs_pair_c3_4_5_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_4_5_prev2_force_negative:
   dist_diff_pair_c3_4_5_prev2
   - 120 dist_diff_abs_pair_c3_4_5_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_4_5_prev2: 120 same_dist_pair_c3_4_5_prev2
   + dist_diff_abs_pair_c3_4_5_prev2 <= 129.8
 same_dist_pair_lower_c3_4_5_prev2: 120 same_dist_pair_c3_4_5_prev2
   + dist_diff_abs_pair_c3_4_5_prev2 >= 10
 dist_diff_pair_def_c3_4_5_prev3: - dist_curr_pair_c3_4_5
   + dist_diff_pair_c3_4_5_prev3 = -29
 dist_diff_abs_pair_c3_4_5_prev3_abs_ge_orig: - dist_diff_pair_c3_4_5_prev3
   + dist_diff_abs_pair_c3_4_5_prev3
   - 120 dist_diff_abs_pair_c3_4_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_4_5_prev3_abs_le_orig: - dist_diff_pair_c3_4_5_prev3
   + dist_diff_abs_pair_c3_4_5_prev3
   + 120 dist_diff_abs_pair_c3_4_5_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_4_5_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_4_5_prev3 + dist_diff_abs_pair_c3_4_5_prev3
   + 120 dist_diff_abs_pair_c3_4_5_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_4_5_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_4_5_prev3 + dist_diff_abs_pair_c3_4_5_prev3
   - 120 dist_diff_abs_pair_c3_4_5_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_4_5_prev3_force_positive:
   dist_diff_pair_c3_4_5_prev3
   - 120 dist_diff_abs_pair_c3_4_5_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_4_5_prev3_force_negative:
   dist_diff_pair_c3_4_5_prev3
   - 120 dist_diff_abs_pair_c3_4_5_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_4_5_prev3: 120 same_dist_pair_c3_4_5_prev3
   + dist_diff_abs_pair_c3_4_5_prev3 <= 129.8
 same_dist_pair_lower_c3_4_5_prev3: 120 same_dist_pair_c3_4_5_prev3
   + dist_diff_abs_pair_c3_4_5_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_4_5_prev0: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_4_5_prev0: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_4_5_prev1: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_4_5_prev1: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_4_5_prev2: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_4_5_prev2: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_4_5_prev3: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_4_5_prev3: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_4_5: diff_dist_pair_c3_4_5
   + same_dist_pair_c3_4_5_prev0 + same_dist_pair_c3_4_5_prev1
   + same_dist_pair_c3_4_5_prev2 + same_dist_pair_c3_4_5_prev3 >= 1
 dx_grid_diff_def_c3_4_6: - x_grid_4 + x_grid_6 + dx_grid_diff_c3_4_6 = 0
 dx_grid_abs_pair_c3_4_6_abs_ge_orig: dx_grid_abs_pair_c3_4_6
   - dx_grid_diff_c3_4_6 - 60 dx_grid_abs_pair_c3_4_6_is_positive >= -60
 dx_grid_abs_pair_c3_4_6_abs_le_orig: dx_grid_abs_pair_c3_4_6
   - dx_grid_diff_c3_4_6 + 60 dx_grid_abs_pair_c3_4_6_is_positive <= 60
 dx_grid_abs_pair_c3_4_6_abs_ge_neg_orig: dx_grid_abs_pair_c3_4_6
   + dx_grid_diff_c3_4_6 + 60 dx_grid_abs_pair_c3_4_6_is_positive >= 0
 dx_grid_abs_pair_c3_4_6_abs_le_neg_orig: dx_grid_abs_pair_c3_4_6
   + dx_grid_diff_c3_4_6 - 60 dx_grid_abs_pair_c3_4_6_is_positive <= 0
 dx_grid_abs_pair_c3_4_6_force_positive: dx_grid_diff_c3_4_6
   - 60 dx_grid_abs_pair_c3_4_6_is_positive >= -60
 dx_grid_abs_pair_c3_4_6_force_negative: dx_grid_diff_c3_4_6
   - 60 dx_grid_abs_pair_c3_4_6_is_positive <= -0.001
 dy_grid_diff_def_c3_4_6: - y_grid_4 + y_grid_6 + dy_grid_diff_c3_4_6 = 0
 dy_grid_abs_pair_c3_4_6_abs_ge_orig: dy_grid_abs_pair_c3_4_6
   - dy_grid_diff_c3_4_6 - 60 dy_grid_abs_pair_c3_4_6_is_positive >= -60
 dy_grid_abs_pair_c3_4_6_abs_le_orig: dy_grid_abs_pair_c3_4_6
   - dy_grid_diff_c3_4_6 + 60 dy_grid_abs_pair_c3_4_6_is_positive <= 60
 dy_grid_abs_pair_c3_4_6_abs_ge_neg_orig: dy_grid_abs_pair_c3_4_6
   + dy_grid_diff_c3_4_6 + 60 dy_grid_abs_pair_c3_4_6_is_positive >= 0
 dy_grid_abs_pair_c3_4_6_abs_le_neg_orig: dy_grid_abs_pair_c3_4_6
   + dy_grid_diff_c3_4_6 - 60 dy_grid_abs_pair_c3_4_6_is_positive <= 0
 dy_grid_abs_pair_c3_4_6_force_positive: dy_grid_diff_c3_4_6
   - 60 dy_grid_abs_pair_c3_4_6_is_positive >= -60
 dy_grid_abs_pair_c3_4_6_force_negative: dy_grid_diff_c3_4_6
   - 60 dy_grid_abs_pair_c3_4_6_is_positive <= -0.001
 dist_curr_pair_def_c3_4_6: - dx_grid_abs_pair_c3_4_6
   - dy_grid_abs_pair_c3_4_6 + dist_curr_pair_c3_4_6 = 0
 dist_diff_pair_def_c3_4_6_prev0: - dist_curr_pair_c3_4_6
   + dist_diff_pair_c3_4_6_prev0 = -42
 dist_diff_abs_pair_c3_4_6_prev0_abs_ge_orig: - dist_diff_pair_c3_4_6_prev0
   + dist_diff_abs_pair_c3_4_6_prev0
   - 120 dist_diff_abs_pair_c3_4_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_4_6_prev0_abs_le_orig: - dist_diff_pair_c3_4_6_prev0
   + dist_diff_abs_pair_c3_4_6_prev0
   + 120 dist_diff_abs_pair_c3_4_6_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_4_6_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_4_6_prev0 + dist_diff_abs_pair_c3_4_6_prev0
   + 120 dist_diff_abs_pair_c3_4_6_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_4_6_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_4_6_prev0 + dist_diff_abs_pair_c3_4_6_prev0
   - 120 dist_diff_abs_pair_c3_4_6_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_4_6_prev0_force_positive:
   dist_diff_pair_c3_4_6_prev0
   - 120 dist_diff_abs_pair_c3_4_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_4_6_prev0_force_negative:
   dist_diff_pair_c3_4_6_prev0
   - 120 dist_diff_abs_pair_c3_4_6_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_4_6_prev0: 120 same_dist_pair_c3_4_6_prev0
   + dist_diff_abs_pair_c3_4_6_prev0 <= 129.8
 same_dist_pair_lower_c3_4_6_prev0: 120 same_dist_pair_c3_4_6_prev0
   + dist_diff_abs_pair_c3_4_6_prev0 >= 10
 dist_diff_pair_def_c3_4_6_prev1: - dist_curr_pair_c3_4_6
   + dist_diff_pair_c3_4_6_prev1 = -29
 dist_diff_abs_pair_c3_4_6_prev1_abs_ge_orig: - dist_diff_pair_c3_4_6_prev1
   + dist_diff_abs_pair_c3_4_6_prev1
   - 120 dist_diff_abs_pair_c3_4_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_4_6_prev1_abs_le_orig: - dist_diff_pair_c3_4_6_prev1
   + dist_diff_abs_pair_c3_4_6_prev1
   + 120 dist_diff_abs_pair_c3_4_6_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_4_6_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_4_6_prev1 + dist_diff_abs_pair_c3_4_6_prev1
   + 120 dist_diff_abs_pair_c3_4_6_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_4_6_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_4_6_prev1 + dist_diff_abs_pair_c3_4_6_prev1
   - 120 dist_diff_abs_pair_c3_4_6_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_4_6_prev1_force_positive:
   dist_diff_pair_c3_4_6_prev1
   - 120 dist_diff_abs_pair_c3_4_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_4_6_prev1_force_negative:
   dist_diff_pair_c3_4_6_prev1
   - 120 dist_diff_abs_pair_c3_4_6_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_4_6_prev1: 120 same_dist_pair_c3_4_6_prev1
   + dist_diff_abs_pair_c3_4_6_prev1 <= 129.8
 same_dist_pair_lower_c3_4_6_prev1: 120 same_dist_pair_c3_4_6_prev1
   + dist_diff_abs_pair_c3_4_6_prev1 >= 10
 dist_diff_pair_def_c3_4_6_prev2: - dist_curr_pair_c3_4_6
   + dist_diff_pair_c3_4_6_prev2 = -36
 dist_diff_abs_pair_c3_4_6_prev2_abs_ge_orig: - dist_diff_pair_c3_4_6_prev2
   + dist_diff_abs_pair_c3_4_6_prev2
   - 120 dist_diff_abs_pair_c3_4_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_4_6_prev2_abs_le_orig: - dist_diff_pair_c3_4_6_prev2
   + dist_diff_abs_pair_c3_4_6_prev2
   + 120 dist_diff_abs_pair_c3_4_6_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_4_6_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_4_6_prev2 + dist_diff_abs_pair_c3_4_6_prev2
   + 120 dist_diff_abs_pair_c3_4_6_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_4_6_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_4_6_prev2 + dist_diff_abs_pair_c3_4_6_prev2
   - 120 dist_diff_abs_pair_c3_4_6_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_4_6_prev2_force_positive:
   dist_diff_pair_c3_4_6_prev2
   - 120 dist_diff_abs_pair_c3_4_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_4_6_prev2_force_negative:
   dist_diff_pair_c3_4_6_prev2
   - 120 dist_diff_abs_pair_c3_4_6_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_4_6_prev2: 120 same_dist_pair_c3_4_6_prev2
   + dist_diff_abs_pair_c3_4_6_prev2 <= 129.8
 same_dist_pair_lower_c3_4_6_prev2: 120 same_dist_pair_c3_4_6_prev2
   + dist_diff_abs_pair_c3_4_6_prev2 >= 10
 dist_diff_pair_def_c3_4_6_prev3: - dist_curr_pair_c3_4_6
   + dist_diff_pair_c3_4_6_prev3 = -19
 dist_diff_abs_pair_c3_4_6_prev3_abs_ge_orig: - dist_diff_pair_c3_4_6_prev3
   + dist_diff_abs_pair_c3_4_6_prev3
   - 120 dist_diff_abs_pair_c3_4_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_4_6_prev3_abs_le_orig: - dist_diff_pair_c3_4_6_prev3
   + dist_diff_abs_pair_c3_4_6_prev3
   + 120 dist_diff_abs_pair_c3_4_6_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_4_6_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_4_6_prev3 + dist_diff_abs_pair_c3_4_6_prev3
   + 120 dist_diff_abs_pair_c3_4_6_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_4_6_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_4_6_prev3 + dist_diff_abs_pair_c3_4_6_prev3
   - 120 dist_diff_abs_pair_c3_4_6_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_4_6_prev3_force_positive:
   dist_diff_pair_c3_4_6_prev3
   - 120 dist_diff_abs_pair_c3_4_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_4_6_prev3_force_negative:
   dist_diff_pair_c3_4_6_prev3
   - 120 dist_diff_abs_pair_c3_4_6_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_4_6_prev3: 120 same_dist_pair_c3_4_6_prev3
   + dist_diff_abs_pair_c3_4_6_prev3 <= 129.8
 same_dist_pair_lower_c3_4_6_prev3: 120 same_dist_pair_c3_4_6_prev3
   + dist_diff_abs_pair_c3_4_6_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_4_6_prev0: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_4_6_prev0: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_4_6_prev1: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_4_6_prev1: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_4_6_prev2: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_4_6_prev2: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_4_6_prev3: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_4_6_prev3: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_4_6: diff_dist_pair_c3_4_6
   + same_dist_pair_c3_4_6_prev0 + same_dist_pair_c3_4_6_prev1
   + same_dist_pair_c3_4_6_prev2 + same_dist_pair_c3_4_6_prev3 >= 1
 dx_grid_diff_def_c3_4_7: - x_grid_4 + x_grid_7 + dx_grid_diff_c3_4_7 = 0
 dx_grid_abs_pair_c3_4_7_abs_ge_orig: dx_grid_abs_pair_c3_4_7
   - dx_grid_diff_c3_4_7 - 60 dx_grid_abs_pair_c3_4_7_is_positive >= -60
 dx_grid_abs_pair_c3_4_7_abs_le_orig: dx_grid_abs_pair_c3_4_7
   - dx_grid_diff_c3_4_7 + 60 dx_grid_abs_pair_c3_4_7_is_positive <= 60
 dx_grid_abs_pair_c3_4_7_abs_ge_neg_orig: dx_grid_abs_pair_c3_4_7
   + dx_grid_diff_c3_4_7 + 60 dx_grid_abs_pair_c3_4_7_is_positive >= 0
 dx_grid_abs_pair_c3_4_7_abs_le_neg_orig: dx_grid_abs_pair_c3_4_7
   + dx_grid_diff_c3_4_7 - 60 dx_grid_abs_pair_c3_4_7_is_positive <= 0
 dx_grid_abs_pair_c3_4_7_force_positive: dx_grid_diff_c3_4_7
   - 60 dx_grid_abs_pair_c3_4_7_is_positive >= -60
 dx_grid_abs_pair_c3_4_7_force_negative: dx_grid_diff_c3_4_7
   - 60 dx_grid_abs_pair_c3_4_7_is_positive <= -0.001
 dy_grid_diff_def_c3_4_7: - y_grid_4 + y_grid_7 + dy_grid_diff_c3_4_7 = 0
 dy_grid_abs_pair_c3_4_7_abs_ge_orig: dy_grid_abs_pair_c3_4_7
   - dy_grid_diff_c3_4_7 - 60 dy_grid_abs_pair_c3_4_7_is_positive >= -60
 dy_grid_abs_pair_c3_4_7_abs_le_orig: dy_grid_abs_pair_c3_4_7
   - dy_grid_diff_c3_4_7 + 60 dy_grid_abs_pair_c3_4_7_is_positive <= 60
 dy_grid_abs_pair_c3_4_7_abs_ge_neg_orig: dy_grid_abs_pair_c3_4_7
   + dy_grid_diff_c3_4_7 + 60 dy_grid_abs_pair_c3_4_7_is_positive >= 0
 dy_grid_abs_pair_c3_4_7_abs_le_neg_orig: dy_grid_abs_pair_c3_4_7
   + dy_grid_diff_c3_4_7 - 60 dy_grid_abs_pair_c3_4_7_is_positive <= 0
 dy_grid_abs_pair_c3_4_7_force_positive: dy_grid_diff_c3_4_7
   - 60 dy_grid_abs_pair_c3_4_7_is_positive >= -60
 dy_grid_abs_pair_c3_4_7_force_negative: dy_grid_diff_c3_4_7
   - 60 dy_grid_abs_pair_c3_4_7_is_positive <= -0.001
 dist_curr_pair_def_c3_4_7: - dx_grid_abs_pair_c3_4_7
   - dy_grid_abs_pair_c3_4_7 + dist_curr_pair_c3_4_7 = 0
 dist_diff_pair_def_c3_4_7_prev0: - dist_curr_pair_c3_4_7
   + dist_diff_pair_c3_4_7_prev0 = -44
 dist_diff_abs_pair_c3_4_7_prev0_abs_ge_orig: - dist_diff_pair_c3_4_7_prev0
   + dist_diff_abs_pair_c3_4_7_prev0
   - 120 dist_diff_abs_pair_c3_4_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_4_7_prev0_abs_le_orig: - dist_diff_pair_c3_4_7_prev0
   + dist_diff_abs_pair_c3_4_7_prev0
   + 120 dist_diff_abs_pair_c3_4_7_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_4_7_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_4_7_prev0 + dist_diff_abs_pair_c3_4_7_prev0
   + 120 dist_diff_abs_pair_c3_4_7_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_4_7_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_4_7_prev0 + dist_diff_abs_pair_c3_4_7_prev0
   - 120 dist_diff_abs_pair_c3_4_7_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_4_7_prev0_force_positive:
   dist_diff_pair_c3_4_7_prev0
   - 120 dist_diff_abs_pair_c3_4_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_4_7_prev0_force_negative:
   dist_diff_pair_c3_4_7_prev0
   - 120 dist_diff_abs_pair_c3_4_7_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_4_7_prev0: 120 same_dist_pair_c3_4_7_prev0
   + dist_diff_abs_pair_c3_4_7_prev0 <= 129.8
 same_dist_pair_lower_c3_4_7_prev0: 120 same_dist_pair_c3_4_7_prev0
   + dist_diff_abs_pair_c3_4_7_prev0 >= 10
 dist_diff_pair_def_c3_4_7_prev1: - dist_curr_pair_c3_4_7
   + dist_diff_pair_c3_4_7_prev1 = -23
 dist_diff_abs_pair_c3_4_7_prev1_abs_ge_orig: - dist_diff_pair_c3_4_7_prev1
   + dist_diff_abs_pair_c3_4_7_prev1
   - 120 dist_diff_abs_pair_c3_4_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_4_7_prev1_abs_le_orig: - dist_diff_pair_c3_4_7_prev1
   + dist_diff_abs_pair_c3_4_7_prev1
   + 120 dist_diff_abs_pair_c3_4_7_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_4_7_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_4_7_prev1 + dist_diff_abs_pair_c3_4_7_prev1
   + 120 dist_diff_abs_pair_c3_4_7_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_4_7_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_4_7_prev1 + dist_diff_abs_pair_c3_4_7_prev1
   - 120 dist_diff_abs_pair_c3_4_7_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_4_7_prev1_force_positive:
   dist_diff_pair_c3_4_7_prev1
   - 120 dist_diff_abs_pair_c3_4_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_4_7_prev1_force_negative:
   dist_diff_pair_c3_4_7_prev1
   - 120 dist_diff_abs_pair_c3_4_7_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_4_7_prev1: 120 same_dist_pair_c3_4_7_prev1
   + dist_diff_abs_pair_c3_4_7_prev1 <= 129.8
 same_dist_pair_lower_c3_4_7_prev1: 120 same_dist_pair_c3_4_7_prev1
   + dist_diff_abs_pair_c3_4_7_prev1 >= 10
 dist_diff_pair_def_c3_4_7_prev2: - dist_curr_pair_c3_4_7
   + dist_diff_pair_c3_4_7_prev2 = -40
 dist_diff_abs_pair_c3_4_7_prev2_abs_ge_orig: - dist_diff_pair_c3_4_7_prev2
   + dist_diff_abs_pair_c3_4_7_prev2
   - 120 dist_diff_abs_pair_c3_4_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_4_7_prev2_abs_le_orig: - dist_diff_pair_c3_4_7_prev2
   + dist_diff_abs_pair_c3_4_7_prev2
   + 120 dist_diff_abs_pair_c3_4_7_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_4_7_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_4_7_prev2 + dist_diff_abs_pair_c3_4_7_prev2
   + 120 dist_diff_abs_pair_c3_4_7_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_4_7_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_4_7_prev2 + dist_diff_abs_pair_c3_4_7_prev2
   - 120 dist_diff_abs_pair_c3_4_7_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_4_7_prev2_force_positive:
   dist_diff_pair_c3_4_7_prev2
   - 120 dist_diff_abs_pair_c3_4_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_4_7_prev2_force_negative:
   dist_diff_pair_c3_4_7_prev2
   - 120 dist_diff_abs_pair_c3_4_7_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_4_7_prev2: 120 same_dist_pair_c3_4_7_prev2
   + dist_diff_abs_pair_c3_4_7_prev2 <= 129.8
 same_dist_pair_lower_c3_4_7_prev2: 120 same_dist_pair_c3_4_7_prev2
   + dist_diff_abs_pair_c3_4_7_prev2 >= 10
 dist_diff_pair_def_c3_4_7_prev3: - dist_curr_pair_c3_4_7
   + dist_diff_pair_c3_4_7_prev3 = -13
 dist_diff_abs_pair_c3_4_7_prev3_abs_ge_orig: - dist_diff_pair_c3_4_7_prev3
   + dist_diff_abs_pair_c3_4_7_prev3
   - 120 dist_diff_abs_pair_c3_4_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_4_7_prev3_abs_le_orig: - dist_diff_pair_c3_4_7_prev3
   + dist_diff_abs_pair_c3_4_7_prev3
   + 120 dist_diff_abs_pair_c3_4_7_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_4_7_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_4_7_prev3 + dist_diff_abs_pair_c3_4_7_prev3
   + 120 dist_diff_abs_pair_c3_4_7_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_4_7_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_4_7_prev3 + dist_diff_abs_pair_c3_4_7_prev3
   - 120 dist_diff_abs_pair_c3_4_7_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_4_7_prev3_force_positive:
   dist_diff_pair_c3_4_7_prev3
   - 120 dist_diff_abs_pair_c3_4_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_4_7_prev3_force_negative:
   dist_diff_pair_c3_4_7_prev3
   - 120 dist_diff_abs_pair_c3_4_7_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_4_7_prev3: 120 same_dist_pair_c3_4_7_prev3
   + dist_diff_abs_pair_c3_4_7_prev3 <= 129.8
 same_dist_pair_lower_c3_4_7_prev3: 120 same_dist_pair_c3_4_7_prev3
   + dist_diff_abs_pair_c3_4_7_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_4_7_prev0: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_4_7_prev0: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_4_7_prev1: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_4_7_prev1: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_4_7_prev2: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_4_7_prev2: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_4_7_prev3: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_4_7_prev3: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_4_7: diff_dist_pair_c3_4_7
   + same_dist_pair_c3_4_7_prev0 + same_dist_pair_c3_4_7_prev1
   + same_dist_pair_c3_4_7_prev2 + same_dist_pair_c3_4_7_prev3 >= 1
 dx_grid_diff_def_c3_5_6: - x_grid_5 + x_grid_6 + dx_grid_diff_c3_5_6 = 0
 dx_grid_abs_pair_c3_5_6_abs_ge_orig: dx_grid_abs_pair_c3_5_6
   - dx_grid_diff_c3_5_6 - 60 dx_grid_abs_pair_c3_5_6_is_positive >= -60
 dx_grid_abs_pair_c3_5_6_abs_le_orig: dx_grid_abs_pair_c3_5_6
   - dx_grid_diff_c3_5_6 + 60 dx_grid_abs_pair_c3_5_6_is_positive <= 60
 dx_grid_abs_pair_c3_5_6_abs_ge_neg_orig: dx_grid_abs_pair_c3_5_6
   + dx_grid_diff_c3_5_6 + 60 dx_grid_abs_pair_c3_5_6_is_positive >= 0
 dx_grid_abs_pair_c3_5_6_abs_le_neg_orig: dx_grid_abs_pair_c3_5_6
   + dx_grid_diff_c3_5_6 - 60 dx_grid_abs_pair_c3_5_6_is_positive <= 0
 dx_grid_abs_pair_c3_5_6_force_positive: dx_grid_diff_c3_5_6
   - 60 dx_grid_abs_pair_c3_5_6_is_positive >= -60
 dx_grid_abs_pair_c3_5_6_force_negative: dx_grid_diff_c3_5_6
   - 60 dx_grid_abs_pair_c3_5_6_is_positive <= -0.001
 dy_grid_diff_def_c3_5_6: - y_grid_5 + y_grid_6 + dy_grid_diff_c3_5_6 = 0
 dy_grid_abs_pair_c3_5_6_abs_ge_orig: dy_grid_abs_pair_c3_5_6
   - dy_grid_diff_c3_5_6 - 60 dy_grid_abs_pair_c3_5_6_is_positive >= -60
 dy_grid_abs_pair_c3_5_6_abs_le_orig: dy_grid_abs_pair_c3_5_6
   - dy_grid_diff_c3_5_6 + 60 dy_grid_abs_pair_c3_5_6_is_positive <= 60
 dy_grid_abs_pair_c3_5_6_abs_ge_neg_orig: dy_grid_abs_pair_c3_5_6
   + dy_grid_diff_c3_5_6 + 60 dy_grid_abs_pair_c3_5_6_is_positive >= 0
 dy_grid_abs_pair_c3_5_6_abs_le_neg_orig: dy_grid_abs_pair_c3_5_6
   + dy_grid_diff_c3_5_6 - 60 dy_grid_abs_pair_c3_5_6_is_positive <= 0
 dy_grid_abs_pair_c3_5_6_force_positive: dy_grid_diff_c3_5_6
   - 60 dy_grid_abs_pair_c3_5_6_is_positive >= -60
 dy_grid_abs_pair_c3_5_6_force_negative: dy_grid_diff_c3_5_6
   - 60 dy_grid_abs_pair_c3_5_6_is_positive <= -0.001
 dist_curr_pair_def_c3_5_6: - dx_grid_abs_pair_c3_5_6
   - dy_grid_abs_pair_c3_5_6 + dist_curr_pair_c3_5_6 = 0
 dist_diff_pair_def_c3_5_6_prev0: - dist_curr_pair_c3_5_6
   + dist_diff_pair_c3_5_6_prev0 = -36
 dist_diff_abs_pair_c3_5_6_prev0_abs_ge_orig: - dist_diff_pair_c3_5_6_prev0
   + dist_diff_abs_pair_c3_5_6_prev0
   - 120 dist_diff_abs_pair_c3_5_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_5_6_prev0_abs_le_orig: - dist_diff_pair_c3_5_6_prev0
   + dist_diff_abs_pair_c3_5_6_prev0
   + 120 dist_diff_abs_pair_c3_5_6_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_5_6_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_5_6_prev0 + dist_diff_abs_pair_c3_5_6_prev0
   + 120 dist_diff_abs_pair_c3_5_6_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_5_6_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_5_6_prev0 + dist_diff_abs_pair_c3_5_6_prev0
   - 120 dist_diff_abs_pair_c3_5_6_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_5_6_prev0_force_positive:
   dist_diff_pair_c3_5_6_prev0
   - 120 dist_diff_abs_pair_c3_5_6_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_5_6_prev0_force_negative:
   dist_diff_pair_c3_5_6_prev0
   - 120 dist_diff_abs_pair_c3_5_6_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_5_6_prev0: 120 same_dist_pair_c3_5_6_prev0
   + dist_diff_abs_pair_c3_5_6_prev0 <= 129.8
 same_dist_pair_lower_c3_5_6_prev0: 120 same_dist_pair_c3_5_6_prev0
   + dist_diff_abs_pair_c3_5_6_prev0 >= 10
 dist_diff_pair_def_c3_5_6_prev1: - dist_curr_pair_c3_5_6
   + dist_diff_pair_c3_5_6_prev1 = -30
 dist_diff_abs_pair_c3_5_6_prev1_abs_ge_orig: - dist_diff_pair_c3_5_6_prev1
   + dist_diff_abs_pair_c3_5_6_prev1
   - 120 dist_diff_abs_pair_c3_5_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_5_6_prev1_abs_le_orig: - dist_diff_pair_c3_5_6_prev1
   + dist_diff_abs_pair_c3_5_6_prev1
   + 120 dist_diff_abs_pair_c3_5_6_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_5_6_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_5_6_prev1 + dist_diff_abs_pair_c3_5_6_prev1
   + 120 dist_diff_abs_pair_c3_5_6_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_5_6_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_5_6_prev1 + dist_diff_abs_pair_c3_5_6_prev1
   - 120 dist_diff_abs_pair_c3_5_6_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_5_6_prev1_force_positive:
   dist_diff_pair_c3_5_6_prev1
   - 120 dist_diff_abs_pair_c3_5_6_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_5_6_prev1_force_negative:
   dist_diff_pair_c3_5_6_prev1
   - 120 dist_diff_abs_pair_c3_5_6_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_5_6_prev1: 120 same_dist_pair_c3_5_6_prev1
   + dist_diff_abs_pair_c3_5_6_prev1 <= 129.8
 same_dist_pair_lower_c3_5_6_prev1: 120 same_dist_pair_c3_5_6_prev1
   + dist_diff_abs_pair_c3_5_6_prev1 >= 10
 dist_diff_pair_def_c3_5_6_prev2: - dist_curr_pair_c3_5_6
   + dist_diff_pair_c3_5_6_prev2 = -25
 dist_diff_abs_pair_c3_5_6_prev2_abs_ge_orig: - dist_diff_pair_c3_5_6_prev2
   + dist_diff_abs_pair_c3_5_6_prev2
   - 120 dist_diff_abs_pair_c3_5_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_5_6_prev2_abs_le_orig: - dist_diff_pair_c3_5_6_prev2
   + dist_diff_abs_pair_c3_5_6_prev2
   + 120 dist_diff_abs_pair_c3_5_6_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_5_6_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_5_6_prev2 + dist_diff_abs_pair_c3_5_6_prev2
   + 120 dist_diff_abs_pair_c3_5_6_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_5_6_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_5_6_prev2 + dist_diff_abs_pair_c3_5_6_prev2
   - 120 dist_diff_abs_pair_c3_5_6_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_5_6_prev2_force_positive:
   dist_diff_pair_c3_5_6_prev2
   - 120 dist_diff_abs_pair_c3_5_6_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_5_6_prev2_force_negative:
   dist_diff_pair_c3_5_6_prev2
   - 120 dist_diff_abs_pair_c3_5_6_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_5_6_prev2: 120 same_dist_pair_c3_5_6_prev2
   + dist_diff_abs_pair_c3_5_6_prev2 <= 129.8
 same_dist_pair_lower_c3_5_6_prev2: 120 same_dist_pair_c3_5_6_prev2
   + dist_diff_abs_pair_c3_5_6_prev2 >= 10
 dist_diff_pair_def_c3_5_6_prev3: - dist_curr_pair_c3_5_6
   + dist_diff_pair_c3_5_6_prev3 = -30
 dist_diff_abs_pair_c3_5_6_prev3_abs_ge_orig: - dist_diff_pair_c3_5_6_prev3
   + dist_diff_abs_pair_c3_5_6_prev3
   - 120 dist_diff_abs_pair_c3_5_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_5_6_prev3_abs_le_orig: - dist_diff_pair_c3_5_6_prev3
   + dist_diff_abs_pair_c3_5_6_prev3
   + 120 dist_diff_abs_pair_c3_5_6_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_5_6_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_5_6_prev3 + dist_diff_abs_pair_c3_5_6_prev3
   + 120 dist_diff_abs_pair_c3_5_6_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_5_6_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_5_6_prev3 + dist_diff_abs_pair_c3_5_6_prev3
   - 120 dist_diff_abs_pair_c3_5_6_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_5_6_prev3_force_positive:
   dist_diff_pair_c3_5_6_prev3
   - 120 dist_diff_abs_pair_c3_5_6_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_5_6_prev3_force_negative:
   dist_diff_pair_c3_5_6_prev3
   - 120 dist_diff_abs_pair_c3_5_6_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_5_6_prev3: 120 same_dist_pair_c3_5_6_prev3
   + dist_diff_abs_pair_c3_5_6_prev3 <= 129.8
 same_dist_pair_lower_c3_5_6_prev3: 120 same_dist_pair_c3_5_6_prev3
   + dist_diff_abs_pair_c3_5_6_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_5_6_prev0: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_5_6_prev0: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_5_6_prev1: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_5_6_prev1: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_5_6_prev2: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_5_6_prev2: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_5_6_prev3: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_5_6_prev3: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_5_6: diff_dist_pair_c3_5_6
   + same_dist_pair_c3_5_6_prev0 + same_dist_pair_c3_5_6_prev1
   + same_dist_pair_c3_5_6_prev2 + same_dist_pair_c3_5_6_prev3 >= 1
 dx_grid_diff_def_c3_5_7: - x_grid_5 + x_grid_7 + dx_grid_diff_c3_5_7 = 0
 dx_grid_abs_pair_c3_5_7_abs_ge_orig: dx_grid_abs_pair_c3_5_7
   - dx_grid_diff_c3_5_7 - 60 dx_grid_abs_pair_c3_5_7_is_positive >= -60
 dx_grid_abs_pair_c3_5_7_abs_le_orig: dx_grid_abs_pair_c3_5_7
   - dx_grid_diff_c3_5_7 + 60 dx_grid_abs_pair_c3_5_7_is_positive <= 60
 dx_grid_abs_pair_c3_5_7_abs_ge_neg_orig: dx_grid_abs_pair_c3_5_7
   + dx_grid_diff_c3_5_7 + 60 dx_grid_abs_pair_c3_5_7_is_positive >= 0
 dx_grid_abs_pair_c3_5_7_abs_le_neg_orig: dx_grid_abs_pair_c3_5_7
   + dx_grid_diff_c3_5_7 - 60 dx_grid_abs_pair_c3_5_7_is_positive <= 0
 dx_grid_abs_pair_c3_5_7_force_positive: dx_grid_diff_c3_5_7
   - 60 dx_grid_abs_pair_c3_5_7_is_positive >= -60
 dx_grid_abs_pair_c3_5_7_force_negative: dx_grid_diff_c3_5_7
   - 60 dx_grid_abs_pair_c3_5_7_is_positive <= -0.001
 dy_grid_diff_def_c3_5_7: - y_grid_5 + y_grid_7 + dy_grid_diff_c3_5_7 = 0
 dy_grid_abs_pair_c3_5_7_abs_ge_orig: dy_grid_abs_pair_c3_5_7
   - dy_grid_diff_c3_5_7 - 60 dy_grid_abs_pair_c3_5_7_is_positive >= -60
 dy_grid_abs_pair_c3_5_7_abs_le_orig: dy_grid_abs_pair_c3_5_7
   - dy_grid_diff_c3_5_7 + 60 dy_grid_abs_pair_c3_5_7_is_positive <= 60
 dy_grid_abs_pair_c3_5_7_abs_ge_neg_orig: dy_grid_abs_pair_c3_5_7
   + dy_grid_diff_c3_5_7 + 60 dy_grid_abs_pair_c3_5_7_is_positive >= 0
 dy_grid_abs_pair_c3_5_7_abs_le_neg_orig: dy_grid_abs_pair_c3_5_7
   + dy_grid_diff_c3_5_7 - 60 dy_grid_abs_pair_c3_5_7_is_positive <= 0
 dy_grid_abs_pair_c3_5_7_force_positive: dy_grid_diff_c3_5_7
   - 60 dy_grid_abs_pair_c3_5_7_is_positive >= -60
 dy_grid_abs_pair_c3_5_7_force_negative: dy_grid_diff_c3_5_7
   - 60 dy_grid_abs_pair_c3_5_7_is_positive <= -0.001
 dist_curr_pair_def_c3_5_7: - dx_grid_abs_pair_c3_5_7
   - dy_grid_abs_pair_c3_5_7 + dist_curr_pair_c3_5_7 = 0
 dist_diff_pair_def_c3_5_7_prev0: - dist_curr_pair_c3_5_7
   + dist_diff_pair_c3_5_7_prev0 = -18
 dist_diff_abs_pair_c3_5_7_prev0_abs_ge_orig: - dist_diff_pair_c3_5_7_prev0
   + dist_diff_abs_pair_c3_5_7_prev0
   - 120 dist_diff_abs_pair_c3_5_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_5_7_prev0_abs_le_orig: - dist_diff_pair_c3_5_7_prev0
   + dist_diff_abs_pair_c3_5_7_prev0
   + 120 dist_diff_abs_pair_c3_5_7_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_5_7_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_5_7_prev0 + dist_diff_abs_pair_c3_5_7_prev0
   + 120 dist_diff_abs_pair_c3_5_7_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_5_7_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_5_7_prev0 + dist_diff_abs_pair_c3_5_7_prev0
   - 120 dist_diff_abs_pair_c3_5_7_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_5_7_prev0_force_positive:
   dist_diff_pair_c3_5_7_prev0
   - 120 dist_diff_abs_pair_c3_5_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_5_7_prev0_force_negative:
   dist_diff_pair_c3_5_7_prev0
   - 120 dist_diff_abs_pair_c3_5_7_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_5_7_prev0: 120 same_dist_pair_c3_5_7_prev0
   + dist_diff_abs_pair_c3_5_7_prev0 <= 129.8
 same_dist_pair_lower_c3_5_7_prev0: 120 same_dist_pair_c3_5_7_prev0
   + dist_diff_abs_pair_c3_5_7_prev0 >= 10
 dist_diff_pair_def_c3_5_7_prev1: - dist_curr_pair_c3_5_7
   + dist_diff_pair_c3_5_7_prev1 = -42
 dist_diff_abs_pair_c3_5_7_prev1_abs_ge_orig: - dist_diff_pair_c3_5_7_prev1
   + dist_diff_abs_pair_c3_5_7_prev1
   - 120 dist_diff_abs_pair_c3_5_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_5_7_prev1_abs_le_orig: - dist_diff_pair_c3_5_7_prev1
   + dist_diff_abs_pair_c3_5_7_prev1
   + 120 dist_diff_abs_pair_c3_5_7_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_5_7_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_5_7_prev1 + dist_diff_abs_pair_c3_5_7_prev1
   + 120 dist_diff_abs_pair_c3_5_7_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_5_7_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_5_7_prev1 + dist_diff_abs_pair_c3_5_7_prev1
   - 120 dist_diff_abs_pair_c3_5_7_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_5_7_prev1_force_positive:
   dist_diff_pair_c3_5_7_prev1
   - 120 dist_diff_abs_pair_c3_5_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_5_7_prev1_force_negative:
   dist_diff_pair_c3_5_7_prev1
   - 120 dist_diff_abs_pair_c3_5_7_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_5_7_prev1: 120 same_dist_pair_c3_5_7_prev1
   + dist_diff_abs_pair_c3_5_7_prev1 <= 129.8
 same_dist_pair_lower_c3_5_7_prev1: 120 same_dist_pair_c3_5_7_prev1
   + dist_diff_abs_pair_c3_5_7_prev1 >= 10
 dist_diff_pair_def_c3_5_7_prev2: - dist_curr_pair_c3_5_7
   + dist_diff_pair_c3_5_7_prev2 = -11
 dist_diff_abs_pair_c3_5_7_prev2_abs_ge_orig: - dist_diff_pair_c3_5_7_prev2
   + dist_diff_abs_pair_c3_5_7_prev2
   - 120 dist_diff_abs_pair_c3_5_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_5_7_prev2_abs_le_orig: - dist_diff_pair_c3_5_7_prev2
   + dist_diff_abs_pair_c3_5_7_prev2
   + 120 dist_diff_abs_pair_c3_5_7_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_5_7_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_5_7_prev2 + dist_diff_abs_pair_c3_5_7_prev2
   + 120 dist_diff_abs_pair_c3_5_7_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_5_7_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_5_7_prev2 + dist_diff_abs_pair_c3_5_7_prev2
   - 120 dist_diff_abs_pair_c3_5_7_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_5_7_prev2_force_positive:
   dist_diff_pair_c3_5_7_prev2
   - 120 dist_diff_abs_pair_c3_5_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_5_7_prev2_force_negative:
   dist_diff_pair_c3_5_7_prev2
   - 120 dist_diff_abs_pair_c3_5_7_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_5_7_prev2: 120 same_dist_pair_c3_5_7_prev2
   + dist_diff_abs_pair_c3_5_7_prev2 <= 129.8
 same_dist_pair_lower_c3_5_7_prev2: 120 same_dist_pair_c3_5_7_prev2
   + dist_diff_abs_pair_c3_5_7_prev2 >= 10
 dist_diff_pair_def_c3_5_7_prev3: - dist_curr_pair_c3_5_7
   + dist_diff_pair_c3_5_7_prev3 = -42
 dist_diff_abs_pair_c3_5_7_prev3_abs_ge_orig: - dist_diff_pair_c3_5_7_prev3
   + dist_diff_abs_pair_c3_5_7_prev3
   - 120 dist_diff_abs_pair_c3_5_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_5_7_prev3_abs_le_orig: - dist_diff_pair_c3_5_7_prev3
   + dist_diff_abs_pair_c3_5_7_prev3
   + 120 dist_diff_abs_pair_c3_5_7_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_5_7_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_5_7_prev3 + dist_diff_abs_pair_c3_5_7_prev3
   + 120 dist_diff_abs_pair_c3_5_7_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_5_7_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_5_7_prev3 + dist_diff_abs_pair_c3_5_7_prev3
   - 120 dist_diff_abs_pair_c3_5_7_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_5_7_prev3_force_positive:
   dist_diff_pair_c3_5_7_prev3
   - 120 dist_diff_abs_pair_c3_5_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_5_7_prev3_force_negative:
   dist_diff_pair_c3_5_7_prev3
   - 120 dist_diff_abs_pair_c3_5_7_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_5_7_prev3: 120 same_dist_pair_c3_5_7_prev3
   + dist_diff_abs_pair_c3_5_7_prev3 <= 129.8
 same_dist_pair_lower_c3_5_7_prev3: 120 same_dist_pair_c3_5_7_prev3
   + dist_diff_abs_pair_c3_5_7_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_5_7_prev0: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_5_7_prev0: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_5_7_prev1: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_5_7_prev1: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_5_7_prev2: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_5_7_prev2: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_5_7_prev3: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_5_7_prev3: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_5_7: diff_dist_pair_c3_5_7
   + same_dist_pair_c3_5_7_prev0 + same_dist_pair_c3_5_7_prev1
   + same_dist_pair_c3_5_7_prev2 + same_dist_pair_c3_5_7_prev3 >= 1
 dx_grid_diff_def_c3_6_7: - x_grid_6 + x_grid_7 + dx_grid_diff_c3_6_7 = 0
 dx_grid_abs_pair_c3_6_7_abs_ge_orig: dx_grid_abs_pair_c3_6_7
   - dx_grid_diff_c3_6_7 - 60 dx_grid_abs_pair_c3_6_7_is_positive >= -60
 dx_grid_abs_pair_c3_6_7_abs_le_orig: dx_grid_abs_pair_c3_6_7
   - dx_grid_diff_c3_6_7 + 60 dx_grid_abs_pair_c3_6_7_is_positive <= 60
 dx_grid_abs_pair_c3_6_7_abs_ge_neg_orig: dx_grid_abs_pair_c3_6_7
   + dx_grid_diff_c3_6_7 + 60 dx_grid_abs_pair_c3_6_7_is_positive >= 0
 dx_grid_abs_pair_c3_6_7_abs_le_neg_orig: dx_grid_abs_pair_c3_6_7
   + dx_grid_diff_c3_6_7 - 60 dx_grid_abs_pair_c3_6_7_is_positive <= 0
 dx_grid_abs_pair_c3_6_7_force_positive: dx_grid_diff_c3_6_7
   - 60 dx_grid_abs_pair_c3_6_7_is_positive >= -60
 dx_grid_abs_pair_c3_6_7_force_negative: dx_grid_diff_c3_6_7
   - 60 dx_grid_abs_pair_c3_6_7_is_positive <= -0.001
 dy_grid_diff_def_c3_6_7: - y_grid_6 + y_grid_7 + dy_grid_diff_c3_6_7 = 0
 dy_grid_abs_pair_c3_6_7_abs_ge_orig: dy_grid_abs_pair_c3_6_7
   - dy_grid_diff_c3_6_7 - 60 dy_grid_abs_pair_c3_6_7_is_positive >= -60
 dy_grid_abs_pair_c3_6_7_abs_le_orig: dy_grid_abs_pair_c3_6_7
   - dy_grid_diff_c3_6_7 + 60 dy_grid_abs_pair_c3_6_7_is_positive <= 60
 dy_grid_abs_pair_c3_6_7_abs_ge_neg_orig: dy_grid_abs_pair_c3_6_7
   + dy_grid_diff_c3_6_7 + 60 dy_grid_abs_pair_c3_6_7_is_positive >= 0
 dy_grid_abs_pair_c3_6_7_abs_le_neg_orig: dy_grid_abs_pair_c3_6_7
   + dy_grid_diff_c3_6_7 - 60 dy_grid_abs_pair_c3_6_7_is_positive <= 0
 dy_grid_abs_pair_c3_6_7_force_positive: dy_grid_diff_c3_6_7
   - 60 dy_grid_abs_pair_c3_6_7_is_positive >= -60
 dy_grid_abs_pair_c3_6_7_force_negative: dy_grid_diff_c3_6_7
   - 60 dy_grid_abs_pair_c3_6_7_is_positive <= -0.001
 dist_curr_pair_def_c3_6_7: - dx_grid_abs_pair_c3_6_7
   - dy_grid_abs_pair_c3_6_7 + dist_curr_pair_c3_6_7 = 0
 dist_diff_pair_def_c3_6_7_prev0: - dist_curr_pair_c3_6_7
   + dist_diff_pair_c3_6_7_prev0 = -18
 dist_diff_abs_pair_c3_6_7_prev0_abs_ge_orig: - dist_diff_pair_c3_6_7_prev0
   + dist_diff_abs_pair_c3_6_7_prev0
   - 120 dist_diff_abs_pair_c3_6_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_6_7_prev0_abs_le_orig: - dist_diff_pair_c3_6_7_prev0
   + dist_diff_abs_pair_c3_6_7_prev0
   + 120 dist_diff_abs_pair_c3_6_7_prev0_is_positive <= 120
 dist_diff_abs_pair_c3_6_7_prev0_abs_ge_neg_orig:
   dist_diff_pair_c3_6_7_prev0 + dist_diff_abs_pair_c3_6_7_prev0
   + 120 dist_diff_abs_pair_c3_6_7_prev0_is_positive >= 0
 dist_diff_abs_pair_c3_6_7_prev0_abs_le_neg_orig:
   dist_diff_pair_c3_6_7_prev0 + dist_diff_abs_pair_c3_6_7_prev0
   - 120 dist_diff_abs_pair_c3_6_7_prev0_is_positive <= 0
 dist_diff_abs_pair_c3_6_7_prev0_force_positive:
   dist_diff_pair_c3_6_7_prev0
   - 120 dist_diff_abs_pair_c3_6_7_prev0_is_positive >= -120
 dist_diff_abs_pair_c3_6_7_prev0_force_negative:
   dist_diff_pair_c3_6_7_prev0
   - 120 dist_diff_abs_pair_c3_6_7_prev0_is_positive <= -0.001
 same_dist_pair_upper_c3_6_7_prev0: 120 same_dist_pair_c3_6_7_prev0
   + dist_diff_abs_pair_c3_6_7_prev0 <= 129.8
 same_dist_pair_lower_c3_6_7_prev0: 120 same_dist_pair_c3_6_7_prev0
   + dist_diff_abs_pair_c3_6_7_prev0 >= 10
 dist_diff_pair_def_c3_6_7_prev1: - dist_curr_pair_c3_6_7
   + dist_diff_pair_c3_6_7_prev1 = -12
 dist_diff_abs_pair_c3_6_7_prev1_abs_ge_orig: - dist_diff_pair_c3_6_7_prev1
   + dist_diff_abs_pair_c3_6_7_prev1
   - 120 dist_diff_abs_pair_c3_6_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_6_7_prev1_abs_le_orig: - dist_diff_pair_c3_6_7_prev1
   + dist_diff_abs_pair_c3_6_7_prev1
   + 120 dist_diff_abs_pair_c3_6_7_prev1_is_positive <= 120
 dist_diff_abs_pair_c3_6_7_prev1_abs_ge_neg_orig:
   dist_diff_pair_c3_6_7_prev1 + dist_diff_abs_pair_c3_6_7_prev1
   + 120 dist_diff_abs_pair_c3_6_7_prev1_is_positive >= 0
 dist_diff_abs_pair_c3_6_7_prev1_abs_le_neg_orig:
   dist_diff_pair_c3_6_7_prev1 + dist_diff_abs_pair_c3_6_7_prev1
   - 120 dist_diff_abs_pair_c3_6_7_prev1_is_positive <= 0
 dist_diff_abs_pair_c3_6_7_prev1_force_positive:
   dist_diff_pair_c3_6_7_prev1
   - 120 dist_diff_abs_pair_c3_6_7_prev1_is_positive >= -120
 dist_diff_abs_pair_c3_6_7_prev1_force_negative:
   dist_diff_pair_c3_6_7_prev1
   - 120 dist_diff_abs_pair_c3_6_7_prev1_is_positive <= -0.001
 same_dist_pair_upper_c3_6_7_prev1: 120 same_dist_pair_c3_6_7_prev1
   + dist_diff_abs_pair_c3_6_7_prev1 <= 129.8
 same_dist_pair_lower_c3_6_7_prev1: 120 same_dist_pair_c3_6_7_prev1
   + dist_diff_abs_pair_c3_6_7_prev1 >= 10
 dist_diff_pair_def_c3_6_7_prev2: - dist_curr_pair_c3_6_7
   + dist_diff_pair_c3_6_7_prev2 = -16
 dist_diff_abs_pair_c3_6_7_prev2_abs_ge_orig: - dist_diff_pair_c3_6_7_prev2
   + dist_diff_abs_pair_c3_6_7_prev2
   - 120 dist_diff_abs_pair_c3_6_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_6_7_prev2_abs_le_orig: - dist_diff_pair_c3_6_7_prev2
   + dist_diff_abs_pair_c3_6_7_prev2
   + 120 dist_diff_abs_pair_c3_6_7_prev2_is_positive <= 120
 dist_diff_abs_pair_c3_6_7_prev2_abs_ge_neg_orig:
   dist_diff_pair_c3_6_7_prev2 + dist_diff_abs_pair_c3_6_7_prev2
   + 120 dist_diff_abs_pair_c3_6_7_prev2_is_positive >= 0
 dist_diff_abs_pair_c3_6_7_prev2_abs_le_neg_orig:
   dist_diff_pair_c3_6_7_prev2 + dist_diff_abs_pair_c3_6_7_prev2
   - 120 dist_diff_abs_pair_c3_6_7_prev2_is_positive <= 0
 dist_diff_abs_pair_c3_6_7_prev2_force_positive:
   dist_diff_pair_c3_6_7_prev2
   - 120 dist_diff_abs_pair_c3_6_7_prev2_is_positive >= -120
 dist_diff_abs_pair_c3_6_7_prev2_force_negative:
   dist_diff_pair_c3_6_7_prev2
   - 120 dist_diff_abs_pair_c3_6_7_prev2_is_positive <= -0.001
 same_dist_pair_upper_c3_6_7_prev2: 120 same_dist_pair_c3_6_7_prev2
   + dist_diff_abs_pair_c3_6_7_prev2 <= 129.8
 same_dist_pair_lower_c3_6_7_prev2: 120 same_dist_pair_c3_6_7_prev2
   + dist_diff_abs_pair_c3_6_7_prev2 >= 10
 dist_diff_pair_def_c3_6_7_prev3: - dist_curr_pair_c3_6_7
   + dist_diff_pair_c3_6_7_prev3 = -12
 dist_diff_abs_pair_c3_6_7_prev3_abs_ge_orig: - dist_diff_pair_c3_6_7_prev3
   + dist_diff_abs_pair_c3_6_7_prev3
   - 120 dist_diff_abs_pair_c3_6_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_6_7_prev3_abs_le_orig: - dist_diff_pair_c3_6_7_prev3
   + dist_diff_abs_pair_c3_6_7_prev3
   + 120 dist_diff_abs_pair_c3_6_7_prev3_is_positive <= 120
 dist_diff_abs_pair_c3_6_7_prev3_abs_ge_neg_orig:
   dist_diff_pair_c3_6_7_prev3 + dist_diff_abs_pair_c3_6_7_prev3
   + 120 dist_diff_abs_pair_c3_6_7_prev3_is_positive >= 0
 dist_diff_abs_pair_c3_6_7_prev3_abs_le_neg_orig:
   dist_diff_pair_c3_6_7_prev3 + dist_diff_abs_pair_c3_6_7_prev3
   - 120 dist_diff_abs_pair_c3_6_7_prev3_is_positive <= 0
 dist_diff_abs_pair_c3_6_7_prev3_force_positive:
   dist_diff_pair_c3_6_7_prev3
   - 120 dist_diff_abs_pair_c3_6_7_prev3_is_positive >= -120
 dist_diff_abs_pair_c3_6_7_prev3_force_negative:
   dist_diff_pair_c3_6_7_prev3
   - 120 dist_diff_abs_pair_c3_6_7_prev3_is_positive <= -0.001
 same_dist_pair_upper_c3_6_7_prev3: 120 same_dist_pair_c3_6_7_prev3
   + dist_diff_abs_pair_c3_6_7_prev3 <= 129.8
 same_dist_pair_lower_c3_6_7_prev3: 120 same_dist_pair_c3_6_7_prev3
   + dist_diff_abs_pair_c3_6_7_prev3 >= 10
 diff_dist_pair_implies_not_same_c3_6_7_prev0: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev0 <= 1
 not_same_implies_diff_dist_pair_c3_6_7_prev0: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev0 <= 1
 diff_dist_pair_implies_not_same_c3_6_7_prev1: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev1 <= 1
 not_same_implies_diff_dist_pair_c3_6_7_prev1: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev1 <= 1
 diff_dist_pair_implies_not_same_c3_6_7_prev2: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev2 <= 1
 not_same_implies_diff_dist_pair_c3_6_7_prev2: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev2 <= 1
 diff_dist_pair_implies_not_same_c3_6_7_prev3: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev3 <= 1
 not_same_implies_diff_dist_pair_c3_6_7_prev3: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev3 <= 1
 all_not_same_implies_diff_dist_pair_c3_6_7: diff_dist_pair_c3_6_7
   + same_dist_pair_c3_6_7_prev0 + same_dist_pair_c3_6_7_prev1
   + same_dist_pair_c3_6_7_prev2 + same_dist_pair_c3_6_7_prev3 >= 1
 exclude_solution_dist_pair_c3: diff_dist_pair_c3_0_1
   + diff_dist_pair_c3_0_2 + diff_dist_pair_c3_0_3 + diff_dist_pair_c3_0_4
   + diff_dist_pair_c3_0_5 + diff_dist_pair_c3_0_6 + diff_dist_pair_c3_0_7
   + diff_dist_pair_c3_1_2 + diff_dist_pair_c3_1_3 + diff_dist_pair_c3_1_4
   + diff_dist_pair_c3_1_5 + diff_dist_pair_c3_1_6 + diff_dist_pair_c3_1_7
   + diff_dist_pair_c3_2_3 + diff_dist_pair_c3_2_4 + diff_dist_pair_c3_2_5
   + diff_dist_pair_c3_2_6 + diff_dist_pair_c3_2_7 + diff_dist_pair_c3_3_4
   + diff_dist_pair_c3_3_5 + diff_dist_pair_c3_3_6 + diff_dist_pair_c3_3_7
   + diff_dist_pair_c3_4_5 + diff_dist_pair_c3_4_6 + diff_dist_pair_c3_4_7
   + diff_dist_pair_c3_5_6 + diff_dist_pair_c3_5_7 + diff_dist_pair_c3_6_7
   >= 1
Bounds
 x_grid_0 <= 59
 y_grid_0 <= 59
 x_grid_1 <= 59
 y_grid_1 <= 59
 x_grid_2 <= 59
 y_grid_2 <= 59
 x_grid_3 <= 59
 y_grid_3 <= 59
 x_grid_4 <= 59
 y_grid_4 <= 59
 x_grid_5 <= 59
 y_grid_5 <= 59
 x_grid_6 <= 59
 y_grid_6 <= 59
 x_grid_7 <= 59
 y_grid_7 <= 59
 x_0 <= 30
 y_0 <= 30
 x_1 <= 30
 y_1 <= 30
 x_2 <= 30
 y_2 <= 30
 x_3 <= 30
 y_3 <= 30
 x_4 <= 30
 y_4 <= 30
 x_5 <= 30
 y_5 <= 30
 x_6 <= 30
 y_6 <= 30
 x_7 <= 30
 y_7 <= 30
 w_0 = 10
 h_0 = 10
 w_1 = 5
 h_1 = 5
 w_2 = 10
 h_2 = 10
 w_3 = 5
 h_3 = 5
 w_4 = 5
 h_4 = 5
 w_5 = 5
 h_5 = 5
 w_6 = 8
 h_6 = 8
 w_7 = 6
 h_7 = 6
 cx_0 <= 30
 cy_0 <= 30
 cx_1 <= 30
 cy_1 <= 30
 cx_2 <= 30
 cy_2 <= 30
 cx_3 <= 30
 cy_3 <= 30
 cx_4 <= 30
 cy_4 <= 30
 cx_5 <= 30
 cy_5 <= 30
 cx_6 <= 30
 cy_6 <= 30
 cx_7 <= 30
 cy_7 <= 30
 shared_y_0_1 <= 10
 shared_x_0_1 <= 10
 shared_y_2_4 <= 10
 shared_x_2_4 <= 10
 shared_y_0_3 <= 10
 shared_x_0_3 <= 10
 shared_y_2_3 <= 10
 shared_x_2_3 <= 10
 shared_y_6_7 <= 8
 shared_x_6_7 <= 8
 shared_y_2_6 <= 10
 shared_x_2_6 <= 10
 shared_y_2_5 <= 10
 shared_x_2_5 <= 10
 bbox_min_x <= 30
 bbox_max_x <= 30
 bbox_min_y <= 30
 bbox_max_y <= 30
 bbox_w <= 30
 bbox_h <= 30
 -30 <= dx_diff_0_1 <= 30
 -30 <= dy_diff_0_1 <= 30
 -30 <= dx_diff_2_4 <= 30
 -30 <= dy_diff_2_4 <= 30
 -30 <= dx_diff_0_3 <= 30
 -30 <= dy_diff_0_3 <= 30
 -30 <= dx_diff_2_3 <= 30
 -30 <= dy_diff_2_3 <= 30
 -30 <= dx_diff_6_7 <= 30
 -30 <= dy_diff_6_7 <= 30
 -30 <= dx_diff_2_6 <= 30
 -30 <= dy_diff_2_6 <= 30
 -30 <= dx_diff_2_5 <= 30
 -30 <= dy_diff_2_5 <= 30
 bbox_area_proxy_t <= 60
 dx_grid_abs_pair_c3_0_1 <= 59
 dy_grid_abs_pair_c3_0_1 <= 59
 -59 <= dx_grid_diff_c3_0_1 <= 59
 -59 <= dy_grid_diff_c3_0_1 <= 59
 dist_curr_pair_c3_0_1 <= 120
 -120 <= dist_diff_pair_c3_0_1_prev0 <= 120
 dist_diff_abs_pair_c3_0_1_prev0 <= 120
 -120 <= dist_diff_pair_c3_0_1_prev1 <= 120
 dist_diff_abs_pair_c3_0_1_prev1 <= 120
 -120 <= dist_diff_pair_c3_0_1_prev2 <= 120
 dist_diff_abs_pair_c3_0_1_prev2 <= 120
 -120 <= dist_diff_pair_c3_0_1_prev3 <= 120
 dist_diff_abs_pair_c3_0_1_prev3 <= 120
 dx_grid_abs_pair_c3_0_2 <= 59
 dy_grid_abs_pair_c3_0_2 <= 59
 -59 <= dx_grid_diff_c3_0_2 <= 59
 -59 <= dy_grid_diff_c3_0_2 <= 59
 dist_curr_pair_c3_0_2 <= 120
 -120 <= dist_diff_pair_c3_0_2_prev0 <= 120
 dist_diff_abs_pair_c3_0_2_prev0 <= 120
 -120 <= dist_diff_pair_c3_0_2_prev1 <= 120
 dist_diff_abs_pair_c3_0_2_prev1 <= 120
 -120 <= dist_diff_pair_c3_0_2_prev2 <= 120
 dist_diff_abs_pair_c3_0_2_prev2 <= 120
 -120 <= dist_diff_pair_c3_0_2_prev3 <= 120
 dist_diff_abs_pair_c3_0_2_prev3 <= 120
 dx_grid_abs_pair_c3_0_3 <= 59
 dy_grid_abs_pair_c3_0_3 <= 59
 -59 <= dx_grid_diff_c3_0_3 <= 59
 -59 <= dy_grid_diff_c3_0_3 <= 59
 dist_curr_pair_c3_0_3 <= 120
 -120 <= dist_diff_pair_c3_0_3_prev0 <= 120
 dist_diff_abs_pair_c3_0_3_prev0 <= 120
 -120 <= dist_diff_pair_c3_0_3_prev1 <= 120
 dist_diff_abs_pair_c3_0_3_prev1 <= 120
 -120 <= dist_diff_pair_c3_0_3_prev2 <= 120
 dist_diff_abs_pair_c3_0_3_prev2 <= 120
 -120 <= dist_diff_pair_c3_0_3_prev3 <= 120
 dist_diff_abs_pair_c3_0_3_prev3 <= 120
 dx_grid_abs_pair_c3_0_4 <= 59
 dy_grid_abs_pair_c3_0_4 <= 59
 -59 <= dx_grid_diff_c3_0_4 <= 59
 -59 <= dy_grid_diff_c3_0_4 <= 59
 dist_curr_pair_c3_0_4 <= 120
 -120 <= dist_diff_pair_c3_0_4_prev0 <= 120
 dist_diff_abs_pair_c3_0_4_prev0 <= 120
 -120 <= dist_diff_pair_c3_0_4_prev1 <= 120
 dist_diff_abs_pair_c3_0_4_prev1 <= 120
 -120 <= dist_diff_pair_c3_0_4_prev2 <= 120
 dist_diff_abs_pair_c3_0_4_prev2 <= 120
 -120 <= dist_diff_pair_c3_0_4_prev3 <= 120
 dist_diff_abs_pair_c3_0_4_prev3 <= 120
 dx_grid_abs_pair_c3_0_5 <= 59
 dy_grid_abs_pair_c3_0_5 <= 59
 -59 <= dx_grid_diff_c3_0_5 <= 59
 -59 <= dy_grid_diff_c3_0_5 <= 59
 dist_curr_pair_c3_0_5 <= 120
 -120 <= dist_diff_pair_c3_0_5_prev0 <= 120
 dist_diff_abs_pair_c3_0_5_prev0 <= 120
 -120 <= dist_diff_pair_c3_0_5_prev1 <= 120
 dist_diff_abs_pair_c3_0_5_prev1 <= 120
 -120 <= dist_diff_pair_c3_0_5_prev2 <= 120
 dist_diff_abs_pair_c3_0_5_prev2 <= 120
 -120 <= dist_diff_pair_c3_0_5_prev3 <= 120
 dist_diff_abs_pair_c3_0_5_prev3 <= 120
 dx_grid_abs_pair_c3_0_6 <= 59
 dy_grid_abs_pair_c3_0_6 <= 59
 -59 <= dx_grid_diff_c3_0_6 <= 59
 -59 <= dy_grid_diff_c3_0_6 <= 59
 dist_curr_pair_c3_0_6 <= 120
 -120 <= dist_diff_pair_c3_0_6_prev0 <= 120
 dist_diff_abs_pair_c3_0_6_prev0 <= 120
 -120 <= dist_diff_pair_c3_0_6_prev1 <= 120
 dist_diff_abs_pair_c3_0_6_prev1 <= 120
 -120 <= dist_diff_pair_c3_0_6_prev2 <= 120
 dist_diff_abs_pair_c3_0_6_prev2 <= 120
 -120 <= dist_diff_pair_c3_0_6_prev3 <= 120
 dist_diff_abs_pair_c3_0_6_prev3 <= 120
 dx_grid_abs_pair_c3_0_7 <= 59
 dy_grid_abs_pair_c3_0_7 <= 59
 -59 <= dx_grid_diff_c3_0_7 <= 59
 -59 <= dy_grid_diff_c3_0_7 <= 59
 dist_curr_pair_c3_0_7 <= 120
 -120 <= dist_diff_pair_c3_0_7_prev0 <= 120
 dist_diff_abs_pair_c3_0_7_prev0 <= 120
 -120 <= dist_diff_pair_c3_0_7_prev1 <= 120
 dist_diff_abs_pair_c3_0_7_prev1 <= 120
 -120 <= dist_diff_pair_c3_0_7_prev2 <= 120
 dist_diff_abs_pair_c3_0_7_prev2 <= 120
 -120 <= dist_diff_pair_c3_0_7_prev3 <= 120
 dist_diff_abs_pair_c3_0_7_prev3 <= 120
 dx_grid_abs_pair_c3_1_2 <= 59
 dy_grid_abs_pair_c3_1_2 <= 59
 -59 <= dx_grid_diff_c3_1_2 <= 59
 -59 <= dy_grid_diff_c3_1_2 <= 59
 dist_curr_pair_c3_1_2 <= 120
 -120 <= dist_diff_pair_c3_1_2_prev0 <= 120
 dist_diff_abs_pair_c3_1_2_prev0 <= 120
 -120 <= dist_diff_pair_c3_1_2_prev1 <= 120
 dist_diff_abs_pair_c3_1_2_prev1 <= 120
 -120 <= dist_diff_pair_c3_1_2_prev2 <= 120
 dist_diff_abs_pair_c3_1_2_prev2 <= 120
 -120 <= dist_diff_pair_c3_1_2_prev3 <= 120
 dist_diff_abs_pair_c3_1_2_prev3 <= 120
 dx_grid_abs_pair_c3_1_3 <= 59
 dy_grid_abs_pair_c3_1_3 <= 59
 -59 <= dx_grid_diff_c3_1_3 <= 59
 -59 <= dy_grid_diff_c3_1_3 <= 59
 dist_curr_pair_c3_1_3 <= 120
 -120 <= dist_diff_pair_c3_1_3_prev0 <= 120
 dist_diff_abs_pair_c3_1_3_prev0 <= 120
 -120 <= dist_diff_pair_c3_1_3_prev1 <= 120
 dist_diff_abs_pair_c3_1_3_prev1 <= 120
 -120 <= dist_diff_pair_c3_1_3_prev2 <= 120
 dist_diff_abs_pair_c3_1_3_prev2 <= 120
 -120 <= dist_diff_pair_c3_1_3_prev3 <= 120
 dist_diff_abs_pair_c3_1_3_prev3 <= 120
 dx_grid_abs_pair_c3_1_4 <= 59
 dy_grid_abs_pair_c3_1_4 <= 59
 -59 <= dx_grid_diff_c3_1_4 <= 59
 -59 <= dy_grid_diff_c3_1_4 <= 59
 dist_curr_pair_c3_1_4 <= 120
 -120 <= dist_diff_pair_c3_1_4_prev0 <= 120
 dist_diff_abs_pair_c3_1_4_prev0 <= 120
 -120 <= dist_diff_pair_c3_1_4_prev1 <= 120
 dist_diff_abs_pair_c3_1_4_prev1 <= 120
 -120 <= dist_diff_pair_c3_1_4_prev2 <= 120
 dist_diff_abs_pair_c3_1_4_prev2 <= 120
 -120 <= dist_diff_pair_c3_1_4_prev3 <= 120
 dist_diff_abs_pair_c3_1_4_prev3 <= 120
 dx_grid_abs_pair_c3_1_5 <= 59
 dy_grid_abs_pair_c3_1_5 <= 59
 -59 <= dx_grid_diff_c3_1_5 <= 59
 -59 <= dy_grid_diff_c3_1_5 <= 59
 dist_curr_pair_c3_1_5 <= 120
 -120 <= dist_diff_pair_c3_1_5_prev0 <= 120
 dist_diff_abs_pair_c3_1_5_prev0 <= 120
 -120 <= dist_diff_pair_c3_1_5_prev1 <= 120
 dist_diff_abs_pair_c3_1_5_prev1 <= 120
 -120 <= dist_diff_pair_c3_1_5_prev2 <= 120
 dist_diff_abs_pair_c3_1_5_prev2 <= 120
 -120 <= dist_diff_pair_c3_1_5_prev3 <= 120
 dist_diff_abs_pair_c3_1_5_prev3 <= 120
 dx_grid_abs_pair_c3_1_6 <= 59
 dy_grid_abs_pair_c3_1_6 <= 59
 -59 <= dx_grid_diff_c3_1_6 <= 59
 -59 <= dy_grid_diff_c3_1_6 <= 59
 dist_curr_pair_c3_1_6 <= 120
 -120 <= dist_diff_pair_c3_1_6_prev0 <= 120
 dist_diff_abs_pair_c3_1_6_prev0 <= 120
 -120 <= dist_diff_pair_c3_1_6_prev1 <= 120
 dist_diff_abs_pair_c3_1_6_prev1 <= 120
 -120 <= dist_diff_pair_c3_1_6_prev2 <= 120
 dist_diff_abs_pair_c3_1_6_prev2 <= 120
 -120 <= dist_diff_pair_c3_1_6_prev3 <= 120
 dist_diff_abs_pair_c3_1_6_prev3 <= 120
 dx_grid_abs_pair_c3_1_7 <= 59
 dy_grid_abs_pair_c3_1_7 <= 59
 -59 <= dx_grid_diff_c3_1_7 <= 59
 -59 <= dy_grid_diff_c3_1_7 <= 59
 dist_curr_pair_c3_1_7 <= 120
 -120 <= dist_diff_pair_c3_1_7_prev0 <= 120
 dist_diff_abs_pair_c3_1_7_prev0 <= 120
 -120 <= dist_diff_pair_c3_1_7_prev1 <= 120
 dist_diff_abs_pair_c3_1_7_prev1 <= 120
 -120 <= dist_diff_pair_c3_1_7_prev2 <= 120
 dist_diff_abs_pair_c3_1_7_prev2 <= 120
 -120 <= dist_diff_pair_c3_1_7_prev3 <= 120
 dist_diff_abs_pair_c3_1_7_prev3 <= 120
 dx_grid_abs_pair_c3_2_3 <= 59
 dy_grid_abs_pair_c3_2_3 <= 59
 -59 <= dx_grid_diff_c3_2_3 <= 59
 -59 <= dy_grid_diff_c3_2_3 <= 59
 dist_curr_pair_c3_2_3 <= 120
 -120 <= dist_diff_pair_c3_2_3_prev0 <= 120
 dist_diff_abs_pair_c3_2_3_prev0 <= 120
 -120 <= dist_diff_pair_c3_2_3_prev1 <= 120
 dist_diff_abs_pair_c3_2_3_prev1 <= 120
 -120 <= dist_diff_pair_c3_2_3_prev2 <= 120
 dist_diff_abs_pair_c3_2_3_prev2 <= 120
 -120 <= dist_diff_pair_c3_2_3_prev3 <= 120
 dist_diff_abs_pair_c3_2_3_prev3 <= 120
 dx_grid_abs_pair_c3_2_4 <= 59
 dy_grid_abs_pair_c3_2_4 <= 59
 -59 <= dx_grid_diff_c3_2_4 <= 59
 -59 <= dy_grid_diff_c3_2_4 <= 59
 dist_curr_pair_c3_2_4 <= 120
 -120 <= dist_diff_pair_c3_2_4_prev0 <= 120
 dist_diff_abs_pair_c3_2_4_prev0 <= 120
 -120 <= dist_diff_pair_c3_2_4_prev1 <= 120
 dist_diff_abs_pair_c3_2_4_prev1 <= 120
 -120 <= dist_diff_pair_c3_2_4_prev2 <= 120
 dist_diff_abs_pair_c3_2_4_prev2 <= 120
 -120 <= dist_diff_pair_c3_2_4_prev3 <= 120
 dist_diff_abs_pair_c3_2_4_prev3 <= 120
 dx_grid_abs_pair_c3_2_5 <= 59
 dy_grid_abs_pair_c3_2_5 <= 59
 -59 <= dx_grid_diff_c3_2_5 <= 59
 -59 <= dy_grid_diff_c3_2_5 <= 59
 dist_curr_pair_c3_2_5 <= 120
 -120 <= dist_diff_pair_c3_2_5_prev0 <= 120
 dist_diff_abs_pair_c3_2_5_prev0 <= 120
 -120 <= dist_diff_pair_c3_2_5_prev1 <= 120
 dist_diff_abs_pair_c3_2_5_prev1 <= 120
 -120 <= dist_diff_pair_c3_2_5_prev2 <= 120
 dist_diff_abs_pair_c3_2_5_prev2 <= 120
 -120 <= dist_diff_pair_c3_2_5_prev3 <= 120
 dist_diff_abs_pair_c3_2_5_prev3 <= 120
 dx_grid_abs_pair_c3_2_6 <= 59
 dy_grid_abs_pair_c3_2_6 <= 59
 -59 <= dx_grid_diff_c3_2_6 <= 59
 -59 <= dy_grid_diff_c3_2_6 <= 59
 dist_curr_pair_c3_2_6 <= 120
 -120 <= dist_diff_pair_c3_2_6_prev0 <= 120
 dist_diff_abs_pair_c3_2_6_prev0 <= 120
 -120 <= dist_diff_pair_c3_2_6_prev1 <= 120
 dist_diff_abs_pair_c3_2_6_prev1 <= 120
 -120 <= dist_diff_pair_c3_2_6_prev2 <= 120
 dist_diff_abs_pair_c3_2_6_prev2 <= 120
 -120 <= dist_diff_pair_c3_2_6_prev3 <= 120
 dist_diff_abs_pair_c3_2_6_prev3 <= 120
 dx_grid_abs_pair_c3_2_7 <= 59
 dy_grid_abs_pair_c3_2_7 <= 59
 -59 <= dx_grid_diff_c3_2_7 <= 59
 -59 <= dy_grid_diff_c3_2_7 <= 59
 dist_curr_pair_c3_2_7 <= 120
 -120 <= dist_diff_pair_c3_2_7_prev0 <= 120
 dist_diff_abs_pair_c3_2_7_prev0 <= 120
 -120 <= dist_diff_pair_c3_2_7_prev1 <= 120
 dist_diff_abs_pair_c3_2_7_prev1 <= 120
 -120 <= dist_diff_pair_c3_2_7_prev2 <= 120
 dist_diff_abs_pair_c3_2_7_prev2 <= 120
 -120 <= dist_diff_pair_c3_2_7_prev3 <= 120
 dist_diff_abs_pair_c3_2_7_prev3 <= 120
 dx_grid_abs_pair_c3_3_4 <= 59
 dy_grid_abs_pair_c3_3_4 <= 59
 -59 <= dx_grid_diff_c3_3_4 <= 59
 -59 <= dy_grid_diff_c3_3_4 <= 59
 dist_curr_pair_c3_3_4 <= 120
 -120 <= dist_diff_pair_c3_3_4_prev0 <= 120
 dist_diff_abs_pair_c3_3_4_prev0 <= 120
 -120 <= dist_diff_pair_c3_3_4_prev1 <= 120
 dist_diff_abs_pair_c3_3_4_prev1 <= 120
 -120 <= dist_diff_pair_c3_3_4_prev2 <= 120
 dist_diff_abs_pair_c3_3_4_prev2 <= 120
 -120 <= dist_diff_pair_c3_3_4_prev3 <= 120
 dist_diff_abs_pair_c3_3_4_prev3 <= 120
 dx_grid_abs_pair_c3_3_5 <= 59
 dy_grid_abs_pair_c3_3_5 <= 59
 -59 <= dx_grid_diff_c3_3_5 <= 59
 -59 <= dy_grid_diff_c3_3_5 <= 59
 dist_curr_pair_c3_3_5 <= 120
 -120 <= dist_diff_pair_c3_3_5_prev0 <= 120
 dist_diff_abs_pair_c3_3_5_prev0 <= 120
 -120 <= dist_diff_pair_c3_3_5_prev1 <= 120
 dist_diff_abs_pair_c3_3_5_prev1 <= 120
 -120 <= dist_diff_pair_c3_3_5_prev2 <= 120
 dist_diff_abs_pair_c3_3_5_prev2 <= 120
 -120 <= dist_diff_pair_c3_3_5_prev3 <= 120
 dist_diff_abs_pair_c3_3_5_prev3 <= 120
 dx_grid_abs_pair_c3_3_6 <= 59
 dy_grid_abs_pair_c3_3_6 <= 59
 -59 <= dx_grid_diff_c3_3_6 <= 59
 -59 <= dy_grid_diff_c3_3_6 <= 59
 dist_curr_pair_c3_3_6 <= 120
 -120 <= dist_diff_pair_c3_3_6_prev0 <= 120
 dist_diff_abs_pair_c3_3_6_prev0 <= 120
 -120 <= dist_diff_pair_c3_3_6_prev1 <= 120
 dist_diff_abs_pair_c3_3_6_prev1 <= 120
 -120 <= dist_diff_pair_c3_3_6_prev2 <= 120
 dist_diff_abs_pair_c3_3_6_prev2 <= 120
 -120 <= dist_diff_pair_c3_3_6_prev3 <= 120
 dist_diff_abs_pair_c3_3_6_prev3 <= 120
 dx_grid_abs_pair_c3_3_7 <= 59
 dy_grid_abs_pair_c3_3_7 <= 59
 -59 <= dx_grid_diff_c3_3_7 <= 59
 -59 <= dy_grid_diff_c3_3_7 <= 59
 dist_curr_pair_c3_3_7 <= 120
 -120 <= dist_diff_pair_c3_3_7_prev0 <= 120
 dist_diff_abs_pair_c3_3_7_prev0 <= 120
 -120 <= dist_diff_pair_c3_3_7_prev1 <= 120
 dist_diff_abs_pair_c3_3_7_prev1 <= 120
 -120 <= dist_diff_pair_c3_3_7_prev2 <= 120
 dist_diff_abs_pair_c3_3_7_prev2 <= 120
 -120 <= dist_diff_pair_c3_3_7_prev3 <= 120
 dist_diff_abs_pair_c3_3_7_prev3 <= 120
 dx_grid_abs_pair_c3_4_5 <= 59
 dy_grid_abs_pair_c3_4_5 <= 59
 -59 <= dx_grid_diff_c3_4_5 <= 59
 -59 <= dy_grid_diff_c3_4_5 <= 59
 dist_curr_pair_c3_4_5 <= 120
 -120 <= dist_diff_pair_c3_4_5_prev0 <= 120
 dist_diff_abs_pair_c3_4_5_prev0 <= 120
 -120 <= dist_diff_pair_c3_4_5_prev1 <= 120
 dist_diff_abs_pair_c3_4_5_prev1 <= 120
 -120 <= dist_diff_pair_c3_4_5_prev2 <= 120
 dist_diff_abs_pair_c3_4_5_prev2 <= 120
 -120 <= dist_diff_pair_c3_4_5_prev3 <= 120
 dist_diff_abs_pair_c3_4_5_prev3 <= 120
 dx_grid_abs_pair_c3_4_6 <= 59
 dy_grid_abs_pair_c3_4_6 <= 59
 -59 <= dx_grid_diff_c3_4_6 <= 59
 -59 <= dy_grid_diff_c3_4_6 <= 59
 dist_curr_pair_c3_4_6 <= 120
 -120 <= dist_diff_pair_c3_4_6_prev0 <= 120
 dist_diff_abs_pair_c3_4_6_prev0 <= 120
 -120 <= dist_diff_pair_c3_4_6_prev1 <= 120
 dist_diff_abs_pair_c3_4_6_prev1 <= 120
 -120 <= dist_diff_pair_c3_4_6_prev2 <= 120
 dist_diff_abs_pair_c3_4_6_prev2 <= 120
 -120 <= dist_diff_pair_c3_4_6_prev3 <= 120
 dist_diff_abs_pair_c3_4_6_prev3 <= 120
 dx_grid_abs_pair_c3_4_7 <= 59
 dy_grid_abs_pair_c3_4_7 <= 59
 -59 <= dx_grid_diff_c3_4_7 <= 59
 -59 <= dy_grid_diff_c3_4_7 <= 59
 dist_curr_pair_c3_4_7 <= 120
 -120 <= dist_diff_pair_c3_4_7_prev0 <= 120
 dist_diff_abs_pair_c3_4_7_prev0 <= 120
 -120 <= dist_diff_pair_c3_4_7_prev1 <= 120
 dist_diff_abs_pair_c3_4_7_prev1 <= 120
 -120 <= dist_diff_pair_c3_4_7_prev2 <= 120
 dist_diff_abs_pair_c3_4_7_prev2 <= 120
 -120 <= dist_diff_pair_c3_4_7_prev3 <= 120
 dist_diff_abs_pair_c3_4_7_prev3 <= 120
 dx_grid_abs_pair_c3_5_6 <= 59
 dy_grid_abs_pair_c3_5_6 <= 59
 -59 <= dx_grid_diff_c3_5_6 <= 59
 -59 <= dy_grid_diff_c3_5_6 <= 59
 dist_curr_pair_c3_5_6 <= 120
 -120 <= dist_diff_pair_c3_5_6_prev0 <= 120
 dist_diff_abs_pair_c3_5_6_prev0 <= 120
 -120 <= dist_diff_pair_c3_5_6_prev1 <= 120
 dist_diff_abs_pair_c3_5_6_prev1 <= 120
 -120 <= dist_diff_pair_c3_5_6_prev2 <= 120
 dist_diff_abs_pair_c3_5_6_prev2 <= 120
 -120 <= dist_diff_pair_c3_5_6_prev3 <= 120
 dist_diff_abs_pair_c3_5_6_prev3 <= 120
 dx_grid_abs_pair_c3_5_7 <= 59
 dy_grid_abs_pair_c3_5_7 <= 59
 -59 <= dx_grid_diff_c3_5_7 <= 59
 -59 <= dy_grid_diff_c3_5_7 <= 59
 dist_curr_pair_c3_5_7 <= 120
 -120 <= dist_diff_pair_c3_5_7_prev0 <= 120
 dist_diff_abs_pair_c3_5_7_prev0 <= 120
 -120 <= dist_diff_pair_c3_5_7_prev1 <= 120
 dist_diff_abs_pair_c3_5_7_prev1 <= 120
 -120 <= dist_diff_pair_c3_5_7_prev2 <= 120
 dist_diff_abs_pair_c3_5_7_prev2 <= 120
 -120 <= dist_diff_pair_c3_5_7_prev3 <= 120
 dist_diff_abs_pair_c3_5_7_prev3 <= 120
 dx_grid_abs_pair_c3_6_7 <= 59
 dy_grid_abs_pair_c3_6_7 <= 59
 -59 <= dx_grid_diff_c3_6_7 <= 59
 -59 <= dy_grid_diff_c3_6_7 <= 59
 dist_curr_pair_c3_6_7 <= 120
 -120 <= dist_diff_pair_c3_6_7_prev0 <= 120
 dist_diff_abs_pair_c3_6_7_prev0 <= 120
 -120 <= dist_diff_pair_c3_6_7_prev1 <= 120
 dist_diff_abs_pair_c3_6_7_prev1 <= 120
 -120 <= dist_diff_pair_c3_6_7_prev2 <= 120
 dist_diff_abs_pair_c3_6_7_prev2 <= 120
 -120 <= dist_diff_pair_c3_6_7_prev3 <= 120
 dist_diff_abs_pair_c3_6_7_prev3 <= 120
Binaries
 r_0 r_1 r_2 r_3 r_4 r_5 r_6 r_7 z1_0_1 z2_0_1 z1L_0_1 z1R_0_1 z2D_0_1
 z2U_0_1 z1_2_4 z2_2_4 z1L_2_4 z1R_2_4 z2D_2_4 z2U_2_4 z1_0_3 z2_0_3
 z1L_0_3 z1R_0_3 z2D_0_3 z2U_0_3 z1_2_3 z2_2_3 z1L_2_3 z1R_2_3 z2D_2_3
 z2U_2_3 z1_6_7 z2_6_7 z1L_6_7 z1R_6_7 z2D_6_7 z2U_6_7 z1_2_6 z2_2_6
 z1L_2_6 z1R_2_6 z2D_2_6 z2U_2_6 z1_2_5 z2_2_5 z1L_2_5 z1R_2_5 z2D_2_5
 z2U_2_5 p_left_0_1 p_right_0_1 p_down_0_1 p_up_0_1 p_left_0_2 p_right_0_2
 p_down_0_2 p_up_0_2 p_left_0_3 p_right_0_3 p_down_0_3 p_up_0_3 p_left_0_4
 p_right_0_4 p_down_0_4 p_up_0_4 p_left_0_5 p_right_0_5 p_down_0_5 p_up_0_5
 p_left_0_6 p_right_0_6 p_down_0_6 p_up_0_6 p_left_0_7 p_right_0_7
 p_down_0_7 p_up_0_7 p_left_1_2 p_right_1_2 p_down_1_2 p_up_1_2 p_left_1_3
 p_right_1_3 p_down_1_3 p_up_1_3 p_left_1_4 p_right_1_4 p_down_1_4 p_up_1_4
 p_left_1_5 p_right_1_5 p_down_1_5 p_up_1_5 p_left_1_6 p_right_1_6
 p_down_1_6 p_up_1_6 p_left_1_7 p_right_1_7 p_down_1_7 p_up_1_7 p_left_2_3
 p_right_2_3 p_down_2_3 p_up_2_3 p_left_2_4 p_right_2_4 p_down_2_4 p_up_2_4
 p_left_2_5 p_right_2_5 p_down_2_5 p_up_2_5 p_left_2_6 p_right_2_6
 p_down_2_6 p_up_2_6 p_left_2_7 p_right_2_7 p_down_2_7 p_up_2_7 p_left_3_4
 p_right_3_4 p_down_3_4 p_up_3_4 p_left_3_5 p_right_3_5 p_down_3_5 p_up_3_5
 p_left_3_6 p_right_3_6 p_down_3_6 p_up_3_6 p_left_3_7 p_right_3_7
 p_down_3_7 p_up_3_7 p_left_4_5 p_right_4_5 p_down_4_5 p_up_4_5 p_left_4_6
 p_right_4_6 p_down_4_6 p_up_4_6 p_left_4_7 p_right_4_7 p_down_4_7 p_up_4_7
 p_left_5_6 p_right_5_6 p_down_5_6 p_up_5_6 p_left_5_7 p_right_5_7
 p_down_5_7 p_up_5_7 p_left_6_7 p_right_6_7 p_down_6_7 p_up_6_7
 dx_abs_0_1_is_positive dy_abs_0_1_is_positive dx_abs_2_4_is_positive
 dy_abs_2_4_is_positive dx_abs_0_3_is_positive dy_abs_0_3_is_positive
 dx_abs_2_3_is_positive dy_abs_2_3_is_positive dx_abs_6_7_is_positive
 dy_abs_6_7_is_positive dx_abs_2_6_is_positive dy_abs_2_6_is_positive
 dx_abs_2_5_is_positive dy_abs_2_5_is_positive
 dx_grid_abs_pair_c3_0_1_is_positive dy_grid_abs_pair_c3_0_1_is_positive
 diff_dist_pair_c3_0_1 same_dist_pair_c3_0_1_prev0
 dist_diff_abs_pair_c3_0_1_prev0_is_positive same_dist_pair_c3_0_1_prev1
 dist_diff_abs_pair_c3_0_1_prev1_is_positive same_dist_pair_c3_0_1_prev2
 dist_diff_abs_pair_c3_0_1_prev2_is_positive same_dist_pair_c3_0_1_prev3
 dist_diff_abs_pair_c3_0_1_prev3_is_positive
 dx_grid_abs_pair_c3_0_2_is_positive dy_grid_abs_pair_c3_0_2_is_positive
 diff_dist_pair_c3_0_2 same_dist_pair_c3_0_2_prev0
 dist_diff_abs_pair_c3_0_2_prev0_is_positive same_dist_pair_c3_0_2_prev1
 dist_diff_abs_pair_c3_0_2_prev1_is_positive same_dist_pair_c3_0_2_prev2
 dist_diff_abs_pair_c3_0_2_prev2_is_positive same_dist_pair_c3_0_2_prev3
 dist_diff_abs_pair_c3_0_2_prev3_is_positive
 dx_grid_abs_pair_c3_0_3_is_positive dy_grid_abs_pair_c3_0_3_is_positive
 diff_dist_pair_c3_0_3 same_dist_pair_c3_0_3_prev0
 dist_diff_abs_pair_c3_0_3_prev0_is_positive same_dist_pair_c3_0_3_prev1
 dist_diff_abs_pair_c3_0_3_prev1_is_positive same_dist_pair_c3_0_3_prev2
 dist_diff_abs_pair_c3_0_3_prev2_is_positive same_dist_pair_c3_0_3_prev3
 dist_diff_abs_pair_c3_0_3_prev3_is_positive
 dx_grid_abs_pair_c3_0_4_is_positive dy_grid_abs_pair_c3_0_4_is_positive
 diff_dist_pair_c3_0_4 same_dist_pair_c3_0_4_prev0
 dist_diff_abs_pair_c3_0_4_prev0_is_positive same_dist_pair_c3_0_4_prev1
 dist_diff_abs_pair_c3_0_4_prev1_is_positive same_dist_pair_c3_0_4_prev2
 dist_diff_abs_pair_c3_0_4_prev2_is_positive same_dist_pair_c3_0_4_prev3
 dist_diff_abs_pair_c3_0_4_prev3_is_positive
 dx_grid_abs_pair_c3_0_5_is_positive dy_grid_abs_pair_c3_0_5_is_positive
 diff_dist_pair_c3_0_5 same_dist_pair_c3_0_5_prev0
 dist_diff_abs_pair_c3_0_5_prev0_is_positive same_dist_pair_c3_0_5_prev1
 dist_diff_abs_pair_c3_0_5_prev1_is_positive same_dist_pair_c3_0_5_prev2
 dist_diff_abs_pair_c3_0_5_prev2_is_positive same_dist_pair_c3_0_5_prev3
 dist_diff_abs_pair_c3_0_5_prev3_is_positive
 dx_grid_abs_pair_c3_0_6_is_positive dy_grid_abs_pair_c3_0_6_is_positive
 diff_dist_pair_c3_0_6 same_dist_pair_c3_0_6_prev0
 dist_diff_abs_pair_c3_0_6_prev0_is_positive same_dist_pair_c3_0_6_prev1
 dist_diff_abs_pair_c3_0_6_prev1_is_positive same_dist_pair_c3_0_6_prev2
 dist_diff_abs_pair_c3_0_6_prev2_is_positive same_dist_pair_c3_0_6_prev3
 dist_diff_abs_pair_c3_0_6_prev3_is_positive
 dx_grid_abs_pair_c3_0_7_is_positive dy_grid_abs_pair_c3_0_7_is_positive
 diff_dist_pair_c3_0_7 same_dist_pair_c3_0_7_prev0
 dist_diff_abs_pair_c3_0_7_prev0_is_positive same_dist_pair_c3_0_7_prev1
 dist_diff_abs_pair_c3_0_7_prev1_is_positive same_dist_pair_c3_0_7_prev2
 dist_diff_abs_pair_c3_0_7_prev2_is_positive same_dist_pair_c3_0_7_prev3
 dist_diff_abs_pair_c3_0_7_prev3_is_positive
 dx_grid_abs_pair_c3_1_2_is_positive dy_grid_abs_pair_c3_1_2_is_positive
 diff_dist_pair_c3_1_2 same_dist_pair_c3_1_2_prev0
 dist_diff_abs_pair_c3_1_2_prev0_is_positive same_dist_pair_c3_1_2_prev1
 dist_diff_abs_pair_c3_1_2_prev1_is_positive same_dist_pair_c3_1_2_prev2
 dist_diff_abs_pair_c3_1_2_prev2_is_positive same_dist_pair_c3_1_2_prev3
 dist_diff_abs_pair_c3_1_2_prev3_is_positive
 dx_grid_abs_pair_c3_1_3_is_positive dy_grid_abs_pair_c3_1_3_is_positive
 diff_dist_pair_c3_1_3 same_dist_pair_c3_1_3_prev0
 dist_diff_abs_pair_c3_1_3_prev0_is_positive same_dist_pair_c3_1_3_prev1
 dist_diff_abs_pair_c3_1_3_prev1_is_positive same_dist_pair_c3_1_3_prev2
 dist_diff_abs_pair_c3_1_3_prev2_is_positive same_dist_pair_c3_1_3_prev3
 dist_diff_abs_pair_c3_1_3_prev3_is_positive
 dx_grid_abs_pair_c3_1_4_is_positive dy_grid_abs_pair_c3_1_4_is_positive
 diff_dist_pair_c3_1_4 same_dist_pair_c3_1_4_prev0
 dist_diff_abs_pair_c3_1_4_prev0_is_positive same_dist_pair_c3_1_4_prev1
 dist_diff_abs_pair_c3_1_4_prev1_is_positive same_dist_pair_c3_1_4_prev2
 dist_diff_abs_pair_c3_1_4_prev2_is_positive same_dist_pair_c3_1_4_prev3
 dist_diff_abs_pair_c3_1_4_prev3_is_positive
 dx_grid_abs_pair_c3_1_5_is_positive dy_grid_abs_pair_c3_1_5_is_positive
 diff_dist_pair_c3_1_5 same_dist_pair_c3_1_5_prev0
 dist_diff_abs_pair_c3_1_5_prev0_is_positive same_dist_pair_c3_1_5_prev1
 dist_diff_abs_pair_c3_1_5_prev1_is_positive same_dist_pair_c3_1_5_prev2
 dist_diff_abs_pair_c3_1_5_prev2_is_positive same_dist_pair_c3_1_5_prev3
 dist_diff_abs_pair_c3_1_5_prev3_is_positive
 dx_grid_abs_pair_c3_1_6_is_positive dy_grid_abs_pair_c3_1_6_is_positive
 diff_dist_pair_c3_1_6 same_dist_pair_c3_1_6_prev0
 dist_diff_abs_pair_c3_1_6_prev0_is_positive same_dist_pair_c3_1_6_prev1
 dist_diff_abs_pair_c3_1_6_prev1_is_positive same_dist_pair_c3_1_6_prev2
 dist_diff_abs_pair_c3_1_6_prev2_is_positive same_dist_pair_c3_1_6_prev3
 dist_diff_abs_pair_c3_1_6_prev3_is_positive
 dx_grid_abs_pair_c3_1_7_is_positive dy_grid_abs_pair_c3_1_7_is_positive
 diff_dist_pair_c3_1_7 same_dist_pair_c3_1_7_prev0
 dist_diff_abs_pair_c3_1_7_prev0_is_positive same_dist_pair_c3_1_7_prev1
 dist_diff_abs_pair_c3_1_7_prev1_is_positive same_dist_pair_c3_1_7_prev2
 dist_diff_abs_pair_c3_1_7_prev2_is_positive same_dist_pair_c3_1_7_prev3
 dist_diff_abs_pair_c3_1_7_prev3_is_positive
 dx_grid_abs_pair_c3_2_3_is_positive dy_grid_abs_pair_c3_2_3_is_positive
 diff_dist_pair_c3_2_3 same_dist_pair_c3_2_3_prev0
 dist_diff_abs_pair_c3_2_3_prev0_is_positive same_dist_pair_c3_2_3_prev1
 dist_diff_abs_pair_c3_2_3_prev1_is_positive same_dist_pair_c3_2_3_prev2
 dist_diff_abs_pair_c3_2_3_prev2_is_positive same_dist_pair_c3_2_3_prev3
 dist_diff_abs_pair_c3_2_3_prev3_is_positive
 dx_grid_abs_pair_c3_2_4_is_positive dy_grid_abs_pair_c3_2_4_is_positive
 diff_dist_pair_c3_2_4 same_dist_pair_c3_2_4_prev0
 dist_diff_abs_pair_c3_2_4_prev0_is_positive same_dist_pair_c3_2_4_prev1
 dist_diff_abs_pair_c3_2_4_prev1_is_positive same_dist_pair_c3_2_4_prev2
 dist_diff_abs_pair_c3_2_4_prev2_is_positive same_dist_pair_c3_2_4_prev3
 dist_diff_abs_pair_c3_2_4_prev3_is_positive
 dx_grid_abs_pair_c3_2_5_is_positive dy_grid_abs_pair_c3_2_5_is_positive
 diff_dist_pair_c3_2_5 same_dist_pair_c3_2_5_prev0
 dist_diff_abs_pair_c3_2_5_prev0_is_positive same_dist_pair_c3_2_5_prev1
 dist_diff_abs_pair_c3_2_5_prev1_is_positive same_dist_pair_c3_2_5_prev2
 dist_diff_abs_pair_c3_2_5_prev2_is_positive same_dist_pair_c3_2_5_prev3
 dist_diff_abs_pair_c3_2_5_prev3_is_positive
 dx_grid_abs_pair_c3_2_6_is_positive dy_grid_abs_pair_c3_2_6_is_positive
 diff_dist_pair_c3_2_6 same_dist_pair_c3_2_6_prev0
 dist_diff_abs_pair_c3_2_6_prev0_is_positive same_dist_pair_c3_2_6_prev1
 dist_diff_abs_pair_c3_2_6_prev1_is_positive same_dist_pair_c3_2_6_prev2
 dist_diff_abs_pair_c3_2_6_prev2_is_positive same_dist_pair_c3_2_6_prev3
 dist_diff_abs_pair_c3_2_6_prev3_is_positive
 dx_grid_abs_pair_c3_2_7_is_positive dy_grid_abs_pair_c3_2_7_is_positive
 diff_dist_pair_c3_2_7 same_dist_pair_c3_2_7_prev0
 dist_diff_abs_pair_c3_2_7_prev0_is_positive same_dist_pair_c3_2_7_prev1
 dist_diff_abs_pair_c3_2_7_prev1_is_positive same_dist_pair_c3_2_7_prev2
 dist_diff_abs_pair_c3_2_7_prev2_is_positive same_dist_pair_c3_2_7_prev3
 dist_diff_abs_pair_c3_2_7_prev3_is_positive
 dx_grid_abs_pair_c3_3_4_is_positive dy_grid_abs_pair_c3_3_4_is_positive
 diff_dist_pair_c3_3_4 same_dist_pair_c3_3_4_prev0
 dist_diff_abs_pair_c3_3_4_prev0_is_positive same_dist_pair_c3_3_4_prev1
 dist_diff_abs_pair_c3_3_4_prev1_is_positive same_dist_pair_c3_3_4_prev2
 dist_diff_abs_pair_c3_3_4_prev2_is_positive same_dist_pair_c3_3_4_prev3
 dist_diff_abs_pair_c3_3_4_prev3_is_positive
 dx_grid_abs_pair_c3_3_5_is_positive dy_grid_abs_pair_c3_3_5_is_positive
 diff_dist_pair_c3_3_5 same_dist_pair_c3_3_5_prev0
 dist_diff_abs_pair_c3_3_5_prev0_is_positive same_dist_pair_c3_3_5_prev1
 dist_diff_abs_pair_c3_3_5_prev1_is_positive same_dist_pair_c3_3_5_prev2
 dist_diff_abs_pair_c3_3_5_prev2_is_positive same_dist_pair_c3_3_5_prev3
 dist_diff_abs_pair_c3_3_5_prev3_is_positive
 dx_grid_abs_pair_c3_3_6_is_positive dy_grid_abs_pair_c3_3_6_is_positive
 diff_dist_pair_c3_3_6 same_dist_pair_c3_3_6_prev0
 dist_diff_abs_pair_c3_3_6_prev0_is_positive same_dist_pair_c3_3_6_prev1
 dist_diff_abs_pair_c3_3_6_prev1_is_positive same_dist_pair_c3_3_6_prev2
 dist_diff_abs_pair_c3_3_6_prev2_is_positive same_dist_pair_c3_3_6_prev3
 dist_diff_abs_pair_c3_3_6_prev3_is_positive
 dx_grid_abs_pair_c3_3_7_is_positive dy_grid_abs_pair_c3_3_7_is_positive
 diff_dist_pair_c3_3_7 same_dist_pair_c3_3_7_prev0
 dist_diff_abs_pair_c3_3_7_prev0_is_positive same_dist_pair_c3_3_7_prev1
 dist_diff_abs_pair_c3_3_7_prev1_is_positive same_dist_pair_c3_3_7_prev2
 dist_diff_abs_pair_c3_3_7_prev2_is_positive same_dist_pair_c3_3_7_prev3
 dist_diff_abs_pair_c3_3_7_prev3_is_positive
 dx_grid_abs_pair_c3_4_5_is_positive dy_grid_abs_pair_c3_4_5_is_positive
 diff_dist_pair_c3_4_5 same_dist_pair_c3_4_5_prev0
 dist_diff_abs_pair_c3_4_5_prev0_is_positive same_dist_pair_c3_4_5_prev1
 dist_diff_abs_pair_c3_4_5_prev1_is_positive same_dist_pair_c3_4_5_prev2
 dist_diff_abs_pair_c3_4_5_prev2_is_positive same_dist_pair_c3_4_5_prev3
 dist_diff_abs_pair_c3_4_5_prev3_is_positive
 dx_grid_abs_pair_c3_4_6_is_positive dy_grid_abs_pair_c3_4_6_is_positive
 diff_dist_pair_c3_4_6 same_dist_pair_c3_4_6_prev0
 dist_diff_abs_pair_c3_4_6_prev0_is_positive same_dist_pair_c3_4_6_prev1
 dist_diff_abs_pair_c3_4_6_prev1_is_positive same_dist_pair_c3_4_6_prev2
 dist_diff_abs_pair_c3_4_6_prev2_is_positive same_dist_pair_c3_4_6_prev3
 dist_diff_abs_pair_c3_4_6_prev3_is_positive
 dx_grid_abs_pair_c3_4_7_is_positive dy_grid_abs_pair_c3_4_7_is_positive
 diff_dist_pair_c3_4_7 same_dist_pair_c3_4_7_prev0
 dist_diff_abs_pair_c3_4_7_prev0_is_positive same_dist_pair_c3_4_7_prev1
 dist_diff_abs_pair_c3_4_7_prev1_is_positive same_dist_pair_c3_4_7_prev2
 dist_diff_abs_pair_c3_4_7_prev2_is_positive same_dist_pair_c3_4_7_prev3
 dist_diff_abs_pair_c3_4_7_prev3_is_positive
 dx_grid_abs_pair_c3_5_6_is_positive dy_grid_abs_pair_c3_5_6_is_positive
 diff_dist_pair_c3_5_6 same_dist_pair_c3_5_6_prev0
 dist_diff_abs_pair_c3_5_6_prev0_is_positive same_dist_pair_c3_5_6_prev1
 dist_diff_abs_pair_c3_5_6_prev1_is_positive same_dist_pair_c3_5_6_prev2
 dist_diff_abs_pair_c3_5_6_prev2_is_positive same_dist_pair_c3_5_6_prev3
 dist_diff_abs_pair_c3_5_6_prev3_is_positive
 dx_grid_abs_pair_c3_5_7_is_positive dy_grid_abs_pair_c3_5_7_is_positive
 diff_dist_pair_c3_5_7 same_dist_pair_c3_5_7_prev0
 dist_diff_abs_pair_c3_5_7_prev0_is_positive same_dist_pair_c3_5_7_prev1
 dist_diff_abs_pair_c3_5_7_prev1_is_positive same_dist_pair_c3_5_7_prev2
 dist_diff_abs_pair_c3_5_7_prev2_is_positive same_dist_pair_c3_5_7_prev3
 dist_diff_abs_pair_c3_5_7_prev3_is_positive
 dx_grid_abs_pair_c3_6_7_is_positive dy_grid_abs_pair_c3_6_7_is_positive
 diff_dist_pair_c3_6_7 same_dist_pair_c3_6_7_prev0
 dist_diff_abs_pair_c3_6_7_prev0_is_positive same_dist_pair_c3_6_7_prev1
 dist_diff_abs_pair_c3_6_7_prev1_is_positive same_dist_pair_c3_6_7_prev2
 dist_diff_abs_pair_c3_6_7_prev2_is_positive same_dist_pair_c3_6_7_prev3
 dist_diff_abs_pair_c3_6_7_prev3_is_positive
Generals
 x_grid_0 y_grid_0 x_grid_1 y_grid_1 x_grid_2 y_grid_2 x_grid_3 y_grid_3
 x_grid_4 y_grid_4 x_grid_5 y_grid_5 x_grid_6 y_grid_6 x_grid_7 y_grid_7
End
