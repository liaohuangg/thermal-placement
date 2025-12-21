\ Model ChipletPlacementGrid
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x5bea5a39b558c153
Minimize
  0 r_0 + 0 r_2 + 0 r_4 + dx_abs_0_1 + dy_abs_0_1 + dx_abs_2_4 + dy_abs_2_4
   + dx_abs_1_2 + dy_abs_1_2 + dx_abs_1_3 + dy_abs_1_3
   + 0.1 bbox_area_proxy_t
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
 width_rotation_0: w_0 = 16
 height_rotation_0: h_0 = 16
 width_rotation_1: - 10 r_1 + w_1 = 10
 height_rotation_1: 10 r_1 + h_1 = 20
 width_rotation_2: w_2 = 20
 height_rotation_2: h_2 = 20
 width_rotation_3: 2 r_3 + w_3 = 16
 height_rotation_3: - 2 r_3 + h_3 = 14
 width_rotation_4: w_4 = 4
 height_rotation_4: h_4 = 4
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
 w_align_grid_0: w_0 = 16
 h_align_grid_0: h_0 = 16
 w_max_0: w_0 <= 60
 h_max_0: h_0 <= 60
 w_align_grid_1: w_1 = 10
 h_align_grid_1: h_1 = 20
 w_max_1: w_1 <= 60
 h_max_1: h_1 <= 60
 w_align_grid_2: w_2 = 20
 h_align_grid_2: h_2 = 20
 w_max_2: w_2 <= 60
 h_max_2: h_2 <= 60
 w_align_grid_3: w_3 = 16
 h_align_grid_3: h_3 = 14
 w_max_3: w_3 <= 60
 h_max_3: h_3 <= 60
 w_align_grid_4: w_4 = 4
 h_align_grid_4: h_4 = 4
 w_max_4: w_4 <= 60
 h_max_4: h_4 <= 60
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
 must_adjacent_1_2: z1_1_2 + z2_1_2 = 1
 horizontal_direction_1_2: - z1_1_2 + z1L_1_2 + z1R_1_2 = 0
 vertical_direction_1_2: - z2_1_2 + z2D_1_2 + z2U_1_2 = 0
 horizontal_left_dist_1_2: - x_1 + x_2 - w_1 + 200 z1L_1_2 <= 200.5
 horizontal_left_dist_lb_1_2: - x_1 + x_2 - w_1 - 200 z1L_1_2 >= -200
 horizontal_right_dist_1_2: x_1 - x_2 - w_2 + 200 z1R_1_2 <= 200.5
 horizontal_right_dist_lb_1_2: x_1 - x_2 - w_2 - 200 z1R_1_2 >= -200
 horizontal_overlap_y1_1_2: y_1 - y_2 - h_2 + 200 z1_1_2 <= 200
 horizontal_overlap_y2_1_2: - y_1 + y_2 - h_1 + 200 z1_1_2 <= 200
 shared_y_ub1_1_2: - y_1 + y_2 - h_1 + 200 z1_1_2 + shared_y_1_2 <= 200
 shared_y_ub2_1_2: y_1 - y_2 - h_2 + 200 z1_1_2 + shared_y_1_2 <= 200
 shared_y_min_1_2: - 200 z1_1_2 + shared_y_1_2 >= -199.9
 shared_y_zero_1_2: - 200 z1_1_2 + shared_y_1_2 <= 0
 vertical_down_dist_1_2: - y_1 + y_2 - h_1 + 200 z2D_1_2 <= 200.5
 vertical_down_dist_lb_1_2: - y_1 + y_2 - h_1 - 200 z2D_1_2 >= -200
 vertical_up_dist_1_2: y_1 - y_2 - h_2 + 200 z2U_1_2 <= 200.5
 vertical_up_dist_lb_1_2: y_1 - y_2 - h_2 - 200 z2U_1_2 >= -200
 vertical_overlap_x1_1_2: x_1 - x_2 - w_2 + 200 z2_1_2 <= 200
 vertical_overlap_x2_1_2: - x_1 + x_2 - w_1 + 200 z2_1_2 <= 200
 shared_x_ub1_1_2: - x_1 + x_2 - w_1 + 200 z2_1_2 + shared_x_1_2 <= 200
 shared_x_ub2_1_2: x_1 - x_2 - w_2 + 200 z2_1_2 + shared_x_1_2 <= 200
 shared_x_min_1_2: - 200 z2_1_2 + shared_x_1_2 >= -199.9
 shared_x_zero_1_2: - 200 z2_1_2 + shared_x_1_2 <= 0
 must_adjacent_1_3: z1_1_3 + z2_1_3 = 1
 horizontal_direction_1_3: - z1_1_3 + z1L_1_3 + z1R_1_3 = 0
 vertical_direction_1_3: - z2_1_3 + z2D_1_3 + z2U_1_3 = 0
 horizontal_left_dist_1_3: - x_1 + x_3 - w_1 + 200 z1L_1_3 <= 200.5
 horizontal_left_dist_lb_1_3: - x_1 + x_3 - w_1 - 200 z1L_1_3 >= -200
 horizontal_right_dist_1_3: x_1 - x_3 - w_3 + 200 z1R_1_3 <= 200.5
 horizontal_right_dist_lb_1_3: x_1 - x_3 - w_3 - 200 z1R_1_3 >= -200
 horizontal_overlap_y1_1_3: y_1 - y_3 - h_3 + 200 z1_1_3 <= 200
 horizontal_overlap_y2_1_3: - y_1 + y_3 - h_1 + 200 z1_1_3 <= 200
 shared_y_ub1_1_3: - y_1 + y_3 - h_1 + 200 z1_1_3 + shared_y_1_3 <= 200
 shared_y_ub2_1_3: y_1 - y_3 - h_3 + 200 z1_1_3 + shared_y_1_3 <= 200
 shared_y_min_1_3: - 200 z1_1_3 + shared_y_1_3 >= -199.9
 shared_y_zero_1_3: - 200 z1_1_3 + shared_y_1_3 <= 0
 vertical_down_dist_1_3: - y_1 + y_3 - h_1 + 200 z2D_1_3 <= 200.5
 vertical_down_dist_lb_1_3: - y_1 + y_3 - h_1 - 200 z2D_1_3 >= -200
 vertical_up_dist_1_3: y_1 - y_3 - h_3 + 200 z2U_1_3 <= 200.5
 vertical_up_dist_lb_1_3: y_1 - y_3 - h_3 - 200 z2U_1_3 >= -200
 vertical_overlap_x1_1_3: x_1 - x_3 - w_3 + 200 z2_1_3 <= 200
 vertical_overlap_x2_1_3: - x_1 + x_3 - w_1 + 200 z2_1_3 <= 200
 shared_x_ub1_1_3: - x_1 + x_3 - w_1 + 200 z2_1_3 + shared_x_1_3 <= 200
 shared_x_ub2_1_3: x_1 - x_3 - w_3 + 200 z2_1_3 + shared_x_1_3 <= 200
 shared_x_min_1_3: - 200 z2_1_3 + shared_x_1_3 >= -199.9
 shared_x_zero_1_3: - 200 z2_1_3 + shared_x_1_3 <= 0
 boundary_x_0: x_0 + w_0 <= 60
 boundary_y_0: y_0 + h_0 <= 60
 boundary_x_1: x_1 + w_1 <= 60
 boundary_y_1: y_1 + h_1 <= 60
 boundary_x_2: x_2 + w_2 <= 60
 boundary_y_2: y_2 + h_2 <= 60
 boundary_x_3: x_3 + w_3 <= 60
 boundary_y_3: y_3 + h_3 <= 60
 boundary_x_4: x_4 + w_4 <= 60
 boundary_y_4: y_4 + h_4 <= 60
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
 non_overlap_any_3_4: p_left_3_4 + p_right_3_4 + p_down_3_4 + p_up_3_4 = 1
 non_overlap_left_3_4: x_3 - x_4 + w_3 + 200 p_left_3_4 <= 200
 non_overlap_left_rev_3_4: - x_3 + x_4 - w_3 - 200 p_left_3_4 <= 0
 non_overlap_right_3_4: - x_3 + x_4 + w_4 + 200 p_right_3_4 <= 200
 non_overlap_right_rev_3_4: x_3 - x_4 - w_4 - 200 p_right_3_4 <= 0
 non_overlap_down_3_4: y_3 - y_4 + h_3 + 200 p_down_3_4 <= 200
 non_overlap_down_rev_3_4: - y_3 + y_4 - h_3 - 200 p_down_3_4 <= 0
 non_overlap_up_3_4: - y_3 + y_4 + h_4 + 200 p_up_3_4 <= 200
 non_overlap_up_rev_3_4: y_3 - y_4 - h_4 - 200 p_up_3_4 <= 0
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
 bbox_w_def: bbox_min_x - bbox_max_x + bbox_w = 0
 bbox_h_def: bbox_min_y - bbox_max_y + bbox_h = 0
 dx_diff_def_0_1: - cx_0 + cx_1 + dx_diff_0_1 = 0
 dy_diff_def_0_1: - cy_0 + cy_1 + dy_diff_0_1 = 0
 dx_abs_0_1_abs_ge_orig: dx_abs_0_1 - dx_diff_0_1
   - 60 dx_abs_0_1_is_positive >= -60
 dx_abs_0_1_abs_le_orig: dx_abs_0_1 - dx_diff_0_1
   + 60 dx_abs_0_1_is_positive <= 60
 dx_abs_0_1_abs_ge_neg_orig: dx_abs_0_1 + dx_diff_0_1
   + 60 dx_abs_0_1_is_positive >= 0
 dx_abs_0_1_abs_le_neg_orig: dx_abs_0_1 + dx_diff_0_1
   - 60 dx_abs_0_1_is_positive <= 0
 dx_abs_0_1_force_positive: dx_diff_0_1 - 60 dx_abs_0_1_is_positive >= -60
 dx_abs_0_1_force_negative: dx_diff_0_1 - 60 dx_abs_0_1_is_positive
   <= -0.001
 dy_abs_0_1_abs_ge_orig: dy_abs_0_1 - dy_diff_0_1
   - 60 dy_abs_0_1_is_positive >= -60
 dy_abs_0_1_abs_le_orig: dy_abs_0_1 - dy_diff_0_1
   + 60 dy_abs_0_1_is_positive <= 60
 dy_abs_0_1_abs_ge_neg_orig: dy_abs_0_1 + dy_diff_0_1
   + 60 dy_abs_0_1_is_positive >= 0
 dy_abs_0_1_abs_le_neg_orig: dy_abs_0_1 + dy_diff_0_1
   - 60 dy_abs_0_1_is_positive <= 0
 dy_abs_0_1_force_positive: dy_diff_0_1 - 60 dy_abs_0_1_is_positive >= -60
 dy_abs_0_1_force_negative: dy_diff_0_1 - 60 dy_abs_0_1_is_positive
   <= -0.001
 dx_diff_def_2_4: - cx_2 + cx_4 + dx_diff_2_4 = 0
 dy_diff_def_2_4: - cy_2 + cy_4 + dy_diff_2_4 = 0
 dx_abs_2_4_abs_ge_orig: dx_abs_2_4 - dx_diff_2_4
   - 60 dx_abs_2_4_is_positive >= -60
 dx_abs_2_4_abs_le_orig: dx_abs_2_4 - dx_diff_2_4
   + 60 dx_abs_2_4_is_positive <= 60
 dx_abs_2_4_abs_ge_neg_orig: dx_abs_2_4 + dx_diff_2_4
   + 60 dx_abs_2_4_is_positive >= 0
 dx_abs_2_4_abs_le_neg_orig: dx_abs_2_4 + dx_diff_2_4
   - 60 dx_abs_2_4_is_positive <= 0
 dx_abs_2_4_force_positive: dx_diff_2_4 - 60 dx_abs_2_4_is_positive >= -60
 dx_abs_2_4_force_negative: dx_diff_2_4 - 60 dx_abs_2_4_is_positive
   <= -0.001
 dy_abs_2_4_abs_ge_orig: dy_abs_2_4 - dy_diff_2_4
   - 60 dy_abs_2_4_is_positive >= -60
 dy_abs_2_4_abs_le_orig: dy_abs_2_4 - dy_diff_2_4
   + 60 dy_abs_2_4_is_positive <= 60
 dy_abs_2_4_abs_ge_neg_orig: dy_abs_2_4 + dy_diff_2_4
   + 60 dy_abs_2_4_is_positive >= 0
 dy_abs_2_4_abs_le_neg_orig: dy_abs_2_4 + dy_diff_2_4
   - 60 dy_abs_2_4_is_positive <= 0
 dy_abs_2_4_force_positive: dy_diff_2_4 - 60 dy_abs_2_4_is_positive >= -60
 dy_abs_2_4_force_negative: dy_diff_2_4 - 60 dy_abs_2_4_is_positive
   <= -0.001
 dx_diff_def_1_2: - cx_1 + cx_2 + dx_diff_1_2 = 0
 dy_diff_def_1_2: - cy_1 + cy_2 + dy_diff_1_2 = 0
 dx_abs_1_2_abs_ge_orig: dx_abs_1_2 - dx_diff_1_2
   - 60 dx_abs_1_2_is_positive >= -60
 dx_abs_1_2_abs_le_orig: dx_abs_1_2 - dx_diff_1_2
   + 60 dx_abs_1_2_is_positive <= 60
 dx_abs_1_2_abs_ge_neg_orig: dx_abs_1_2 + dx_diff_1_2
   + 60 dx_abs_1_2_is_positive >= 0
 dx_abs_1_2_abs_le_neg_orig: dx_abs_1_2 + dx_diff_1_2
   - 60 dx_abs_1_2_is_positive <= 0
 dx_abs_1_2_force_positive: dx_diff_1_2 - 60 dx_abs_1_2_is_positive >= -60
 dx_abs_1_2_force_negative: dx_diff_1_2 - 60 dx_abs_1_2_is_positive
   <= -0.001
 dy_abs_1_2_abs_ge_orig: dy_abs_1_2 - dy_diff_1_2
   - 60 dy_abs_1_2_is_positive >= -60
 dy_abs_1_2_abs_le_orig: dy_abs_1_2 - dy_diff_1_2
   + 60 dy_abs_1_2_is_positive <= 60
 dy_abs_1_2_abs_ge_neg_orig: dy_abs_1_2 + dy_diff_1_2
   + 60 dy_abs_1_2_is_positive >= 0
 dy_abs_1_2_abs_le_neg_orig: dy_abs_1_2 + dy_diff_1_2
   - 60 dy_abs_1_2_is_positive <= 0
 dy_abs_1_2_force_positive: dy_diff_1_2 - 60 dy_abs_1_2_is_positive >= -60
 dy_abs_1_2_force_negative: dy_diff_1_2 - 60 dy_abs_1_2_is_positive
   <= -0.001
 dx_diff_def_1_3: - cx_1 + cx_3 + dx_diff_1_3 = 0
 dy_diff_def_1_3: - cy_1 + cy_3 + dy_diff_1_3 = 0
 dx_abs_1_3_abs_ge_orig: dx_abs_1_3 - dx_diff_1_3
   - 60 dx_abs_1_3_is_positive >= -60
 dx_abs_1_3_abs_le_orig: dx_abs_1_3 - dx_diff_1_3
   + 60 dx_abs_1_3_is_positive <= 60
 dx_abs_1_3_abs_ge_neg_orig: dx_abs_1_3 + dx_diff_1_3
   + 60 dx_abs_1_3_is_positive >= 0
 dx_abs_1_3_abs_le_neg_orig: dx_abs_1_3 + dx_diff_1_3
   - 60 dx_abs_1_3_is_positive <= 0
 dx_abs_1_3_force_positive: dx_diff_1_3 - 60 dx_abs_1_3_is_positive >= -60
 dx_abs_1_3_force_negative: dx_diff_1_3 - 60 dx_abs_1_3_is_positive
   <= -0.001
 dy_abs_1_3_abs_ge_orig: dy_abs_1_3 - dy_diff_1_3
   - 60 dy_abs_1_3_is_positive >= -60
 dy_abs_1_3_abs_le_orig: dy_abs_1_3 - dy_diff_1_3
   + 60 dy_abs_1_3_is_positive <= 60
 dy_abs_1_3_abs_ge_neg_orig: dy_abs_1_3 + dy_diff_1_3
   + 60 dy_abs_1_3_is_positive >= 0
 dy_abs_1_3_abs_le_neg_orig: dy_abs_1_3 + dy_diff_1_3
   - 60 dy_abs_1_3_is_positive <= 0
 dy_abs_1_3_force_positive: dy_diff_1_3 - 60 dy_abs_1_3_is_positive >= -60
 dy_abs_1_3_force_negative: dy_diff_1_3 - 60 dy_abs_1_3_is_positive
   <= -0.001
 t_ge_width: - bbox_w + bbox_area_proxy_t >= 0
 t_ge_height: - bbox_h + bbox_area_proxy_t >= 0
 t_ge_scaled_mean: - 0.8 bbox_w - 0.8 bbox_h + bbox_area_proxy_t >= 0
 cx_grid_center_def_c25_0: - x_grid_0 + cx_grid_center_c25_0 = 16
 cy_grid_center_def_c25_0: - y_grid_0 + cy_grid_center_c25_0 = 16
 cx_grid_center_def_c25_1: - x_grid_1 + cx_grid_center_c25_1 = 10
 cy_grid_center_def_c25_1: - y_grid_1 + cy_grid_center_c25_1 = 20
 cx_grid_center_def_c25_2: - x_grid_2 + cx_grid_center_c25_2 = 20
 cy_grid_center_def_c25_2: - y_grid_2 + cy_grid_center_c25_2 = 20
 cx_grid_center_def_c25_3: - x_grid_3 + cx_grid_center_c25_3 = 16
 cy_grid_center_def_c25_3: - y_grid_3 + cy_grid_center_c25_3 = 14
 cx_grid_center_def_c25_4: - x_grid_4 + cx_grid_center_c25_4 = 4
 cy_grid_center_def_c25_4: - y_grid_4 + cy_grid_center_c25_4 = 4
 dx_grid_diff_def_c25_0_1: - cx_grid_center_c25_0 + cx_grid_center_c25_1
   + dx_grid_diff_c25_0_1 = 0
 dx_grid_abs_pair_c25_0_1_abs_ge_orig: dx_grid_abs_pair_c25_0_1
   - dx_grid_diff_c25_0_1 - 130 dx_grid_abs_pair_c25_0_1_is_positive
   >= -130
 dx_grid_abs_pair_c25_0_1_abs_le_orig: dx_grid_abs_pair_c25_0_1
   - dx_grid_diff_c25_0_1 + 130 dx_grid_abs_pair_c25_0_1_is_positive
   <= 130
 dx_grid_abs_pair_c25_0_1_abs_ge_neg_orig: dx_grid_abs_pair_c25_0_1
   + dx_grid_diff_c25_0_1 + 130 dx_grid_abs_pair_c25_0_1_is_positive >= 0
 dx_grid_abs_pair_c25_0_1_abs_le_neg_orig: dx_grid_abs_pair_c25_0_1
   + dx_grid_diff_c25_0_1 - 130 dx_grid_abs_pair_c25_0_1_is_positive <= 0
 dx_grid_abs_pair_c25_0_1_force_positive: dx_grid_diff_c25_0_1
   - 130 dx_grid_abs_pair_c25_0_1_is_positive >= -130
 dx_grid_abs_pair_c25_0_1_force_negative: dx_grid_diff_c25_0_1
   - 130 dx_grid_abs_pair_c25_0_1_is_positive <= -0.001
 dy_grid_diff_def_c25_0_1: - cy_grid_center_c25_0 + cy_grid_center_c25_1
   + dy_grid_diff_c25_0_1 = 0
 dy_grid_abs_pair_c25_0_1_abs_ge_orig: dy_grid_abs_pair_c25_0_1
   - dy_grid_diff_c25_0_1 - 130 dy_grid_abs_pair_c25_0_1_is_positive
   >= -130
 dy_grid_abs_pair_c25_0_1_abs_le_orig: dy_grid_abs_pair_c25_0_1
   - dy_grid_diff_c25_0_1 + 130 dy_grid_abs_pair_c25_0_1_is_positive
   <= 130
 dy_grid_abs_pair_c25_0_1_abs_ge_neg_orig: dy_grid_abs_pair_c25_0_1
   + dy_grid_diff_c25_0_1 + 130 dy_grid_abs_pair_c25_0_1_is_positive >= 0
 dy_grid_abs_pair_c25_0_1_abs_le_neg_orig: dy_grid_abs_pair_c25_0_1
   + dy_grid_diff_c25_0_1 - 130 dy_grid_abs_pair_c25_0_1_is_positive <= 0
 dy_grid_abs_pair_c25_0_1_force_positive: dy_grid_diff_c25_0_1
   - 130 dy_grid_abs_pair_c25_0_1_is_positive >= -130
 dy_grid_abs_pair_c25_0_1_force_negative: dy_grid_diff_c25_0_1
   - 130 dy_grid_abs_pair_c25_0_1_is_positive <= -0.001
 dist_curr_pair_def_c25_0_1: - dx_grid_abs_pair_c25_0_1
   - dy_grid_abs_pair_c25_0_1 + dist_curr_pair_c25_0_1 = 0
 dist_diff_pair_def_c25_0_1_prev0: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev0 = -30
 dist_diff_abs_pair_c25_0_1_prev0_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev0 + dist_diff_abs_pair_c25_0_1_prev0
   - 240 dist_diff_abs_pair_c25_0_1_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev0_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev0 + dist_diff_abs_pair_c25_0_1_prev0
   + 240 dist_diff_abs_pair_c25_0_1_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev0 + dist_diff_abs_pair_c25_0_1_prev0
   + 240 dist_diff_abs_pair_c25_0_1_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev0 + dist_diff_abs_pair_c25_0_1_prev0
   - 240 dist_diff_abs_pair_c25_0_1_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev0_force_positive:
   dist_diff_pair_c25_0_1_prev0
   - 240 dist_diff_abs_pair_c25_0_1_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev0_force_negative:
   dist_diff_pair_c25_0_1_prev0
   - 240 dist_diff_abs_pair_c25_0_1_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev0: 240 same_dist_pair_c25_0_1_prev0
   + dist_diff_abs_pair_c25_0_1_prev0 <= 249.8
 same_dist_pair_lower_c25_0_1_prev0: 240 same_dist_pair_c25_0_1_prev0
   + dist_diff_abs_pair_c25_0_1_prev0 >= 10
 dist_diff_pair_def_c25_0_1_prev1: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev1 = -36
 dist_diff_abs_pair_c25_0_1_prev1_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev1 + dist_diff_abs_pair_c25_0_1_prev1
   - 240 dist_diff_abs_pair_c25_0_1_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev1_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev1 + dist_diff_abs_pair_c25_0_1_prev1
   + 240 dist_diff_abs_pair_c25_0_1_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev1 + dist_diff_abs_pair_c25_0_1_prev1
   + 240 dist_diff_abs_pair_c25_0_1_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev1 + dist_diff_abs_pair_c25_0_1_prev1
   - 240 dist_diff_abs_pair_c25_0_1_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev1_force_positive:
   dist_diff_pair_c25_0_1_prev1
   - 240 dist_diff_abs_pair_c25_0_1_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev1_force_negative:
   dist_diff_pair_c25_0_1_prev1
   - 240 dist_diff_abs_pair_c25_0_1_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev1: 240 same_dist_pair_c25_0_1_prev1
   + dist_diff_abs_pair_c25_0_1_prev1 <= 249.8
 same_dist_pair_lower_c25_0_1_prev1: 240 same_dist_pair_c25_0_1_prev1
   + dist_diff_abs_pair_c25_0_1_prev1 >= 10
 dist_diff_pair_def_c25_0_1_prev2: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev2 = -26
 dist_diff_abs_pair_c25_0_1_prev2_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev2 + dist_diff_abs_pair_c25_0_1_prev2
   - 240 dist_diff_abs_pair_c25_0_1_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev2_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev2 + dist_diff_abs_pair_c25_0_1_prev2
   + 240 dist_diff_abs_pair_c25_0_1_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev2 + dist_diff_abs_pair_c25_0_1_prev2
   + 240 dist_diff_abs_pair_c25_0_1_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev2 + dist_diff_abs_pair_c25_0_1_prev2
   - 240 dist_diff_abs_pair_c25_0_1_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev2_force_positive:
   dist_diff_pair_c25_0_1_prev2
   - 240 dist_diff_abs_pair_c25_0_1_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev2_force_negative:
   dist_diff_pair_c25_0_1_prev2
   - 240 dist_diff_abs_pair_c25_0_1_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev2: 240 same_dist_pair_c25_0_1_prev2
   + dist_diff_abs_pair_c25_0_1_prev2 <= 249.8
 same_dist_pair_lower_c25_0_1_prev2: 240 same_dist_pair_c25_0_1_prev2
   + dist_diff_abs_pair_c25_0_1_prev2 >= 10
 dist_diff_pair_def_c25_0_1_prev3: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev3 = -26
 dist_diff_abs_pair_c25_0_1_prev3_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev3 + dist_diff_abs_pair_c25_0_1_prev3
   - 240 dist_diff_abs_pair_c25_0_1_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev3_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev3 + dist_diff_abs_pair_c25_0_1_prev3
   + 240 dist_diff_abs_pair_c25_0_1_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev3 + dist_diff_abs_pair_c25_0_1_prev3
   + 240 dist_diff_abs_pair_c25_0_1_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev3 + dist_diff_abs_pair_c25_0_1_prev3
   - 240 dist_diff_abs_pair_c25_0_1_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev3_force_positive:
   dist_diff_pair_c25_0_1_prev3
   - 240 dist_diff_abs_pair_c25_0_1_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev3_force_negative:
   dist_diff_pair_c25_0_1_prev3
   - 240 dist_diff_abs_pair_c25_0_1_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev3: 240 same_dist_pair_c25_0_1_prev3
   + dist_diff_abs_pair_c25_0_1_prev3 <= 249.8
 same_dist_pair_lower_c25_0_1_prev3: 240 same_dist_pair_c25_0_1_prev3
   + dist_diff_abs_pair_c25_0_1_prev3 >= 10
 dist_diff_pair_def_c25_0_1_prev4: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev4 = -40
 dist_diff_abs_pair_c25_0_1_prev4_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev4 + dist_diff_abs_pair_c25_0_1_prev4
   - 240 dist_diff_abs_pair_c25_0_1_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev4_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev4 + dist_diff_abs_pair_c25_0_1_prev4
   + 240 dist_diff_abs_pair_c25_0_1_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev4 + dist_diff_abs_pair_c25_0_1_prev4
   + 240 dist_diff_abs_pair_c25_0_1_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev4 + dist_diff_abs_pair_c25_0_1_prev4
   - 240 dist_diff_abs_pair_c25_0_1_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev4_force_positive:
   dist_diff_pair_c25_0_1_prev4
   - 240 dist_diff_abs_pair_c25_0_1_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev4_force_negative:
   dist_diff_pair_c25_0_1_prev4
   - 240 dist_diff_abs_pair_c25_0_1_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev4: 240 same_dist_pair_c25_0_1_prev4
   + dist_diff_abs_pair_c25_0_1_prev4 <= 249.8
 same_dist_pair_lower_c25_0_1_prev4: 240 same_dist_pair_c25_0_1_prev4
   + dist_diff_abs_pair_c25_0_1_prev4 >= 10
 dist_diff_pair_def_c25_0_1_prev5: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev5 = -30
 dist_diff_abs_pair_c25_0_1_prev5_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev5 + dist_diff_abs_pair_c25_0_1_prev5
   - 240 dist_diff_abs_pair_c25_0_1_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev5_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev5 + dist_diff_abs_pair_c25_0_1_prev5
   + 240 dist_diff_abs_pair_c25_0_1_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev5 + dist_diff_abs_pair_c25_0_1_prev5
   + 240 dist_diff_abs_pair_c25_0_1_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev5 + dist_diff_abs_pair_c25_0_1_prev5
   - 240 dist_diff_abs_pair_c25_0_1_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev5_force_positive:
   dist_diff_pair_c25_0_1_prev5
   - 240 dist_diff_abs_pair_c25_0_1_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev5_force_negative:
   dist_diff_pair_c25_0_1_prev5
   - 240 dist_diff_abs_pair_c25_0_1_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev5: 240 same_dist_pair_c25_0_1_prev5
   + dist_diff_abs_pair_c25_0_1_prev5 <= 249.8
 same_dist_pair_lower_c25_0_1_prev5: 240 same_dist_pair_c25_0_1_prev5
   + dist_diff_abs_pair_c25_0_1_prev5 >= 10
 dist_diff_pair_def_c25_0_1_prev6: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev6 = -30
 dist_diff_abs_pair_c25_0_1_prev6_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev6 + dist_diff_abs_pair_c25_0_1_prev6
   - 240 dist_diff_abs_pair_c25_0_1_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev6_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev6 + dist_diff_abs_pair_c25_0_1_prev6
   + 240 dist_diff_abs_pair_c25_0_1_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev6 + dist_diff_abs_pair_c25_0_1_prev6
   + 240 dist_diff_abs_pair_c25_0_1_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev6 + dist_diff_abs_pair_c25_0_1_prev6
   - 240 dist_diff_abs_pair_c25_0_1_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev6_force_positive:
   dist_diff_pair_c25_0_1_prev6
   - 240 dist_diff_abs_pair_c25_0_1_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev6_force_negative:
   dist_diff_pair_c25_0_1_prev6
   - 240 dist_diff_abs_pair_c25_0_1_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev6: 240 same_dist_pair_c25_0_1_prev6
   + dist_diff_abs_pair_c25_0_1_prev6 <= 249.8
 same_dist_pair_lower_c25_0_1_prev6: 240 same_dist_pair_c25_0_1_prev6
   + dist_diff_abs_pair_c25_0_1_prev6 >= 10
 dist_diff_pair_def_c25_0_1_prev7: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev7 = -30
 dist_diff_abs_pair_c25_0_1_prev7_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev7 + dist_diff_abs_pair_c25_0_1_prev7
   - 240 dist_diff_abs_pair_c25_0_1_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev7_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev7 + dist_diff_abs_pair_c25_0_1_prev7
   + 240 dist_diff_abs_pair_c25_0_1_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev7 + dist_diff_abs_pair_c25_0_1_prev7
   + 240 dist_diff_abs_pair_c25_0_1_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev7 + dist_diff_abs_pair_c25_0_1_prev7
   - 240 dist_diff_abs_pair_c25_0_1_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev7_force_positive:
   dist_diff_pair_c25_0_1_prev7
   - 240 dist_diff_abs_pair_c25_0_1_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev7_force_negative:
   dist_diff_pair_c25_0_1_prev7
   - 240 dist_diff_abs_pair_c25_0_1_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev7: 240 same_dist_pair_c25_0_1_prev7
   + dist_diff_abs_pair_c25_0_1_prev7 <= 249.8
 same_dist_pair_lower_c25_0_1_prev7: 240 same_dist_pair_c25_0_1_prev7
   + dist_diff_abs_pair_c25_0_1_prev7 >= 10
 dist_diff_pair_def_c25_0_1_prev8: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev8 = -45
 dist_diff_abs_pair_c25_0_1_prev8_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev8 + dist_diff_abs_pair_c25_0_1_prev8
   - 240 dist_diff_abs_pair_c25_0_1_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev8_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev8 + dist_diff_abs_pair_c25_0_1_prev8
   + 240 dist_diff_abs_pair_c25_0_1_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev8 + dist_diff_abs_pair_c25_0_1_prev8
   + 240 dist_diff_abs_pair_c25_0_1_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev8 + dist_diff_abs_pair_c25_0_1_prev8
   - 240 dist_diff_abs_pair_c25_0_1_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev8_force_positive:
   dist_diff_pair_c25_0_1_prev8
   - 240 dist_diff_abs_pair_c25_0_1_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev8_force_negative:
   dist_diff_pair_c25_0_1_prev8
   - 240 dist_diff_abs_pair_c25_0_1_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev8: 240 same_dist_pair_c25_0_1_prev8
   + dist_diff_abs_pair_c25_0_1_prev8 <= 249.8
 same_dist_pair_lower_c25_0_1_prev8: 240 same_dist_pair_c25_0_1_prev8
   + dist_diff_abs_pair_c25_0_1_prev8 >= 10
 dist_diff_pair_def_c25_0_1_prev9: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev9 = -26
 dist_diff_abs_pair_c25_0_1_prev9_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev9 + dist_diff_abs_pair_c25_0_1_prev9
   - 240 dist_diff_abs_pair_c25_0_1_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev9_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev9 + dist_diff_abs_pair_c25_0_1_prev9
   + 240 dist_diff_abs_pair_c25_0_1_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev9 + dist_diff_abs_pair_c25_0_1_prev9
   + 240 dist_diff_abs_pair_c25_0_1_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev9 + dist_diff_abs_pair_c25_0_1_prev9
   - 240 dist_diff_abs_pair_c25_0_1_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev9_force_positive:
   dist_diff_pair_c25_0_1_prev9
   - 240 dist_diff_abs_pair_c25_0_1_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev9_force_negative:
   dist_diff_pair_c25_0_1_prev9
   - 240 dist_diff_abs_pair_c25_0_1_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev9: 240 same_dist_pair_c25_0_1_prev9
   + dist_diff_abs_pair_c25_0_1_prev9 <= 249.8
 same_dist_pair_lower_c25_0_1_prev9: 240 same_dist_pair_c25_0_1_prev9
   + dist_diff_abs_pair_c25_0_1_prev9 >= 10
 dist_diff_pair_def_c25_0_1_prev10: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev10 = -36
 dist_diff_abs_pair_c25_0_1_prev10_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev10 + dist_diff_abs_pair_c25_0_1_prev10
   - 240 dist_diff_abs_pair_c25_0_1_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev10_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev10 + dist_diff_abs_pair_c25_0_1_prev10
   + 240 dist_diff_abs_pair_c25_0_1_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev10 + dist_diff_abs_pair_c25_0_1_prev10
   + 240 dist_diff_abs_pair_c25_0_1_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev10 + dist_diff_abs_pair_c25_0_1_prev10
   - 240 dist_diff_abs_pair_c25_0_1_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev10_force_positive:
   dist_diff_pair_c25_0_1_prev10
   - 240 dist_diff_abs_pair_c25_0_1_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev10_force_negative:
   dist_diff_pair_c25_0_1_prev10
   - 240 dist_diff_abs_pair_c25_0_1_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev10: 240 same_dist_pair_c25_0_1_prev10
   + dist_diff_abs_pair_c25_0_1_prev10 <= 249.8
 same_dist_pair_lower_c25_0_1_prev10: 240 same_dist_pair_c25_0_1_prev10
   + dist_diff_abs_pair_c25_0_1_prev10 >= 10
 dist_diff_pair_def_c25_0_1_prev11: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev11 = -26
 dist_diff_abs_pair_c25_0_1_prev11_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev11 + dist_diff_abs_pair_c25_0_1_prev11
   - 240 dist_diff_abs_pair_c25_0_1_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev11_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev11 + dist_diff_abs_pair_c25_0_1_prev11
   + 240 dist_diff_abs_pair_c25_0_1_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev11 + dist_diff_abs_pair_c25_0_1_prev11
   + 240 dist_diff_abs_pair_c25_0_1_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev11 + dist_diff_abs_pair_c25_0_1_prev11
   - 240 dist_diff_abs_pair_c25_0_1_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev11_force_positive:
   dist_diff_pair_c25_0_1_prev11
   - 240 dist_diff_abs_pair_c25_0_1_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev11_force_negative:
   dist_diff_pair_c25_0_1_prev11
   - 240 dist_diff_abs_pair_c25_0_1_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev11: 240 same_dist_pair_c25_0_1_prev11
   + dist_diff_abs_pair_c25_0_1_prev11 <= 249.8
 same_dist_pair_lower_c25_0_1_prev11: 240 same_dist_pair_c25_0_1_prev11
   + dist_diff_abs_pair_c25_0_1_prev11 >= 10
 dist_diff_pair_def_c25_0_1_prev12: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev12 = -38
 dist_diff_abs_pair_c25_0_1_prev12_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev12 + dist_diff_abs_pair_c25_0_1_prev12
   - 240 dist_diff_abs_pair_c25_0_1_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev12_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev12 + dist_diff_abs_pair_c25_0_1_prev12
   + 240 dist_diff_abs_pair_c25_0_1_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev12 + dist_diff_abs_pair_c25_0_1_prev12
   + 240 dist_diff_abs_pair_c25_0_1_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev12 + dist_diff_abs_pair_c25_0_1_prev12
   - 240 dist_diff_abs_pair_c25_0_1_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev12_force_positive:
   dist_diff_pair_c25_0_1_prev12
   - 240 dist_diff_abs_pair_c25_0_1_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev12_force_negative:
   dist_diff_pair_c25_0_1_prev12
   - 240 dist_diff_abs_pair_c25_0_1_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev12: 240 same_dist_pair_c25_0_1_prev12
   + dist_diff_abs_pair_c25_0_1_prev12 <= 249.8
 same_dist_pair_lower_c25_0_1_prev12: 240 same_dist_pair_c25_0_1_prev12
   + dist_diff_abs_pair_c25_0_1_prev12 >= 10
 dist_diff_pair_def_c25_0_1_prev13: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev13 = -30
 dist_diff_abs_pair_c25_0_1_prev13_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev13 + dist_diff_abs_pair_c25_0_1_prev13
   - 240 dist_diff_abs_pair_c25_0_1_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev13_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev13 + dist_diff_abs_pair_c25_0_1_prev13
   + 240 dist_diff_abs_pair_c25_0_1_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev13 + dist_diff_abs_pair_c25_0_1_prev13
   + 240 dist_diff_abs_pair_c25_0_1_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev13 + dist_diff_abs_pair_c25_0_1_prev13
   - 240 dist_diff_abs_pair_c25_0_1_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev13_force_positive:
   dist_diff_pair_c25_0_1_prev13
   - 240 dist_diff_abs_pair_c25_0_1_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev13_force_negative:
   dist_diff_pair_c25_0_1_prev13
   - 240 dist_diff_abs_pair_c25_0_1_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev13: 240 same_dist_pair_c25_0_1_prev13
   + dist_diff_abs_pair_c25_0_1_prev13 <= 249.8
 same_dist_pair_lower_c25_0_1_prev13: 240 same_dist_pair_c25_0_1_prev13
   + dist_diff_abs_pair_c25_0_1_prev13 >= 10
 dist_diff_pair_def_c25_0_1_prev14: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev14 = -26
 dist_diff_abs_pair_c25_0_1_prev14_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev14 + dist_diff_abs_pair_c25_0_1_prev14
   - 240 dist_diff_abs_pair_c25_0_1_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev14_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev14 + dist_diff_abs_pair_c25_0_1_prev14
   + 240 dist_diff_abs_pair_c25_0_1_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev14 + dist_diff_abs_pair_c25_0_1_prev14
   + 240 dist_diff_abs_pair_c25_0_1_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev14 + dist_diff_abs_pair_c25_0_1_prev14
   - 240 dist_diff_abs_pair_c25_0_1_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev14_force_positive:
   dist_diff_pair_c25_0_1_prev14
   - 240 dist_diff_abs_pair_c25_0_1_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev14_force_negative:
   dist_diff_pair_c25_0_1_prev14
   - 240 dist_diff_abs_pair_c25_0_1_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev14: 240 same_dist_pair_c25_0_1_prev14
   + dist_diff_abs_pair_c25_0_1_prev14 <= 249.8
 same_dist_pair_lower_c25_0_1_prev14: 240 same_dist_pair_c25_0_1_prev14
   + dist_diff_abs_pair_c25_0_1_prev14 >= 10
 dist_diff_pair_def_c25_0_1_prev15: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev15 = -55
 dist_diff_abs_pair_c25_0_1_prev15_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev15 + dist_diff_abs_pair_c25_0_1_prev15
   - 240 dist_diff_abs_pair_c25_0_1_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev15_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev15 + dist_diff_abs_pair_c25_0_1_prev15
   + 240 dist_diff_abs_pair_c25_0_1_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev15 + dist_diff_abs_pair_c25_0_1_prev15
   + 240 dist_diff_abs_pair_c25_0_1_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev15 + dist_diff_abs_pair_c25_0_1_prev15
   - 240 dist_diff_abs_pair_c25_0_1_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev15_force_positive:
   dist_diff_pair_c25_0_1_prev15
   - 240 dist_diff_abs_pair_c25_0_1_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev15_force_negative:
   dist_diff_pair_c25_0_1_prev15
   - 240 dist_diff_abs_pair_c25_0_1_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev15: 240 same_dist_pair_c25_0_1_prev15
   + dist_diff_abs_pair_c25_0_1_prev15 <= 249.8
 same_dist_pair_lower_c25_0_1_prev15: 240 same_dist_pair_c25_0_1_prev15
   + dist_diff_abs_pair_c25_0_1_prev15 >= 10
 dist_diff_pair_def_c25_0_1_prev16: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev16 = -30
 dist_diff_abs_pair_c25_0_1_prev16_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev16 + dist_diff_abs_pair_c25_0_1_prev16
   - 240 dist_diff_abs_pair_c25_0_1_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev16_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev16 + dist_diff_abs_pair_c25_0_1_prev16
   + 240 dist_diff_abs_pair_c25_0_1_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev16 + dist_diff_abs_pair_c25_0_1_prev16
   + 240 dist_diff_abs_pair_c25_0_1_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev16 + dist_diff_abs_pair_c25_0_1_prev16
   - 240 dist_diff_abs_pair_c25_0_1_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev16_force_positive:
   dist_diff_pair_c25_0_1_prev16
   - 240 dist_diff_abs_pair_c25_0_1_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev16_force_negative:
   dist_diff_pair_c25_0_1_prev16
   - 240 dist_diff_abs_pair_c25_0_1_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev16: 240 same_dist_pair_c25_0_1_prev16
   + dist_diff_abs_pair_c25_0_1_prev16 <= 249.8
 same_dist_pair_lower_c25_0_1_prev16: 240 same_dist_pair_c25_0_1_prev16
   + dist_diff_abs_pair_c25_0_1_prev16 >= 10
 dist_diff_pair_def_c25_0_1_prev17: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev17 = -30
 dist_diff_abs_pair_c25_0_1_prev17_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev17 + dist_diff_abs_pair_c25_0_1_prev17
   - 240 dist_diff_abs_pair_c25_0_1_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev17_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev17 + dist_diff_abs_pair_c25_0_1_prev17
   + 240 dist_diff_abs_pair_c25_0_1_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev17 + dist_diff_abs_pair_c25_0_1_prev17
   + 240 dist_diff_abs_pair_c25_0_1_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev17 + dist_diff_abs_pair_c25_0_1_prev17
   - 240 dist_diff_abs_pair_c25_0_1_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev17_force_positive:
   dist_diff_pair_c25_0_1_prev17
   - 240 dist_diff_abs_pair_c25_0_1_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev17_force_negative:
   dist_diff_pair_c25_0_1_prev17
   - 240 dist_diff_abs_pair_c25_0_1_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev17: 240 same_dist_pair_c25_0_1_prev17
   + dist_diff_abs_pair_c25_0_1_prev17 <= 249.8
 same_dist_pair_lower_c25_0_1_prev17: 240 same_dist_pair_c25_0_1_prev17
   + dist_diff_abs_pair_c25_0_1_prev17 >= 10
 dist_diff_pair_def_c25_0_1_prev18: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev18 = -43
 dist_diff_abs_pair_c25_0_1_prev18_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev18 + dist_diff_abs_pair_c25_0_1_prev18
   - 240 dist_diff_abs_pair_c25_0_1_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev18_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev18 + dist_diff_abs_pair_c25_0_1_prev18
   + 240 dist_diff_abs_pair_c25_0_1_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev18 + dist_diff_abs_pair_c25_0_1_prev18
   + 240 dist_diff_abs_pair_c25_0_1_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev18 + dist_diff_abs_pair_c25_0_1_prev18
   - 240 dist_diff_abs_pair_c25_0_1_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev18_force_positive:
   dist_diff_pair_c25_0_1_prev18
   - 240 dist_diff_abs_pair_c25_0_1_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev18_force_negative:
   dist_diff_pair_c25_0_1_prev18
   - 240 dist_diff_abs_pair_c25_0_1_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev18: 240 same_dist_pair_c25_0_1_prev18
   + dist_diff_abs_pair_c25_0_1_prev18 <= 249.8
 same_dist_pair_lower_c25_0_1_prev18: 240 same_dist_pair_c25_0_1_prev18
   + dist_diff_abs_pair_c25_0_1_prev18 >= 10
 dist_diff_pair_def_c25_0_1_prev19: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev19 = -38
 dist_diff_abs_pair_c25_0_1_prev19_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev19 + dist_diff_abs_pair_c25_0_1_prev19
   - 240 dist_diff_abs_pair_c25_0_1_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev19_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev19 + dist_diff_abs_pair_c25_0_1_prev19
   + 240 dist_diff_abs_pair_c25_0_1_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev19 + dist_diff_abs_pair_c25_0_1_prev19
   + 240 dist_diff_abs_pair_c25_0_1_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev19 + dist_diff_abs_pair_c25_0_1_prev19
   - 240 dist_diff_abs_pair_c25_0_1_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev19_force_positive:
   dist_diff_pair_c25_0_1_prev19
   - 240 dist_diff_abs_pair_c25_0_1_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev19_force_negative:
   dist_diff_pair_c25_0_1_prev19
   - 240 dist_diff_abs_pair_c25_0_1_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev19: 240 same_dist_pair_c25_0_1_prev19
   + dist_diff_abs_pair_c25_0_1_prev19 <= 249.8
 same_dist_pair_lower_c25_0_1_prev19: 240 same_dist_pair_c25_0_1_prev19
   + dist_diff_abs_pair_c25_0_1_prev19 >= 10
 dist_diff_pair_def_c25_0_1_prev20: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev20 = -26
 dist_diff_abs_pair_c25_0_1_prev20_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev20 + dist_diff_abs_pair_c25_0_1_prev20
   - 240 dist_diff_abs_pair_c25_0_1_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev20_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev20 + dist_diff_abs_pair_c25_0_1_prev20
   + 240 dist_diff_abs_pair_c25_0_1_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev20 + dist_diff_abs_pair_c25_0_1_prev20
   + 240 dist_diff_abs_pair_c25_0_1_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev20 + dist_diff_abs_pair_c25_0_1_prev20
   - 240 dist_diff_abs_pair_c25_0_1_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev20_force_positive:
   dist_diff_pair_c25_0_1_prev20
   - 240 dist_diff_abs_pair_c25_0_1_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev20_force_negative:
   dist_diff_pair_c25_0_1_prev20
   - 240 dist_diff_abs_pair_c25_0_1_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev20: 240 same_dist_pair_c25_0_1_prev20
   + dist_diff_abs_pair_c25_0_1_prev20 <= 249.8
 same_dist_pair_lower_c25_0_1_prev20: 240 same_dist_pair_c25_0_1_prev20
   + dist_diff_abs_pair_c25_0_1_prev20 >= 10
 dist_diff_pair_def_c25_0_1_prev21: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev21 = -30
 dist_diff_abs_pair_c25_0_1_prev21_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev21 + dist_diff_abs_pair_c25_0_1_prev21
   - 240 dist_diff_abs_pair_c25_0_1_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev21_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev21 + dist_diff_abs_pair_c25_0_1_prev21
   + 240 dist_diff_abs_pair_c25_0_1_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev21 + dist_diff_abs_pair_c25_0_1_prev21
   + 240 dist_diff_abs_pair_c25_0_1_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev21 + dist_diff_abs_pair_c25_0_1_prev21
   - 240 dist_diff_abs_pair_c25_0_1_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev21_force_positive:
   dist_diff_pair_c25_0_1_prev21
   - 240 dist_diff_abs_pair_c25_0_1_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev21_force_negative:
   dist_diff_pair_c25_0_1_prev21
   - 240 dist_diff_abs_pair_c25_0_1_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev21: 240 same_dist_pair_c25_0_1_prev21
   + dist_diff_abs_pair_c25_0_1_prev21 <= 249.8
 same_dist_pair_lower_c25_0_1_prev21: 240 same_dist_pair_c25_0_1_prev21
   + dist_diff_abs_pair_c25_0_1_prev21 >= 10
 dist_diff_pair_def_c25_0_1_prev22: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev22 = -36
 dist_diff_abs_pair_c25_0_1_prev22_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev22 + dist_diff_abs_pair_c25_0_1_prev22
   - 240 dist_diff_abs_pair_c25_0_1_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev22_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev22 + dist_diff_abs_pair_c25_0_1_prev22
   + 240 dist_diff_abs_pair_c25_0_1_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev22 + dist_diff_abs_pair_c25_0_1_prev22
   + 240 dist_diff_abs_pair_c25_0_1_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev22 + dist_diff_abs_pair_c25_0_1_prev22
   - 240 dist_diff_abs_pair_c25_0_1_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev22_force_positive:
   dist_diff_pair_c25_0_1_prev22
   - 240 dist_diff_abs_pair_c25_0_1_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev22_force_negative:
   dist_diff_pair_c25_0_1_prev22
   - 240 dist_diff_abs_pair_c25_0_1_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev22: 240 same_dist_pair_c25_0_1_prev22
   + dist_diff_abs_pair_c25_0_1_prev22 <= 249.8
 same_dist_pair_lower_c25_0_1_prev22: 240 same_dist_pair_c25_0_1_prev22
   + dist_diff_abs_pair_c25_0_1_prev22 >= 10
 dist_diff_pair_def_c25_0_1_prev23: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev23 = -31
 dist_diff_abs_pair_c25_0_1_prev23_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev23 + dist_diff_abs_pair_c25_0_1_prev23
   - 240 dist_diff_abs_pair_c25_0_1_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev23_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev23 + dist_diff_abs_pair_c25_0_1_prev23
   + 240 dist_diff_abs_pair_c25_0_1_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev23 + dist_diff_abs_pair_c25_0_1_prev23
   + 240 dist_diff_abs_pair_c25_0_1_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev23 + dist_diff_abs_pair_c25_0_1_prev23
   - 240 dist_diff_abs_pair_c25_0_1_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev23_force_positive:
   dist_diff_pair_c25_0_1_prev23
   - 240 dist_diff_abs_pair_c25_0_1_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev23_force_negative:
   dist_diff_pair_c25_0_1_prev23
   - 240 dist_diff_abs_pair_c25_0_1_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev23: 240 same_dist_pair_c25_0_1_prev23
   + dist_diff_abs_pair_c25_0_1_prev23 <= 249.8
 same_dist_pair_lower_c25_0_1_prev23: 240 same_dist_pair_c25_0_1_prev23
   + dist_diff_abs_pair_c25_0_1_prev23 >= 10
 dist_diff_pair_def_c25_0_1_prev24: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev24 = -58
 dist_diff_abs_pair_c25_0_1_prev24_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev24 + dist_diff_abs_pair_c25_0_1_prev24
   - 240 dist_diff_abs_pair_c25_0_1_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev24_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev24 + dist_diff_abs_pair_c25_0_1_prev24
   + 240 dist_diff_abs_pair_c25_0_1_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev24 + dist_diff_abs_pair_c25_0_1_prev24
   + 240 dist_diff_abs_pair_c25_0_1_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev24 + dist_diff_abs_pair_c25_0_1_prev24
   - 240 dist_diff_abs_pair_c25_0_1_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev24_force_positive:
   dist_diff_pair_c25_0_1_prev24
   - 240 dist_diff_abs_pair_c25_0_1_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev24_force_negative:
   dist_diff_pair_c25_0_1_prev24
   - 240 dist_diff_abs_pair_c25_0_1_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev24: 240 same_dist_pair_c25_0_1_prev24
   + dist_diff_abs_pair_c25_0_1_prev24 <= 249.8
 same_dist_pair_lower_c25_0_1_prev24: 240 same_dist_pair_c25_0_1_prev24
   + dist_diff_abs_pair_c25_0_1_prev24 >= 10
 dist_diff_pair_def_c25_0_1_prev25: - dist_curr_pair_c25_0_1
   + dist_diff_pair_c25_0_1_prev25 = -26
 dist_diff_abs_pair_c25_0_1_prev25_abs_ge_orig:
   - dist_diff_pair_c25_0_1_prev25 + dist_diff_abs_pair_c25_0_1_prev25
   - 240 dist_diff_abs_pair_c25_0_1_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev25_abs_le_orig:
   - dist_diff_pair_c25_0_1_prev25 + dist_diff_abs_pair_c25_0_1_prev25
   + 240 dist_diff_abs_pair_c25_0_1_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_0_1_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_0_1_prev25 + dist_diff_abs_pair_c25_0_1_prev25
   + 240 dist_diff_abs_pair_c25_0_1_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_0_1_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_0_1_prev25 + dist_diff_abs_pair_c25_0_1_prev25
   - 240 dist_diff_abs_pair_c25_0_1_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_0_1_prev25_force_positive:
   dist_diff_pair_c25_0_1_prev25
   - 240 dist_diff_abs_pair_c25_0_1_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_0_1_prev25_force_negative:
   dist_diff_pair_c25_0_1_prev25
   - 240 dist_diff_abs_pair_c25_0_1_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_0_1_prev25: 240 same_dist_pair_c25_0_1_prev25
   + dist_diff_abs_pair_c25_0_1_prev25 <= 249.8
 same_dist_pair_lower_c25_0_1_prev25: 240 same_dist_pair_c25_0_1_prev25
   + dist_diff_abs_pair_c25_0_1_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_0_1_prev0: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev0: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev1: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev1: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev2: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev2: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev3: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev3: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev4: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev4: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev5: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev5: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev6: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev6: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev7: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev7: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev8: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev8: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev9: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev9: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev10: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev10: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev11: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev11: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev12: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev12: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev13: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev13: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev14: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev14: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev15: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev15: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev16: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev16: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev17: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev17: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev18: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev18: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev19: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev19: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev20: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev20: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev21: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev21: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev22: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev22: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev23: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev23: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev24: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev24: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_0_1_prev25: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_0_1_prev25: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_0_1: diff_dist_pair_c25_0_1
   + same_dist_pair_c25_0_1_prev0 + same_dist_pair_c25_0_1_prev1
   + same_dist_pair_c25_0_1_prev2 + same_dist_pair_c25_0_1_prev3
   + same_dist_pair_c25_0_1_prev4 + same_dist_pair_c25_0_1_prev5
   + same_dist_pair_c25_0_1_prev6 + same_dist_pair_c25_0_1_prev7
   + same_dist_pair_c25_0_1_prev8 + same_dist_pair_c25_0_1_prev9
   + same_dist_pair_c25_0_1_prev10 + same_dist_pair_c25_0_1_prev11
   + same_dist_pair_c25_0_1_prev12 + same_dist_pair_c25_0_1_prev13
   + same_dist_pair_c25_0_1_prev14 + same_dist_pair_c25_0_1_prev15
   + same_dist_pair_c25_0_1_prev16 + same_dist_pair_c25_0_1_prev17
   + same_dist_pair_c25_0_1_prev18 + same_dist_pair_c25_0_1_prev19
   + same_dist_pair_c25_0_1_prev20 + same_dist_pair_c25_0_1_prev21
   + same_dist_pair_c25_0_1_prev22 + same_dist_pair_c25_0_1_prev23
   + same_dist_pair_c25_0_1_prev24 + same_dist_pair_c25_0_1_prev25 >= 1
 dx_grid_diff_def_c25_0_2: - cx_grid_center_c25_0 + cx_grid_center_c25_2
   + dx_grid_diff_c25_0_2 = 0
 dx_grid_abs_pair_c25_0_2_abs_ge_orig: dx_grid_abs_pair_c25_0_2
   - dx_grid_diff_c25_0_2 - 130 dx_grid_abs_pair_c25_0_2_is_positive
   >= -130
 dx_grid_abs_pair_c25_0_2_abs_le_orig: dx_grid_abs_pair_c25_0_2
   - dx_grid_diff_c25_0_2 + 130 dx_grid_abs_pair_c25_0_2_is_positive
   <= 130
 dx_grid_abs_pair_c25_0_2_abs_ge_neg_orig: dx_grid_abs_pair_c25_0_2
   + dx_grid_diff_c25_0_2 + 130 dx_grid_abs_pair_c25_0_2_is_positive >= 0
 dx_grid_abs_pair_c25_0_2_abs_le_neg_orig: dx_grid_abs_pair_c25_0_2
   + dx_grid_diff_c25_0_2 - 130 dx_grid_abs_pair_c25_0_2_is_positive <= 0
 dx_grid_abs_pair_c25_0_2_force_positive: dx_grid_diff_c25_0_2
   - 130 dx_grid_abs_pair_c25_0_2_is_positive >= -130
 dx_grid_abs_pair_c25_0_2_force_negative: dx_grid_diff_c25_0_2
   - 130 dx_grid_abs_pair_c25_0_2_is_positive <= -0.001
 dy_grid_diff_def_c25_0_2: - cy_grid_center_c25_0 + cy_grid_center_c25_2
   + dy_grid_diff_c25_0_2 = 0
 dy_grid_abs_pair_c25_0_2_abs_ge_orig: dy_grid_abs_pair_c25_0_2
   - dy_grid_diff_c25_0_2 - 130 dy_grid_abs_pair_c25_0_2_is_positive
   >= -130
 dy_grid_abs_pair_c25_0_2_abs_le_orig: dy_grid_abs_pair_c25_0_2
   - dy_grid_diff_c25_0_2 + 130 dy_grid_abs_pair_c25_0_2_is_positive
   <= 130
 dy_grid_abs_pair_c25_0_2_abs_ge_neg_orig: dy_grid_abs_pair_c25_0_2
   + dy_grid_diff_c25_0_2 + 130 dy_grid_abs_pair_c25_0_2_is_positive >= 0
 dy_grid_abs_pair_c25_0_2_abs_le_neg_orig: dy_grid_abs_pair_c25_0_2
   + dy_grid_diff_c25_0_2 - 130 dy_grid_abs_pair_c25_0_2_is_positive <= 0
 dy_grid_abs_pair_c25_0_2_force_positive: dy_grid_diff_c25_0_2
   - 130 dy_grid_abs_pair_c25_0_2_is_positive >= -130
 dy_grid_abs_pair_c25_0_2_force_negative: dy_grid_diff_c25_0_2
   - 130 dy_grid_abs_pair_c25_0_2_is_positive <= -0.001
 dist_curr_pair_def_c25_0_2: - dx_grid_abs_pair_c25_0_2
   - dy_grid_abs_pair_c25_0_2 + dist_curr_pair_c25_0_2 = 0
 dist_diff_pair_def_c25_0_2_prev0: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev0 = -60
 dist_diff_abs_pair_c25_0_2_prev0_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev0 + dist_diff_abs_pair_c25_0_2_prev0
   - 240 dist_diff_abs_pair_c25_0_2_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev0_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev0 + dist_diff_abs_pair_c25_0_2_prev0
   + 240 dist_diff_abs_pair_c25_0_2_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev0 + dist_diff_abs_pair_c25_0_2_prev0
   + 240 dist_diff_abs_pair_c25_0_2_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev0 + dist_diff_abs_pair_c25_0_2_prev0
   - 240 dist_diff_abs_pair_c25_0_2_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev0_force_positive:
   dist_diff_pair_c25_0_2_prev0
   - 240 dist_diff_abs_pair_c25_0_2_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev0_force_negative:
   dist_diff_pair_c25_0_2_prev0
   - 240 dist_diff_abs_pair_c25_0_2_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev0: 240 same_dist_pair_c25_0_2_prev0
   + dist_diff_abs_pair_c25_0_2_prev0 <= 249.8
 same_dist_pair_lower_c25_0_2_prev0: 240 same_dist_pair_c25_0_2_prev0
   + dist_diff_abs_pair_c25_0_2_prev0 >= 10
 dist_diff_pair_def_c25_0_2_prev1: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev1 = -66
 dist_diff_abs_pair_c25_0_2_prev1_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev1 + dist_diff_abs_pair_c25_0_2_prev1
   - 240 dist_diff_abs_pair_c25_0_2_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev1_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev1 + dist_diff_abs_pair_c25_0_2_prev1
   + 240 dist_diff_abs_pair_c25_0_2_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev1 + dist_diff_abs_pair_c25_0_2_prev1
   + 240 dist_diff_abs_pair_c25_0_2_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev1 + dist_diff_abs_pair_c25_0_2_prev1
   - 240 dist_diff_abs_pair_c25_0_2_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev1_force_positive:
   dist_diff_pair_c25_0_2_prev1
   - 240 dist_diff_abs_pair_c25_0_2_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev1_force_negative:
   dist_diff_pair_c25_0_2_prev1
   - 240 dist_diff_abs_pair_c25_0_2_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev1: 240 same_dist_pair_c25_0_2_prev1
   + dist_diff_abs_pair_c25_0_2_prev1 <= 249.8
 same_dist_pair_lower_c25_0_2_prev1: 240 same_dist_pair_c25_0_2_prev1
   + dist_diff_abs_pair_c25_0_2_prev1 >= 10
 dist_diff_pair_def_c25_0_2_prev2: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev2 = -60
 dist_diff_abs_pair_c25_0_2_prev2_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev2 + dist_diff_abs_pair_c25_0_2_prev2
   - 240 dist_diff_abs_pair_c25_0_2_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev2_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev2 + dist_diff_abs_pair_c25_0_2_prev2
   + 240 dist_diff_abs_pair_c25_0_2_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev2 + dist_diff_abs_pair_c25_0_2_prev2
   + 240 dist_diff_abs_pair_c25_0_2_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev2 + dist_diff_abs_pair_c25_0_2_prev2
   - 240 dist_diff_abs_pair_c25_0_2_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev2_force_positive:
   dist_diff_pair_c25_0_2_prev2
   - 240 dist_diff_abs_pair_c25_0_2_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev2_force_negative:
   dist_diff_pair_c25_0_2_prev2
   - 240 dist_diff_abs_pair_c25_0_2_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev2: 240 same_dist_pair_c25_0_2_prev2
   + dist_diff_abs_pair_c25_0_2_prev2 <= 249.8
 same_dist_pair_lower_c25_0_2_prev2: 240 same_dist_pair_c25_0_2_prev2
   + dist_diff_abs_pair_c25_0_2_prev2 >= 10
 dist_diff_pair_def_c25_0_2_prev3: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev3 = -60
 dist_diff_abs_pair_c25_0_2_prev3_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev3 + dist_diff_abs_pair_c25_0_2_prev3
   - 240 dist_diff_abs_pair_c25_0_2_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev3_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev3 + dist_diff_abs_pair_c25_0_2_prev3
   + 240 dist_diff_abs_pair_c25_0_2_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev3 + dist_diff_abs_pair_c25_0_2_prev3
   + 240 dist_diff_abs_pair_c25_0_2_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev3 + dist_diff_abs_pair_c25_0_2_prev3
   - 240 dist_diff_abs_pair_c25_0_2_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev3_force_positive:
   dist_diff_pair_c25_0_2_prev3
   - 240 dist_diff_abs_pair_c25_0_2_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev3_force_negative:
   dist_diff_pair_c25_0_2_prev3
   - 240 dist_diff_abs_pair_c25_0_2_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev3: 240 same_dist_pair_c25_0_2_prev3
   + dist_diff_abs_pair_c25_0_2_prev3 <= 249.8
 same_dist_pair_lower_c25_0_2_prev3: 240 same_dist_pair_c25_0_2_prev3
   + dist_diff_abs_pair_c25_0_2_prev3 >= 10
 dist_diff_pair_def_c25_0_2_prev4: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev4 = -76
 dist_diff_abs_pair_c25_0_2_prev4_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev4 + dist_diff_abs_pair_c25_0_2_prev4
   - 240 dist_diff_abs_pair_c25_0_2_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev4_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev4 + dist_diff_abs_pair_c25_0_2_prev4
   + 240 dist_diff_abs_pair_c25_0_2_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev4 + dist_diff_abs_pair_c25_0_2_prev4
   + 240 dist_diff_abs_pair_c25_0_2_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev4 + dist_diff_abs_pair_c25_0_2_prev4
   - 240 dist_diff_abs_pair_c25_0_2_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev4_force_positive:
   dist_diff_pair_c25_0_2_prev4
   - 240 dist_diff_abs_pair_c25_0_2_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev4_force_negative:
   dist_diff_pair_c25_0_2_prev4
   - 240 dist_diff_abs_pair_c25_0_2_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev4: 240 same_dist_pair_c25_0_2_prev4
   + dist_diff_abs_pair_c25_0_2_prev4 <= 249.8
 same_dist_pair_lower_c25_0_2_prev4: 240 same_dist_pair_c25_0_2_prev4
   + dist_diff_abs_pair_c25_0_2_prev4 >= 10
 dist_diff_pair_def_c25_0_2_prev5: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev5 = -60
 dist_diff_abs_pair_c25_0_2_prev5_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev5 + dist_diff_abs_pair_c25_0_2_prev5
   - 240 dist_diff_abs_pair_c25_0_2_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev5_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev5 + dist_diff_abs_pair_c25_0_2_prev5
   + 240 dist_diff_abs_pair_c25_0_2_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev5 + dist_diff_abs_pair_c25_0_2_prev5
   + 240 dist_diff_abs_pair_c25_0_2_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev5 + dist_diff_abs_pair_c25_0_2_prev5
   - 240 dist_diff_abs_pair_c25_0_2_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev5_force_positive:
   dist_diff_pair_c25_0_2_prev5
   - 240 dist_diff_abs_pair_c25_0_2_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev5_force_negative:
   dist_diff_pair_c25_0_2_prev5
   - 240 dist_diff_abs_pair_c25_0_2_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev5: 240 same_dist_pair_c25_0_2_prev5
   + dist_diff_abs_pair_c25_0_2_prev5 <= 249.8
 same_dist_pair_lower_c25_0_2_prev5: 240 same_dist_pair_c25_0_2_prev5
   + dist_diff_abs_pair_c25_0_2_prev5 >= 10
 dist_diff_pair_def_c25_0_2_prev6: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev6 = -60
 dist_diff_abs_pair_c25_0_2_prev6_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev6 + dist_diff_abs_pair_c25_0_2_prev6
   - 240 dist_diff_abs_pair_c25_0_2_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev6_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev6 + dist_diff_abs_pair_c25_0_2_prev6
   + 240 dist_diff_abs_pair_c25_0_2_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev6 + dist_diff_abs_pair_c25_0_2_prev6
   + 240 dist_diff_abs_pair_c25_0_2_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev6 + dist_diff_abs_pair_c25_0_2_prev6
   - 240 dist_diff_abs_pair_c25_0_2_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev6_force_positive:
   dist_diff_pair_c25_0_2_prev6
   - 240 dist_diff_abs_pair_c25_0_2_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev6_force_negative:
   dist_diff_pair_c25_0_2_prev6
   - 240 dist_diff_abs_pair_c25_0_2_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev6: 240 same_dist_pair_c25_0_2_prev6
   + dist_diff_abs_pair_c25_0_2_prev6 <= 249.8
 same_dist_pair_lower_c25_0_2_prev6: 240 same_dist_pair_c25_0_2_prev6
   + dist_diff_abs_pair_c25_0_2_prev6 >= 10
 dist_diff_pair_def_c25_0_2_prev7: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev7 = -60
 dist_diff_abs_pair_c25_0_2_prev7_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev7 + dist_diff_abs_pair_c25_0_2_prev7
   - 240 dist_diff_abs_pair_c25_0_2_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev7_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev7 + dist_diff_abs_pair_c25_0_2_prev7
   + 240 dist_diff_abs_pair_c25_0_2_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev7 + dist_diff_abs_pair_c25_0_2_prev7
   + 240 dist_diff_abs_pair_c25_0_2_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev7 + dist_diff_abs_pair_c25_0_2_prev7
   - 240 dist_diff_abs_pair_c25_0_2_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev7_force_positive:
   dist_diff_pair_c25_0_2_prev7
   - 240 dist_diff_abs_pair_c25_0_2_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev7_force_negative:
   dist_diff_pair_c25_0_2_prev7
   - 240 dist_diff_abs_pair_c25_0_2_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev7: 240 same_dist_pair_c25_0_2_prev7
   + dist_diff_abs_pair_c25_0_2_prev7 <= 249.8
 same_dist_pair_lower_c25_0_2_prev7: 240 same_dist_pair_c25_0_2_prev7
   + dist_diff_abs_pair_c25_0_2_prev7 >= 10
 dist_diff_pair_def_c25_0_2_prev8: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev8 = -57
 dist_diff_abs_pair_c25_0_2_prev8_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev8 + dist_diff_abs_pair_c25_0_2_prev8
   - 240 dist_diff_abs_pair_c25_0_2_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev8_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev8 + dist_diff_abs_pair_c25_0_2_prev8
   + 240 dist_diff_abs_pair_c25_0_2_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev8 + dist_diff_abs_pair_c25_0_2_prev8
   + 240 dist_diff_abs_pair_c25_0_2_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev8 + dist_diff_abs_pair_c25_0_2_prev8
   - 240 dist_diff_abs_pair_c25_0_2_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev8_force_positive:
   dist_diff_pair_c25_0_2_prev8
   - 240 dist_diff_abs_pair_c25_0_2_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev8_force_negative:
   dist_diff_pair_c25_0_2_prev8
   - 240 dist_diff_abs_pair_c25_0_2_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev8: 240 same_dist_pair_c25_0_2_prev8
   + dist_diff_abs_pair_c25_0_2_prev8 <= 249.8
 same_dist_pair_lower_c25_0_2_prev8: 240 same_dist_pair_c25_0_2_prev8
   + dist_diff_abs_pair_c25_0_2_prev8 >= 10
 dist_diff_pair_def_c25_0_2_prev9: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev9 = -76
 dist_diff_abs_pair_c25_0_2_prev9_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev9 + dist_diff_abs_pair_c25_0_2_prev9
   - 240 dist_diff_abs_pair_c25_0_2_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev9_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev9 + dist_diff_abs_pair_c25_0_2_prev9
   + 240 dist_diff_abs_pair_c25_0_2_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev9 + dist_diff_abs_pair_c25_0_2_prev9
   + 240 dist_diff_abs_pair_c25_0_2_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev9 + dist_diff_abs_pair_c25_0_2_prev9
   - 240 dist_diff_abs_pair_c25_0_2_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev9_force_positive:
   dist_diff_pair_c25_0_2_prev9
   - 240 dist_diff_abs_pair_c25_0_2_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev9_force_negative:
   dist_diff_pair_c25_0_2_prev9
   - 240 dist_diff_abs_pair_c25_0_2_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev9: 240 same_dist_pair_c25_0_2_prev9
   + dist_diff_abs_pair_c25_0_2_prev9 <= 249.8
 same_dist_pair_lower_c25_0_2_prev9: 240 same_dist_pair_c25_0_2_prev9
   + dist_diff_abs_pair_c25_0_2_prev9 >= 10
 dist_diff_pair_def_c25_0_2_prev10: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev10 = -67
 dist_diff_abs_pair_c25_0_2_prev10_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev10 + dist_diff_abs_pair_c25_0_2_prev10
   - 240 dist_diff_abs_pair_c25_0_2_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev10_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev10 + dist_diff_abs_pair_c25_0_2_prev10
   + 240 dist_diff_abs_pair_c25_0_2_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev10 + dist_diff_abs_pair_c25_0_2_prev10
   + 240 dist_diff_abs_pair_c25_0_2_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev10 + dist_diff_abs_pair_c25_0_2_prev10
   - 240 dist_diff_abs_pair_c25_0_2_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev10_force_positive:
   dist_diff_pair_c25_0_2_prev10
   - 240 dist_diff_abs_pair_c25_0_2_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev10_force_negative:
   dist_diff_pair_c25_0_2_prev10
   - 240 dist_diff_abs_pair_c25_0_2_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev10: 240 same_dist_pair_c25_0_2_prev10
   + dist_diff_abs_pair_c25_0_2_prev10 <= 249.8
 same_dist_pair_lower_c25_0_2_prev10: 240 same_dist_pair_c25_0_2_prev10
   + dist_diff_abs_pair_c25_0_2_prev10 >= 10
 dist_diff_pair_def_c25_0_2_prev11: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev11 = -66
 dist_diff_abs_pair_c25_0_2_prev11_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev11 + dist_diff_abs_pair_c25_0_2_prev11
   - 240 dist_diff_abs_pair_c25_0_2_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev11_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev11 + dist_diff_abs_pair_c25_0_2_prev11
   + 240 dist_diff_abs_pair_c25_0_2_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev11 + dist_diff_abs_pair_c25_0_2_prev11
   + 240 dist_diff_abs_pair_c25_0_2_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev11 + dist_diff_abs_pair_c25_0_2_prev11
   - 240 dist_diff_abs_pair_c25_0_2_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev11_force_positive:
   dist_diff_pair_c25_0_2_prev11
   - 240 dist_diff_abs_pair_c25_0_2_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev11_force_negative:
   dist_diff_pair_c25_0_2_prev11
   - 240 dist_diff_abs_pair_c25_0_2_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev11: 240 same_dist_pair_c25_0_2_prev11
   + dist_diff_abs_pair_c25_0_2_prev11 <= 249.8
 same_dist_pair_lower_c25_0_2_prev11: 240 same_dist_pair_c25_0_2_prev11
   + dist_diff_abs_pair_c25_0_2_prev11 >= 10
 dist_diff_pair_def_c25_0_2_prev12: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev12 = -68
 dist_diff_abs_pair_c25_0_2_prev12_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev12 + dist_diff_abs_pair_c25_0_2_prev12
   - 240 dist_diff_abs_pair_c25_0_2_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev12_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev12 + dist_diff_abs_pair_c25_0_2_prev12
   + 240 dist_diff_abs_pair_c25_0_2_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev12 + dist_diff_abs_pair_c25_0_2_prev12
   + 240 dist_diff_abs_pair_c25_0_2_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev12 + dist_diff_abs_pair_c25_0_2_prev12
   - 240 dist_diff_abs_pair_c25_0_2_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev12_force_positive:
   dist_diff_pair_c25_0_2_prev12
   - 240 dist_diff_abs_pair_c25_0_2_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev12_force_negative:
   dist_diff_pair_c25_0_2_prev12
   - 240 dist_diff_abs_pair_c25_0_2_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev12: 240 same_dist_pair_c25_0_2_prev12
   + dist_diff_abs_pair_c25_0_2_prev12 <= 249.8
 same_dist_pair_lower_c25_0_2_prev12: 240 same_dist_pair_c25_0_2_prev12
   + dist_diff_abs_pair_c25_0_2_prev12 >= 10
 dist_diff_pair_def_c25_0_2_prev13: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev13 = -60
 dist_diff_abs_pair_c25_0_2_prev13_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev13 + dist_diff_abs_pair_c25_0_2_prev13
   - 240 dist_diff_abs_pair_c25_0_2_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev13_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev13 + dist_diff_abs_pair_c25_0_2_prev13
   + 240 dist_diff_abs_pair_c25_0_2_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev13 + dist_diff_abs_pair_c25_0_2_prev13
   + 240 dist_diff_abs_pair_c25_0_2_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev13 + dist_diff_abs_pair_c25_0_2_prev13
   - 240 dist_diff_abs_pair_c25_0_2_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev13_force_positive:
   dist_diff_pair_c25_0_2_prev13
   - 240 dist_diff_abs_pair_c25_0_2_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev13_force_negative:
   dist_diff_pair_c25_0_2_prev13
   - 240 dist_diff_abs_pair_c25_0_2_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev13: 240 same_dist_pair_c25_0_2_prev13
   + dist_diff_abs_pair_c25_0_2_prev13 <= 249.8
 same_dist_pair_lower_c25_0_2_prev13: 240 same_dist_pair_c25_0_2_prev13
   + dist_diff_abs_pair_c25_0_2_prev13 >= 10
 dist_diff_pair_def_c25_0_2_prev14: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev14 = -66
 dist_diff_abs_pair_c25_0_2_prev14_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev14 + dist_diff_abs_pair_c25_0_2_prev14
   - 240 dist_diff_abs_pair_c25_0_2_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev14_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev14 + dist_diff_abs_pair_c25_0_2_prev14
   + 240 dist_diff_abs_pair_c25_0_2_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev14 + dist_diff_abs_pair_c25_0_2_prev14
   + 240 dist_diff_abs_pair_c25_0_2_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev14 + dist_diff_abs_pair_c25_0_2_prev14
   - 240 dist_diff_abs_pair_c25_0_2_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev14_force_positive:
   dist_diff_pair_c25_0_2_prev14
   - 240 dist_diff_abs_pair_c25_0_2_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev14_force_negative:
   dist_diff_pair_c25_0_2_prev14
   - 240 dist_diff_abs_pair_c25_0_2_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev14: 240 same_dist_pair_c25_0_2_prev14
   + dist_diff_abs_pair_c25_0_2_prev14 <= 249.8
 same_dist_pair_lower_c25_0_2_prev14: 240 same_dist_pair_c25_0_2_prev14
   + dist_diff_abs_pair_c25_0_2_prev14 >= 10
 dist_diff_pair_def_c25_0_2_prev15: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev15 = -85
 dist_diff_abs_pair_c25_0_2_prev15_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev15 + dist_diff_abs_pair_c25_0_2_prev15
   - 240 dist_diff_abs_pair_c25_0_2_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev15_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev15 + dist_diff_abs_pair_c25_0_2_prev15
   + 240 dist_diff_abs_pair_c25_0_2_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev15 + dist_diff_abs_pair_c25_0_2_prev15
   + 240 dist_diff_abs_pair_c25_0_2_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev15 + dist_diff_abs_pair_c25_0_2_prev15
   - 240 dist_diff_abs_pair_c25_0_2_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev15_force_positive:
   dist_diff_pair_c25_0_2_prev15
   - 240 dist_diff_abs_pair_c25_0_2_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev15_force_negative:
   dist_diff_pair_c25_0_2_prev15
   - 240 dist_diff_abs_pair_c25_0_2_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev15: 240 same_dist_pair_c25_0_2_prev15
   + dist_diff_abs_pair_c25_0_2_prev15 <= 249.8
 same_dist_pair_lower_c25_0_2_prev15: 240 same_dist_pair_c25_0_2_prev15
   + dist_diff_abs_pair_c25_0_2_prev15 >= 10
 dist_diff_pair_def_c25_0_2_prev16: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev16 = -82
 dist_diff_abs_pair_c25_0_2_prev16_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev16 + dist_diff_abs_pair_c25_0_2_prev16
   - 240 dist_diff_abs_pair_c25_0_2_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev16_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev16 + dist_diff_abs_pair_c25_0_2_prev16
   + 240 dist_diff_abs_pair_c25_0_2_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev16 + dist_diff_abs_pair_c25_0_2_prev16
   + 240 dist_diff_abs_pair_c25_0_2_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev16 + dist_diff_abs_pair_c25_0_2_prev16
   - 240 dist_diff_abs_pair_c25_0_2_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev16_force_positive:
   dist_diff_pair_c25_0_2_prev16
   - 240 dist_diff_abs_pair_c25_0_2_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev16_force_negative:
   dist_diff_pair_c25_0_2_prev16
   - 240 dist_diff_abs_pair_c25_0_2_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev16: 240 same_dist_pair_c25_0_2_prev16
   + dist_diff_abs_pair_c25_0_2_prev16 <= 249.8
 same_dist_pair_lower_c25_0_2_prev16: 240 same_dist_pair_c25_0_2_prev16
   + dist_diff_abs_pair_c25_0_2_prev16 >= 10
 dist_diff_pair_def_c25_0_2_prev17: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev17 = -76
 dist_diff_abs_pair_c25_0_2_prev17_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev17 + dist_diff_abs_pair_c25_0_2_prev17
   - 240 dist_diff_abs_pair_c25_0_2_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev17_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev17 + dist_diff_abs_pair_c25_0_2_prev17
   + 240 dist_diff_abs_pair_c25_0_2_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev17 + dist_diff_abs_pair_c25_0_2_prev17
   + 240 dist_diff_abs_pair_c25_0_2_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev17 + dist_diff_abs_pair_c25_0_2_prev17
   - 240 dist_diff_abs_pair_c25_0_2_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev17_force_positive:
   dist_diff_pair_c25_0_2_prev17
   - 240 dist_diff_abs_pair_c25_0_2_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev17_force_negative:
   dist_diff_pair_c25_0_2_prev17
   - 240 dist_diff_abs_pair_c25_0_2_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev17: 240 same_dist_pair_c25_0_2_prev17
   + dist_diff_abs_pair_c25_0_2_prev17 <= 249.8
 same_dist_pair_lower_c25_0_2_prev17: 240 same_dist_pair_c25_0_2_prev17
   + dist_diff_abs_pair_c25_0_2_prev17 >= 10
 dist_diff_pair_def_c25_0_2_prev18: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev18 = -61
 dist_diff_abs_pair_c25_0_2_prev18_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev18 + dist_diff_abs_pair_c25_0_2_prev18
   - 240 dist_diff_abs_pair_c25_0_2_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev18_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev18 + dist_diff_abs_pair_c25_0_2_prev18
   + 240 dist_diff_abs_pair_c25_0_2_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev18 + dist_diff_abs_pair_c25_0_2_prev18
   + 240 dist_diff_abs_pair_c25_0_2_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev18 + dist_diff_abs_pair_c25_0_2_prev18
   - 240 dist_diff_abs_pair_c25_0_2_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev18_force_positive:
   dist_diff_pair_c25_0_2_prev18
   - 240 dist_diff_abs_pair_c25_0_2_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev18_force_negative:
   dist_diff_pair_c25_0_2_prev18
   - 240 dist_diff_abs_pair_c25_0_2_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev18: 240 same_dist_pair_c25_0_2_prev18
   + dist_diff_abs_pair_c25_0_2_prev18 <= 249.8
 same_dist_pair_lower_c25_0_2_prev18: 240 same_dist_pair_c25_0_2_prev18
   + dist_diff_abs_pair_c25_0_2_prev18 >= 10
 dist_diff_pair_def_c25_0_2_prev19: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev19 = -40
 dist_diff_abs_pair_c25_0_2_prev19_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev19 + dist_diff_abs_pair_c25_0_2_prev19
   - 240 dist_diff_abs_pair_c25_0_2_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev19_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev19 + dist_diff_abs_pair_c25_0_2_prev19
   + 240 dist_diff_abs_pair_c25_0_2_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev19 + dist_diff_abs_pair_c25_0_2_prev19
   + 240 dist_diff_abs_pair_c25_0_2_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev19 + dist_diff_abs_pair_c25_0_2_prev19
   - 240 dist_diff_abs_pair_c25_0_2_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev19_force_positive:
   dist_diff_pair_c25_0_2_prev19
   - 240 dist_diff_abs_pair_c25_0_2_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev19_force_negative:
   dist_diff_pair_c25_0_2_prev19
   - 240 dist_diff_abs_pair_c25_0_2_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev19: 240 same_dist_pair_c25_0_2_prev19
   + dist_diff_abs_pair_c25_0_2_prev19 <= 249.8
 same_dist_pair_lower_c25_0_2_prev19: 240 same_dist_pair_c25_0_2_prev19
   + dist_diff_abs_pair_c25_0_2_prev19 >= 10
 dist_diff_pair_def_c25_0_2_prev20: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev20 = -74
 dist_diff_abs_pair_c25_0_2_prev20_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev20 + dist_diff_abs_pair_c25_0_2_prev20
   - 240 dist_diff_abs_pair_c25_0_2_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev20_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev20 + dist_diff_abs_pair_c25_0_2_prev20
   + 240 dist_diff_abs_pair_c25_0_2_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev20 + dist_diff_abs_pair_c25_0_2_prev20
   + 240 dist_diff_abs_pair_c25_0_2_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev20 + dist_diff_abs_pair_c25_0_2_prev20
   - 240 dist_diff_abs_pair_c25_0_2_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev20_force_positive:
   dist_diff_pair_c25_0_2_prev20
   - 240 dist_diff_abs_pair_c25_0_2_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev20_force_negative:
   dist_diff_pair_c25_0_2_prev20
   - 240 dist_diff_abs_pair_c25_0_2_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev20: 240 same_dist_pair_c25_0_2_prev20
   + dist_diff_abs_pair_c25_0_2_prev20 <= 249.8
 same_dist_pair_lower_c25_0_2_prev20: 240 same_dist_pair_c25_0_2_prev20
   + dist_diff_abs_pair_c25_0_2_prev20 >= 10
 dist_diff_pair_def_c25_0_2_prev21: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev21 = -76
 dist_diff_abs_pair_c25_0_2_prev21_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev21 + dist_diff_abs_pair_c25_0_2_prev21
   - 240 dist_diff_abs_pair_c25_0_2_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev21_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev21 + dist_diff_abs_pair_c25_0_2_prev21
   + 240 dist_diff_abs_pair_c25_0_2_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev21 + dist_diff_abs_pair_c25_0_2_prev21
   + 240 dist_diff_abs_pair_c25_0_2_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev21 + dist_diff_abs_pair_c25_0_2_prev21
   - 240 dist_diff_abs_pair_c25_0_2_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev21_force_positive:
   dist_diff_pair_c25_0_2_prev21
   - 240 dist_diff_abs_pair_c25_0_2_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev21_force_negative:
   dist_diff_pair_c25_0_2_prev21
   - 240 dist_diff_abs_pair_c25_0_2_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev21: 240 same_dist_pair_c25_0_2_prev21
   + dist_diff_abs_pair_c25_0_2_prev21 <= 249.8
 same_dist_pair_lower_c25_0_2_prev21: 240 same_dist_pair_c25_0_2_prev21
   + dist_diff_abs_pair_c25_0_2_prev21 >= 10
 dist_diff_pair_def_c25_0_2_prev22: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev22 = -95
 dist_diff_abs_pair_c25_0_2_prev22_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev22 + dist_diff_abs_pair_c25_0_2_prev22
   - 240 dist_diff_abs_pair_c25_0_2_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev22_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev22 + dist_diff_abs_pair_c25_0_2_prev22
   + 240 dist_diff_abs_pair_c25_0_2_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev22 + dist_diff_abs_pair_c25_0_2_prev22
   + 240 dist_diff_abs_pair_c25_0_2_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev22 + dist_diff_abs_pair_c25_0_2_prev22
   - 240 dist_diff_abs_pair_c25_0_2_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev22_force_positive:
   dist_diff_pair_c25_0_2_prev22
   - 240 dist_diff_abs_pair_c25_0_2_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev22_force_negative:
   dist_diff_pair_c25_0_2_prev22
   - 240 dist_diff_abs_pair_c25_0_2_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev22: 240 same_dist_pair_c25_0_2_prev22
   + dist_diff_abs_pair_c25_0_2_prev22 <= 249.8
 same_dist_pair_lower_c25_0_2_prev22: 240 same_dist_pair_c25_0_2_prev22
   + dist_diff_abs_pair_c25_0_2_prev22 >= 10
 dist_diff_pair_def_c25_0_2_prev23: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev23 = -91
 dist_diff_abs_pair_c25_0_2_prev23_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev23 + dist_diff_abs_pair_c25_0_2_prev23
   - 240 dist_diff_abs_pair_c25_0_2_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev23_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev23 + dist_diff_abs_pair_c25_0_2_prev23
   + 240 dist_diff_abs_pair_c25_0_2_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev23 + dist_diff_abs_pair_c25_0_2_prev23
   + 240 dist_diff_abs_pair_c25_0_2_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev23 + dist_diff_abs_pair_c25_0_2_prev23
   - 240 dist_diff_abs_pair_c25_0_2_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev23_force_positive:
   dist_diff_pair_c25_0_2_prev23
   - 240 dist_diff_abs_pair_c25_0_2_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev23_force_negative:
   dist_diff_pair_c25_0_2_prev23
   - 240 dist_diff_abs_pair_c25_0_2_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev23: 240 same_dist_pair_c25_0_2_prev23
   + dist_diff_abs_pair_c25_0_2_prev23 <= 249.8
 same_dist_pair_lower_c25_0_2_prev23: 240 same_dist_pair_c25_0_2_prev23
   + dist_diff_abs_pair_c25_0_2_prev23 >= 10
 dist_diff_pair_def_c25_0_2_prev24: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev24 = -44
 dist_diff_abs_pair_c25_0_2_prev24_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev24 + dist_diff_abs_pair_c25_0_2_prev24
   - 240 dist_diff_abs_pair_c25_0_2_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev24_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev24 + dist_diff_abs_pair_c25_0_2_prev24
   + 240 dist_diff_abs_pair_c25_0_2_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev24 + dist_diff_abs_pair_c25_0_2_prev24
   + 240 dist_diff_abs_pair_c25_0_2_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev24 + dist_diff_abs_pair_c25_0_2_prev24
   - 240 dist_diff_abs_pair_c25_0_2_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev24_force_positive:
   dist_diff_pair_c25_0_2_prev24
   - 240 dist_diff_abs_pair_c25_0_2_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev24_force_negative:
   dist_diff_pair_c25_0_2_prev24
   - 240 dist_diff_abs_pair_c25_0_2_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev24: 240 same_dist_pair_c25_0_2_prev24
   + dist_diff_abs_pair_c25_0_2_prev24 <= 249.8
 same_dist_pair_lower_c25_0_2_prev24: 240 same_dist_pair_c25_0_2_prev24
   + dist_diff_abs_pair_c25_0_2_prev24 >= 10
 dist_diff_pair_def_c25_0_2_prev25: - dist_curr_pair_c25_0_2
   + dist_diff_pair_c25_0_2_prev25 = -41
 dist_diff_abs_pair_c25_0_2_prev25_abs_ge_orig:
   - dist_diff_pair_c25_0_2_prev25 + dist_diff_abs_pair_c25_0_2_prev25
   - 240 dist_diff_abs_pair_c25_0_2_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev25_abs_le_orig:
   - dist_diff_pair_c25_0_2_prev25 + dist_diff_abs_pair_c25_0_2_prev25
   + 240 dist_diff_abs_pair_c25_0_2_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_0_2_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_0_2_prev25 + dist_diff_abs_pair_c25_0_2_prev25
   + 240 dist_diff_abs_pair_c25_0_2_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_0_2_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_0_2_prev25 + dist_diff_abs_pair_c25_0_2_prev25
   - 240 dist_diff_abs_pair_c25_0_2_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_0_2_prev25_force_positive:
   dist_diff_pair_c25_0_2_prev25
   - 240 dist_diff_abs_pair_c25_0_2_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_0_2_prev25_force_negative:
   dist_diff_pair_c25_0_2_prev25
   - 240 dist_diff_abs_pair_c25_0_2_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_0_2_prev25: 240 same_dist_pair_c25_0_2_prev25
   + dist_diff_abs_pair_c25_0_2_prev25 <= 249.8
 same_dist_pair_lower_c25_0_2_prev25: 240 same_dist_pair_c25_0_2_prev25
   + dist_diff_abs_pair_c25_0_2_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_0_2_prev0: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev0: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev1: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev1: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev2: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev2: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev3: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev3: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev4: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev4: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev5: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev5: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev6: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev6: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev7: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev7: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev8: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev8: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev9: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev9: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev10: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev10: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev11: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev11: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev12: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev12: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev13: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev13: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev14: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev14: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev15: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev15: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev16: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev16: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev17: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev17: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev18: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev18: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev19: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev19: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev20: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev20: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev21: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev21: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev22: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev22: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev23: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev23: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev24: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev24: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_0_2_prev25: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_0_2_prev25: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_0_2: diff_dist_pair_c25_0_2
   + same_dist_pair_c25_0_2_prev0 + same_dist_pair_c25_0_2_prev1
   + same_dist_pair_c25_0_2_prev2 + same_dist_pair_c25_0_2_prev3
   + same_dist_pair_c25_0_2_prev4 + same_dist_pair_c25_0_2_prev5
   + same_dist_pair_c25_0_2_prev6 + same_dist_pair_c25_0_2_prev7
   + same_dist_pair_c25_0_2_prev8 + same_dist_pair_c25_0_2_prev9
   + same_dist_pair_c25_0_2_prev10 + same_dist_pair_c25_0_2_prev11
   + same_dist_pair_c25_0_2_prev12 + same_dist_pair_c25_0_2_prev13
   + same_dist_pair_c25_0_2_prev14 + same_dist_pair_c25_0_2_prev15
   + same_dist_pair_c25_0_2_prev16 + same_dist_pair_c25_0_2_prev17
   + same_dist_pair_c25_0_2_prev18 + same_dist_pair_c25_0_2_prev19
   + same_dist_pair_c25_0_2_prev20 + same_dist_pair_c25_0_2_prev21
   + same_dist_pair_c25_0_2_prev22 + same_dist_pair_c25_0_2_prev23
   + same_dist_pair_c25_0_2_prev24 + same_dist_pair_c25_0_2_prev25 >= 1
 dx_grid_diff_def_c25_0_3: - cx_grid_center_c25_0 + cx_grid_center_c25_3
   + dx_grid_diff_c25_0_3 = 0
 dx_grid_abs_pair_c25_0_3_abs_ge_orig: dx_grid_abs_pair_c25_0_3
   - dx_grid_diff_c25_0_3 - 130 dx_grid_abs_pair_c25_0_3_is_positive
   >= -130
 dx_grid_abs_pair_c25_0_3_abs_le_orig: dx_grid_abs_pair_c25_0_3
   - dx_grid_diff_c25_0_3 + 130 dx_grid_abs_pair_c25_0_3_is_positive
   <= 130
 dx_grid_abs_pair_c25_0_3_abs_ge_neg_orig: dx_grid_abs_pair_c25_0_3
   + dx_grid_diff_c25_0_3 + 130 dx_grid_abs_pair_c25_0_3_is_positive >= 0
 dx_grid_abs_pair_c25_0_3_abs_le_neg_orig: dx_grid_abs_pair_c25_0_3
   + dx_grid_diff_c25_0_3 - 130 dx_grid_abs_pair_c25_0_3_is_positive <= 0
 dx_grid_abs_pair_c25_0_3_force_positive: dx_grid_diff_c25_0_3
   - 130 dx_grid_abs_pair_c25_0_3_is_positive >= -130
 dx_grid_abs_pair_c25_0_3_force_negative: dx_grid_diff_c25_0_3
   - 130 dx_grid_abs_pair_c25_0_3_is_positive <= -0.001
 dy_grid_diff_def_c25_0_3: - cy_grid_center_c25_0 + cy_grid_center_c25_3
   + dy_grid_diff_c25_0_3 = 0
 dy_grid_abs_pair_c25_0_3_abs_ge_orig: dy_grid_abs_pair_c25_0_3
   - dy_grid_diff_c25_0_3 - 130 dy_grid_abs_pair_c25_0_3_is_positive
   >= -130
 dy_grid_abs_pair_c25_0_3_abs_le_orig: dy_grid_abs_pair_c25_0_3
   - dy_grid_diff_c25_0_3 + 130 dy_grid_abs_pair_c25_0_3_is_positive
   <= 130
 dy_grid_abs_pair_c25_0_3_abs_ge_neg_orig: dy_grid_abs_pair_c25_0_3
   + dy_grid_diff_c25_0_3 + 130 dy_grid_abs_pair_c25_0_3_is_positive >= 0
 dy_grid_abs_pair_c25_0_3_abs_le_neg_orig: dy_grid_abs_pair_c25_0_3
   + dy_grid_diff_c25_0_3 - 130 dy_grid_abs_pair_c25_0_3_is_positive <= 0
 dy_grid_abs_pair_c25_0_3_force_positive: dy_grid_diff_c25_0_3
   - 130 dy_grid_abs_pair_c25_0_3_is_positive >= -130
 dy_grid_abs_pair_c25_0_3_force_negative: dy_grid_diff_c25_0_3
   - 130 dy_grid_abs_pair_c25_0_3_is_positive <= -0.001
 dist_curr_pair_def_c25_0_3: - dx_grid_abs_pair_c25_0_3
   - dy_grid_abs_pair_c25_0_3 + dist_curr_pair_c25_0_3 = 0
 dist_diff_pair_def_c25_0_3_prev0: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev0 = -56
 dist_diff_abs_pair_c25_0_3_prev0_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev0 + dist_diff_abs_pair_c25_0_3_prev0
   - 240 dist_diff_abs_pair_c25_0_3_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev0_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev0 + dist_diff_abs_pair_c25_0_3_prev0
   + 240 dist_diff_abs_pair_c25_0_3_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev0 + dist_diff_abs_pair_c25_0_3_prev0
   + 240 dist_diff_abs_pair_c25_0_3_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev0 + dist_diff_abs_pair_c25_0_3_prev0
   - 240 dist_diff_abs_pair_c25_0_3_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev0_force_positive:
   dist_diff_pair_c25_0_3_prev0
   - 240 dist_diff_abs_pair_c25_0_3_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev0_force_negative:
   dist_diff_pair_c25_0_3_prev0
   - 240 dist_diff_abs_pair_c25_0_3_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev0: 240 same_dist_pair_c25_0_3_prev0
   + dist_diff_abs_pair_c25_0_3_prev0 <= 249.8
 same_dist_pair_lower_c25_0_3_prev0: 240 same_dist_pair_c25_0_3_prev0
   + dist_diff_abs_pair_c25_0_3_prev0 >= 10
 dist_diff_pair_def_c25_0_3_prev1: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev1 = -56
 dist_diff_abs_pair_c25_0_3_prev1_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev1 + dist_diff_abs_pair_c25_0_3_prev1
   - 240 dist_diff_abs_pair_c25_0_3_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev1_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev1 + dist_diff_abs_pair_c25_0_3_prev1
   + 240 dist_diff_abs_pair_c25_0_3_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev1 + dist_diff_abs_pair_c25_0_3_prev1
   + 240 dist_diff_abs_pair_c25_0_3_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev1 + dist_diff_abs_pair_c25_0_3_prev1
   - 240 dist_diff_abs_pair_c25_0_3_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev1_force_positive:
   dist_diff_pair_c25_0_3_prev1
   - 240 dist_diff_abs_pair_c25_0_3_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev1_force_negative:
   dist_diff_pair_c25_0_3_prev1
   - 240 dist_diff_abs_pair_c25_0_3_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev1: 240 same_dist_pair_c25_0_3_prev1
   + dist_diff_abs_pair_c25_0_3_prev1 <= 249.8
 same_dist_pair_lower_c25_0_3_prev1: 240 same_dist_pair_c25_0_3_prev1
   + dist_diff_abs_pair_c25_0_3_prev1 >= 10
 dist_diff_pair_def_c25_0_3_prev2: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev2 = -66
 dist_diff_abs_pair_c25_0_3_prev2_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev2 + dist_diff_abs_pair_c25_0_3_prev2
   - 240 dist_diff_abs_pair_c25_0_3_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev2_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev2 + dist_diff_abs_pair_c25_0_3_prev2
   + 240 dist_diff_abs_pair_c25_0_3_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev2 + dist_diff_abs_pair_c25_0_3_prev2
   + 240 dist_diff_abs_pair_c25_0_3_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev2 + dist_diff_abs_pair_c25_0_3_prev2
   - 240 dist_diff_abs_pair_c25_0_3_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev2_force_positive:
   dist_diff_pair_c25_0_3_prev2
   - 240 dist_diff_abs_pair_c25_0_3_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev2_force_negative:
   dist_diff_pair_c25_0_3_prev2
   - 240 dist_diff_abs_pair_c25_0_3_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev2: 240 same_dist_pair_c25_0_3_prev2
   + dist_diff_abs_pair_c25_0_3_prev2 <= 249.8
 same_dist_pair_lower_c25_0_3_prev2: 240 same_dist_pair_c25_0_3_prev2
   + dist_diff_abs_pair_c25_0_3_prev2 >= 10
 dist_diff_pair_def_c25_0_3_prev3: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev3 = -57
 dist_diff_abs_pair_c25_0_3_prev3_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev3 + dist_diff_abs_pair_c25_0_3_prev3
   - 240 dist_diff_abs_pair_c25_0_3_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev3_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev3 + dist_diff_abs_pair_c25_0_3_prev3
   + 240 dist_diff_abs_pair_c25_0_3_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev3 + dist_diff_abs_pair_c25_0_3_prev3
   + 240 dist_diff_abs_pair_c25_0_3_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev3 + dist_diff_abs_pair_c25_0_3_prev3
   - 240 dist_diff_abs_pair_c25_0_3_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev3_force_positive:
   dist_diff_pair_c25_0_3_prev3
   - 240 dist_diff_abs_pair_c25_0_3_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev3_force_negative:
   dist_diff_pair_c25_0_3_prev3
   - 240 dist_diff_abs_pair_c25_0_3_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev3: 240 same_dist_pair_c25_0_3_prev3
   + dist_diff_abs_pair_c25_0_3_prev3 <= 249.8
 same_dist_pair_lower_c25_0_3_prev3: 240 same_dist_pair_c25_0_3_prev3
   + dist_diff_abs_pair_c25_0_3_prev3 >= 10
 dist_diff_pair_def_c25_0_3_prev4: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev4 = -58
 dist_diff_abs_pair_c25_0_3_prev4_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev4 + dist_diff_abs_pair_c25_0_3_prev4
   - 240 dist_diff_abs_pair_c25_0_3_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev4_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev4 + dist_diff_abs_pair_c25_0_3_prev4
   + 240 dist_diff_abs_pair_c25_0_3_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev4 + dist_diff_abs_pair_c25_0_3_prev4
   + 240 dist_diff_abs_pair_c25_0_3_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev4 + dist_diff_abs_pair_c25_0_3_prev4
   - 240 dist_diff_abs_pair_c25_0_3_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev4_force_positive:
   dist_diff_pair_c25_0_3_prev4
   - 240 dist_diff_abs_pair_c25_0_3_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev4_force_negative:
   dist_diff_pair_c25_0_3_prev4
   - 240 dist_diff_abs_pair_c25_0_3_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev4: 240 same_dist_pair_c25_0_3_prev4
   + dist_diff_abs_pair_c25_0_3_prev4 <= 249.8
 same_dist_pair_lower_c25_0_3_prev4: 240 same_dist_pair_c25_0_3_prev4
   + dist_diff_abs_pair_c25_0_3_prev4 >= 10
 dist_diff_pair_def_c25_0_3_prev5: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev5 = -56
 dist_diff_abs_pair_c25_0_3_prev5_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev5 + dist_diff_abs_pair_c25_0_3_prev5
   - 240 dist_diff_abs_pair_c25_0_3_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev5_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev5 + dist_diff_abs_pair_c25_0_3_prev5
   + 240 dist_diff_abs_pair_c25_0_3_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev5 + dist_diff_abs_pair_c25_0_3_prev5
   + 240 dist_diff_abs_pair_c25_0_3_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev5 + dist_diff_abs_pair_c25_0_3_prev5
   - 240 dist_diff_abs_pair_c25_0_3_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev5_force_positive:
   dist_diff_pair_c25_0_3_prev5
   - 240 dist_diff_abs_pair_c25_0_3_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev5_force_negative:
   dist_diff_pair_c25_0_3_prev5
   - 240 dist_diff_abs_pair_c25_0_3_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev5: 240 same_dist_pair_c25_0_3_prev5
   + dist_diff_abs_pair_c25_0_3_prev5 <= 249.8
 same_dist_pair_lower_c25_0_3_prev5: 240 same_dist_pair_c25_0_3_prev5
   + dist_diff_abs_pair_c25_0_3_prev5 >= 10
 dist_diff_pair_def_c25_0_3_prev6: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev6 = -46
 dist_diff_abs_pair_c25_0_3_prev6_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev6 + dist_diff_abs_pair_c25_0_3_prev6
   - 240 dist_diff_abs_pair_c25_0_3_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev6_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev6 + dist_diff_abs_pair_c25_0_3_prev6
   + 240 dist_diff_abs_pair_c25_0_3_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev6 + dist_diff_abs_pair_c25_0_3_prev6
   + 240 dist_diff_abs_pair_c25_0_3_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev6 + dist_diff_abs_pair_c25_0_3_prev6
   - 240 dist_diff_abs_pair_c25_0_3_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev6_force_positive:
   dist_diff_pair_c25_0_3_prev6
   - 240 dist_diff_abs_pair_c25_0_3_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev6_force_negative:
   dist_diff_pair_c25_0_3_prev6
   - 240 dist_diff_abs_pair_c25_0_3_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev6: 240 same_dist_pair_c25_0_3_prev6
   + dist_diff_abs_pair_c25_0_3_prev6 <= 249.8
 same_dist_pair_lower_c25_0_3_prev6: 240 same_dist_pair_c25_0_3_prev6
   + dist_diff_abs_pair_c25_0_3_prev6 >= 10
 dist_diff_pair_def_c25_0_3_prev7: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev7 = -68
 dist_diff_abs_pair_c25_0_3_prev7_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev7 + dist_diff_abs_pair_c25_0_3_prev7
   - 240 dist_diff_abs_pair_c25_0_3_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev7_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev7 + dist_diff_abs_pair_c25_0_3_prev7
   + 240 dist_diff_abs_pair_c25_0_3_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev7 + dist_diff_abs_pair_c25_0_3_prev7
   + 240 dist_diff_abs_pair_c25_0_3_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev7 + dist_diff_abs_pair_c25_0_3_prev7
   - 240 dist_diff_abs_pair_c25_0_3_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev7_force_positive:
   dist_diff_pair_c25_0_3_prev7
   - 240 dist_diff_abs_pair_c25_0_3_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev7_force_negative:
   dist_diff_pair_c25_0_3_prev7
   - 240 dist_diff_abs_pair_c25_0_3_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev7: 240 same_dist_pair_c25_0_3_prev7
   + dist_diff_abs_pair_c25_0_3_prev7 <= 249.8
 same_dist_pair_lower_c25_0_3_prev7: 240 same_dist_pair_c25_0_3_prev7
   + dist_diff_abs_pair_c25_0_3_prev7 >= 10
 dist_diff_pair_def_c25_0_3_prev8: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev8 = -65
 dist_diff_abs_pair_c25_0_3_prev8_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev8 + dist_diff_abs_pair_c25_0_3_prev8
   - 240 dist_diff_abs_pair_c25_0_3_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev8_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev8 + dist_diff_abs_pair_c25_0_3_prev8
   + 240 dist_diff_abs_pair_c25_0_3_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev8 + dist_diff_abs_pair_c25_0_3_prev8
   + 240 dist_diff_abs_pair_c25_0_3_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev8 + dist_diff_abs_pair_c25_0_3_prev8
   - 240 dist_diff_abs_pair_c25_0_3_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev8_force_positive:
   dist_diff_pair_c25_0_3_prev8
   - 240 dist_diff_abs_pair_c25_0_3_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev8_force_negative:
   dist_diff_pair_c25_0_3_prev8
   - 240 dist_diff_abs_pair_c25_0_3_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev8: 240 same_dist_pair_c25_0_3_prev8
   + dist_diff_abs_pair_c25_0_3_prev8 <= 249.8
 same_dist_pair_lower_c25_0_3_prev8: 240 same_dist_pair_c25_0_3_prev8
   + dist_diff_abs_pair_c25_0_3_prev8 >= 10
 dist_diff_pair_def_c25_0_3_prev9: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev9 = -54
 dist_diff_abs_pair_c25_0_3_prev9_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev9 + dist_diff_abs_pair_c25_0_3_prev9
   - 240 dist_diff_abs_pair_c25_0_3_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev9_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev9 + dist_diff_abs_pair_c25_0_3_prev9
   + 240 dist_diff_abs_pair_c25_0_3_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev9 + dist_diff_abs_pair_c25_0_3_prev9
   + 240 dist_diff_abs_pair_c25_0_3_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev9 + dist_diff_abs_pair_c25_0_3_prev9
   - 240 dist_diff_abs_pair_c25_0_3_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev9_force_positive:
   dist_diff_pair_c25_0_3_prev9
   - 240 dist_diff_abs_pair_c25_0_3_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev9_force_negative:
   dist_diff_pair_c25_0_3_prev9
   - 240 dist_diff_abs_pair_c25_0_3_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev9: 240 same_dist_pair_c25_0_3_prev9
   + dist_diff_abs_pair_c25_0_3_prev9 <= 249.8
 same_dist_pair_lower_c25_0_3_prev9: 240 same_dist_pair_c25_0_3_prev9
   + dist_diff_abs_pair_c25_0_3_prev9 >= 10
 dist_diff_pair_def_c25_0_3_prev10: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev10 = -56
 dist_diff_abs_pair_c25_0_3_prev10_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev10 + dist_diff_abs_pair_c25_0_3_prev10
   - 240 dist_diff_abs_pair_c25_0_3_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev10_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev10 + dist_diff_abs_pair_c25_0_3_prev10
   + 240 dist_diff_abs_pair_c25_0_3_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev10 + dist_diff_abs_pair_c25_0_3_prev10
   + 240 dist_diff_abs_pair_c25_0_3_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev10 + dist_diff_abs_pair_c25_0_3_prev10
   - 240 dist_diff_abs_pair_c25_0_3_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev10_force_positive:
   dist_diff_pair_c25_0_3_prev10
   - 240 dist_diff_abs_pair_c25_0_3_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev10_force_negative:
   dist_diff_pair_c25_0_3_prev10
   - 240 dist_diff_abs_pair_c25_0_3_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev10: 240 same_dist_pair_c25_0_3_prev10
   + dist_diff_abs_pair_c25_0_3_prev10 <= 249.8
 same_dist_pair_lower_c25_0_3_prev10: 240 same_dist_pair_c25_0_3_prev10
   + dist_diff_abs_pair_c25_0_3_prev10 >= 10
 dist_diff_pair_def_c25_0_3_prev11: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev11 = -52
 dist_diff_abs_pair_c25_0_3_prev11_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev11 + dist_diff_abs_pair_c25_0_3_prev11
   - 240 dist_diff_abs_pair_c25_0_3_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev11_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev11 + dist_diff_abs_pair_c25_0_3_prev11
   + 240 dist_diff_abs_pair_c25_0_3_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev11 + dist_diff_abs_pair_c25_0_3_prev11
   + 240 dist_diff_abs_pair_c25_0_3_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev11 + dist_diff_abs_pair_c25_0_3_prev11
   - 240 dist_diff_abs_pair_c25_0_3_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev11_force_positive:
   dist_diff_pair_c25_0_3_prev11
   - 240 dist_diff_abs_pair_c25_0_3_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev11_force_negative:
   dist_diff_pair_c25_0_3_prev11
   - 240 dist_diff_abs_pair_c25_0_3_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev11: 240 same_dist_pair_c25_0_3_prev11
   + dist_diff_abs_pair_c25_0_3_prev11 <= 249.8
 same_dist_pair_lower_c25_0_3_prev11: 240 same_dist_pair_c25_0_3_prev11
   + dist_diff_abs_pair_c25_0_3_prev11 >= 10
 dist_diff_pair_def_c25_0_3_prev12: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev12 = -78
 dist_diff_abs_pair_c25_0_3_prev12_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev12 + dist_diff_abs_pair_c25_0_3_prev12
   - 240 dist_diff_abs_pair_c25_0_3_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev12_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev12 + dist_diff_abs_pair_c25_0_3_prev12
   + 240 dist_diff_abs_pair_c25_0_3_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev12 + dist_diff_abs_pair_c25_0_3_prev12
   + 240 dist_diff_abs_pair_c25_0_3_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev12 + dist_diff_abs_pair_c25_0_3_prev12
   - 240 dist_diff_abs_pair_c25_0_3_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev12_force_positive:
   dist_diff_pair_c25_0_3_prev12
   - 240 dist_diff_abs_pair_c25_0_3_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev12_force_negative:
   dist_diff_pair_c25_0_3_prev12
   - 240 dist_diff_abs_pair_c25_0_3_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev12: 240 same_dist_pair_c25_0_3_prev12
   + dist_diff_abs_pair_c25_0_3_prev12 <= 249.8
 same_dist_pair_lower_c25_0_3_prev12: 240 same_dist_pair_c25_0_3_prev12
   + dist_diff_abs_pair_c25_0_3_prev12 >= 10
 dist_diff_pair_def_c25_0_3_prev13: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev13 = -36
 dist_diff_abs_pair_c25_0_3_prev13_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev13 + dist_diff_abs_pair_c25_0_3_prev13
   - 240 dist_diff_abs_pair_c25_0_3_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev13_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev13 + dist_diff_abs_pair_c25_0_3_prev13
   + 240 dist_diff_abs_pair_c25_0_3_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev13 + dist_diff_abs_pair_c25_0_3_prev13
   + 240 dist_diff_abs_pair_c25_0_3_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev13 + dist_diff_abs_pair_c25_0_3_prev13
   - 240 dist_diff_abs_pair_c25_0_3_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev13_force_positive:
   dist_diff_pair_c25_0_3_prev13
   - 240 dist_diff_abs_pair_c25_0_3_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev13_force_negative:
   dist_diff_pair_c25_0_3_prev13
   - 240 dist_diff_abs_pair_c25_0_3_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev13: 240 same_dist_pair_c25_0_3_prev13
   + dist_diff_abs_pair_c25_0_3_prev13 <= 249.8
 same_dist_pair_lower_c25_0_3_prev13: 240 same_dist_pair_c25_0_3_prev13
   + dist_diff_abs_pair_c25_0_3_prev13 >= 10
 dist_diff_pair_def_c25_0_3_prev14: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev14 = -54
 dist_diff_abs_pair_c25_0_3_prev14_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev14 + dist_diff_abs_pair_c25_0_3_prev14
   - 240 dist_diff_abs_pair_c25_0_3_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev14_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev14 + dist_diff_abs_pair_c25_0_3_prev14
   + 240 dist_diff_abs_pair_c25_0_3_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev14 + dist_diff_abs_pair_c25_0_3_prev14
   + 240 dist_diff_abs_pair_c25_0_3_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev14 + dist_diff_abs_pair_c25_0_3_prev14
   - 240 dist_diff_abs_pair_c25_0_3_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev14_force_positive:
   dist_diff_pair_c25_0_3_prev14
   - 240 dist_diff_abs_pair_c25_0_3_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev14_force_negative:
   dist_diff_pair_c25_0_3_prev14
   - 240 dist_diff_abs_pair_c25_0_3_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev14: 240 same_dist_pair_c25_0_3_prev14
   + dist_diff_abs_pair_c25_0_3_prev14 <= 249.8
 same_dist_pair_lower_c25_0_3_prev14: 240 same_dist_pair_c25_0_3_prev14
   + dist_diff_abs_pair_c25_0_3_prev14 >= 10
 dist_diff_pair_def_c25_0_3_prev15: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev15 = -37
 dist_diff_abs_pair_c25_0_3_prev15_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev15 + dist_diff_abs_pair_c25_0_3_prev15
   - 240 dist_diff_abs_pair_c25_0_3_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev15_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev15 + dist_diff_abs_pair_c25_0_3_prev15
   + 240 dist_diff_abs_pair_c25_0_3_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev15 + dist_diff_abs_pair_c25_0_3_prev15
   + 240 dist_diff_abs_pair_c25_0_3_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev15 + dist_diff_abs_pair_c25_0_3_prev15
   - 240 dist_diff_abs_pair_c25_0_3_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev15_force_positive:
   dist_diff_pair_c25_0_3_prev15
   - 240 dist_diff_abs_pair_c25_0_3_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev15_force_negative:
   dist_diff_pair_c25_0_3_prev15
   - 240 dist_diff_abs_pair_c25_0_3_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev15: 240 same_dist_pair_c25_0_3_prev15
   + dist_diff_abs_pair_c25_0_3_prev15 <= 249.8
 same_dist_pair_lower_c25_0_3_prev15: 240 same_dist_pair_c25_0_3_prev15
   + dist_diff_abs_pair_c25_0_3_prev15 >= 10
 dist_diff_pair_def_c25_0_3_prev16: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev16 = -58
 dist_diff_abs_pair_c25_0_3_prev16_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev16 + dist_diff_abs_pair_c25_0_3_prev16
   - 240 dist_diff_abs_pair_c25_0_3_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev16_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev16 + dist_diff_abs_pair_c25_0_3_prev16
   + 240 dist_diff_abs_pair_c25_0_3_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev16 + dist_diff_abs_pair_c25_0_3_prev16
   + 240 dist_diff_abs_pair_c25_0_3_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev16 + dist_diff_abs_pair_c25_0_3_prev16
   - 240 dist_diff_abs_pair_c25_0_3_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev16_force_positive:
   dist_diff_pair_c25_0_3_prev16
   - 240 dist_diff_abs_pair_c25_0_3_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev16_force_negative:
   dist_diff_pair_c25_0_3_prev16
   - 240 dist_diff_abs_pair_c25_0_3_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev16: 240 same_dist_pair_c25_0_3_prev16
   + dist_diff_abs_pair_c25_0_3_prev16 <= 249.8
 same_dist_pair_lower_c25_0_3_prev16: 240 same_dist_pair_c25_0_3_prev16
   + dist_diff_abs_pair_c25_0_3_prev16 >= 10
 dist_diff_pair_def_c25_0_3_prev17: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev17 = -66
 dist_diff_abs_pair_c25_0_3_prev17_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev17 + dist_diff_abs_pair_c25_0_3_prev17
   - 240 dist_diff_abs_pair_c25_0_3_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev17_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev17 + dist_diff_abs_pair_c25_0_3_prev17
   + 240 dist_diff_abs_pair_c25_0_3_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev17 + dist_diff_abs_pair_c25_0_3_prev17
   + 240 dist_diff_abs_pair_c25_0_3_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev17 + dist_diff_abs_pair_c25_0_3_prev17
   - 240 dist_diff_abs_pair_c25_0_3_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev17_force_positive:
   dist_diff_pair_c25_0_3_prev17
   - 240 dist_diff_abs_pair_c25_0_3_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev17_force_negative:
   dist_diff_pair_c25_0_3_prev17
   - 240 dist_diff_abs_pair_c25_0_3_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev17: 240 same_dist_pair_c25_0_3_prev17
   + dist_diff_abs_pair_c25_0_3_prev17 <= 249.8
 same_dist_pair_lower_c25_0_3_prev17: 240 same_dist_pair_c25_0_3_prev17
   + dist_diff_abs_pair_c25_0_3_prev17 >= 10
 dist_diff_pair_def_c25_0_3_prev18: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev18 = -88
 dist_diff_abs_pair_c25_0_3_prev18_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev18 + dist_diff_abs_pair_c25_0_3_prev18
   - 240 dist_diff_abs_pair_c25_0_3_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev18_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev18 + dist_diff_abs_pair_c25_0_3_prev18
   + 240 dist_diff_abs_pair_c25_0_3_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev18 + dist_diff_abs_pair_c25_0_3_prev18
   + 240 dist_diff_abs_pair_c25_0_3_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev18 + dist_diff_abs_pair_c25_0_3_prev18
   - 240 dist_diff_abs_pair_c25_0_3_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev18_force_positive:
   dist_diff_pair_c25_0_3_prev18
   - 240 dist_diff_abs_pair_c25_0_3_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev18_force_negative:
   dist_diff_pair_c25_0_3_prev18
   - 240 dist_diff_abs_pair_c25_0_3_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev18: 240 same_dist_pair_c25_0_3_prev18
   + dist_diff_abs_pair_c25_0_3_prev18 <= 249.8
 same_dist_pair_lower_c25_0_3_prev18: 240 same_dist_pair_c25_0_3_prev18
   + dist_diff_abs_pair_c25_0_3_prev18 >= 10
 dist_diff_pair_def_c25_0_3_prev19: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev19 = -70
 dist_diff_abs_pair_c25_0_3_prev19_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev19 + dist_diff_abs_pair_c25_0_3_prev19
   - 240 dist_diff_abs_pair_c25_0_3_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev19_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev19 + dist_diff_abs_pair_c25_0_3_prev19
   + 240 dist_diff_abs_pair_c25_0_3_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev19 + dist_diff_abs_pair_c25_0_3_prev19
   + 240 dist_diff_abs_pair_c25_0_3_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev19 + dist_diff_abs_pair_c25_0_3_prev19
   - 240 dist_diff_abs_pair_c25_0_3_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev19_force_positive:
   dist_diff_pair_c25_0_3_prev19
   - 240 dist_diff_abs_pair_c25_0_3_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev19_force_negative:
   dist_diff_pair_c25_0_3_prev19
   - 240 dist_diff_abs_pair_c25_0_3_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev19: 240 same_dist_pair_c25_0_3_prev19
   + dist_diff_abs_pair_c25_0_3_prev19 <= 249.8
 same_dist_pair_lower_c25_0_3_prev19: 240 same_dist_pair_c25_0_3_prev19
   + dist_diff_abs_pair_c25_0_3_prev19 >= 10
 dist_diff_pair_def_c25_0_3_prev20: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev20 = -60
 dist_diff_abs_pair_c25_0_3_prev20_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev20 + dist_diff_abs_pair_c25_0_3_prev20
   - 240 dist_diff_abs_pair_c25_0_3_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev20_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev20 + dist_diff_abs_pair_c25_0_3_prev20
   + 240 dist_diff_abs_pair_c25_0_3_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev20 + dist_diff_abs_pair_c25_0_3_prev20
   + 240 dist_diff_abs_pair_c25_0_3_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev20 + dist_diff_abs_pair_c25_0_3_prev20
   - 240 dist_diff_abs_pair_c25_0_3_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev20_force_positive:
   dist_diff_pair_c25_0_3_prev20
   - 240 dist_diff_abs_pair_c25_0_3_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev20_force_negative:
   dist_diff_pair_c25_0_3_prev20
   - 240 dist_diff_abs_pair_c25_0_3_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev20: 240 same_dist_pair_c25_0_3_prev20
   + dist_diff_abs_pair_c25_0_3_prev20 <= 249.8
 same_dist_pair_lower_c25_0_3_prev20: 240 same_dist_pair_c25_0_3_prev20
   + dist_diff_abs_pair_c25_0_3_prev20 >= 10
 dist_diff_pair_def_c25_0_3_prev21: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev21 = -58
 dist_diff_abs_pair_c25_0_3_prev21_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev21 + dist_diff_abs_pair_c25_0_3_prev21
   - 240 dist_diff_abs_pair_c25_0_3_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev21_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev21 + dist_diff_abs_pair_c25_0_3_prev21
   + 240 dist_diff_abs_pair_c25_0_3_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev21 + dist_diff_abs_pair_c25_0_3_prev21
   + 240 dist_diff_abs_pair_c25_0_3_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev21 + dist_diff_abs_pair_c25_0_3_prev21
   - 240 dist_diff_abs_pair_c25_0_3_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev21_force_positive:
   dist_diff_pair_c25_0_3_prev21
   - 240 dist_diff_abs_pair_c25_0_3_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev21_force_negative:
   dist_diff_pair_c25_0_3_prev21
   - 240 dist_diff_abs_pair_c25_0_3_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev21: 240 same_dist_pair_c25_0_3_prev21
   + dist_diff_abs_pair_c25_0_3_prev21 <= 249.8
 same_dist_pair_lower_c25_0_3_prev21: 240 same_dist_pair_c25_0_3_prev21
   + dist_diff_abs_pair_c25_0_3_prev21 >= 10
 dist_diff_pair_def_c25_0_3_prev22: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev22 = -57
 dist_diff_abs_pair_c25_0_3_prev22_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev22 + dist_diff_abs_pair_c25_0_3_prev22
   - 240 dist_diff_abs_pair_c25_0_3_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev22_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev22 + dist_diff_abs_pair_c25_0_3_prev22
   + 240 dist_diff_abs_pair_c25_0_3_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev22 + dist_diff_abs_pair_c25_0_3_prev22
   + 240 dist_diff_abs_pair_c25_0_3_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev22 + dist_diff_abs_pair_c25_0_3_prev22
   - 240 dist_diff_abs_pair_c25_0_3_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev22_force_positive:
   dist_diff_pair_c25_0_3_prev22
   - 240 dist_diff_abs_pair_c25_0_3_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev22_force_negative:
   dist_diff_pair_c25_0_3_prev22
   - 240 dist_diff_abs_pair_c25_0_3_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev22: 240 same_dist_pair_c25_0_3_prev22
   + dist_diff_abs_pair_c25_0_3_prev22 <= 249.8
 same_dist_pair_lower_c25_0_3_prev22: 240 same_dist_pair_c25_0_3_prev22
   + dist_diff_abs_pair_c25_0_3_prev22 >= 10
 dist_diff_pair_def_c25_0_3_prev23: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev23 = -58
 dist_diff_abs_pair_c25_0_3_prev23_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev23 + dist_diff_abs_pair_c25_0_3_prev23
   - 240 dist_diff_abs_pair_c25_0_3_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev23_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev23 + dist_diff_abs_pair_c25_0_3_prev23
   + 240 dist_diff_abs_pair_c25_0_3_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev23 + dist_diff_abs_pair_c25_0_3_prev23
   + 240 dist_diff_abs_pair_c25_0_3_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev23 + dist_diff_abs_pair_c25_0_3_prev23
   - 240 dist_diff_abs_pair_c25_0_3_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev23_force_positive:
   dist_diff_pair_c25_0_3_prev23
   - 240 dist_diff_abs_pair_c25_0_3_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev23_force_negative:
   dist_diff_pair_c25_0_3_prev23
   - 240 dist_diff_abs_pair_c25_0_3_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev23: 240 same_dist_pair_c25_0_3_prev23
   + dist_diff_abs_pair_c25_0_3_prev23 <= 249.8
 same_dist_pair_lower_c25_0_3_prev23: 240 same_dist_pair_c25_0_3_prev23
   + dist_diff_abs_pair_c25_0_3_prev23 >= 10
 dist_diff_pair_def_c25_0_3_prev24: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev24 = -34
 dist_diff_abs_pair_c25_0_3_prev24_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev24 + dist_diff_abs_pair_c25_0_3_prev24
   - 240 dist_diff_abs_pair_c25_0_3_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev24_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev24 + dist_diff_abs_pair_c25_0_3_prev24
   + 240 dist_diff_abs_pair_c25_0_3_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev24 + dist_diff_abs_pair_c25_0_3_prev24
   + 240 dist_diff_abs_pair_c25_0_3_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev24 + dist_diff_abs_pair_c25_0_3_prev24
   - 240 dist_diff_abs_pair_c25_0_3_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev24_force_positive:
   dist_diff_pair_c25_0_3_prev24
   - 240 dist_diff_abs_pair_c25_0_3_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev24_force_negative:
   dist_diff_pair_c25_0_3_prev24
   - 240 dist_diff_abs_pair_c25_0_3_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev24: 240 same_dist_pair_c25_0_3_prev24
   + dist_diff_abs_pair_c25_0_3_prev24 <= 249.8
 same_dist_pair_lower_c25_0_3_prev24: 240 same_dist_pair_c25_0_3_prev24
   + dist_diff_abs_pair_c25_0_3_prev24 >= 10
 dist_diff_pair_def_c25_0_3_prev25: - dist_curr_pair_c25_0_3
   + dist_diff_pair_c25_0_3_prev25 = -65
 dist_diff_abs_pair_c25_0_3_prev25_abs_ge_orig:
   - dist_diff_pair_c25_0_3_prev25 + dist_diff_abs_pair_c25_0_3_prev25
   - 240 dist_diff_abs_pair_c25_0_3_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev25_abs_le_orig:
   - dist_diff_pair_c25_0_3_prev25 + dist_diff_abs_pair_c25_0_3_prev25
   + 240 dist_diff_abs_pair_c25_0_3_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_0_3_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_0_3_prev25 + dist_diff_abs_pair_c25_0_3_prev25
   + 240 dist_diff_abs_pair_c25_0_3_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_0_3_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_0_3_prev25 + dist_diff_abs_pair_c25_0_3_prev25
   - 240 dist_diff_abs_pair_c25_0_3_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_0_3_prev25_force_positive:
   dist_diff_pair_c25_0_3_prev25
   - 240 dist_diff_abs_pair_c25_0_3_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_0_3_prev25_force_negative:
   dist_diff_pair_c25_0_3_prev25
   - 240 dist_diff_abs_pair_c25_0_3_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_0_3_prev25: 240 same_dist_pair_c25_0_3_prev25
   + dist_diff_abs_pair_c25_0_3_prev25 <= 249.8
 same_dist_pair_lower_c25_0_3_prev25: 240 same_dist_pair_c25_0_3_prev25
   + dist_diff_abs_pair_c25_0_3_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_0_3_prev0: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev0: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev1: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev1: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev2: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev2: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev3: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev3: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev4: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev4: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev5: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev5: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev6: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev6: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev7: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev7: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev8: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev8: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev9: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev9: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev10: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev10: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev11: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev11: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev12: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev12: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev13: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev13: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev14: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev14: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev15: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev15: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev16: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev16: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev17: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev17: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev18: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev18: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev19: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev19: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev20: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev20: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev21: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev21: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev22: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev22: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev23: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev23: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev24: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev24: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_0_3_prev25: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_0_3_prev25: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_0_3: diff_dist_pair_c25_0_3
   + same_dist_pair_c25_0_3_prev0 + same_dist_pair_c25_0_3_prev1
   + same_dist_pair_c25_0_3_prev2 + same_dist_pair_c25_0_3_prev3
   + same_dist_pair_c25_0_3_prev4 + same_dist_pair_c25_0_3_prev5
   + same_dist_pair_c25_0_3_prev6 + same_dist_pair_c25_0_3_prev7
   + same_dist_pair_c25_0_3_prev8 + same_dist_pair_c25_0_3_prev9
   + same_dist_pair_c25_0_3_prev10 + same_dist_pair_c25_0_3_prev11
   + same_dist_pair_c25_0_3_prev12 + same_dist_pair_c25_0_3_prev13
   + same_dist_pair_c25_0_3_prev14 + same_dist_pair_c25_0_3_prev15
   + same_dist_pair_c25_0_3_prev16 + same_dist_pair_c25_0_3_prev17
   + same_dist_pair_c25_0_3_prev18 + same_dist_pair_c25_0_3_prev19
   + same_dist_pair_c25_0_3_prev20 + same_dist_pair_c25_0_3_prev21
   + same_dist_pair_c25_0_3_prev22 + same_dist_pair_c25_0_3_prev23
   + same_dist_pair_c25_0_3_prev24 + same_dist_pair_c25_0_3_prev25 >= 1
 dx_grid_diff_def_c25_0_4: - cx_grid_center_c25_0 + cx_grid_center_c25_4
   + dx_grid_diff_c25_0_4 = 0
 dx_grid_abs_pair_c25_0_4_abs_ge_orig: dx_grid_abs_pair_c25_0_4
   - dx_grid_diff_c25_0_4 - 130 dx_grid_abs_pair_c25_0_4_is_positive
   >= -130
 dx_grid_abs_pair_c25_0_4_abs_le_orig: dx_grid_abs_pair_c25_0_4
   - dx_grid_diff_c25_0_4 + 130 dx_grid_abs_pair_c25_0_4_is_positive
   <= 130
 dx_grid_abs_pair_c25_0_4_abs_ge_neg_orig: dx_grid_abs_pair_c25_0_4
   + dx_grid_diff_c25_0_4 + 130 dx_grid_abs_pair_c25_0_4_is_positive >= 0
 dx_grid_abs_pair_c25_0_4_abs_le_neg_orig: dx_grid_abs_pair_c25_0_4
   + dx_grid_diff_c25_0_4 - 130 dx_grid_abs_pair_c25_0_4_is_positive <= 0
 dx_grid_abs_pair_c25_0_4_force_positive: dx_grid_diff_c25_0_4
   - 130 dx_grid_abs_pair_c25_0_4_is_positive >= -130
 dx_grid_abs_pair_c25_0_4_force_negative: dx_grid_diff_c25_0_4
   - 130 dx_grid_abs_pair_c25_0_4_is_positive <= -0.001
 dy_grid_diff_def_c25_0_4: - cy_grid_center_c25_0 + cy_grid_center_c25_4
   + dy_grid_diff_c25_0_4 = 0
 dy_grid_abs_pair_c25_0_4_abs_ge_orig: dy_grid_abs_pair_c25_0_4
   - dy_grid_diff_c25_0_4 - 130 dy_grid_abs_pair_c25_0_4_is_positive
   >= -130
 dy_grid_abs_pair_c25_0_4_abs_le_orig: dy_grid_abs_pair_c25_0_4
   - dy_grid_diff_c25_0_4 + 130 dy_grid_abs_pair_c25_0_4_is_positive
   <= 130
 dy_grid_abs_pair_c25_0_4_abs_ge_neg_orig: dy_grid_abs_pair_c25_0_4
   + dy_grid_diff_c25_0_4 + 130 dy_grid_abs_pair_c25_0_4_is_positive >= 0
 dy_grid_abs_pair_c25_0_4_abs_le_neg_orig: dy_grid_abs_pair_c25_0_4
   + dy_grid_diff_c25_0_4 - 130 dy_grid_abs_pair_c25_0_4_is_positive <= 0
 dy_grid_abs_pair_c25_0_4_force_positive: dy_grid_diff_c25_0_4
   - 130 dy_grid_abs_pair_c25_0_4_is_positive >= -130
 dy_grid_abs_pair_c25_0_4_force_negative: dy_grid_diff_c25_0_4
   - 130 dy_grid_abs_pair_c25_0_4_is_positive <= -0.001
 dist_curr_pair_def_c25_0_4: - dx_grid_abs_pair_c25_0_4
   - dy_grid_abs_pair_c25_0_4 + dist_curr_pair_c25_0_4 = 0
 dist_diff_pair_def_c25_0_4_prev0: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev0 = -76
 dist_diff_abs_pair_c25_0_4_prev0_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev0 + dist_diff_abs_pair_c25_0_4_prev0
   - 240 dist_diff_abs_pair_c25_0_4_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev0_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev0 + dist_diff_abs_pair_c25_0_4_prev0
   + 240 dist_diff_abs_pair_c25_0_4_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev0 + dist_diff_abs_pair_c25_0_4_prev0
   + 240 dist_diff_abs_pair_c25_0_4_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev0 + dist_diff_abs_pair_c25_0_4_prev0
   - 240 dist_diff_abs_pair_c25_0_4_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev0_force_positive:
   dist_diff_pair_c25_0_4_prev0
   - 240 dist_diff_abs_pair_c25_0_4_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev0_force_negative:
   dist_diff_pair_c25_0_4_prev0
   - 240 dist_diff_abs_pair_c25_0_4_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev0: 240 same_dist_pair_c25_0_4_prev0
   + dist_diff_abs_pair_c25_0_4_prev0 <= 249.8
 same_dist_pair_lower_c25_0_4_prev0: 240 same_dist_pair_c25_0_4_prev0
   + dist_diff_abs_pair_c25_0_4_prev0 >= 10
 dist_diff_pair_def_c25_0_4_prev1: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev1 = -42
 dist_diff_abs_pair_c25_0_4_prev1_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev1 + dist_diff_abs_pair_c25_0_4_prev1
   - 240 dist_diff_abs_pair_c25_0_4_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev1_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev1 + dist_diff_abs_pair_c25_0_4_prev1
   + 240 dist_diff_abs_pair_c25_0_4_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev1 + dist_diff_abs_pair_c25_0_4_prev1
   + 240 dist_diff_abs_pair_c25_0_4_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev1 + dist_diff_abs_pair_c25_0_4_prev1
   - 240 dist_diff_abs_pair_c25_0_4_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev1_force_positive:
   dist_diff_pair_c25_0_4_prev1
   - 240 dist_diff_abs_pair_c25_0_4_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev1_force_negative:
   dist_diff_pair_c25_0_4_prev1
   - 240 dist_diff_abs_pair_c25_0_4_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev1: 240 same_dist_pair_c25_0_4_prev1
   + dist_diff_abs_pair_c25_0_4_prev1 <= 249.8
 same_dist_pair_lower_c25_0_4_prev1: 240 same_dist_pair_c25_0_4_prev1
   + dist_diff_abs_pair_c25_0_4_prev1 >= 10
 dist_diff_pair_def_c25_0_4_prev2: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev2 = -76
 dist_diff_abs_pair_c25_0_4_prev2_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev2 + dist_diff_abs_pair_c25_0_4_prev2
   - 240 dist_diff_abs_pair_c25_0_4_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev2_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev2 + dist_diff_abs_pair_c25_0_4_prev2
   + 240 dist_diff_abs_pair_c25_0_4_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev2 + dist_diff_abs_pair_c25_0_4_prev2
   + 240 dist_diff_abs_pair_c25_0_4_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev2 + dist_diff_abs_pair_c25_0_4_prev2
   - 240 dist_diff_abs_pair_c25_0_4_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev2_force_positive:
   dist_diff_pair_c25_0_4_prev2
   - 240 dist_diff_abs_pair_c25_0_4_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev2_force_negative:
   dist_diff_pair_c25_0_4_prev2
   - 240 dist_diff_abs_pair_c25_0_4_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev2: 240 same_dist_pair_c25_0_4_prev2
   + dist_diff_abs_pair_c25_0_4_prev2 <= 249.8
 same_dist_pair_lower_c25_0_4_prev2: 240 same_dist_pair_c25_0_4_prev2
   + dist_diff_abs_pair_c25_0_4_prev2 >= 10
 dist_diff_pair_def_c25_0_4_prev3: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev3 = -79
 dist_diff_abs_pair_c25_0_4_prev3_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev3 + dist_diff_abs_pair_c25_0_4_prev3
   - 240 dist_diff_abs_pair_c25_0_4_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev3_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev3 + dist_diff_abs_pair_c25_0_4_prev3
   + 240 dist_diff_abs_pair_c25_0_4_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev3 + dist_diff_abs_pair_c25_0_4_prev3
   + 240 dist_diff_abs_pair_c25_0_4_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev3 + dist_diff_abs_pair_c25_0_4_prev3
   - 240 dist_diff_abs_pair_c25_0_4_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev3_force_positive:
   dist_diff_pair_c25_0_4_prev3
   - 240 dist_diff_abs_pair_c25_0_4_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev3_force_negative:
   dist_diff_pair_c25_0_4_prev3
   - 240 dist_diff_abs_pair_c25_0_4_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev3: 240 same_dist_pair_c25_0_4_prev3
   + dist_diff_abs_pair_c25_0_4_prev3 <= 249.8
 same_dist_pair_lower_c25_0_4_prev3: 240 same_dist_pair_c25_0_4_prev3
   + dist_diff_abs_pair_c25_0_4_prev3 >= 10
 dist_diff_pair_def_c25_0_4_prev4: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev4 = -52
 dist_diff_abs_pair_c25_0_4_prev4_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev4 + dist_diff_abs_pair_c25_0_4_prev4
   - 240 dist_diff_abs_pair_c25_0_4_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev4_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev4 + dist_diff_abs_pair_c25_0_4_prev4
   + 240 dist_diff_abs_pair_c25_0_4_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev4 + dist_diff_abs_pair_c25_0_4_prev4
   + 240 dist_diff_abs_pair_c25_0_4_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev4 + dist_diff_abs_pair_c25_0_4_prev4
   - 240 dist_diff_abs_pair_c25_0_4_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev4_force_positive:
   dist_diff_pair_c25_0_4_prev4
   - 240 dist_diff_abs_pair_c25_0_4_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev4_force_negative:
   dist_diff_pair_c25_0_4_prev4
   - 240 dist_diff_abs_pair_c25_0_4_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev4: 240 same_dist_pair_c25_0_4_prev4
   + dist_diff_abs_pair_c25_0_4_prev4 <= 249.8
 same_dist_pair_lower_c25_0_4_prev4: 240 same_dist_pair_c25_0_4_prev4
   + dist_diff_abs_pair_c25_0_4_prev4 >= 10
 dist_diff_pair_def_c25_0_4_prev5: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev5 = -66
 dist_diff_abs_pair_c25_0_4_prev5_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev5 + dist_diff_abs_pair_c25_0_4_prev5
   - 240 dist_diff_abs_pair_c25_0_4_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev5_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev5 + dist_diff_abs_pair_c25_0_4_prev5
   + 240 dist_diff_abs_pair_c25_0_4_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev5 + dist_diff_abs_pair_c25_0_4_prev5
   + 240 dist_diff_abs_pair_c25_0_4_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev5 + dist_diff_abs_pair_c25_0_4_prev5
   - 240 dist_diff_abs_pair_c25_0_4_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev5_force_positive:
   dist_diff_pair_c25_0_4_prev5
   - 240 dist_diff_abs_pair_c25_0_4_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev5_force_negative:
   dist_diff_pair_c25_0_4_prev5
   - 240 dist_diff_abs_pair_c25_0_4_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev5: 240 same_dist_pair_c25_0_4_prev5
   + dist_diff_abs_pair_c25_0_4_prev5 <= 249.8
 same_dist_pair_lower_c25_0_4_prev5: 240 same_dist_pair_c25_0_4_prev5
   + dist_diff_abs_pair_c25_0_4_prev5 >= 10
 dist_diff_pair_def_c25_0_4_prev6: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev6 = -76
 dist_diff_abs_pair_c25_0_4_prev6_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev6 + dist_diff_abs_pair_c25_0_4_prev6
   - 240 dist_diff_abs_pair_c25_0_4_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev6_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev6 + dist_diff_abs_pair_c25_0_4_prev6
   + 240 dist_diff_abs_pair_c25_0_4_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev6 + dist_diff_abs_pair_c25_0_4_prev6
   + 240 dist_diff_abs_pair_c25_0_4_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev6 + dist_diff_abs_pair_c25_0_4_prev6
   - 240 dist_diff_abs_pair_c25_0_4_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev6_force_positive:
   dist_diff_pair_c25_0_4_prev6
   - 240 dist_diff_abs_pair_c25_0_4_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev6_force_negative:
   dist_diff_pair_c25_0_4_prev6
   - 240 dist_diff_abs_pair_c25_0_4_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev6: 240 same_dist_pair_c25_0_4_prev6
   + dist_diff_abs_pair_c25_0_4_prev6 <= 249.8
 same_dist_pair_lower_c25_0_4_prev6: 240 same_dist_pair_c25_0_4_prev6
   + dist_diff_abs_pair_c25_0_4_prev6 >= 10
 dist_diff_pair_def_c25_0_4_prev7: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev7 = -78
 dist_diff_abs_pair_c25_0_4_prev7_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev7 + dist_diff_abs_pair_c25_0_4_prev7
   - 240 dist_diff_abs_pair_c25_0_4_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev7_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev7 + dist_diff_abs_pair_c25_0_4_prev7
   + 240 dist_diff_abs_pair_c25_0_4_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev7 + dist_diff_abs_pair_c25_0_4_prev7
   + 240 dist_diff_abs_pair_c25_0_4_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev7 + dist_diff_abs_pair_c25_0_4_prev7
   - 240 dist_diff_abs_pair_c25_0_4_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev7_force_positive:
   dist_diff_pair_c25_0_4_prev7
   - 240 dist_diff_abs_pair_c25_0_4_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev7_force_negative:
   dist_diff_pair_c25_0_4_prev7
   - 240 dist_diff_abs_pair_c25_0_4_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev7: 240 same_dist_pair_c25_0_4_prev7
   + dist_diff_abs_pair_c25_0_4_prev7 <= 249.8
 same_dist_pair_lower_c25_0_4_prev7: 240 same_dist_pair_c25_0_4_prev7
   + dist_diff_abs_pair_c25_0_4_prev7 >= 10
 dist_diff_pair_def_c25_0_4_prev8: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev8 = -32
 dist_diff_abs_pair_c25_0_4_prev8_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev8 + dist_diff_abs_pair_c25_0_4_prev8
   - 240 dist_diff_abs_pair_c25_0_4_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev8_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev8 + dist_diff_abs_pair_c25_0_4_prev8
   + 240 dist_diff_abs_pair_c25_0_4_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev8 + dist_diff_abs_pair_c25_0_4_prev8
   + 240 dist_diff_abs_pair_c25_0_4_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev8 + dist_diff_abs_pair_c25_0_4_prev8
   - 240 dist_diff_abs_pair_c25_0_4_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev8_force_positive:
   dist_diff_pair_c25_0_4_prev8
   - 240 dist_diff_abs_pair_c25_0_4_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev8_force_negative:
   dist_diff_pair_c25_0_4_prev8
   - 240 dist_diff_abs_pair_c25_0_4_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev8: 240 same_dist_pair_c25_0_4_prev8
   + dist_diff_abs_pair_c25_0_4_prev8 <= 249.8
 same_dist_pair_lower_c25_0_4_prev8: 240 same_dist_pair_c25_0_4_prev8
   + dist_diff_abs_pair_c25_0_4_prev8 >= 10
 dist_diff_pair_def_c25_0_4_prev9: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev9 = -56
 dist_diff_abs_pair_c25_0_4_prev9_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev9 + dist_diff_abs_pair_c25_0_4_prev9
   - 240 dist_diff_abs_pair_c25_0_4_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev9_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev9 + dist_diff_abs_pair_c25_0_4_prev9
   + 240 dist_diff_abs_pair_c25_0_4_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev9 + dist_diff_abs_pair_c25_0_4_prev9
   + 240 dist_diff_abs_pair_c25_0_4_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev9 + dist_diff_abs_pair_c25_0_4_prev9
   - 240 dist_diff_abs_pair_c25_0_4_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev9_force_positive:
   dist_diff_pair_c25_0_4_prev9
   - 240 dist_diff_abs_pair_c25_0_4_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev9_force_negative:
   dist_diff_pair_c25_0_4_prev9
   - 240 dist_diff_abs_pair_c25_0_4_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev9: 240 same_dist_pair_c25_0_4_prev9
   + dist_diff_abs_pair_c25_0_4_prev9 <= 249.8
 same_dist_pair_lower_c25_0_4_prev9: 240 same_dist_pair_c25_0_4_prev9
   + dist_diff_abs_pair_c25_0_4_prev9 >= 10
 dist_diff_pair_def_c25_0_4_prev10: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev10 = -33
 dist_diff_abs_pair_c25_0_4_prev10_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev10 + dist_diff_abs_pair_c25_0_4_prev10
   - 240 dist_diff_abs_pair_c25_0_4_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev10_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev10 + dist_diff_abs_pair_c25_0_4_prev10
   + 240 dist_diff_abs_pair_c25_0_4_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev10 + dist_diff_abs_pair_c25_0_4_prev10
   + 240 dist_diff_abs_pair_c25_0_4_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev10 + dist_diff_abs_pair_c25_0_4_prev10
   - 240 dist_diff_abs_pair_c25_0_4_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev10_force_positive:
   dist_diff_pair_c25_0_4_prev10
   - 240 dist_diff_abs_pair_c25_0_4_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev10_force_negative:
   dist_diff_pair_c25_0_4_prev10
   - 240 dist_diff_abs_pair_c25_0_4_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev10: 240 same_dist_pair_c25_0_4_prev10
   + dist_diff_abs_pair_c25_0_4_prev10 <= 249.8
 same_dist_pair_lower_c25_0_4_prev10: 240 same_dist_pair_c25_0_4_prev10
   + dist_diff_abs_pair_c25_0_4_prev10 >= 10
 dist_diff_pair_def_c25_0_4_prev11: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev11 = -70
 dist_diff_abs_pair_c25_0_4_prev11_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev11 + dist_diff_abs_pair_c25_0_4_prev11
   - 240 dist_diff_abs_pair_c25_0_4_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev11_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev11 + dist_diff_abs_pair_c25_0_4_prev11
   + 240 dist_diff_abs_pair_c25_0_4_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev11 + dist_diff_abs_pair_c25_0_4_prev11
   + 240 dist_diff_abs_pair_c25_0_4_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev11 + dist_diff_abs_pair_c25_0_4_prev11
   - 240 dist_diff_abs_pair_c25_0_4_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev11_force_positive:
   dist_diff_pair_c25_0_4_prev11
   - 240 dist_diff_abs_pair_c25_0_4_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev11_force_negative:
   dist_diff_pair_c25_0_4_prev11
   - 240 dist_diff_abs_pair_c25_0_4_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev11: 240 same_dist_pair_c25_0_4_prev11
   + dist_diff_abs_pair_c25_0_4_prev11 <= 249.8
 same_dist_pair_lower_c25_0_4_prev11: 240 same_dist_pair_c25_0_4_prev11
   + dist_diff_abs_pair_c25_0_4_prev11 >= 10
 dist_diff_pair_def_c25_0_4_prev12: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev12 = -64
 dist_diff_abs_pair_c25_0_4_prev12_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev12 + dist_diff_abs_pair_c25_0_4_prev12
   - 240 dist_diff_abs_pair_c25_0_4_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev12_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev12 + dist_diff_abs_pair_c25_0_4_prev12
   + 240 dist_diff_abs_pair_c25_0_4_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev12 + dist_diff_abs_pair_c25_0_4_prev12
   + 240 dist_diff_abs_pair_c25_0_4_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev12 + dist_diff_abs_pair_c25_0_4_prev12
   - 240 dist_diff_abs_pair_c25_0_4_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev12_force_positive:
   dist_diff_pair_c25_0_4_prev12
   - 240 dist_diff_abs_pair_c25_0_4_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev12_force_negative:
   dist_diff_pair_c25_0_4_prev12
   - 240 dist_diff_abs_pair_c25_0_4_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev12: 240 same_dist_pair_c25_0_4_prev12
   + dist_diff_abs_pair_c25_0_4_prev12 <= 249.8
 same_dist_pair_lower_c25_0_4_prev12: 240 same_dist_pair_c25_0_4_prev12
   + dist_diff_abs_pair_c25_0_4_prev12 >= 10
 dist_diff_pair_def_c25_0_4_prev13: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev13 = -76
 dist_diff_abs_pair_c25_0_4_prev13_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev13 + dist_diff_abs_pair_c25_0_4_prev13
   - 240 dist_diff_abs_pair_c25_0_4_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev13_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev13 + dist_diff_abs_pair_c25_0_4_prev13
   + 240 dist_diff_abs_pair_c25_0_4_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev13 + dist_diff_abs_pair_c25_0_4_prev13
   + 240 dist_diff_abs_pair_c25_0_4_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev13 + dist_diff_abs_pair_c25_0_4_prev13
   - 240 dist_diff_abs_pair_c25_0_4_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev13_force_positive:
   dist_diff_pair_c25_0_4_prev13
   - 240 dist_diff_abs_pair_c25_0_4_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev13_force_negative:
   dist_diff_pair_c25_0_4_prev13
   - 240 dist_diff_abs_pair_c25_0_4_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev13: 240 same_dist_pair_c25_0_4_prev13
   + dist_diff_abs_pair_c25_0_4_prev13 <= 249.8
 same_dist_pair_lower_c25_0_4_prev13: 240 same_dist_pair_c25_0_4_prev13
   + dist_diff_abs_pair_c25_0_4_prev13 >= 10
 dist_diff_pair_def_c25_0_4_prev14: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev14 = -22
 dist_diff_abs_pair_c25_0_4_prev14_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev14 + dist_diff_abs_pair_c25_0_4_prev14
   - 240 dist_diff_abs_pair_c25_0_4_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev14_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev14 + dist_diff_abs_pair_c25_0_4_prev14
   + 240 dist_diff_abs_pair_c25_0_4_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev14 + dist_diff_abs_pair_c25_0_4_prev14
   + 240 dist_diff_abs_pair_c25_0_4_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev14 + dist_diff_abs_pair_c25_0_4_prev14
   - 240 dist_diff_abs_pair_c25_0_4_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev14_force_positive:
   dist_diff_pair_c25_0_4_prev14
   - 240 dist_diff_abs_pair_c25_0_4_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev14_force_negative:
   dist_diff_pair_c25_0_4_prev14
   - 240 dist_diff_abs_pair_c25_0_4_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev14: 240 same_dist_pair_c25_0_4_prev14
   + dist_diff_abs_pair_c25_0_4_prev14 <= 249.8
 same_dist_pair_lower_c25_0_4_prev14: 240 same_dist_pair_c25_0_4_prev14
   + dist_diff_abs_pair_c25_0_4_prev14 >= 10
 dist_diff_pair_def_c25_0_4_prev15: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev15 = -61
 dist_diff_abs_pair_c25_0_4_prev15_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev15 + dist_diff_abs_pair_c25_0_4_prev15
   - 240 dist_diff_abs_pair_c25_0_4_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev15_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev15 + dist_diff_abs_pair_c25_0_4_prev15
   + 240 dist_diff_abs_pair_c25_0_4_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev15 + dist_diff_abs_pair_c25_0_4_prev15
   + 240 dist_diff_abs_pair_c25_0_4_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev15 + dist_diff_abs_pair_c25_0_4_prev15
   - 240 dist_diff_abs_pair_c25_0_4_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev15_force_positive:
   dist_diff_pair_c25_0_4_prev15
   - 240 dist_diff_abs_pair_c25_0_4_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev15_force_negative:
   dist_diff_pair_c25_0_4_prev15
   - 240 dist_diff_abs_pair_c25_0_4_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev15: 240 same_dist_pair_c25_0_4_prev15
   + dist_diff_abs_pair_c25_0_4_prev15 <= 249.8
 same_dist_pair_lower_c25_0_4_prev15: 240 same_dist_pair_c25_0_4_prev15
   + dist_diff_abs_pair_c25_0_4_prev15 >= 10
 dist_diff_pair_def_c25_0_4_prev16: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev16 = -58
 dist_diff_abs_pair_c25_0_4_prev16_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev16 + dist_diff_abs_pair_c25_0_4_prev16
   - 240 dist_diff_abs_pair_c25_0_4_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev16_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev16 + dist_diff_abs_pair_c25_0_4_prev16
   + 240 dist_diff_abs_pair_c25_0_4_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev16 + dist_diff_abs_pair_c25_0_4_prev16
   + 240 dist_diff_abs_pair_c25_0_4_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev16 + dist_diff_abs_pair_c25_0_4_prev16
   - 240 dist_diff_abs_pair_c25_0_4_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev16_force_positive:
   dist_diff_pair_c25_0_4_prev16
   - 240 dist_diff_abs_pair_c25_0_4_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev16_force_negative:
   dist_diff_pair_c25_0_4_prev16
   - 240 dist_diff_abs_pair_c25_0_4_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev16: 240 same_dist_pair_c25_0_4_prev16
   + dist_diff_abs_pair_c25_0_4_prev16 <= 249.8
 same_dist_pair_lower_c25_0_4_prev16: 240 same_dist_pair_c25_0_4_prev16
   + dist_diff_abs_pair_c25_0_4_prev16 >= 10
 dist_diff_pair_def_c25_0_4_prev17: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev17 = -52
 dist_diff_abs_pair_c25_0_4_prev17_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev17 + dist_diff_abs_pair_c25_0_4_prev17
   - 240 dist_diff_abs_pair_c25_0_4_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev17_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev17 + dist_diff_abs_pair_c25_0_4_prev17
   + 240 dist_diff_abs_pair_c25_0_4_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev17 + dist_diff_abs_pair_c25_0_4_prev17
   + 240 dist_diff_abs_pair_c25_0_4_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev17 + dist_diff_abs_pair_c25_0_4_prev17
   - 240 dist_diff_abs_pair_c25_0_4_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev17_force_positive:
   dist_diff_pair_c25_0_4_prev17
   - 240 dist_diff_abs_pair_c25_0_4_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev17_force_negative:
   dist_diff_pair_c25_0_4_prev17
   - 240 dist_diff_abs_pair_c25_0_4_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev17: 240 same_dist_pair_c25_0_4_prev17
   + dist_diff_abs_pair_c25_0_4_prev17 <= 249.8
 same_dist_pair_lower_c25_0_4_prev17: 240 same_dist_pair_c25_0_4_prev17
   + dist_diff_abs_pair_c25_0_4_prev17 >= 10
 dist_diff_pair_def_c25_0_4_prev18: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev18 = -81
 dist_diff_abs_pair_c25_0_4_prev18_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev18 + dist_diff_abs_pair_c25_0_4_prev18
   - 240 dist_diff_abs_pair_c25_0_4_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev18_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev18 + dist_diff_abs_pair_c25_0_4_prev18
   + 240 dist_diff_abs_pair_c25_0_4_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev18 + dist_diff_abs_pair_c25_0_4_prev18
   + 240 dist_diff_abs_pair_c25_0_4_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev18 + dist_diff_abs_pair_c25_0_4_prev18
   - 240 dist_diff_abs_pair_c25_0_4_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev18_force_positive:
   dist_diff_pair_c25_0_4_prev18
   - 240 dist_diff_abs_pair_c25_0_4_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev18_force_negative:
   dist_diff_pair_c25_0_4_prev18
   - 240 dist_diff_abs_pair_c25_0_4_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev18: 240 same_dist_pair_c25_0_4_prev18
   + dist_diff_abs_pair_c25_0_4_prev18 <= 249.8
 same_dist_pair_lower_c25_0_4_prev18: 240 same_dist_pair_c25_0_4_prev18
   + dist_diff_abs_pair_c25_0_4_prev18 >= 10
 dist_diff_pair_def_c25_0_4_prev19: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev19 = -56
 dist_diff_abs_pair_c25_0_4_prev19_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev19 + dist_diff_abs_pair_c25_0_4_prev19
   - 240 dist_diff_abs_pair_c25_0_4_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev19_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev19 + dist_diff_abs_pair_c25_0_4_prev19
   + 240 dist_diff_abs_pair_c25_0_4_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev19 + dist_diff_abs_pair_c25_0_4_prev19
   + 240 dist_diff_abs_pair_c25_0_4_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev19 + dist_diff_abs_pair_c25_0_4_prev19
   - 240 dist_diff_abs_pair_c25_0_4_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev19_force_positive:
   dist_diff_pair_c25_0_4_prev19
   - 240 dist_diff_abs_pair_c25_0_4_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev19_force_negative:
   dist_diff_pair_c25_0_4_prev19
   - 240 dist_diff_abs_pair_c25_0_4_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev19: 240 same_dist_pair_c25_0_4_prev19
   + dist_diff_abs_pair_c25_0_4_prev19 <= 249.8
 same_dist_pair_lower_c25_0_4_prev19: 240 same_dist_pair_c25_0_4_prev19
   + dist_diff_abs_pair_c25_0_4_prev19 >= 10
 dist_diff_pair_def_c25_0_4_prev20: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev20 = -46
 dist_diff_abs_pair_c25_0_4_prev20_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev20 + dist_diff_abs_pair_c25_0_4_prev20
   - 240 dist_diff_abs_pair_c25_0_4_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev20_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev20 + dist_diff_abs_pair_c25_0_4_prev20
   + 240 dist_diff_abs_pair_c25_0_4_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev20 + dist_diff_abs_pair_c25_0_4_prev20
   + 240 dist_diff_abs_pair_c25_0_4_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev20 + dist_diff_abs_pair_c25_0_4_prev20
   - 240 dist_diff_abs_pair_c25_0_4_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev20_force_positive:
   dist_diff_pair_c25_0_4_prev20
   - 240 dist_diff_abs_pair_c25_0_4_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev20_force_negative:
   dist_diff_pair_c25_0_4_prev20
   - 240 dist_diff_abs_pair_c25_0_4_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev20: 240 same_dist_pair_c25_0_4_prev20
   + dist_diff_abs_pair_c25_0_4_prev20 <= 249.8
 same_dist_pair_lower_c25_0_4_prev20: 240 same_dist_pair_c25_0_4_prev20
   + dist_diff_abs_pair_c25_0_4_prev20 >= 10
 dist_diff_pair_def_c25_0_4_prev21: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev21 = -52
 dist_diff_abs_pair_c25_0_4_prev21_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev21 + dist_diff_abs_pair_c25_0_4_prev21
   - 240 dist_diff_abs_pair_c25_0_4_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev21_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev21 + dist_diff_abs_pair_c25_0_4_prev21
   + 240 dist_diff_abs_pair_c25_0_4_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev21 + dist_diff_abs_pair_c25_0_4_prev21
   + 240 dist_diff_abs_pair_c25_0_4_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev21 + dist_diff_abs_pair_c25_0_4_prev21
   - 240 dist_diff_abs_pair_c25_0_4_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev21_force_positive:
   dist_diff_pair_c25_0_4_prev21
   - 240 dist_diff_abs_pair_c25_0_4_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev21_force_negative:
   dist_diff_pair_c25_0_4_prev21
   - 240 dist_diff_abs_pair_c25_0_4_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev21: 240 same_dist_pair_c25_0_4_prev21
   + dist_diff_abs_pair_c25_0_4_prev21 <= 249.8
 same_dist_pair_lower_c25_0_4_prev21: 240 same_dist_pair_c25_0_4_prev21
   + dist_diff_abs_pair_c25_0_4_prev21 >= 10
 dist_diff_pair_def_c25_0_4_prev22: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev22 = -71
 dist_diff_abs_pair_c25_0_4_prev22_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev22 + dist_diff_abs_pair_c25_0_4_prev22
   - 240 dist_diff_abs_pair_c25_0_4_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev22_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev22 + dist_diff_abs_pair_c25_0_4_prev22
   + 240 dist_diff_abs_pair_c25_0_4_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev22 + dist_diff_abs_pair_c25_0_4_prev22
   + 240 dist_diff_abs_pair_c25_0_4_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev22 + dist_diff_abs_pair_c25_0_4_prev22
   - 240 dist_diff_abs_pair_c25_0_4_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev22_force_positive:
   dist_diff_pair_c25_0_4_prev22
   - 240 dist_diff_abs_pair_c25_0_4_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev22_force_negative:
   dist_diff_pair_c25_0_4_prev22
   - 240 dist_diff_abs_pair_c25_0_4_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev22: 240 same_dist_pair_c25_0_4_prev22
   + dist_diff_abs_pair_c25_0_4_prev22 <= 249.8
 same_dist_pair_lower_c25_0_4_prev22: 240 same_dist_pair_c25_0_4_prev22
   + dist_diff_abs_pair_c25_0_4_prev22 >= 10
 dist_diff_pair_def_c25_0_4_prev23: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev23 = -67
 dist_diff_abs_pair_c25_0_4_prev23_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev23 + dist_diff_abs_pair_c25_0_4_prev23
   - 240 dist_diff_abs_pair_c25_0_4_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev23_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev23 + dist_diff_abs_pair_c25_0_4_prev23
   + 240 dist_diff_abs_pair_c25_0_4_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev23 + dist_diff_abs_pair_c25_0_4_prev23
   + 240 dist_diff_abs_pair_c25_0_4_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev23 + dist_diff_abs_pair_c25_0_4_prev23
   - 240 dist_diff_abs_pair_c25_0_4_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev23_force_positive:
   dist_diff_pair_c25_0_4_prev23
   - 240 dist_diff_abs_pair_c25_0_4_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev23_force_negative:
   dist_diff_pair_c25_0_4_prev23
   - 240 dist_diff_abs_pair_c25_0_4_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev23: 240 same_dist_pair_c25_0_4_prev23
   + dist_diff_abs_pair_c25_0_4_prev23 <= 249.8
 same_dist_pair_lower_c25_0_4_prev23: 240 same_dist_pair_c25_0_4_prev23
   + dist_diff_abs_pair_c25_0_4_prev23 >= 10
 dist_diff_pair_def_c25_0_4_prev24: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev24 = -32
 dist_diff_abs_pair_c25_0_4_prev24_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev24 + dist_diff_abs_pair_c25_0_4_prev24
   - 240 dist_diff_abs_pair_c25_0_4_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev24_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev24 + dist_diff_abs_pair_c25_0_4_prev24
   + 240 dist_diff_abs_pair_c25_0_4_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev24 + dist_diff_abs_pair_c25_0_4_prev24
   + 240 dist_diff_abs_pair_c25_0_4_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev24 + dist_diff_abs_pair_c25_0_4_prev24
   - 240 dist_diff_abs_pair_c25_0_4_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev24_force_positive:
   dist_diff_pair_c25_0_4_prev24
   - 240 dist_diff_abs_pair_c25_0_4_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev24_force_negative:
   dist_diff_pair_c25_0_4_prev24
   - 240 dist_diff_abs_pair_c25_0_4_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev24: 240 same_dist_pair_c25_0_4_prev24
   + dist_diff_abs_pair_c25_0_4_prev24 <= 249.8
 same_dist_pair_lower_c25_0_4_prev24: 240 same_dist_pair_c25_0_4_prev24
   + dist_diff_abs_pair_c25_0_4_prev24 >= 10
 dist_diff_pair_def_c25_0_4_prev25: - dist_curr_pair_c25_0_4
   + dist_diff_pair_c25_0_4_prev25 = -50
 dist_diff_abs_pair_c25_0_4_prev25_abs_ge_orig:
   - dist_diff_pair_c25_0_4_prev25 + dist_diff_abs_pair_c25_0_4_prev25
   - 240 dist_diff_abs_pair_c25_0_4_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev25_abs_le_orig:
   - dist_diff_pair_c25_0_4_prev25 + dist_diff_abs_pair_c25_0_4_prev25
   + 240 dist_diff_abs_pair_c25_0_4_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_0_4_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_0_4_prev25 + dist_diff_abs_pair_c25_0_4_prev25
   + 240 dist_diff_abs_pair_c25_0_4_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_0_4_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_0_4_prev25 + dist_diff_abs_pair_c25_0_4_prev25
   - 240 dist_diff_abs_pair_c25_0_4_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_0_4_prev25_force_positive:
   dist_diff_pair_c25_0_4_prev25
   - 240 dist_diff_abs_pair_c25_0_4_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_0_4_prev25_force_negative:
   dist_diff_pair_c25_0_4_prev25
   - 240 dist_diff_abs_pair_c25_0_4_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_0_4_prev25: 240 same_dist_pair_c25_0_4_prev25
   + dist_diff_abs_pair_c25_0_4_prev25 <= 249.8
 same_dist_pair_lower_c25_0_4_prev25: 240 same_dist_pair_c25_0_4_prev25
   + dist_diff_abs_pair_c25_0_4_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_0_4_prev0: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev0: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev1: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev1: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev2: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev2: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev3: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev3: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev4: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev4: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev5: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev5: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev6: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev6: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev7: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev7: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev8: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev8: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev9: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev9: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev10: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev10: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev11: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev11: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev12: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev12: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev13: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev13: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev14: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev14: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev15: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev15: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev16: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev16: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev17: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev17: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev18: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev18: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev19: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev19: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev20: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev20: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev21: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev21: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev22: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev22: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev23: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev23: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev24: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev24: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_0_4_prev25: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_0_4_prev25: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_0_4: diff_dist_pair_c25_0_4
   + same_dist_pair_c25_0_4_prev0 + same_dist_pair_c25_0_4_prev1
   + same_dist_pair_c25_0_4_prev2 + same_dist_pair_c25_0_4_prev3
   + same_dist_pair_c25_0_4_prev4 + same_dist_pair_c25_0_4_prev5
   + same_dist_pair_c25_0_4_prev6 + same_dist_pair_c25_0_4_prev7
   + same_dist_pair_c25_0_4_prev8 + same_dist_pair_c25_0_4_prev9
   + same_dist_pair_c25_0_4_prev10 + same_dist_pair_c25_0_4_prev11
   + same_dist_pair_c25_0_4_prev12 + same_dist_pair_c25_0_4_prev13
   + same_dist_pair_c25_0_4_prev14 + same_dist_pair_c25_0_4_prev15
   + same_dist_pair_c25_0_4_prev16 + same_dist_pair_c25_0_4_prev17
   + same_dist_pair_c25_0_4_prev18 + same_dist_pair_c25_0_4_prev19
   + same_dist_pair_c25_0_4_prev20 + same_dist_pair_c25_0_4_prev21
   + same_dist_pair_c25_0_4_prev22 + same_dist_pair_c25_0_4_prev23
   + same_dist_pair_c25_0_4_prev24 + same_dist_pair_c25_0_4_prev25 >= 1
 dx_grid_diff_def_c25_1_2: - cx_grid_center_c25_1 + cx_grid_center_c25_2
   + dx_grid_diff_c25_1_2 = 0
 dx_grid_abs_pair_c25_1_2_abs_ge_orig: dx_grid_abs_pair_c25_1_2
   - dx_grid_diff_c25_1_2 - 130 dx_grid_abs_pair_c25_1_2_is_positive
   >= -130
 dx_grid_abs_pair_c25_1_2_abs_le_orig: dx_grid_abs_pair_c25_1_2
   - dx_grid_diff_c25_1_2 + 130 dx_grid_abs_pair_c25_1_2_is_positive
   <= 130
 dx_grid_abs_pair_c25_1_2_abs_ge_neg_orig: dx_grid_abs_pair_c25_1_2
   + dx_grid_diff_c25_1_2 + 130 dx_grid_abs_pair_c25_1_2_is_positive >= 0
 dx_grid_abs_pair_c25_1_2_abs_le_neg_orig: dx_grid_abs_pair_c25_1_2
   + dx_grid_diff_c25_1_2 - 130 dx_grid_abs_pair_c25_1_2_is_positive <= 0
 dx_grid_abs_pair_c25_1_2_force_positive: dx_grid_diff_c25_1_2
   - 130 dx_grid_abs_pair_c25_1_2_is_positive >= -130
 dx_grid_abs_pair_c25_1_2_force_negative: dx_grid_diff_c25_1_2
   - 130 dx_grid_abs_pair_c25_1_2_is_positive <= -0.001
 dy_grid_diff_def_c25_1_2: - cy_grid_center_c25_1 + cy_grid_center_c25_2
   + dy_grid_diff_c25_1_2 = 0
 dy_grid_abs_pair_c25_1_2_abs_ge_orig: dy_grid_abs_pair_c25_1_2
   - dy_grid_diff_c25_1_2 - 130 dy_grid_abs_pair_c25_1_2_is_positive
   >= -130
 dy_grid_abs_pair_c25_1_2_abs_le_orig: dy_grid_abs_pair_c25_1_2
   - dy_grid_diff_c25_1_2 + 130 dy_grid_abs_pair_c25_1_2_is_positive
   <= 130
 dy_grid_abs_pair_c25_1_2_abs_ge_neg_orig: dy_grid_abs_pair_c25_1_2
   + dy_grid_diff_c25_1_2 + 130 dy_grid_abs_pair_c25_1_2_is_positive >= 0
 dy_grid_abs_pair_c25_1_2_abs_le_neg_orig: dy_grid_abs_pair_c25_1_2
   + dy_grid_diff_c25_1_2 - 130 dy_grid_abs_pair_c25_1_2_is_positive <= 0
 dy_grid_abs_pair_c25_1_2_force_positive: dy_grid_diff_c25_1_2
   - 130 dy_grid_abs_pair_c25_1_2_is_positive >= -130
 dy_grid_abs_pair_c25_1_2_force_negative: dy_grid_diff_c25_1_2
   - 130 dy_grid_abs_pair_c25_1_2_is_positive <= -0.001
 dist_curr_pair_def_c25_1_2: - dx_grid_abs_pair_c25_1_2
   - dy_grid_abs_pair_c25_1_2 + dist_curr_pair_c25_1_2 = 0
 dist_diff_pair_def_c25_1_2_prev0: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev0 = -30
 dist_diff_abs_pair_c25_1_2_prev0_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev0 + dist_diff_abs_pair_c25_1_2_prev0
   - 240 dist_diff_abs_pair_c25_1_2_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev0_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev0 + dist_diff_abs_pair_c25_1_2_prev0
   + 240 dist_diff_abs_pair_c25_1_2_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev0 + dist_diff_abs_pair_c25_1_2_prev0
   + 240 dist_diff_abs_pair_c25_1_2_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev0 + dist_diff_abs_pair_c25_1_2_prev0
   - 240 dist_diff_abs_pair_c25_1_2_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev0_force_positive:
   dist_diff_pair_c25_1_2_prev0
   - 240 dist_diff_abs_pair_c25_1_2_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev0_force_negative:
   dist_diff_pair_c25_1_2_prev0
   - 240 dist_diff_abs_pair_c25_1_2_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev0: 240 same_dist_pair_c25_1_2_prev0
   + dist_diff_abs_pair_c25_1_2_prev0 <= 249.8
 same_dist_pair_lower_c25_1_2_prev0: 240 same_dist_pair_c25_1_2_prev0
   + dist_diff_abs_pair_c25_1_2_prev0 >= 10
 dist_diff_pair_def_c25_1_2_prev1: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev1 = -30
 dist_diff_abs_pair_c25_1_2_prev1_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev1 + dist_diff_abs_pair_c25_1_2_prev1
   - 240 dist_diff_abs_pair_c25_1_2_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev1_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev1 + dist_diff_abs_pair_c25_1_2_prev1
   + 240 dist_diff_abs_pair_c25_1_2_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev1 + dist_diff_abs_pair_c25_1_2_prev1
   + 240 dist_diff_abs_pair_c25_1_2_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev1 + dist_diff_abs_pair_c25_1_2_prev1
   - 240 dist_diff_abs_pair_c25_1_2_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev1_force_positive:
   dist_diff_pair_c25_1_2_prev1
   - 240 dist_diff_abs_pair_c25_1_2_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev1_force_negative:
   dist_diff_pair_c25_1_2_prev1
   - 240 dist_diff_abs_pair_c25_1_2_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev1: 240 same_dist_pair_c25_1_2_prev1
   + dist_diff_abs_pair_c25_1_2_prev1 <= 249.8
 same_dist_pair_lower_c25_1_2_prev1: 240 same_dist_pair_c25_1_2_prev1
   + dist_diff_abs_pair_c25_1_2_prev1 >= 10
 dist_diff_pair_def_c25_1_2_prev2: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev2 = -34
 dist_diff_abs_pair_c25_1_2_prev2_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev2 + dist_diff_abs_pair_c25_1_2_prev2
   - 240 dist_diff_abs_pair_c25_1_2_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev2_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev2 + dist_diff_abs_pair_c25_1_2_prev2
   + 240 dist_diff_abs_pair_c25_1_2_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev2 + dist_diff_abs_pair_c25_1_2_prev2
   + 240 dist_diff_abs_pair_c25_1_2_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev2 + dist_diff_abs_pair_c25_1_2_prev2
   - 240 dist_diff_abs_pair_c25_1_2_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev2_force_positive:
   dist_diff_pair_c25_1_2_prev2
   - 240 dist_diff_abs_pair_c25_1_2_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev2_force_negative:
   dist_diff_pair_c25_1_2_prev2
   - 240 dist_diff_abs_pair_c25_1_2_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev2: 240 same_dist_pair_c25_1_2_prev2
   + dist_diff_abs_pair_c25_1_2_prev2 <= 249.8
 same_dist_pair_lower_c25_1_2_prev2: 240 same_dist_pair_c25_1_2_prev2
   + dist_diff_abs_pair_c25_1_2_prev2 >= 10
 dist_diff_pair_def_c25_1_2_prev3: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev3 = -34
 dist_diff_abs_pair_c25_1_2_prev3_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev3 + dist_diff_abs_pair_c25_1_2_prev3
   - 240 dist_diff_abs_pair_c25_1_2_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev3_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev3 + dist_diff_abs_pair_c25_1_2_prev3
   + 240 dist_diff_abs_pair_c25_1_2_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev3 + dist_diff_abs_pair_c25_1_2_prev3
   + 240 dist_diff_abs_pair_c25_1_2_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev3 + dist_diff_abs_pair_c25_1_2_prev3
   - 240 dist_diff_abs_pair_c25_1_2_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev3_force_positive:
   dist_diff_pair_c25_1_2_prev3
   - 240 dist_diff_abs_pair_c25_1_2_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev3_force_negative:
   dist_diff_pair_c25_1_2_prev3
   - 240 dist_diff_abs_pair_c25_1_2_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev3: 240 same_dist_pair_c25_1_2_prev3
   + dist_diff_abs_pair_c25_1_2_prev3 <= 249.8
 same_dist_pair_lower_c25_1_2_prev3: 240 same_dist_pair_c25_1_2_prev3
   + dist_diff_abs_pair_c25_1_2_prev3 >= 10
 dist_diff_pair_def_c25_1_2_prev4: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev4 = -36
 dist_diff_abs_pair_c25_1_2_prev4_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev4 + dist_diff_abs_pair_c25_1_2_prev4
   - 240 dist_diff_abs_pair_c25_1_2_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev4_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev4 + dist_diff_abs_pair_c25_1_2_prev4
   + 240 dist_diff_abs_pair_c25_1_2_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev4 + dist_diff_abs_pair_c25_1_2_prev4
   + 240 dist_diff_abs_pair_c25_1_2_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev4 + dist_diff_abs_pair_c25_1_2_prev4
   - 240 dist_diff_abs_pair_c25_1_2_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev4_force_positive:
   dist_diff_pair_c25_1_2_prev4
   - 240 dist_diff_abs_pair_c25_1_2_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev4_force_negative:
   dist_diff_pair_c25_1_2_prev4
   - 240 dist_diff_abs_pair_c25_1_2_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev4: 240 same_dist_pair_c25_1_2_prev4
   + dist_diff_abs_pair_c25_1_2_prev4 <= 249.8
 same_dist_pair_lower_c25_1_2_prev4: 240 same_dist_pair_c25_1_2_prev4
   + dist_diff_abs_pair_c25_1_2_prev4 >= 10
 dist_diff_pair_def_c25_1_2_prev5: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev5 = -30
 dist_diff_abs_pair_c25_1_2_prev5_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev5 + dist_diff_abs_pair_c25_1_2_prev5
   - 240 dist_diff_abs_pair_c25_1_2_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev5_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev5 + dist_diff_abs_pair_c25_1_2_prev5
   + 240 dist_diff_abs_pair_c25_1_2_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev5 + dist_diff_abs_pair_c25_1_2_prev5
   + 240 dist_diff_abs_pair_c25_1_2_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev5 + dist_diff_abs_pair_c25_1_2_prev5
   - 240 dist_diff_abs_pair_c25_1_2_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev5_force_positive:
   dist_diff_pair_c25_1_2_prev5
   - 240 dist_diff_abs_pair_c25_1_2_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev5_force_negative:
   dist_diff_pair_c25_1_2_prev5
   - 240 dist_diff_abs_pair_c25_1_2_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev5: 240 same_dist_pair_c25_1_2_prev5
   + dist_diff_abs_pair_c25_1_2_prev5 <= 249.8
 same_dist_pair_lower_c25_1_2_prev5: 240 same_dist_pair_c25_1_2_prev5
   + dist_diff_abs_pair_c25_1_2_prev5 >= 10
 dist_diff_pair_def_c25_1_2_prev6: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev6 = -30
 dist_diff_abs_pair_c25_1_2_prev6_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev6 + dist_diff_abs_pair_c25_1_2_prev6
   - 240 dist_diff_abs_pair_c25_1_2_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev6_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev6 + dist_diff_abs_pair_c25_1_2_prev6
   + 240 dist_diff_abs_pair_c25_1_2_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev6 + dist_diff_abs_pair_c25_1_2_prev6
   + 240 dist_diff_abs_pair_c25_1_2_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev6 + dist_diff_abs_pair_c25_1_2_prev6
   - 240 dist_diff_abs_pair_c25_1_2_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev6_force_positive:
   dist_diff_pair_c25_1_2_prev6
   - 240 dist_diff_abs_pair_c25_1_2_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev6_force_negative:
   dist_diff_pair_c25_1_2_prev6
   - 240 dist_diff_abs_pair_c25_1_2_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev6: 240 same_dist_pair_c25_1_2_prev6
   + dist_diff_abs_pair_c25_1_2_prev6 <= 249.8
 same_dist_pair_lower_c25_1_2_prev6: 240 same_dist_pair_c25_1_2_prev6
   + dist_diff_abs_pair_c25_1_2_prev6 >= 10
 dist_diff_pair_def_c25_1_2_prev7: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev7 = -30
 dist_diff_abs_pair_c25_1_2_prev7_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev7 + dist_diff_abs_pair_c25_1_2_prev7
   - 240 dist_diff_abs_pair_c25_1_2_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev7_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev7 + dist_diff_abs_pair_c25_1_2_prev7
   + 240 dist_diff_abs_pair_c25_1_2_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev7 + dist_diff_abs_pair_c25_1_2_prev7
   + 240 dist_diff_abs_pair_c25_1_2_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev7 + dist_diff_abs_pair_c25_1_2_prev7
   - 240 dist_diff_abs_pair_c25_1_2_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev7_force_positive:
   dist_diff_pair_c25_1_2_prev7
   - 240 dist_diff_abs_pair_c25_1_2_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev7_force_negative:
   dist_diff_pair_c25_1_2_prev7
   - 240 dist_diff_abs_pair_c25_1_2_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev7: 240 same_dist_pair_c25_1_2_prev7
   + dist_diff_abs_pair_c25_1_2_prev7 <= 249.8
 same_dist_pair_lower_c25_1_2_prev7: 240 same_dist_pair_c25_1_2_prev7
   + dist_diff_abs_pair_c25_1_2_prev7 >= 10
 dist_diff_pair_def_c25_1_2_prev8: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev8 = -30
 dist_diff_abs_pair_c25_1_2_prev8_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev8 + dist_diff_abs_pair_c25_1_2_prev8
   - 240 dist_diff_abs_pair_c25_1_2_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev8_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev8 + dist_diff_abs_pair_c25_1_2_prev8
   + 240 dist_diff_abs_pair_c25_1_2_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev8 + dist_diff_abs_pair_c25_1_2_prev8
   + 240 dist_diff_abs_pair_c25_1_2_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev8 + dist_diff_abs_pair_c25_1_2_prev8
   - 240 dist_diff_abs_pair_c25_1_2_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev8_force_positive:
   dist_diff_pair_c25_1_2_prev8
   - 240 dist_diff_abs_pair_c25_1_2_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev8_force_negative:
   dist_diff_pair_c25_1_2_prev8
   - 240 dist_diff_abs_pair_c25_1_2_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev8: 240 same_dist_pair_c25_1_2_prev8
   + dist_diff_abs_pair_c25_1_2_prev8 <= 249.8
 same_dist_pair_lower_c25_1_2_prev8: 240 same_dist_pair_c25_1_2_prev8
   + dist_diff_abs_pair_c25_1_2_prev8 >= 10
 dist_diff_pair_def_c25_1_2_prev9: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev9 = -50
 dist_diff_abs_pair_c25_1_2_prev9_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev9 + dist_diff_abs_pair_c25_1_2_prev9
   - 240 dist_diff_abs_pair_c25_1_2_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev9_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev9 + dist_diff_abs_pair_c25_1_2_prev9
   + 240 dist_diff_abs_pair_c25_1_2_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev9 + dist_diff_abs_pair_c25_1_2_prev9
   + 240 dist_diff_abs_pair_c25_1_2_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev9 + dist_diff_abs_pair_c25_1_2_prev9
   - 240 dist_diff_abs_pair_c25_1_2_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev9_force_positive:
   dist_diff_pair_c25_1_2_prev9
   - 240 dist_diff_abs_pair_c25_1_2_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev9_force_negative:
   dist_diff_pair_c25_1_2_prev9
   - 240 dist_diff_abs_pair_c25_1_2_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev9: 240 same_dist_pair_c25_1_2_prev9
   + dist_diff_abs_pair_c25_1_2_prev9 <= 249.8
 same_dist_pair_lower_c25_1_2_prev9: 240 same_dist_pair_c25_1_2_prev9
   + dist_diff_abs_pair_c25_1_2_prev9 >= 10
 dist_diff_pair_def_c25_1_2_prev10: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev10 = -31
 dist_diff_abs_pair_c25_1_2_prev10_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev10 + dist_diff_abs_pair_c25_1_2_prev10
   - 240 dist_diff_abs_pair_c25_1_2_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev10_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev10 + dist_diff_abs_pair_c25_1_2_prev10
   + 240 dist_diff_abs_pair_c25_1_2_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev10 + dist_diff_abs_pair_c25_1_2_prev10
   + 240 dist_diff_abs_pair_c25_1_2_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev10 + dist_diff_abs_pair_c25_1_2_prev10
   - 240 dist_diff_abs_pair_c25_1_2_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev10_force_positive:
   dist_diff_pair_c25_1_2_prev10
   - 240 dist_diff_abs_pair_c25_1_2_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev10_force_negative:
   dist_diff_pair_c25_1_2_prev10
   - 240 dist_diff_abs_pair_c25_1_2_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev10: 240 same_dist_pair_c25_1_2_prev10
   + dist_diff_abs_pair_c25_1_2_prev10 <= 249.8
 same_dist_pair_lower_c25_1_2_prev10: 240 same_dist_pair_c25_1_2_prev10
   + dist_diff_abs_pair_c25_1_2_prev10 >= 10
 dist_diff_pair_def_c25_1_2_prev11: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev11 = -40
 dist_diff_abs_pair_c25_1_2_prev11_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev11 + dist_diff_abs_pair_c25_1_2_prev11
   - 240 dist_diff_abs_pair_c25_1_2_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev11_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev11 + dist_diff_abs_pair_c25_1_2_prev11
   + 240 dist_diff_abs_pair_c25_1_2_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev11 + dist_diff_abs_pair_c25_1_2_prev11
   + 240 dist_diff_abs_pair_c25_1_2_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev11 + dist_diff_abs_pair_c25_1_2_prev11
   - 240 dist_diff_abs_pair_c25_1_2_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev11_force_positive:
   dist_diff_pair_c25_1_2_prev11
   - 240 dist_diff_abs_pair_c25_1_2_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev11_force_negative:
   dist_diff_pair_c25_1_2_prev11
   - 240 dist_diff_abs_pair_c25_1_2_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev11: 240 same_dist_pair_c25_1_2_prev11
   + dist_diff_abs_pair_c25_1_2_prev11 <= 249.8
 same_dist_pair_lower_c25_1_2_prev11: 240 same_dist_pair_c25_1_2_prev11
   + dist_diff_abs_pair_c25_1_2_prev11 >= 10
 dist_diff_pair_def_c25_1_2_prev12: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev12 = -30
 dist_diff_abs_pair_c25_1_2_prev12_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev12 + dist_diff_abs_pair_c25_1_2_prev12
   - 240 dist_diff_abs_pair_c25_1_2_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev12_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev12 + dist_diff_abs_pair_c25_1_2_prev12
   + 240 dist_diff_abs_pair_c25_1_2_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev12 + dist_diff_abs_pair_c25_1_2_prev12
   + 240 dist_diff_abs_pair_c25_1_2_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev12 + dist_diff_abs_pair_c25_1_2_prev12
   - 240 dist_diff_abs_pair_c25_1_2_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev12_force_positive:
   dist_diff_pair_c25_1_2_prev12
   - 240 dist_diff_abs_pair_c25_1_2_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev12_force_negative:
   dist_diff_pair_c25_1_2_prev12
   - 240 dist_diff_abs_pair_c25_1_2_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev12: 240 same_dist_pair_c25_1_2_prev12
   + dist_diff_abs_pair_c25_1_2_prev12 <= 249.8
 same_dist_pair_lower_c25_1_2_prev12: 240 same_dist_pair_c25_1_2_prev12
   + dist_diff_abs_pair_c25_1_2_prev12 >= 10
 dist_diff_pair_def_c25_1_2_prev13: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev13 = -30
 dist_diff_abs_pair_c25_1_2_prev13_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev13 + dist_diff_abs_pair_c25_1_2_prev13
   - 240 dist_diff_abs_pair_c25_1_2_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev13_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev13 + dist_diff_abs_pair_c25_1_2_prev13
   + 240 dist_diff_abs_pair_c25_1_2_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev13 + dist_diff_abs_pair_c25_1_2_prev13
   + 240 dist_diff_abs_pair_c25_1_2_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev13 + dist_diff_abs_pair_c25_1_2_prev13
   - 240 dist_diff_abs_pair_c25_1_2_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev13_force_positive:
   dist_diff_pair_c25_1_2_prev13
   - 240 dist_diff_abs_pair_c25_1_2_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev13_force_negative:
   dist_diff_pair_c25_1_2_prev13
   - 240 dist_diff_abs_pair_c25_1_2_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev13: 240 same_dist_pair_c25_1_2_prev13
   + dist_diff_abs_pair_c25_1_2_prev13 <= 249.8
 same_dist_pair_lower_c25_1_2_prev13: 240 same_dist_pair_c25_1_2_prev13
   + dist_diff_abs_pair_c25_1_2_prev13 >= 10
 dist_diff_pair_def_c25_1_2_prev14: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev14 = -40
 dist_diff_abs_pair_c25_1_2_prev14_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev14 + dist_diff_abs_pair_c25_1_2_prev14
   - 240 dist_diff_abs_pair_c25_1_2_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev14_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev14 + dist_diff_abs_pair_c25_1_2_prev14
   + 240 dist_diff_abs_pair_c25_1_2_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev14 + dist_diff_abs_pair_c25_1_2_prev14
   + 240 dist_diff_abs_pair_c25_1_2_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev14 + dist_diff_abs_pair_c25_1_2_prev14
   - 240 dist_diff_abs_pair_c25_1_2_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev14_force_positive:
   dist_diff_pair_c25_1_2_prev14
   - 240 dist_diff_abs_pair_c25_1_2_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev14_force_negative:
   dist_diff_pair_c25_1_2_prev14
   - 240 dist_diff_abs_pair_c25_1_2_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev14: 240 same_dist_pair_c25_1_2_prev14
   + dist_diff_abs_pair_c25_1_2_prev14 <= 249.8
 same_dist_pair_lower_c25_1_2_prev14: 240 same_dist_pair_c25_1_2_prev14
   + dist_diff_abs_pair_c25_1_2_prev14 >= 10
 dist_diff_pair_def_c25_1_2_prev15: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev15 = -30
 dist_diff_abs_pair_c25_1_2_prev15_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev15 + dist_diff_abs_pair_c25_1_2_prev15
   - 240 dist_diff_abs_pair_c25_1_2_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev15_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev15 + dist_diff_abs_pair_c25_1_2_prev15
   + 240 dist_diff_abs_pair_c25_1_2_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev15 + dist_diff_abs_pair_c25_1_2_prev15
   + 240 dist_diff_abs_pair_c25_1_2_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev15 + dist_diff_abs_pair_c25_1_2_prev15
   - 240 dist_diff_abs_pair_c25_1_2_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev15_force_positive:
   dist_diff_pair_c25_1_2_prev15
   - 240 dist_diff_abs_pair_c25_1_2_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev15_force_negative:
   dist_diff_pair_c25_1_2_prev15
   - 240 dist_diff_abs_pair_c25_1_2_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev15: 240 same_dist_pair_c25_1_2_prev15
   + dist_diff_abs_pair_c25_1_2_prev15 <= 249.8
 same_dist_pair_lower_c25_1_2_prev15: 240 same_dist_pair_c25_1_2_prev15
   + dist_diff_abs_pair_c25_1_2_prev15 >= 10
 dist_diff_pair_def_c25_1_2_prev16: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev16 = -60
 dist_diff_abs_pair_c25_1_2_prev16_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev16 + dist_diff_abs_pair_c25_1_2_prev16
   - 240 dist_diff_abs_pair_c25_1_2_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev16_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev16 + dist_diff_abs_pair_c25_1_2_prev16
   + 240 dist_diff_abs_pair_c25_1_2_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev16 + dist_diff_abs_pair_c25_1_2_prev16
   + 240 dist_diff_abs_pair_c25_1_2_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev16 + dist_diff_abs_pair_c25_1_2_prev16
   - 240 dist_diff_abs_pair_c25_1_2_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev16_force_positive:
   dist_diff_pair_c25_1_2_prev16
   - 240 dist_diff_abs_pair_c25_1_2_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev16_force_negative:
   dist_diff_pair_c25_1_2_prev16
   - 240 dist_diff_abs_pair_c25_1_2_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev16: 240 same_dist_pair_c25_1_2_prev16
   + dist_diff_abs_pair_c25_1_2_prev16 <= 249.8
 same_dist_pair_lower_c25_1_2_prev16: 240 same_dist_pair_c25_1_2_prev16
   + dist_diff_abs_pair_c25_1_2_prev16 >= 10
 dist_diff_pair_def_c25_1_2_prev17: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev17 = -54
 dist_diff_abs_pair_c25_1_2_prev17_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev17 + dist_diff_abs_pair_c25_1_2_prev17
   - 240 dist_diff_abs_pair_c25_1_2_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev17_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev17 + dist_diff_abs_pair_c25_1_2_prev17
   + 240 dist_diff_abs_pair_c25_1_2_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev17 + dist_diff_abs_pair_c25_1_2_prev17
   + 240 dist_diff_abs_pair_c25_1_2_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev17 + dist_diff_abs_pair_c25_1_2_prev17
   - 240 dist_diff_abs_pair_c25_1_2_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev17_force_positive:
   dist_diff_pair_c25_1_2_prev17
   - 240 dist_diff_abs_pair_c25_1_2_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev17_force_negative:
   dist_diff_pair_c25_1_2_prev17
   - 240 dist_diff_abs_pair_c25_1_2_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev17: 240 same_dist_pair_c25_1_2_prev17
   + dist_diff_abs_pair_c25_1_2_prev17 <= 249.8
 same_dist_pair_lower_c25_1_2_prev17: 240 same_dist_pair_c25_1_2_prev17
   + dist_diff_abs_pair_c25_1_2_prev17 >= 10
 dist_diff_pair_def_c25_1_2_prev18: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev18 = -30
 dist_diff_abs_pair_c25_1_2_prev18_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev18 + dist_diff_abs_pair_c25_1_2_prev18
   - 240 dist_diff_abs_pair_c25_1_2_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev18_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev18 + dist_diff_abs_pair_c25_1_2_prev18
   + 240 dist_diff_abs_pair_c25_1_2_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev18 + dist_diff_abs_pair_c25_1_2_prev18
   + 240 dist_diff_abs_pair_c25_1_2_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev18 + dist_diff_abs_pair_c25_1_2_prev18
   - 240 dist_diff_abs_pair_c25_1_2_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev18_force_positive:
   dist_diff_pair_c25_1_2_prev18
   - 240 dist_diff_abs_pair_c25_1_2_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev18_force_negative:
   dist_diff_pair_c25_1_2_prev18
   - 240 dist_diff_abs_pair_c25_1_2_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev18: 240 same_dist_pair_c25_1_2_prev18
   + dist_diff_abs_pair_c25_1_2_prev18 <= 249.8
 same_dist_pair_lower_c25_1_2_prev18: 240 same_dist_pair_c25_1_2_prev18
   + dist_diff_abs_pair_c25_1_2_prev18 >= 10
 dist_diff_pair_def_c25_1_2_prev19: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev19 = -54
 dist_diff_abs_pair_c25_1_2_prev19_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev19 + dist_diff_abs_pair_c25_1_2_prev19
   - 240 dist_diff_abs_pair_c25_1_2_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev19_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev19 + dist_diff_abs_pair_c25_1_2_prev19
   + 240 dist_diff_abs_pair_c25_1_2_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev19 + dist_diff_abs_pair_c25_1_2_prev19
   + 240 dist_diff_abs_pair_c25_1_2_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev19 + dist_diff_abs_pair_c25_1_2_prev19
   - 240 dist_diff_abs_pair_c25_1_2_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev19_force_positive:
   dist_diff_pair_c25_1_2_prev19
   - 240 dist_diff_abs_pair_c25_1_2_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev19_force_negative:
   dist_diff_pair_c25_1_2_prev19
   - 240 dist_diff_abs_pair_c25_1_2_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev19: 240 same_dist_pair_c25_1_2_prev19
   + dist_diff_abs_pair_c25_1_2_prev19 <= 249.8
 same_dist_pair_lower_c25_1_2_prev19: 240 same_dist_pair_c25_1_2_prev19
   + dist_diff_abs_pair_c25_1_2_prev19 >= 10
 dist_diff_pair_def_c25_1_2_prev20: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev20 = -48
 dist_diff_abs_pair_c25_1_2_prev20_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev20 + dist_diff_abs_pair_c25_1_2_prev20
   - 240 dist_diff_abs_pair_c25_1_2_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev20_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev20 + dist_diff_abs_pair_c25_1_2_prev20
   + 240 dist_diff_abs_pair_c25_1_2_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev20 + dist_diff_abs_pair_c25_1_2_prev20
   + 240 dist_diff_abs_pair_c25_1_2_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev20 + dist_diff_abs_pair_c25_1_2_prev20
   - 240 dist_diff_abs_pair_c25_1_2_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev20_force_positive:
   dist_diff_pair_c25_1_2_prev20
   - 240 dist_diff_abs_pair_c25_1_2_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev20_force_negative:
   dist_diff_pair_c25_1_2_prev20
   - 240 dist_diff_abs_pair_c25_1_2_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev20: 240 same_dist_pair_c25_1_2_prev20
   + dist_diff_abs_pair_c25_1_2_prev20 <= 249.8
 same_dist_pair_lower_c25_1_2_prev20: 240 same_dist_pair_c25_1_2_prev20
   + dist_diff_abs_pair_c25_1_2_prev20 >= 10
 dist_diff_pair_def_c25_1_2_prev21: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev21 = -54
 dist_diff_abs_pair_c25_1_2_prev21_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev21 + dist_diff_abs_pair_c25_1_2_prev21
   - 240 dist_diff_abs_pair_c25_1_2_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev21_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev21 + dist_diff_abs_pair_c25_1_2_prev21
   + 240 dist_diff_abs_pair_c25_1_2_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev21 + dist_diff_abs_pair_c25_1_2_prev21
   + 240 dist_diff_abs_pair_c25_1_2_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev21 + dist_diff_abs_pair_c25_1_2_prev21
   - 240 dist_diff_abs_pair_c25_1_2_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev21_force_positive:
   dist_diff_pair_c25_1_2_prev21
   - 240 dist_diff_abs_pair_c25_1_2_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev21_force_negative:
   dist_diff_pair_c25_1_2_prev21
   - 240 dist_diff_abs_pair_c25_1_2_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev21: 240 same_dist_pair_c25_1_2_prev21
   + dist_diff_abs_pair_c25_1_2_prev21 <= 249.8
 same_dist_pair_lower_c25_1_2_prev21: 240 same_dist_pair_c25_1_2_prev21
   + dist_diff_abs_pair_c25_1_2_prev21 >= 10
 dist_diff_pair_def_c25_1_2_prev22: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev22 = -59
 dist_diff_abs_pair_c25_1_2_prev22_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev22 + dist_diff_abs_pair_c25_1_2_prev22
   - 240 dist_diff_abs_pair_c25_1_2_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev22_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev22 + dist_diff_abs_pair_c25_1_2_prev22
   + 240 dist_diff_abs_pair_c25_1_2_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev22 + dist_diff_abs_pair_c25_1_2_prev22
   + 240 dist_diff_abs_pair_c25_1_2_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev22 + dist_diff_abs_pair_c25_1_2_prev22
   - 240 dist_diff_abs_pair_c25_1_2_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev22_force_positive:
   dist_diff_pair_c25_1_2_prev22
   - 240 dist_diff_abs_pair_c25_1_2_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev22_force_negative:
   dist_diff_pair_c25_1_2_prev22
   - 240 dist_diff_abs_pair_c25_1_2_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev22: 240 same_dist_pair_c25_1_2_prev22
   + dist_diff_abs_pair_c25_1_2_prev22 <= 249.8
 same_dist_pair_lower_c25_1_2_prev22: 240 same_dist_pair_c25_1_2_prev22
   + dist_diff_abs_pair_c25_1_2_prev22 >= 10
 dist_diff_pair_def_c25_1_2_prev23: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev23 = -70
 dist_diff_abs_pair_c25_1_2_prev23_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev23 + dist_diff_abs_pair_c25_1_2_prev23
   - 240 dist_diff_abs_pair_c25_1_2_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev23_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev23 + dist_diff_abs_pair_c25_1_2_prev23
   + 240 dist_diff_abs_pair_c25_1_2_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev23 + dist_diff_abs_pair_c25_1_2_prev23
   + 240 dist_diff_abs_pair_c25_1_2_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev23 + dist_diff_abs_pair_c25_1_2_prev23
   - 240 dist_diff_abs_pair_c25_1_2_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev23_force_positive:
   dist_diff_pair_c25_1_2_prev23
   - 240 dist_diff_abs_pair_c25_1_2_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev23_force_negative:
   dist_diff_pair_c25_1_2_prev23
   - 240 dist_diff_abs_pair_c25_1_2_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev23: 240 same_dist_pair_c25_1_2_prev23
   + dist_diff_abs_pair_c25_1_2_prev23 <= 249.8
 same_dist_pair_lower_c25_1_2_prev23: 240 same_dist_pair_c25_1_2_prev23
   + dist_diff_abs_pair_c25_1_2_prev23 >= 10
 dist_diff_pair_def_c25_1_2_prev24: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev24 = -30
 dist_diff_abs_pair_c25_1_2_prev24_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev24 + dist_diff_abs_pair_c25_1_2_prev24
   - 240 dist_diff_abs_pair_c25_1_2_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev24_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev24 + dist_diff_abs_pair_c25_1_2_prev24
   + 240 dist_diff_abs_pair_c25_1_2_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev24 + dist_diff_abs_pair_c25_1_2_prev24
   + 240 dist_diff_abs_pair_c25_1_2_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev24 + dist_diff_abs_pair_c25_1_2_prev24
   - 240 dist_diff_abs_pair_c25_1_2_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev24_force_positive:
   dist_diff_pair_c25_1_2_prev24
   - 240 dist_diff_abs_pair_c25_1_2_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev24_force_negative:
   dist_diff_pair_c25_1_2_prev24
   - 240 dist_diff_abs_pair_c25_1_2_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev24: 240 same_dist_pair_c25_1_2_prev24
   + dist_diff_abs_pair_c25_1_2_prev24 <= 249.8
 same_dist_pair_lower_c25_1_2_prev24: 240 same_dist_pair_c25_1_2_prev24
   + dist_diff_abs_pair_c25_1_2_prev24 >= 10
 dist_diff_pair_def_c25_1_2_prev25: - dist_curr_pair_c25_1_2
   + dist_diff_pair_c25_1_2_prev25 = -67
 dist_diff_abs_pair_c25_1_2_prev25_abs_ge_orig:
   - dist_diff_pair_c25_1_2_prev25 + dist_diff_abs_pair_c25_1_2_prev25
   - 240 dist_diff_abs_pair_c25_1_2_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev25_abs_le_orig:
   - dist_diff_pair_c25_1_2_prev25 + dist_diff_abs_pair_c25_1_2_prev25
   + 240 dist_diff_abs_pair_c25_1_2_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_1_2_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_1_2_prev25 + dist_diff_abs_pair_c25_1_2_prev25
   + 240 dist_diff_abs_pair_c25_1_2_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_1_2_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_1_2_prev25 + dist_diff_abs_pair_c25_1_2_prev25
   - 240 dist_diff_abs_pair_c25_1_2_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_1_2_prev25_force_positive:
   dist_diff_pair_c25_1_2_prev25
   - 240 dist_diff_abs_pair_c25_1_2_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_1_2_prev25_force_negative:
   dist_diff_pair_c25_1_2_prev25
   - 240 dist_diff_abs_pair_c25_1_2_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_1_2_prev25: 240 same_dist_pair_c25_1_2_prev25
   + dist_diff_abs_pair_c25_1_2_prev25 <= 249.8
 same_dist_pair_lower_c25_1_2_prev25: 240 same_dist_pair_c25_1_2_prev25
   + dist_diff_abs_pair_c25_1_2_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_1_2_prev0: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev0: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev1: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev1: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev2: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev2: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev3: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev3: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev4: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev4: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev5: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev5: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev6: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev6: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev7: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev7: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev8: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev8: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev9: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev9: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev10: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev10: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev11: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev11: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev12: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev12: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev13: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev13: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev14: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev14: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev15: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev15: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev16: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev16: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev17: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev17: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev18: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev18: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev19: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev19: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev20: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev20: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev21: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev21: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev22: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev22: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev23: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev23: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev24: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev24: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_1_2_prev25: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_1_2_prev25: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_1_2: diff_dist_pair_c25_1_2
   + same_dist_pair_c25_1_2_prev0 + same_dist_pair_c25_1_2_prev1
   + same_dist_pair_c25_1_2_prev2 + same_dist_pair_c25_1_2_prev3
   + same_dist_pair_c25_1_2_prev4 + same_dist_pair_c25_1_2_prev5
   + same_dist_pair_c25_1_2_prev6 + same_dist_pair_c25_1_2_prev7
   + same_dist_pair_c25_1_2_prev8 + same_dist_pair_c25_1_2_prev9
   + same_dist_pair_c25_1_2_prev10 + same_dist_pair_c25_1_2_prev11
   + same_dist_pair_c25_1_2_prev12 + same_dist_pair_c25_1_2_prev13
   + same_dist_pair_c25_1_2_prev14 + same_dist_pair_c25_1_2_prev15
   + same_dist_pair_c25_1_2_prev16 + same_dist_pair_c25_1_2_prev17
   + same_dist_pair_c25_1_2_prev18 + same_dist_pair_c25_1_2_prev19
   + same_dist_pair_c25_1_2_prev20 + same_dist_pair_c25_1_2_prev21
   + same_dist_pair_c25_1_2_prev22 + same_dist_pair_c25_1_2_prev23
   + same_dist_pair_c25_1_2_prev24 + same_dist_pair_c25_1_2_prev25 >= 1
 dx_grid_diff_def_c25_1_3: - cx_grid_center_c25_1 + cx_grid_center_c25_3
   + dx_grid_diff_c25_1_3 = 0
 dx_grid_abs_pair_c25_1_3_abs_ge_orig: dx_grid_abs_pair_c25_1_3
   - dx_grid_diff_c25_1_3 - 130 dx_grid_abs_pair_c25_1_3_is_positive
   >= -130
 dx_grid_abs_pair_c25_1_3_abs_le_orig: dx_grid_abs_pair_c25_1_3
   - dx_grid_diff_c25_1_3 + 130 dx_grid_abs_pair_c25_1_3_is_positive
   <= 130
 dx_grid_abs_pair_c25_1_3_abs_ge_neg_orig: dx_grid_abs_pair_c25_1_3
   + dx_grid_diff_c25_1_3 + 130 dx_grid_abs_pair_c25_1_3_is_positive >= 0
 dx_grid_abs_pair_c25_1_3_abs_le_neg_orig: dx_grid_abs_pair_c25_1_3
   + dx_grid_diff_c25_1_3 - 130 dx_grid_abs_pair_c25_1_3_is_positive <= 0
 dx_grid_abs_pair_c25_1_3_force_positive: dx_grid_diff_c25_1_3
   - 130 dx_grid_abs_pair_c25_1_3_is_positive >= -130
 dx_grid_abs_pair_c25_1_3_force_negative: dx_grid_diff_c25_1_3
   - 130 dx_grid_abs_pair_c25_1_3_is_positive <= -0.001
 dy_grid_diff_def_c25_1_3: - cy_grid_center_c25_1 + cy_grid_center_c25_3
   + dy_grid_diff_c25_1_3 = 0
 dy_grid_abs_pair_c25_1_3_abs_ge_orig: dy_grid_abs_pair_c25_1_3
   - dy_grid_diff_c25_1_3 - 130 dy_grid_abs_pair_c25_1_3_is_positive
   >= -130
 dy_grid_abs_pair_c25_1_3_abs_le_orig: dy_grid_abs_pair_c25_1_3
   - dy_grid_diff_c25_1_3 + 130 dy_grid_abs_pair_c25_1_3_is_positive
   <= 130
 dy_grid_abs_pair_c25_1_3_abs_ge_neg_orig: dy_grid_abs_pair_c25_1_3
   + dy_grid_diff_c25_1_3 + 130 dy_grid_abs_pair_c25_1_3_is_positive >= 0
 dy_grid_abs_pair_c25_1_3_abs_le_neg_orig: dy_grid_abs_pair_c25_1_3
   + dy_grid_diff_c25_1_3 - 130 dy_grid_abs_pair_c25_1_3_is_positive <= 0
 dy_grid_abs_pair_c25_1_3_force_positive: dy_grid_diff_c25_1_3
   - 130 dy_grid_abs_pair_c25_1_3_is_positive >= -130
 dy_grid_abs_pair_c25_1_3_force_negative: dy_grid_diff_c25_1_3
   - 130 dy_grid_abs_pair_c25_1_3_is_positive <= -0.001
 dist_curr_pair_def_c25_1_3: - dx_grid_abs_pair_c25_1_3
   - dy_grid_abs_pair_c25_1_3 + dist_curr_pair_c25_1_3 = 0
 dist_diff_pair_def_c25_1_3_prev0: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev0 = -34
 dist_diff_abs_pair_c25_1_3_prev0_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev0 + dist_diff_abs_pair_c25_1_3_prev0
   - 240 dist_diff_abs_pair_c25_1_3_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev0_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev0 + dist_diff_abs_pair_c25_1_3_prev0
   + 240 dist_diff_abs_pair_c25_1_3_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev0 + dist_diff_abs_pair_c25_1_3_prev0
   + 240 dist_diff_abs_pair_c25_1_3_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev0 + dist_diff_abs_pair_c25_1_3_prev0
   - 240 dist_diff_abs_pair_c25_1_3_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev0_force_positive:
   dist_diff_pair_c25_1_3_prev0
   - 240 dist_diff_abs_pair_c25_1_3_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev0_force_negative:
   dist_diff_pair_c25_1_3_prev0
   - 240 dist_diff_abs_pair_c25_1_3_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev0: 240 same_dist_pair_c25_1_3_prev0
   + dist_diff_abs_pair_c25_1_3_prev0 <= 249.8
 same_dist_pair_lower_c25_1_3_prev0: 240 same_dist_pair_c25_1_3_prev0
   + dist_diff_abs_pair_c25_1_3_prev0 >= 10
 dist_diff_pair_def_c25_1_3_prev1: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev1 = -32
 dist_diff_abs_pair_c25_1_3_prev1_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev1 + dist_diff_abs_pair_c25_1_3_prev1
   - 240 dist_diff_abs_pair_c25_1_3_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev1_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev1 + dist_diff_abs_pair_c25_1_3_prev1
   + 240 dist_diff_abs_pair_c25_1_3_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev1 + dist_diff_abs_pair_c25_1_3_prev1
   + 240 dist_diff_abs_pair_c25_1_3_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev1 + dist_diff_abs_pair_c25_1_3_prev1
   - 240 dist_diff_abs_pair_c25_1_3_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev1_force_positive:
   dist_diff_pair_c25_1_3_prev1
   - 240 dist_diff_abs_pair_c25_1_3_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev1_force_negative:
   dist_diff_pair_c25_1_3_prev1
   - 240 dist_diff_abs_pair_c25_1_3_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev1: 240 same_dist_pair_c25_1_3_prev1
   + dist_diff_abs_pair_c25_1_3_prev1 <= 249.8
 same_dist_pair_lower_c25_1_3_prev1: 240 same_dist_pair_c25_1_3_prev1
   + dist_diff_abs_pair_c25_1_3_prev1 >= 10
 dist_diff_pair_def_c25_1_3_prev2: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev2 = -40
 dist_diff_abs_pair_c25_1_3_prev2_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev2 + dist_diff_abs_pair_c25_1_3_prev2
   - 240 dist_diff_abs_pair_c25_1_3_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev2_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev2 + dist_diff_abs_pair_c25_1_3_prev2
   + 240 dist_diff_abs_pair_c25_1_3_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev2 + dist_diff_abs_pair_c25_1_3_prev2
   + 240 dist_diff_abs_pair_c25_1_3_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev2 + dist_diff_abs_pair_c25_1_3_prev2
   - 240 dist_diff_abs_pair_c25_1_3_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev2_force_positive:
   dist_diff_pair_c25_1_3_prev2
   - 240 dist_diff_abs_pair_c25_1_3_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev2_force_negative:
   dist_diff_pair_c25_1_3_prev2
   - 240 dist_diff_abs_pair_c25_1_3_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev2: 240 same_dist_pair_c25_1_3_prev2
   + dist_diff_abs_pair_c25_1_3_prev2 <= 249.8
 same_dist_pair_lower_c25_1_3_prev2: 240 same_dist_pair_c25_1_3_prev2
   + dist_diff_abs_pair_c25_1_3_prev2 >= 10
 dist_diff_pair_def_c25_1_3_prev3: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev3 = -37
 dist_diff_abs_pair_c25_1_3_prev3_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev3 + dist_diff_abs_pair_c25_1_3_prev3
   - 240 dist_diff_abs_pair_c25_1_3_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev3_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev3 + dist_diff_abs_pair_c25_1_3_prev3
   + 240 dist_diff_abs_pair_c25_1_3_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev3 + dist_diff_abs_pair_c25_1_3_prev3
   + 240 dist_diff_abs_pair_c25_1_3_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev3 + dist_diff_abs_pair_c25_1_3_prev3
   - 240 dist_diff_abs_pair_c25_1_3_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev3_force_positive:
   dist_diff_pair_c25_1_3_prev3
   - 240 dist_diff_abs_pair_c25_1_3_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev3_force_negative:
   dist_diff_pair_c25_1_3_prev3
   - 240 dist_diff_abs_pair_c25_1_3_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev3: 240 same_dist_pair_c25_1_3_prev3
   + dist_diff_abs_pair_c25_1_3_prev3 <= 249.8
 same_dist_pair_lower_c25_1_3_prev3: 240 same_dist_pair_c25_1_3_prev3
   + dist_diff_abs_pair_c25_1_3_prev3 >= 10
 dist_diff_pair_def_c25_1_3_prev4: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev4 = -26
 dist_diff_abs_pair_c25_1_3_prev4_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev4 + dist_diff_abs_pair_c25_1_3_prev4
   - 240 dist_diff_abs_pair_c25_1_3_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev4_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev4 + dist_diff_abs_pair_c25_1_3_prev4
   + 240 dist_diff_abs_pair_c25_1_3_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev4 + dist_diff_abs_pair_c25_1_3_prev4
   + 240 dist_diff_abs_pair_c25_1_3_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev4 + dist_diff_abs_pair_c25_1_3_prev4
   - 240 dist_diff_abs_pair_c25_1_3_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev4_force_positive:
   dist_diff_pair_c25_1_3_prev4
   - 240 dist_diff_abs_pair_c25_1_3_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev4_force_negative:
   dist_diff_pair_c25_1_3_prev4
   - 240 dist_diff_abs_pair_c25_1_3_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev4: 240 same_dist_pair_c25_1_3_prev4
   + dist_diff_abs_pair_c25_1_3_prev4 <= 249.8
 same_dist_pair_lower_c25_1_3_prev4: 240 same_dist_pair_c25_1_3_prev4
   + dist_diff_abs_pair_c25_1_3_prev4 >= 10
 dist_diff_pair_def_c25_1_3_prev5: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev5 = -34
 dist_diff_abs_pair_c25_1_3_prev5_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev5 + dist_diff_abs_pair_c25_1_3_prev5
   - 240 dist_diff_abs_pair_c25_1_3_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev5_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev5 + dist_diff_abs_pair_c25_1_3_prev5
   + 240 dist_diff_abs_pair_c25_1_3_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev5 + dist_diff_abs_pair_c25_1_3_prev5
   + 240 dist_diff_abs_pair_c25_1_3_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev5 + dist_diff_abs_pair_c25_1_3_prev5
   - 240 dist_diff_abs_pair_c25_1_3_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev5_force_positive:
   dist_diff_pair_c25_1_3_prev5
   - 240 dist_diff_abs_pair_c25_1_3_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev5_force_negative:
   dist_diff_pair_c25_1_3_prev5
   - 240 dist_diff_abs_pair_c25_1_3_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev5: 240 same_dist_pair_c25_1_3_prev5
   + dist_diff_abs_pair_c25_1_3_prev5 <= 249.8
 same_dist_pair_lower_c25_1_3_prev5: 240 same_dist_pair_c25_1_3_prev5
   + dist_diff_abs_pair_c25_1_3_prev5 >= 10
 dist_diff_pair_def_c25_1_3_prev6: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev6 = -44
 dist_diff_abs_pair_c25_1_3_prev6_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev6 + dist_diff_abs_pair_c25_1_3_prev6
   - 240 dist_diff_abs_pair_c25_1_3_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev6_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev6 + dist_diff_abs_pair_c25_1_3_prev6
   + 240 dist_diff_abs_pair_c25_1_3_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev6 + dist_diff_abs_pair_c25_1_3_prev6
   + 240 dist_diff_abs_pair_c25_1_3_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev6 + dist_diff_abs_pair_c25_1_3_prev6
   - 240 dist_diff_abs_pair_c25_1_3_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev6_force_positive:
   dist_diff_pair_c25_1_3_prev6
   - 240 dist_diff_abs_pair_c25_1_3_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev6_force_negative:
   dist_diff_pair_c25_1_3_prev6
   - 240 dist_diff_abs_pair_c25_1_3_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev6: 240 same_dist_pair_c25_1_3_prev6
   + dist_diff_abs_pair_c25_1_3_prev6 <= 249.8
 same_dist_pair_lower_c25_1_3_prev6: 240 same_dist_pair_c25_1_3_prev6
   + dist_diff_abs_pair_c25_1_3_prev6 >= 10
 dist_diff_pair_def_c25_1_3_prev7: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev7 = -46
 dist_diff_abs_pair_c25_1_3_prev7_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev7 + dist_diff_abs_pair_c25_1_3_prev7
   - 240 dist_diff_abs_pair_c25_1_3_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev7_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev7 + dist_diff_abs_pair_c25_1_3_prev7
   + 240 dist_diff_abs_pair_c25_1_3_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev7 + dist_diff_abs_pair_c25_1_3_prev7
   + 240 dist_diff_abs_pair_c25_1_3_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev7 + dist_diff_abs_pair_c25_1_3_prev7
   - 240 dist_diff_abs_pair_c25_1_3_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev7_force_positive:
   dist_diff_pair_c25_1_3_prev7
   - 240 dist_diff_abs_pair_c25_1_3_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev7_force_negative:
   dist_diff_pair_c25_1_3_prev7
   - 240 dist_diff_abs_pair_c25_1_3_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev7: 240 same_dist_pair_c25_1_3_prev7
   + dist_diff_abs_pair_c25_1_3_prev7 <= 249.8
 same_dist_pair_lower_c25_1_3_prev7: 240 same_dist_pair_c25_1_3_prev7
   + dist_diff_abs_pair_c25_1_3_prev7 >= 10
 dist_diff_pair_def_c25_1_3_prev8: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev8 = -32
 dist_diff_abs_pair_c25_1_3_prev8_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev8 + dist_diff_abs_pair_c25_1_3_prev8
   - 240 dist_diff_abs_pair_c25_1_3_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev8_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev8 + dist_diff_abs_pair_c25_1_3_prev8
   + 240 dist_diff_abs_pair_c25_1_3_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev8 + dist_diff_abs_pair_c25_1_3_prev8
   + 240 dist_diff_abs_pair_c25_1_3_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev8 + dist_diff_abs_pair_c25_1_3_prev8
   - 240 dist_diff_abs_pair_c25_1_3_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev8_force_positive:
   dist_diff_pair_c25_1_3_prev8
   - 240 dist_diff_abs_pair_c25_1_3_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev8_force_negative:
   dist_diff_pair_c25_1_3_prev8
   - 240 dist_diff_abs_pair_c25_1_3_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev8: 240 same_dist_pair_c25_1_3_prev8
   + dist_diff_abs_pair_c25_1_3_prev8 <= 249.8
 same_dist_pair_lower_c25_1_3_prev8: 240 same_dist_pair_c25_1_3_prev8
   + dist_diff_abs_pair_c25_1_3_prev8 >= 10
 dist_diff_pair_def_c25_1_3_prev9: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev9 = -28
 dist_diff_abs_pair_c25_1_3_prev9_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev9 + dist_diff_abs_pair_c25_1_3_prev9
   - 240 dist_diff_abs_pair_c25_1_3_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev9_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev9 + dist_diff_abs_pair_c25_1_3_prev9
   + 240 dist_diff_abs_pair_c25_1_3_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev9 + dist_diff_abs_pair_c25_1_3_prev9
   + 240 dist_diff_abs_pair_c25_1_3_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev9 + dist_diff_abs_pair_c25_1_3_prev9
   - 240 dist_diff_abs_pair_c25_1_3_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev9_force_positive:
   dist_diff_pair_c25_1_3_prev9
   - 240 dist_diff_abs_pair_c25_1_3_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev9_force_negative:
   dist_diff_pair_c25_1_3_prev9
   - 240 dist_diff_abs_pair_c25_1_3_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev9: 240 same_dist_pair_c25_1_3_prev9
   + dist_diff_abs_pair_c25_1_3_prev9 <= 249.8
 same_dist_pair_lower_c25_1_3_prev9: 240 same_dist_pair_c25_1_3_prev9
   + dist_diff_abs_pair_c25_1_3_prev9 >= 10
 dist_diff_pair_def_c25_1_3_prev10: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev10 = -32
 dist_diff_abs_pair_c25_1_3_prev10_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev10 + dist_diff_abs_pair_c25_1_3_prev10
   - 240 dist_diff_abs_pair_c25_1_3_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev10_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev10 + dist_diff_abs_pair_c25_1_3_prev10
   + 240 dist_diff_abs_pair_c25_1_3_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev10 + dist_diff_abs_pair_c25_1_3_prev10
   + 240 dist_diff_abs_pair_c25_1_3_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev10 + dist_diff_abs_pair_c25_1_3_prev10
   - 240 dist_diff_abs_pair_c25_1_3_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev10_force_positive:
   dist_diff_pair_c25_1_3_prev10
   - 240 dist_diff_abs_pair_c25_1_3_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev10_force_negative:
   dist_diff_pair_c25_1_3_prev10
   - 240 dist_diff_abs_pair_c25_1_3_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev10: 240 same_dist_pair_c25_1_3_prev10
   + dist_diff_abs_pair_c25_1_3_prev10 <= 249.8
 same_dist_pair_lower_c25_1_3_prev10: 240 same_dist_pair_c25_1_3_prev10
   + dist_diff_abs_pair_c25_1_3_prev10 >= 10
 dist_diff_pair_def_c25_1_3_prev11: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev11 = -26
 dist_diff_abs_pair_c25_1_3_prev11_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev11 + dist_diff_abs_pair_c25_1_3_prev11
   - 240 dist_diff_abs_pair_c25_1_3_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev11_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev11 + dist_diff_abs_pair_c25_1_3_prev11
   + 240 dist_diff_abs_pair_c25_1_3_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev11 + dist_diff_abs_pair_c25_1_3_prev11
   + 240 dist_diff_abs_pair_c25_1_3_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev11 + dist_diff_abs_pair_c25_1_3_prev11
   - 240 dist_diff_abs_pair_c25_1_3_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev11_force_positive:
   dist_diff_pair_c25_1_3_prev11
   - 240 dist_diff_abs_pair_c25_1_3_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev11_force_negative:
   dist_diff_pair_c25_1_3_prev11
   - 240 dist_diff_abs_pair_c25_1_3_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev11: 240 same_dist_pair_c25_1_3_prev11
   + dist_diff_abs_pair_c25_1_3_prev11 <= 249.8
 same_dist_pair_lower_c25_1_3_prev11: 240 same_dist_pair_c25_1_3_prev11
   + dist_diff_abs_pair_c25_1_3_prev11 >= 10
 dist_diff_pair_def_c25_1_3_prev12: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev12 = -40
 dist_diff_abs_pair_c25_1_3_prev12_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev12 + dist_diff_abs_pair_c25_1_3_prev12
   - 240 dist_diff_abs_pair_c25_1_3_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev12_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev12 + dist_diff_abs_pair_c25_1_3_prev12
   + 240 dist_diff_abs_pair_c25_1_3_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev12 + dist_diff_abs_pair_c25_1_3_prev12
   + 240 dist_diff_abs_pair_c25_1_3_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev12 + dist_diff_abs_pair_c25_1_3_prev12
   - 240 dist_diff_abs_pair_c25_1_3_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev12_force_positive:
   dist_diff_pair_c25_1_3_prev12
   - 240 dist_diff_abs_pair_c25_1_3_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev12_force_negative:
   dist_diff_pair_c25_1_3_prev12
   - 240 dist_diff_abs_pair_c25_1_3_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev12: 240 same_dist_pair_c25_1_3_prev12
   + dist_diff_abs_pair_c25_1_3_prev12 <= 249.8
 same_dist_pair_lower_c25_1_3_prev12: 240 same_dist_pair_c25_1_3_prev12
   + dist_diff_abs_pair_c25_1_3_prev12 >= 10
 dist_diff_pair_def_c25_1_3_prev13: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev13 = -54
 dist_diff_abs_pair_c25_1_3_prev13_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev13 + dist_diff_abs_pair_c25_1_3_prev13
   - 240 dist_diff_abs_pair_c25_1_3_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev13_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev13 + dist_diff_abs_pair_c25_1_3_prev13
   + 240 dist_diff_abs_pair_c25_1_3_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev13 + dist_diff_abs_pair_c25_1_3_prev13
   + 240 dist_diff_abs_pair_c25_1_3_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev13 + dist_diff_abs_pair_c25_1_3_prev13
   - 240 dist_diff_abs_pair_c25_1_3_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev13_force_positive:
   dist_diff_pair_c25_1_3_prev13
   - 240 dist_diff_abs_pair_c25_1_3_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev13_force_negative:
   dist_diff_pair_c25_1_3_prev13
   - 240 dist_diff_abs_pair_c25_1_3_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev13: 240 same_dist_pair_c25_1_3_prev13
   + dist_diff_abs_pair_c25_1_3_prev13 <= 249.8
 same_dist_pair_lower_c25_1_3_prev13: 240 same_dist_pair_c25_1_3_prev13
   + dist_diff_abs_pair_c25_1_3_prev13 >= 10
 dist_diff_pair_def_c25_1_3_prev14: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev14 = -28
 dist_diff_abs_pair_c25_1_3_prev14_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev14 + dist_diff_abs_pair_c25_1_3_prev14
   - 240 dist_diff_abs_pair_c25_1_3_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev14_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev14 + dist_diff_abs_pair_c25_1_3_prev14
   + 240 dist_diff_abs_pair_c25_1_3_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev14 + dist_diff_abs_pair_c25_1_3_prev14
   + 240 dist_diff_abs_pair_c25_1_3_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev14 + dist_diff_abs_pair_c25_1_3_prev14
   - 240 dist_diff_abs_pair_c25_1_3_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev14_force_positive:
   dist_diff_pair_c25_1_3_prev14
   - 240 dist_diff_abs_pair_c25_1_3_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev14_force_negative:
   dist_diff_pair_c25_1_3_prev14
   - 240 dist_diff_abs_pair_c25_1_3_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev14: 240 same_dist_pair_c25_1_3_prev14
   + dist_diff_abs_pair_c25_1_3_prev14 <= 249.8
 same_dist_pair_lower_c25_1_3_prev14: 240 same_dist_pair_c25_1_3_prev14
   + dist_diff_abs_pair_c25_1_3_prev14 >= 10
 dist_diff_pair_def_c25_1_3_prev15: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev15 = -32
 dist_diff_abs_pair_c25_1_3_prev15_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev15 + dist_diff_abs_pair_c25_1_3_prev15
   - 240 dist_diff_abs_pair_c25_1_3_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev15_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev15 + dist_diff_abs_pair_c25_1_3_prev15
   + 240 dist_diff_abs_pair_c25_1_3_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev15 + dist_diff_abs_pair_c25_1_3_prev15
   + 240 dist_diff_abs_pair_c25_1_3_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev15 + dist_diff_abs_pair_c25_1_3_prev15
   - 240 dist_diff_abs_pair_c25_1_3_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev15_force_positive:
   dist_diff_pair_c25_1_3_prev15
   - 240 dist_diff_abs_pair_c25_1_3_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev15_force_negative:
   dist_diff_pair_c25_1_3_prev15
   - 240 dist_diff_abs_pair_c25_1_3_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev15: 240 same_dist_pair_c25_1_3_prev15
   + dist_diff_abs_pair_c25_1_3_prev15 <= 249.8
 same_dist_pair_lower_c25_1_3_prev15: 240 same_dist_pair_c25_1_3_prev15
   + dist_diff_abs_pair_c25_1_3_prev15 >= 10
 dist_diff_pair_def_c25_1_3_prev16: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev16 = -28
 dist_diff_abs_pair_c25_1_3_prev16_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev16 + dist_diff_abs_pair_c25_1_3_prev16
   - 240 dist_diff_abs_pair_c25_1_3_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev16_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev16 + dist_diff_abs_pair_c25_1_3_prev16
   + 240 dist_diff_abs_pair_c25_1_3_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev16 + dist_diff_abs_pair_c25_1_3_prev16
   + 240 dist_diff_abs_pair_c25_1_3_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev16 + dist_diff_abs_pair_c25_1_3_prev16
   - 240 dist_diff_abs_pair_c25_1_3_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev16_force_positive:
   dist_diff_pair_c25_1_3_prev16
   - 240 dist_diff_abs_pair_c25_1_3_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev16_force_negative:
   dist_diff_pair_c25_1_3_prev16
   - 240 dist_diff_abs_pair_c25_1_3_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev16: 240 same_dist_pair_c25_1_3_prev16
   + dist_diff_abs_pair_c25_1_3_prev16 <= 249.8
 same_dist_pair_lower_c25_1_3_prev16: 240 same_dist_pair_c25_1_3_prev16
   + dist_diff_abs_pair_c25_1_3_prev16 >= 10
 dist_diff_pair_def_c25_1_3_prev17: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev17 = -36
 dist_diff_abs_pair_c25_1_3_prev17_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev17 + dist_diff_abs_pair_c25_1_3_prev17
   - 240 dist_diff_abs_pair_c25_1_3_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev17_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev17 + dist_diff_abs_pair_c25_1_3_prev17
   + 240 dist_diff_abs_pair_c25_1_3_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev17 + dist_diff_abs_pair_c25_1_3_prev17
   + 240 dist_diff_abs_pair_c25_1_3_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev17 + dist_diff_abs_pair_c25_1_3_prev17
   - 240 dist_diff_abs_pair_c25_1_3_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev17_force_positive:
   dist_diff_pair_c25_1_3_prev17
   - 240 dist_diff_abs_pair_c25_1_3_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev17_force_negative:
   dist_diff_pair_c25_1_3_prev17
   - 240 dist_diff_abs_pair_c25_1_3_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev17: 240 same_dist_pair_c25_1_3_prev17
   + dist_diff_abs_pair_c25_1_3_prev17 <= 249.8
 same_dist_pair_lower_c25_1_3_prev17: 240 same_dist_pair_c25_1_3_prev17
   + dist_diff_abs_pair_c25_1_3_prev17 >= 10
 dist_diff_pair_def_c25_1_3_prev18: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev18 = -45
 dist_diff_abs_pair_c25_1_3_prev18_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev18 + dist_diff_abs_pair_c25_1_3_prev18
   - 240 dist_diff_abs_pair_c25_1_3_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev18_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev18 + dist_diff_abs_pair_c25_1_3_prev18
   + 240 dist_diff_abs_pair_c25_1_3_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev18 + dist_diff_abs_pair_c25_1_3_prev18
   + 240 dist_diff_abs_pair_c25_1_3_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev18 + dist_diff_abs_pair_c25_1_3_prev18
   - 240 dist_diff_abs_pair_c25_1_3_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev18_force_positive:
   dist_diff_pair_c25_1_3_prev18
   - 240 dist_diff_abs_pair_c25_1_3_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev18_force_negative:
   dist_diff_pair_c25_1_3_prev18
   - 240 dist_diff_abs_pair_c25_1_3_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev18: 240 same_dist_pair_c25_1_3_prev18
   + dist_diff_abs_pair_c25_1_3_prev18 <= 249.8
 same_dist_pair_lower_c25_1_3_prev18: 240 same_dist_pair_c25_1_3_prev18
   + dist_diff_abs_pair_c25_1_3_prev18 >= 10
 dist_diff_pair_def_c25_1_3_prev19: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev19 = -32
 dist_diff_abs_pair_c25_1_3_prev19_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev19 + dist_diff_abs_pair_c25_1_3_prev19
   - 240 dist_diff_abs_pair_c25_1_3_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev19_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev19 + dist_diff_abs_pair_c25_1_3_prev19
   + 240 dist_diff_abs_pair_c25_1_3_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev19 + dist_diff_abs_pair_c25_1_3_prev19
   + 240 dist_diff_abs_pair_c25_1_3_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev19 + dist_diff_abs_pair_c25_1_3_prev19
   - 240 dist_diff_abs_pair_c25_1_3_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev19_force_positive:
   dist_diff_pair_c25_1_3_prev19
   - 240 dist_diff_abs_pair_c25_1_3_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev19_force_negative:
   dist_diff_pair_c25_1_3_prev19
   - 240 dist_diff_abs_pair_c25_1_3_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev19: 240 same_dist_pair_c25_1_3_prev19
   + dist_diff_abs_pair_c25_1_3_prev19 <= 249.8
 same_dist_pair_lower_c25_1_3_prev19: 240 same_dist_pair_c25_1_3_prev19
   + dist_diff_abs_pair_c25_1_3_prev19 >= 10
 dist_diff_pair_def_c25_1_3_prev20: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev20 = -34
 dist_diff_abs_pair_c25_1_3_prev20_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev20 + dist_diff_abs_pair_c25_1_3_prev20
   - 240 dist_diff_abs_pair_c25_1_3_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev20_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev20 + dist_diff_abs_pair_c25_1_3_prev20
   + 240 dist_diff_abs_pair_c25_1_3_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev20 + dist_diff_abs_pair_c25_1_3_prev20
   + 240 dist_diff_abs_pair_c25_1_3_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev20 + dist_diff_abs_pair_c25_1_3_prev20
   - 240 dist_diff_abs_pair_c25_1_3_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev20_force_positive:
   dist_diff_pair_c25_1_3_prev20
   - 240 dist_diff_abs_pair_c25_1_3_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev20_force_negative:
   dist_diff_pair_c25_1_3_prev20
   - 240 dist_diff_abs_pair_c25_1_3_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev20: 240 same_dist_pair_c25_1_3_prev20
   + dist_diff_abs_pair_c25_1_3_prev20 <= 249.8
 same_dist_pair_lower_c25_1_3_prev20: 240 same_dist_pair_c25_1_3_prev20
   + dist_diff_abs_pair_c25_1_3_prev20 >= 10
 dist_diff_pair_def_c25_1_3_prev21: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev21 = -40
 dist_diff_abs_pair_c25_1_3_prev21_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev21 + dist_diff_abs_pair_c25_1_3_prev21
   - 240 dist_diff_abs_pair_c25_1_3_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev21_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev21 + dist_diff_abs_pair_c25_1_3_prev21
   + 240 dist_diff_abs_pair_c25_1_3_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev21 + dist_diff_abs_pair_c25_1_3_prev21
   + 240 dist_diff_abs_pair_c25_1_3_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev21 + dist_diff_abs_pair_c25_1_3_prev21
   - 240 dist_diff_abs_pair_c25_1_3_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev21_force_positive:
   dist_diff_pair_c25_1_3_prev21
   - 240 dist_diff_abs_pair_c25_1_3_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev21_force_negative:
   dist_diff_pair_c25_1_3_prev21
   - 240 dist_diff_abs_pair_c25_1_3_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev21: 240 same_dist_pair_c25_1_3_prev21
   + dist_diff_abs_pair_c25_1_3_prev21 <= 249.8
 same_dist_pair_lower_c25_1_3_prev21: 240 same_dist_pair_c25_1_3_prev21
   + dist_diff_abs_pair_c25_1_3_prev21 >= 10
 dist_diff_pair_def_c25_1_3_prev22: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev22 = -31
 dist_diff_abs_pair_c25_1_3_prev22_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev22 + dist_diff_abs_pair_c25_1_3_prev22
   - 240 dist_diff_abs_pair_c25_1_3_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev22_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev22 + dist_diff_abs_pair_c25_1_3_prev22
   + 240 dist_diff_abs_pair_c25_1_3_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev22 + dist_diff_abs_pair_c25_1_3_prev22
   + 240 dist_diff_abs_pair_c25_1_3_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev22 + dist_diff_abs_pair_c25_1_3_prev22
   - 240 dist_diff_abs_pair_c25_1_3_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev22_force_positive:
   dist_diff_pair_c25_1_3_prev22
   - 240 dist_diff_abs_pair_c25_1_3_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev22_force_negative:
   dist_diff_pair_c25_1_3_prev22
   - 240 dist_diff_abs_pair_c25_1_3_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev22: 240 same_dist_pair_c25_1_3_prev22
   + dist_diff_abs_pair_c25_1_3_prev22 <= 249.8
 same_dist_pair_lower_c25_1_3_prev22: 240 same_dist_pair_c25_1_3_prev22
   + dist_diff_abs_pair_c25_1_3_prev22 >= 10
 dist_diff_pair_def_c25_1_3_prev23: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev23 = -27
 dist_diff_abs_pair_c25_1_3_prev23_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev23 + dist_diff_abs_pair_c25_1_3_prev23
   - 240 dist_diff_abs_pair_c25_1_3_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev23_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev23 + dist_diff_abs_pair_c25_1_3_prev23
   + 240 dist_diff_abs_pair_c25_1_3_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev23 + dist_diff_abs_pair_c25_1_3_prev23
   + 240 dist_diff_abs_pair_c25_1_3_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev23 + dist_diff_abs_pair_c25_1_3_prev23
   - 240 dist_diff_abs_pair_c25_1_3_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev23_force_positive:
   dist_diff_pair_c25_1_3_prev23
   - 240 dist_diff_abs_pair_c25_1_3_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev23_force_negative:
   dist_diff_pair_c25_1_3_prev23
   - 240 dist_diff_abs_pair_c25_1_3_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev23: 240 same_dist_pair_c25_1_3_prev23
   + dist_diff_abs_pair_c25_1_3_prev23 <= 249.8
 same_dist_pair_lower_c25_1_3_prev23: 240 same_dist_pair_c25_1_3_prev23
   + dist_diff_abs_pair_c25_1_3_prev23 >= 10
 dist_diff_pair_def_c25_1_3_prev24: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev24 = -44
 dist_diff_abs_pair_c25_1_3_prev24_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev24 + dist_diff_abs_pair_c25_1_3_prev24
   - 240 dist_diff_abs_pair_c25_1_3_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev24_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev24 + dist_diff_abs_pair_c25_1_3_prev24
   + 240 dist_diff_abs_pair_c25_1_3_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev24 + dist_diff_abs_pair_c25_1_3_prev24
   + 240 dist_diff_abs_pair_c25_1_3_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev24 + dist_diff_abs_pair_c25_1_3_prev24
   - 240 dist_diff_abs_pair_c25_1_3_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev24_force_positive:
   dist_diff_pair_c25_1_3_prev24
   - 240 dist_diff_abs_pair_c25_1_3_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev24_force_negative:
   dist_diff_pair_c25_1_3_prev24
   - 240 dist_diff_abs_pair_c25_1_3_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev24: 240 same_dist_pair_c25_1_3_prev24
   + dist_diff_abs_pair_c25_1_3_prev24 <= 249.8
 same_dist_pair_lower_c25_1_3_prev24: 240 same_dist_pair_c25_1_3_prev24
   + dist_diff_abs_pair_c25_1_3_prev24 >= 10
 dist_diff_pair_def_c25_1_3_prev25: - dist_curr_pair_c25_1_3
   + dist_diff_pair_c25_1_3_prev25 = -39
 dist_diff_abs_pair_c25_1_3_prev25_abs_ge_orig:
   - dist_diff_pair_c25_1_3_prev25 + dist_diff_abs_pair_c25_1_3_prev25
   - 240 dist_diff_abs_pair_c25_1_3_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev25_abs_le_orig:
   - dist_diff_pair_c25_1_3_prev25 + dist_diff_abs_pair_c25_1_3_prev25
   + 240 dist_diff_abs_pair_c25_1_3_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_1_3_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_1_3_prev25 + dist_diff_abs_pair_c25_1_3_prev25
   + 240 dist_diff_abs_pair_c25_1_3_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_1_3_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_1_3_prev25 + dist_diff_abs_pair_c25_1_3_prev25
   - 240 dist_diff_abs_pair_c25_1_3_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_1_3_prev25_force_positive:
   dist_diff_pair_c25_1_3_prev25
   - 240 dist_diff_abs_pair_c25_1_3_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_1_3_prev25_force_negative:
   dist_diff_pair_c25_1_3_prev25
   - 240 dist_diff_abs_pair_c25_1_3_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_1_3_prev25: 240 same_dist_pair_c25_1_3_prev25
   + dist_diff_abs_pair_c25_1_3_prev25 <= 249.8
 same_dist_pair_lower_c25_1_3_prev25: 240 same_dist_pair_c25_1_3_prev25
   + dist_diff_abs_pair_c25_1_3_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_1_3_prev0: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev0: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev1: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev1: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev2: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev2: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev3: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev3: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev4: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev4: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev5: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev5: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev6: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev6: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev7: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev7: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev8: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev8: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev9: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev9: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev10: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev10: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev11: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev11: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev12: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev12: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev13: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev13: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev14: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev14: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev15: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev15: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev16: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev16: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev17: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev17: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev18: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev18: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev19: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev19: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev20: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev20: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev21: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev21: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev22: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev22: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev23: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev23: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev24: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev24: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_1_3_prev25: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_1_3_prev25: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_1_3: diff_dist_pair_c25_1_3
   + same_dist_pair_c25_1_3_prev0 + same_dist_pair_c25_1_3_prev1
   + same_dist_pair_c25_1_3_prev2 + same_dist_pair_c25_1_3_prev3
   + same_dist_pair_c25_1_3_prev4 + same_dist_pair_c25_1_3_prev5
   + same_dist_pair_c25_1_3_prev6 + same_dist_pair_c25_1_3_prev7
   + same_dist_pair_c25_1_3_prev8 + same_dist_pair_c25_1_3_prev9
   + same_dist_pair_c25_1_3_prev10 + same_dist_pair_c25_1_3_prev11
   + same_dist_pair_c25_1_3_prev12 + same_dist_pair_c25_1_3_prev13
   + same_dist_pair_c25_1_3_prev14 + same_dist_pair_c25_1_3_prev15
   + same_dist_pair_c25_1_3_prev16 + same_dist_pair_c25_1_3_prev17
   + same_dist_pair_c25_1_3_prev18 + same_dist_pair_c25_1_3_prev19
   + same_dist_pair_c25_1_3_prev20 + same_dist_pair_c25_1_3_prev21
   + same_dist_pair_c25_1_3_prev22 + same_dist_pair_c25_1_3_prev23
   + same_dist_pair_c25_1_3_prev24 + same_dist_pair_c25_1_3_prev25 >= 1
 dx_grid_diff_def_c25_1_4: - cx_grid_center_c25_1 + cx_grid_center_c25_4
   + dx_grid_diff_c25_1_4 = 0
 dx_grid_abs_pair_c25_1_4_abs_ge_orig: dx_grid_abs_pair_c25_1_4
   - dx_grid_diff_c25_1_4 - 130 dx_grid_abs_pair_c25_1_4_is_positive
   >= -130
 dx_grid_abs_pair_c25_1_4_abs_le_orig: dx_grid_abs_pair_c25_1_4
   - dx_grid_diff_c25_1_4 + 130 dx_grid_abs_pair_c25_1_4_is_positive
   <= 130
 dx_grid_abs_pair_c25_1_4_abs_ge_neg_orig: dx_grid_abs_pair_c25_1_4
   + dx_grid_diff_c25_1_4 + 130 dx_grid_abs_pair_c25_1_4_is_positive >= 0
 dx_grid_abs_pair_c25_1_4_abs_le_neg_orig: dx_grid_abs_pair_c25_1_4
   + dx_grid_diff_c25_1_4 - 130 dx_grid_abs_pair_c25_1_4_is_positive <= 0
 dx_grid_abs_pair_c25_1_4_force_positive: dx_grid_diff_c25_1_4
   - 130 dx_grid_abs_pair_c25_1_4_is_positive >= -130
 dx_grid_abs_pair_c25_1_4_force_negative: dx_grid_diff_c25_1_4
   - 130 dx_grid_abs_pair_c25_1_4_is_positive <= -0.001
 dy_grid_diff_def_c25_1_4: - cy_grid_center_c25_1 + cy_grid_center_c25_4
   + dy_grid_diff_c25_1_4 = 0
 dy_grid_abs_pair_c25_1_4_abs_ge_orig: dy_grid_abs_pair_c25_1_4
   - dy_grid_diff_c25_1_4 - 130 dy_grid_abs_pair_c25_1_4_is_positive
   >= -130
 dy_grid_abs_pair_c25_1_4_abs_le_orig: dy_grid_abs_pair_c25_1_4
   - dy_grid_diff_c25_1_4 + 130 dy_grid_abs_pair_c25_1_4_is_positive
   <= 130
 dy_grid_abs_pair_c25_1_4_abs_ge_neg_orig: dy_grid_abs_pair_c25_1_4
   + dy_grid_diff_c25_1_4 + 130 dy_grid_abs_pair_c25_1_4_is_positive >= 0
 dy_grid_abs_pair_c25_1_4_abs_le_neg_orig: dy_grid_abs_pair_c25_1_4
   + dy_grid_diff_c25_1_4 - 130 dy_grid_abs_pair_c25_1_4_is_positive <= 0
 dy_grid_abs_pair_c25_1_4_force_positive: dy_grid_diff_c25_1_4
   - 130 dy_grid_abs_pair_c25_1_4_is_positive >= -130
 dy_grid_abs_pair_c25_1_4_force_negative: dy_grid_diff_c25_1_4
   - 130 dy_grid_abs_pair_c25_1_4_is_positive <= -0.001
 dist_curr_pair_def_c25_1_4: - dx_grid_abs_pair_c25_1_4
   - dy_grid_abs_pair_c25_1_4 + dist_curr_pair_c25_1_4 = 0
 dist_diff_pair_def_c25_1_4_prev0: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev0 = -54
 dist_diff_abs_pair_c25_1_4_prev0_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev0 + dist_diff_abs_pair_c25_1_4_prev0
   - 240 dist_diff_abs_pair_c25_1_4_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev0_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev0 + dist_diff_abs_pair_c25_1_4_prev0
   + 240 dist_diff_abs_pair_c25_1_4_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev0 + dist_diff_abs_pair_c25_1_4_prev0
   + 240 dist_diff_abs_pair_c25_1_4_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev0 + dist_diff_abs_pair_c25_1_4_prev0
   - 240 dist_diff_abs_pair_c25_1_4_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev0_force_positive:
   dist_diff_pair_c25_1_4_prev0
   - 240 dist_diff_abs_pair_c25_1_4_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev0_force_negative:
   dist_diff_pair_c25_1_4_prev0
   - 240 dist_diff_abs_pair_c25_1_4_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev0: 240 same_dist_pair_c25_1_4_prev0
   + dist_diff_abs_pair_c25_1_4_prev0 <= 249.8
 same_dist_pair_lower_c25_1_4_prev0: 240 same_dist_pair_c25_1_4_prev0
   + dist_diff_abs_pair_c25_1_4_prev0 >= 10
 dist_diff_pair_def_c25_1_4_prev1: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev1 = -54
 dist_diff_abs_pair_c25_1_4_prev1_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev1 + dist_diff_abs_pair_c25_1_4_prev1
   - 240 dist_diff_abs_pair_c25_1_4_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev1_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev1 + dist_diff_abs_pair_c25_1_4_prev1
   + 240 dist_diff_abs_pair_c25_1_4_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev1 + dist_diff_abs_pair_c25_1_4_prev1
   + 240 dist_diff_abs_pair_c25_1_4_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev1 + dist_diff_abs_pair_c25_1_4_prev1
   - 240 dist_diff_abs_pair_c25_1_4_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev1_force_positive:
   dist_diff_pair_c25_1_4_prev1
   - 240 dist_diff_abs_pair_c25_1_4_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev1_force_negative:
   dist_diff_pair_c25_1_4_prev1
   - 240 dist_diff_abs_pair_c25_1_4_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev1: 240 same_dist_pair_c25_1_4_prev1
   + dist_diff_abs_pair_c25_1_4_prev1 <= 249.8
 same_dist_pair_lower_c25_1_4_prev1: 240 same_dist_pair_c25_1_4_prev1
   + dist_diff_abs_pair_c25_1_4_prev1 >= 10
 dist_diff_pair_def_c25_1_4_prev2: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev2 = -50
 dist_diff_abs_pair_c25_1_4_prev2_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev2 + dist_diff_abs_pair_c25_1_4_prev2
   - 240 dist_diff_abs_pair_c25_1_4_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev2_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev2 + dist_diff_abs_pair_c25_1_4_prev2
   + 240 dist_diff_abs_pair_c25_1_4_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev2 + dist_diff_abs_pair_c25_1_4_prev2
   + 240 dist_diff_abs_pair_c25_1_4_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev2 + dist_diff_abs_pair_c25_1_4_prev2
   - 240 dist_diff_abs_pair_c25_1_4_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev2_force_positive:
   dist_diff_pair_c25_1_4_prev2
   - 240 dist_diff_abs_pair_c25_1_4_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev2_force_negative:
   dist_diff_pair_c25_1_4_prev2
   - 240 dist_diff_abs_pair_c25_1_4_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev2: 240 same_dist_pair_c25_1_4_prev2
   + dist_diff_abs_pair_c25_1_4_prev2 <= 249.8
 same_dist_pair_lower_c25_1_4_prev2: 240 same_dist_pair_c25_1_4_prev2
   + dist_diff_abs_pair_c25_1_4_prev2 >= 10
 dist_diff_pair_def_c25_1_4_prev3: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev3 = -53
 dist_diff_abs_pair_c25_1_4_prev3_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev3 + dist_diff_abs_pair_c25_1_4_prev3
   - 240 dist_diff_abs_pair_c25_1_4_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev3_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev3 + dist_diff_abs_pair_c25_1_4_prev3
   + 240 dist_diff_abs_pair_c25_1_4_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev3 + dist_diff_abs_pair_c25_1_4_prev3
   + 240 dist_diff_abs_pair_c25_1_4_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev3 + dist_diff_abs_pair_c25_1_4_prev3
   - 240 dist_diff_abs_pair_c25_1_4_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev3_force_positive:
   dist_diff_pair_c25_1_4_prev3
   - 240 dist_diff_abs_pair_c25_1_4_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev3_force_negative:
   dist_diff_pair_c25_1_4_prev3
   - 240 dist_diff_abs_pair_c25_1_4_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev3: 240 same_dist_pair_c25_1_4_prev3
   + dist_diff_abs_pair_c25_1_4_prev3 <= 249.8
 same_dist_pair_lower_c25_1_4_prev3: 240 same_dist_pair_c25_1_4_prev3
   + dist_diff_abs_pair_c25_1_4_prev3 >= 10
 dist_diff_pair_def_c25_1_4_prev4: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev4 = -48
 dist_diff_abs_pair_c25_1_4_prev4_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev4 + dist_diff_abs_pair_c25_1_4_prev4
   - 240 dist_diff_abs_pair_c25_1_4_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev4_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev4 + dist_diff_abs_pair_c25_1_4_prev4
   + 240 dist_diff_abs_pair_c25_1_4_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev4 + dist_diff_abs_pair_c25_1_4_prev4
   + 240 dist_diff_abs_pair_c25_1_4_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev4 + dist_diff_abs_pair_c25_1_4_prev4
   - 240 dist_diff_abs_pair_c25_1_4_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev4_force_positive:
   dist_diff_pair_c25_1_4_prev4
   - 240 dist_diff_abs_pair_c25_1_4_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev4_force_negative:
   dist_diff_pair_c25_1_4_prev4
   - 240 dist_diff_abs_pair_c25_1_4_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev4: 240 same_dist_pair_c25_1_4_prev4
   + dist_diff_abs_pair_c25_1_4_prev4 <= 249.8
 same_dist_pair_lower_c25_1_4_prev4: 240 same_dist_pair_c25_1_4_prev4
   + dist_diff_abs_pair_c25_1_4_prev4 >= 10
 dist_diff_pair_def_c25_1_4_prev5: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev5 = -44
 dist_diff_abs_pair_c25_1_4_prev5_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev5 + dist_diff_abs_pair_c25_1_4_prev5
   - 240 dist_diff_abs_pair_c25_1_4_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev5_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev5 + dist_diff_abs_pair_c25_1_4_prev5
   + 240 dist_diff_abs_pair_c25_1_4_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev5 + dist_diff_abs_pair_c25_1_4_prev5
   + 240 dist_diff_abs_pair_c25_1_4_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev5 + dist_diff_abs_pair_c25_1_4_prev5
   - 240 dist_diff_abs_pair_c25_1_4_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev5_force_positive:
   dist_diff_pair_c25_1_4_prev5
   - 240 dist_diff_abs_pair_c25_1_4_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev5_force_negative:
   dist_diff_pair_c25_1_4_prev5
   - 240 dist_diff_abs_pair_c25_1_4_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev5: 240 same_dist_pair_c25_1_4_prev5
   + dist_diff_abs_pair_c25_1_4_prev5 <= 249.8
 same_dist_pair_lower_c25_1_4_prev5: 240 same_dist_pair_c25_1_4_prev5
   + dist_diff_abs_pair_c25_1_4_prev5 >= 10
 dist_diff_pair_def_c25_1_4_prev6: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev6 = -54
 dist_diff_abs_pair_c25_1_4_prev6_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev6 + dist_diff_abs_pair_c25_1_4_prev6
   - 240 dist_diff_abs_pair_c25_1_4_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev6_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev6 + dist_diff_abs_pair_c25_1_4_prev6
   + 240 dist_diff_abs_pair_c25_1_4_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev6 + dist_diff_abs_pair_c25_1_4_prev6
   + 240 dist_diff_abs_pair_c25_1_4_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev6 + dist_diff_abs_pair_c25_1_4_prev6
   - 240 dist_diff_abs_pair_c25_1_4_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev6_force_positive:
   dist_diff_pair_c25_1_4_prev6
   - 240 dist_diff_abs_pair_c25_1_4_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev6_force_negative:
   dist_diff_pair_c25_1_4_prev6
   - 240 dist_diff_abs_pair_c25_1_4_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev6: 240 same_dist_pair_c25_1_4_prev6
   + dist_diff_abs_pair_c25_1_4_prev6 <= 249.8
 same_dist_pair_lower_c25_1_4_prev6: 240 same_dist_pair_c25_1_4_prev6
   + dist_diff_abs_pair_c25_1_4_prev6 >= 10
 dist_diff_pair_def_c25_1_4_prev7: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev7 = -56
 dist_diff_abs_pair_c25_1_4_prev7_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev7 + dist_diff_abs_pair_c25_1_4_prev7
   - 240 dist_diff_abs_pair_c25_1_4_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev7_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev7 + dist_diff_abs_pair_c25_1_4_prev7
   + 240 dist_diff_abs_pair_c25_1_4_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev7 + dist_diff_abs_pair_c25_1_4_prev7
   + 240 dist_diff_abs_pair_c25_1_4_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev7 + dist_diff_abs_pair_c25_1_4_prev7
   - 240 dist_diff_abs_pair_c25_1_4_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev7_force_positive:
   dist_diff_pair_c25_1_4_prev7
   - 240 dist_diff_abs_pair_c25_1_4_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev7_force_negative:
   dist_diff_pair_c25_1_4_prev7
   - 240 dist_diff_abs_pair_c25_1_4_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev7: 240 same_dist_pair_c25_1_4_prev7
   + dist_diff_abs_pair_c25_1_4_prev7 <= 249.8
 same_dist_pair_lower_c25_1_4_prev7: 240 same_dist_pair_c25_1_4_prev7
   + dist_diff_abs_pair_c25_1_4_prev7 >= 10
 dist_diff_pair_def_c25_1_4_prev8: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev8 = -53
 dist_diff_abs_pair_c25_1_4_prev8_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev8 + dist_diff_abs_pair_c25_1_4_prev8
   - 240 dist_diff_abs_pair_c25_1_4_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev8_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev8 + dist_diff_abs_pair_c25_1_4_prev8
   + 240 dist_diff_abs_pair_c25_1_4_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev8 + dist_diff_abs_pair_c25_1_4_prev8
   + 240 dist_diff_abs_pair_c25_1_4_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev8 + dist_diff_abs_pair_c25_1_4_prev8
   - 240 dist_diff_abs_pair_c25_1_4_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev8_force_positive:
   dist_diff_pair_c25_1_4_prev8
   - 240 dist_diff_abs_pair_c25_1_4_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev8_force_negative:
   dist_diff_pair_c25_1_4_prev8
   - 240 dist_diff_abs_pair_c25_1_4_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev8: 240 same_dist_pair_c25_1_4_prev8
   + dist_diff_abs_pair_c25_1_4_prev8 <= 249.8
 same_dist_pair_lower_c25_1_4_prev8: 240 same_dist_pair_c25_1_4_prev8
   + dist_diff_abs_pair_c25_1_4_prev8 >= 10
 dist_diff_pair_def_c25_1_4_prev9: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev9 = -30
 dist_diff_abs_pair_c25_1_4_prev9_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev9 + dist_diff_abs_pair_c25_1_4_prev9
   - 240 dist_diff_abs_pair_c25_1_4_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev9_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev9 + dist_diff_abs_pair_c25_1_4_prev9
   + 240 dist_diff_abs_pair_c25_1_4_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev9 + dist_diff_abs_pair_c25_1_4_prev9
   + 240 dist_diff_abs_pair_c25_1_4_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev9 + dist_diff_abs_pair_c25_1_4_prev9
   - 240 dist_diff_abs_pair_c25_1_4_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev9_force_positive:
   dist_diff_pair_c25_1_4_prev9
   - 240 dist_diff_abs_pair_c25_1_4_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev9_force_negative:
   dist_diff_pair_c25_1_4_prev9
   - 240 dist_diff_abs_pair_c25_1_4_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev9: 240 same_dist_pair_c25_1_4_prev9
   + dist_diff_abs_pair_c25_1_4_prev9 <= 249.8
 same_dist_pair_lower_c25_1_4_prev9: 240 same_dist_pair_c25_1_4_prev9
   + dist_diff_abs_pair_c25_1_4_prev9 >= 10
 dist_diff_pair_def_c25_1_4_prev10: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev10 = -43
 dist_diff_abs_pair_c25_1_4_prev10_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev10 + dist_diff_abs_pair_c25_1_4_prev10
   - 240 dist_diff_abs_pair_c25_1_4_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev10_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev10 + dist_diff_abs_pair_c25_1_4_prev10
   + 240 dist_diff_abs_pair_c25_1_4_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev10 + dist_diff_abs_pair_c25_1_4_prev10
   + 240 dist_diff_abs_pair_c25_1_4_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev10 + dist_diff_abs_pair_c25_1_4_prev10
   - 240 dist_diff_abs_pair_c25_1_4_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev10_force_positive:
   dist_diff_pair_c25_1_4_prev10
   - 240 dist_diff_abs_pair_c25_1_4_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev10_force_negative:
   dist_diff_pair_c25_1_4_prev10
   - 240 dist_diff_abs_pair_c25_1_4_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev10: 240 same_dist_pair_c25_1_4_prev10
   + dist_diff_abs_pair_c25_1_4_prev10 <= 249.8
 same_dist_pair_lower_c25_1_4_prev10: 240 same_dist_pair_c25_1_4_prev10
   + dist_diff_abs_pair_c25_1_4_prev10 >= 10
 dist_diff_pair_def_c25_1_4_prev11: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev11 = -44
 dist_diff_abs_pair_c25_1_4_prev11_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev11 + dist_diff_abs_pair_c25_1_4_prev11
   - 240 dist_diff_abs_pair_c25_1_4_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev11_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev11 + dist_diff_abs_pair_c25_1_4_prev11
   + 240 dist_diff_abs_pair_c25_1_4_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev11 + dist_diff_abs_pair_c25_1_4_prev11
   + 240 dist_diff_abs_pair_c25_1_4_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev11 + dist_diff_abs_pair_c25_1_4_prev11
   - 240 dist_diff_abs_pair_c25_1_4_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev11_force_positive:
   dist_diff_pair_c25_1_4_prev11
   - 240 dist_diff_abs_pair_c25_1_4_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev11_force_negative:
   dist_diff_pair_c25_1_4_prev11
   - 240 dist_diff_abs_pair_c25_1_4_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev11: 240 same_dist_pair_c25_1_4_prev11
   + dist_diff_abs_pair_c25_1_4_prev11 <= 249.8
 same_dist_pair_lower_c25_1_4_prev11: 240 same_dist_pair_c25_1_4_prev11
   + dist_diff_abs_pair_c25_1_4_prev11 >= 10
 dist_diff_pair_def_c25_1_4_prev12: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev12 = -50
 dist_diff_abs_pair_c25_1_4_prev12_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev12 + dist_diff_abs_pair_c25_1_4_prev12
   - 240 dist_diff_abs_pair_c25_1_4_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev12_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev12 + dist_diff_abs_pair_c25_1_4_prev12
   + 240 dist_diff_abs_pair_c25_1_4_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev12 + dist_diff_abs_pair_c25_1_4_prev12
   + 240 dist_diff_abs_pair_c25_1_4_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev12 + dist_diff_abs_pair_c25_1_4_prev12
   - 240 dist_diff_abs_pair_c25_1_4_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev12_force_positive:
   dist_diff_pair_c25_1_4_prev12
   - 240 dist_diff_abs_pair_c25_1_4_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev12_force_negative:
   dist_diff_pair_c25_1_4_prev12
   - 240 dist_diff_abs_pair_c25_1_4_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev12: 240 same_dist_pair_c25_1_4_prev12
   + dist_diff_abs_pair_c25_1_4_prev12 <= 249.8
 same_dist_pair_lower_c25_1_4_prev12: 240 same_dist_pair_c25_1_4_prev12
   + dist_diff_abs_pair_c25_1_4_prev12 >= 10
 dist_diff_pair_def_c25_1_4_prev13: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev13 = -54
 dist_diff_abs_pair_c25_1_4_prev13_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev13 + dist_diff_abs_pair_c25_1_4_prev13
   - 240 dist_diff_abs_pair_c25_1_4_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev13_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev13 + dist_diff_abs_pair_c25_1_4_prev13
   + 240 dist_diff_abs_pair_c25_1_4_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev13 + dist_diff_abs_pair_c25_1_4_prev13
   + 240 dist_diff_abs_pair_c25_1_4_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev13 + dist_diff_abs_pair_c25_1_4_prev13
   - 240 dist_diff_abs_pair_c25_1_4_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev13_force_positive:
   dist_diff_pair_c25_1_4_prev13
   - 240 dist_diff_abs_pair_c25_1_4_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev13_force_negative:
   dist_diff_pair_c25_1_4_prev13
   - 240 dist_diff_abs_pair_c25_1_4_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev13: 240 same_dist_pair_c25_1_4_prev13
   + dist_diff_abs_pair_c25_1_4_prev13 <= 249.8
 same_dist_pair_lower_c25_1_4_prev13: 240 same_dist_pair_c25_1_4_prev13
   + dist_diff_abs_pair_c25_1_4_prev13 >= 10
 dist_diff_pair_def_c25_1_4_prev14: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev14 = -44
 dist_diff_abs_pair_c25_1_4_prev14_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev14 + dist_diff_abs_pair_c25_1_4_prev14
   - 240 dist_diff_abs_pair_c25_1_4_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev14_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev14 + dist_diff_abs_pair_c25_1_4_prev14
   + 240 dist_diff_abs_pair_c25_1_4_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev14 + dist_diff_abs_pair_c25_1_4_prev14
   + 240 dist_diff_abs_pair_c25_1_4_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev14 + dist_diff_abs_pair_c25_1_4_prev14
   - 240 dist_diff_abs_pair_c25_1_4_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev14_force_positive:
   dist_diff_pair_c25_1_4_prev14
   - 240 dist_diff_abs_pair_c25_1_4_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev14_force_negative:
   dist_diff_pair_c25_1_4_prev14
   - 240 dist_diff_abs_pair_c25_1_4_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev14: 240 same_dist_pair_c25_1_4_prev14
   + dist_diff_abs_pair_c25_1_4_prev14 <= 249.8
 same_dist_pair_lower_c25_1_4_prev14: 240 same_dist_pair_c25_1_4_prev14
   + dist_diff_abs_pair_c25_1_4_prev14 >= 10
 dist_diff_pair_def_c25_1_4_prev15: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev15 = -54
 dist_diff_abs_pair_c25_1_4_prev15_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev15 + dist_diff_abs_pair_c25_1_4_prev15
   - 240 dist_diff_abs_pair_c25_1_4_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev15_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev15 + dist_diff_abs_pair_c25_1_4_prev15
   + 240 dist_diff_abs_pair_c25_1_4_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev15 + dist_diff_abs_pair_c25_1_4_prev15
   + 240 dist_diff_abs_pair_c25_1_4_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev15 + dist_diff_abs_pair_c25_1_4_prev15
   - 240 dist_diff_abs_pair_c25_1_4_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev15_force_positive:
   dist_diff_pair_c25_1_4_prev15
   - 240 dist_diff_abs_pair_c25_1_4_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev15_force_negative:
   dist_diff_pair_c25_1_4_prev15
   - 240 dist_diff_abs_pair_c25_1_4_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev15: 240 same_dist_pair_c25_1_4_prev15
   + dist_diff_abs_pair_c25_1_4_prev15 <= 249.8
 same_dist_pair_lower_c25_1_4_prev15: 240 same_dist_pair_c25_1_4_prev15
   + dist_diff_abs_pair_c25_1_4_prev15 >= 10
 dist_diff_pair_def_c25_1_4_prev16: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev16 = -36
 dist_diff_abs_pair_c25_1_4_prev16_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev16 + dist_diff_abs_pair_c25_1_4_prev16
   - 240 dist_diff_abs_pair_c25_1_4_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev16_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev16 + dist_diff_abs_pair_c25_1_4_prev16
   + 240 dist_diff_abs_pair_c25_1_4_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev16 + dist_diff_abs_pair_c25_1_4_prev16
   + 240 dist_diff_abs_pair_c25_1_4_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev16 + dist_diff_abs_pair_c25_1_4_prev16
   - 240 dist_diff_abs_pair_c25_1_4_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev16_force_positive:
   dist_diff_pair_c25_1_4_prev16
   - 240 dist_diff_abs_pair_c25_1_4_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev16_force_negative:
   dist_diff_pair_c25_1_4_prev16
   - 240 dist_diff_abs_pair_c25_1_4_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev16: 240 same_dist_pair_c25_1_4_prev16
   + dist_diff_abs_pair_c25_1_4_prev16 <= 249.8
 same_dist_pair_lower_c25_1_4_prev16: 240 same_dist_pair_c25_1_4_prev16
   + dist_diff_abs_pair_c25_1_4_prev16 >= 10
 dist_diff_pair_def_c25_1_4_prev17: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev17 = -30
 dist_diff_abs_pair_c25_1_4_prev17_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev17 + dist_diff_abs_pair_c25_1_4_prev17
   - 240 dist_diff_abs_pair_c25_1_4_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev17_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev17 + dist_diff_abs_pair_c25_1_4_prev17
   + 240 dist_diff_abs_pair_c25_1_4_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev17 + dist_diff_abs_pair_c25_1_4_prev17
   + 240 dist_diff_abs_pair_c25_1_4_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev17 + dist_diff_abs_pair_c25_1_4_prev17
   - 240 dist_diff_abs_pair_c25_1_4_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev17_force_positive:
   dist_diff_pair_c25_1_4_prev17
   - 240 dist_diff_abs_pair_c25_1_4_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev17_force_negative:
   dist_diff_pair_c25_1_4_prev17
   - 240 dist_diff_abs_pair_c25_1_4_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev17: 240 same_dist_pair_c25_1_4_prev17
   + dist_diff_abs_pair_c25_1_4_prev17 <= 249.8
 same_dist_pair_lower_c25_1_4_prev17: 240 same_dist_pair_c25_1_4_prev17
   + dist_diff_abs_pair_c25_1_4_prev17 >= 10
 dist_diff_pair_def_c25_1_4_prev18: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev18 = -50
 dist_diff_abs_pair_c25_1_4_prev18_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev18 + dist_diff_abs_pair_c25_1_4_prev18
   - 240 dist_diff_abs_pair_c25_1_4_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev18_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev18 + dist_diff_abs_pair_c25_1_4_prev18
   + 240 dist_diff_abs_pair_c25_1_4_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev18 + dist_diff_abs_pair_c25_1_4_prev18
   + 240 dist_diff_abs_pair_c25_1_4_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev18 + dist_diff_abs_pair_c25_1_4_prev18
   - 240 dist_diff_abs_pair_c25_1_4_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev18_force_positive:
   dist_diff_pair_c25_1_4_prev18
   - 240 dist_diff_abs_pair_c25_1_4_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev18_force_negative:
   dist_diff_pair_c25_1_4_prev18
   - 240 dist_diff_abs_pair_c25_1_4_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev18: 240 same_dist_pair_c25_1_4_prev18
   + dist_diff_abs_pair_c25_1_4_prev18 <= 249.8
 same_dist_pair_lower_c25_1_4_prev18: 240 same_dist_pair_c25_1_4_prev18
   + dist_diff_abs_pair_c25_1_4_prev18 >= 10
 dist_diff_pair_def_c25_1_4_prev19: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev19 = -30
 dist_diff_abs_pair_c25_1_4_prev19_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev19 + dist_diff_abs_pair_c25_1_4_prev19
   - 240 dist_diff_abs_pair_c25_1_4_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev19_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev19 + dist_diff_abs_pair_c25_1_4_prev19
   + 240 dist_diff_abs_pair_c25_1_4_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev19 + dist_diff_abs_pair_c25_1_4_prev19
   + 240 dist_diff_abs_pair_c25_1_4_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev19 + dist_diff_abs_pair_c25_1_4_prev19
   - 240 dist_diff_abs_pair_c25_1_4_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev19_force_positive:
   dist_diff_pair_c25_1_4_prev19
   - 240 dist_diff_abs_pair_c25_1_4_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev19_force_negative:
   dist_diff_pair_c25_1_4_prev19
   - 240 dist_diff_abs_pair_c25_1_4_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev19: 240 same_dist_pair_c25_1_4_prev19
   + dist_diff_abs_pair_c25_1_4_prev19 <= 249.8
 same_dist_pair_lower_c25_1_4_prev19: 240 same_dist_pair_c25_1_4_prev19
   + dist_diff_abs_pair_c25_1_4_prev19 >= 10
 dist_diff_pair_def_c25_1_4_prev20: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev20 = -20
 dist_diff_abs_pair_c25_1_4_prev20_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev20 + dist_diff_abs_pair_c25_1_4_prev20
   - 240 dist_diff_abs_pair_c25_1_4_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev20_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev20 + dist_diff_abs_pair_c25_1_4_prev20
   + 240 dist_diff_abs_pair_c25_1_4_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev20 + dist_diff_abs_pair_c25_1_4_prev20
   + 240 dist_diff_abs_pair_c25_1_4_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev20 + dist_diff_abs_pair_c25_1_4_prev20
   - 240 dist_diff_abs_pair_c25_1_4_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev20_force_positive:
   dist_diff_pair_c25_1_4_prev20
   - 240 dist_diff_abs_pair_c25_1_4_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev20_force_negative:
   dist_diff_pair_c25_1_4_prev20
   - 240 dist_diff_abs_pair_c25_1_4_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev20: 240 same_dist_pair_c25_1_4_prev20
   + dist_diff_abs_pair_c25_1_4_prev20 <= 249.8
 same_dist_pair_lower_c25_1_4_prev20: 240 same_dist_pair_c25_1_4_prev20
   + dist_diff_abs_pair_c25_1_4_prev20 >= 10
 dist_diff_pair_def_c25_1_4_prev21: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev21 = -30
 dist_diff_abs_pair_c25_1_4_prev21_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev21 + dist_diff_abs_pair_c25_1_4_prev21
   - 240 dist_diff_abs_pair_c25_1_4_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev21_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev21 + dist_diff_abs_pair_c25_1_4_prev21
   + 240 dist_diff_abs_pair_c25_1_4_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev21 + dist_diff_abs_pair_c25_1_4_prev21
   + 240 dist_diff_abs_pair_c25_1_4_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev21 + dist_diff_abs_pair_c25_1_4_prev21
   - 240 dist_diff_abs_pair_c25_1_4_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev21_force_positive:
   dist_diff_pair_c25_1_4_prev21
   - 240 dist_diff_abs_pair_c25_1_4_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev21_force_negative:
   dist_diff_pair_c25_1_4_prev21
   - 240 dist_diff_abs_pair_c25_1_4_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev21: 240 same_dist_pair_c25_1_4_prev21
   + dist_diff_abs_pair_c25_1_4_prev21 <= 249.8
 same_dist_pair_lower_c25_1_4_prev21: 240 same_dist_pair_c25_1_4_prev21
   + dist_diff_abs_pair_c25_1_4_prev21 >= 10
 dist_diff_pair_def_c25_1_4_prev22: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev22 = -35
 dist_diff_abs_pair_c25_1_4_prev22_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev22 + dist_diff_abs_pair_c25_1_4_prev22
   - 240 dist_diff_abs_pair_c25_1_4_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev22_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev22 + dist_diff_abs_pair_c25_1_4_prev22
   + 240 dist_diff_abs_pair_c25_1_4_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev22 + dist_diff_abs_pair_c25_1_4_prev22
   + 240 dist_diff_abs_pair_c25_1_4_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev22 + dist_diff_abs_pair_c25_1_4_prev22
   - 240 dist_diff_abs_pair_c25_1_4_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev22_force_positive:
   dist_diff_pair_c25_1_4_prev22
   - 240 dist_diff_abs_pair_c25_1_4_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev22_force_negative:
   dist_diff_pair_c25_1_4_prev22
   - 240 dist_diff_abs_pair_c25_1_4_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev22: 240 same_dist_pair_c25_1_4_prev22
   + dist_diff_abs_pair_c25_1_4_prev22 <= 249.8
 same_dist_pair_lower_c25_1_4_prev22: 240 same_dist_pair_c25_1_4_prev22
   + dist_diff_abs_pair_c25_1_4_prev22 >= 10
 dist_diff_pair_def_c25_1_4_prev23: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev23 = -46
 dist_diff_abs_pair_c25_1_4_prev23_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev23 + dist_diff_abs_pair_c25_1_4_prev23
   - 240 dist_diff_abs_pair_c25_1_4_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev23_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev23 + dist_diff_abs_pair_c25_1_4_prev23
   + 240 dist_diff_abs_pair_c25_1_4_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev23 + dist_diff_abs_pair_c25_1_4_prev23
   + 240 dist_diff_abs_pair_c25_1_4_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev23 + dist_diff_abs_pair_c25_1_4_prev23
   - 240 dist_diff_abs_pair_c25_1_4_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev23_force_positive:
   dist_diff_pair_c25_1_4_prev23
   - 240 dist_diff_abs_pair_c25_1_4_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev23_force_negative:
   dist_diff_pair_c25_1_4_prev23
   - 240 dist_diff_abs_pair_c25_1_4_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev23: 240 same_dist_pair_c25_1_4_prev23
   + dist_diff_abs_pair_c25_1_4_prev23 <= 249.8
 same_dist_pair_lower_c25_1_4_prev23: 240 same_dist_pair_c25_1_4_prev23
   + dist_diff_abs_pair_c25_1_4_prev23 >= 10
 dist_diff_pair_def_c25_1_4_prev24: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev24 = -66
 dist_diff_abs_pair_c25_1_4_prev24_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev24 + dist_diff_abs_pair_c25_1_4_prev24
   - 240 dist_diff_abs_pair_c25_1_4_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev24_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev24 + dist_diff_abs_pair_c25_1_4_prev24
   + 240 dist_diff_abs_pair_c25_1_4_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev24 + dist_diff_abs_pair_c25_1_4_prev24
   + 240 dist_diff_abs_pair_c25_1_4_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev24 + dist_diff_abs_pair_c25_1_4_prev24
   - 240 dist_diff_abs_pair_c25_1_4_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev24_force_positive:
   dist_diff_pair_c25_1_4_prev24
   - 240 dist_diff_abs_pair_c25_1_4_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev24_force_negative:
   dist_diff_pair_c25_1_4_prev24
   - 240 dist_diff_abs_pair_c25_1_4_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev24: 240 same_dist_pair_c25_1_4_prev24
   + dist_diff_abs_pair_c25_1_4_prev24 <= 249.8
 same_dist_pair_lower_c25_1_4_prev24: 240 same_dist_pair_c25_1_4_prev24
   + dist_diff_abs_pair_c25_1_4_prev24 >= 10
 dist_diff_pair_def_c25_1_4_prev25: - dist_curr_pair_c25_1_4
   + dist_diff_pair_c25_1_4_prev25 = -76
 dist_diff_abs_pair_c25_1_4_prev25_abs_ge_orig:
   - dist_diff_pair_c25_1_4_prev25 + dist_diff_abs_pair_c25_1_4_prev25
   - 240 dist_diff_abs_pair_c25_1_4_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev25_abs_le_orig:
   - dist_diff_pair_c25_1_4_prev25 + dist_diff_abs_pair_c25_1_4_prev25
   + 240 dist_diff_abs_pair_c25_1_4_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_1_4_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_1_4_prev25 + dist_diff_abs_pair_c25_1_4_prev25
   + 240 dist_diff_abs_pair_c25_1_4_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_1_4_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_1_4_prev25 + dist_diff_abs_pair_c25_1_4_prev25
   - 240 dist_diff_abs_pair_c25_1_4_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_1_4_prev25_force_positive:
   dist_diff_pair_c25_1_4_prev25
   - 240 dist_diff_abs_pair_c25_1_4_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_1_4_prev25_force_negative:
   dist_diff_pair_c25_1_4_prev25
   - 240 dist_diff_abs_pair_c25_1_4_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_1_4_prev25: 240 same_dist_pair_c25_1_4_prev25
   + dist_diff_abs_pair_c25_1_4_prev25 <= 249.8
 same_dist_pair_lower_c25_1_4_prev25: 240 same_dist_pair_c25_1_4_prev25
   + dist_diff_abs_pair_c25_1_4_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_1_4_prev0: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev0: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev1: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev1: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev2: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev2: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev3: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev3: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev4: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev4: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev5: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev5: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev6: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev6: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev7: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev7: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev8: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev8: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev9: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev9: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev10: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev10: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev11: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev11: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev12: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev12: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev13: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev13: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev14: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev14: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev15: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev15: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev16: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev16: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev17: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev17: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev18: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev18: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev19: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev19: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev20: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev20: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev21: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev21: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev22: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev22: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev23: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev23: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev24: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev24: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_1_4_prev25: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_1_4_prev25: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_1_4: diff_dist_pair_c25_1_4
   + same_dist_pair_c25_1_4_prev0 + same_dist_pair_c25_1_4_prev1
   + same_dist_pair_c25_1_4_prev2 + same_dist_pair_c25_1_4_prev3
   + same_dist_pair_c25_1_4_prev4 + same_dist_pair_c25_1_4_prev5
   + same_dist_pair_c25_1_4_prev6 + same_dist_pair_c25_1_4_prev7
   + same_dist_pair_c25_1_4_prev8 + same_dist_pair_c25_1_4_prev9
   + same_dist_pair_c25_1_4_prev10 + same_dist_pair_c25_1_4_prev11
   + same_dist_pair_c25_1_4_prev12 + same_dist_pair_c25_1_4_prev13
   + same_dist_pair_c25_1_4_prev14 + same_dist_pair_c25_1_4_prev15
   + same_dist_pair_c25_1_4_prev16 + same_dist_pair_c25_1_4_prev17
   + same_dist_pair_c25_1_4_prev18 + same_dist_pair_c25_1_4_prev19
   + same_dist_pair_c25_1_4_prev20 + same_dist_pair_c25_1_4_prev21
   + same_dist_pair_c25_1_4_prev22 + same_dist_pair_c25_1_4_prev23
   + same_dist_pair_c25_1_4_prev24 + same_dist_pair_c25_1_4_prev25 >= 1
 dx_grid_diff_def_c25_2_3: - cx_grid_center_c25_2 + cx_grid_center_c25_3
   + dx_grid_diff_c25_2_3 = 0
 dx_grid_abs_pair_c25_2_3_abs_ge_orig: dx_grid_abs_pair_c25_2_3
   - dx_grid_diff_c25_2_3 - 130 dx_grid_abs_pair_c25_2_3_is_positive
   >= -130
 dx_grid_abs_pair_c25_2_3_abs_le_orig: dx_grid_abs_pair_c25_2_3
   - dx_grid_diff_c25_2_3 + 130 dx_grid_abs_pair_c25_2_3_is_positive
   <= 130
 dx_grid_abs_pair_c25_2_3_abs_ge_neg_orig: dx_grid_abs_pair_c25_2_3
   + dx_grid_diff_c25_2_3 + 130 dx_grid_abs_pair_c25_2_3_is_positive >= 0
 dx_grid_abs_pair_c25_2_3_abs_le_neg_orig: dx_grid_abs_pair_c25_2_3
   + dx_grid_diff_c25_2_3 - 130 dx_grid_abs_pair_c25_2_3_is_positive <= 0
 dx_grid_abs_pair_c25_2_3_force_positive: dx_grid_diff_c25_2_3
   - 130 dx_grid_abs_pair_c25_2_3_is_positive >= -130
 dx_grid_abs_pair_c25_2_3_force_negative: dx_grid_diff_c25_2_3
   - 130 dx_grid_abs_pair_c25_2_3_is_positive <= -0.001
 dy_grid_diff_def_c25_2_3: - cy_grid_center_c25_2 + cy_grid_center_c25_3
   + dy_grid_diff_c25_2_3 = 0
 dy_grid_abs_pair_c25_2_3_abs_ge_orig: dy_grid_abs_pair_c25_2_3
   - dy_grid_diff_c25_2_3 - 130 dy_grid_abs_pair_c25_2_3_is_positive
   >= -130
 dy_grid_abs_pair_c25_2_3_abs_le_orig: dy_grid_abs_pair_c25_2_3
   - dy_grid_diff_c25_2_3 + 130 dy_grid_abs_pair_c25_2_3_is_positive
   <= 130
 dy_grid_abs_pair_c25_2_3_abs_ge_neg_orig: dy_grid_abs_pair_c25_2_3
   + dy_grid_diff_c25_2_3 + 130 dy_grid_abs_pair_c25_2_3_is_positive >= 0
 dy_grid_abs_pair_c25_2_3_abs_le_neg_orig: dy_grid_abs_pair_c25_2_3
   + dy_grid_diff_c25_2_3 - 130 dy_grid_abs_pair_c25_2_3_is_positive <= 0
 dy_grid_abs_pair_c25_2_3_force_positive: dy_grid_diff_c25_2_3
   - 130 dy_grid_abs_pair_c25_2_3_is_positive >= -130
 dy_grid_abs_pair_c25_2_3_force_negative: dy_grid_diff_c25_2_3
   - 130 dy_grid_abs_pair_c25_2_3_is_positive <= -0.001
 dist_curr_pair_def_c25_2_3: - dx_grid_abs_pair_c25_2_3
   - dy_grid_abs_pair_c25_2_3 + dist_curr_pair_c25_2_3 = 0
 dist_diff_pair_def_c25_2_3_prev0: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev0 = -64
 dist_diff_abs_pair_c25_2_3_prev0_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev0 + dist_diff_abs_pair_c25_2_3_prev0
   - 240 dist_diff_abs_pair_c25_2_3_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev0_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev0 + dist_diff_abs_pair_c25_2_3_prev0
   + 240 dist_diff_abs_pair_c25_2_3_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev0 + dist_diff_abs_pair_c25_2_3_prev0
   + 240 dist_diff_abs_pair_c25_2_3_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev0 + dist_diff_abs_pair_c25_2_3_prev0
   - 240 dist_diff_abs_pair_c25_2_3_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev0_force_positive:
   dist_diff_pair_c25_2_3_prev0
   - 240 dist_diff_abs_pair_c25_2_3_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev0_force_negative:
   dist_diff_pair_c25_2_3_prev0
   - 240 dist_diff_abs_pair_c25_2_3_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev0: 240 same_dist_pair_c25_2_3_prev0
   + dist_diff_abs_pair_c25_2_3_prev0 <= 249.8
 same_dist_pair_lower_c25_2_3_prev0: 240 same_dist_pair_c25_2_3_prev0
   + dist_diff_abs_pair_c25_2_3_prev0 >= 10
 dist_diff_pair_def_c25_2_3_prev1: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev1 = -62
 dist_diff_abs_pair_c25_2_3_prev1_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev1 + dist_diff_abs_pair_c25_2_3_prev1
   - 240 dist_diff_abs_pair_c25_2_3_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev1_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev1 + dist_diff_abs_pair_c25_2_3_prev1
   + 240 dist_diff_abs_pair_c25_2_3_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev1 + dist_diff_abs_pair_c25_2_3_prev1
   + 240 dist_diff_abs_pair_c25_2_3_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev1 + dist_diff_abs_pair_c25_2_3_prev1
   - 240 dist_diff_abs_pair_c25_2_3_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev1_force_positive:
   dist_diff_pair_c25_2_3_prev1
   - 240 dist_diff_abs_pair_c25_2_3_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev1_force_negative:
   dist_diff_pair_c25_2_3_prev1
   - 240 dist_diff_abs_pair_c25_2_3_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev1: 240 same_dist_pair_c25_2_3_prev1
   + dist_diff_abs_pair_c25_2_3_prev1 <= 249.8
 same_dist_pair_lower_c25_2_3_prev1: 240 same_dist_pair_c25_2_3_prev1
   + dist_diff_abs_pair_c25_2_3_prev1 >= 10
 dist_diff_pair_def_c25_2_3_prev2: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev2 = -62
 dist_diff_abs_pair_c25_2_3_prev2_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev2 + dist_diff_abs_pair_c25_2_3_prev2
   - 240 dist_diff_abs_pair_c25_2_3_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev2_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev2 + dist_diff_abs_pair_c25_2_3_prev2
   + 240 dist_diff_abs_pair_c25_2_3_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev2 + dist_diff_abs_pair_c25_2_3_prev2
   + 240 dist_diff_abs_pair_c25_2_3_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev2 + dist_diff_abs_pair_c25_2_3_prev2
   - 240 dist_diff_abs_pair_c25_2_3_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev2_force_positive:
   dist_diff_pair_c25_2_3_prev2
   - 240 dist_diff_abs_pair_c25_2_3_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev2_force_negative:
   dist_diff_pair_c25_2_3_prev2
   - 240 dist_diff_abs_pair_c25_2_3_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev2: 240 same_dist_pair_c25_2_3_prev2
   + dist_diff_abs_pair_c25_2_3_prev2 <= 249.8
 same_dist_pair_lower_c25_2_3_prev2: 240 same_dist_pair_c25_2_3_prev2
   + dist_diff_abs_pair_c25_2_3_prev2 >= 10
 dist_diff_pair_def_c25_2_3_prev3: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev3 = -71
 dist_diff_abs_pair_c25_2_3_prev3_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev3 + dist_diff_abs_pair_c25_2_3_prev3
   - 240 dist_diff_abs_pair_c25_2_3_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev3_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev3 + dist_diff_abs_pair_c25_2_3_prev3
   + 240 dist_diff_abs_pair_c25_2_3_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev3 + dist_diff_abs_pair_c25_2_3_prev3
   + 240 dist_diff_abs_pair_c25_2_3_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev3 + dist_diff_abs_pair_c25_2_3_prev3
   - 240 dist_diff_abs_pair_c25_2_3_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev3_force_positive:
   dist_diff_pair_c25_2_3_prev3
   - 240 dist_diff_abs_pair_c25_2_3_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev3_force_negative:
   dist_diff_pair_c25_2_3_prev3
   - 240 dist_diff_abs_pair_c25_2_3_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev3: 240 same_dist_pair_c25_2_3_prev3
   + dist_diff_abs_pair_c25_2_3_prev3 <= 249.8
 same_dist_pair_lower_c25_2_3_prev3: 240 same_dist_pair_c25_2_3_prev3
   + dist_diff_abs_pair_c25_2_3_prev3 >= 10
 dist_diff_pair_def_c25_2_3_prev4: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev4 = -62
 dist_diff_abs_pair_c25_2_3_prev4_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev4 + dist_diff_abs_pair_c25_2_3_prev4
   - 240 dist_diff_abs_pair_c25_2_3_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev4_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev4 + dist_diff_abs_pair_c25_2_3_prev4
   + 240 dist_diff_abs_pair_c25_2_3_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev4 + dist_diff_abs_pair_c25_2_3_prev4
   + 240 dist_diff_abs_pair_c25_2_3_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev4 + dist_diff_abs_pair_c25_2_3_prev4
   - 240 dist_diff_abs_pair_c25_2_3_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev4_force_positive:
   dist_diff_pair_c25_2_3_prev4
   - 240 dist_diff_abs_pair_c25_2_3_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev4_force_negative:
   dist_diff_pair_c25_2_3_prev4
   - 240 dist_diff_abs_pair_c25_2_3_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev4: 240 same_dist_pair_c25_2_3_prev4
   + dist_diff_abs_pair_c25_2_3_prev4 <= 249.8
 same_dist_pair_lower_c25_2_3_prev4: 240 same_dist_pair_c25_2_3_prev4
   + dist_diff_abs_pair_c25_2_3_prev4 >= 10
 dist_diff_pair_def_c25_2_3_prev5: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev5 = -64
 dist_diff_abs_pair_c25_2_3_prev5_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev5 + dist_diff_abs_pair_c25_2_3_prev5
   - 240 dist_diff_abs_pair_c25_2_3_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev5_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev5 + dist_diff_abs_pair_c25_2_3_prev5
   + 240 dist_diff_abs_pair_c25_2_3_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev5 + dist_diff_abs_pair_c25_2_3_prev5
   + 240 dist_diff_abs_pair_c25_2_3_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev5 + dist_diff_abs_pair_c25_2_3_prev5
   - 240 dist_diff_abs_pair_c25_2_3_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev5_force_positive:
   dist_diff_pair_c25_2_3_prev5
   - 240 dist_diff_abs_pair_c25_2_3_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev5_force_negative:
   dist_diff_pair_c25_2_3_prev5
   - 240 dist_diff_abs_pair_c25_2_3_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev5: 240 same_dist_pair_c25_2_3_prev5
   + dist_diff_abs_pair_c25_2_3_prev5 <= 249.8
 same_dist_pair_lower_c25_2_3_prev5: 240 same_dist_pair_c25_2_3_prev5
   + dist_diff_abs_pair_c25_2_3_prev5 >= 10
 dist_diff_pair_def_c25_2_3_prev6: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev6 = -74
 dist_diff_abs_pair_c25_2_3_prev6_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev6 + dist_diff_abs_pair_c25_2_3_prev6
   - 240 dist_diff_abs_pair_c25_2_3_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev6_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev6 + dist_diff_abs_pair_c25_2_3_prev6
   + 240 dist_diff_abs_pair_c25_2_3_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev6 + dist_diff_abs_pair_c25_2_3_prev6
   + 240 dist_diff_abs_pair_c25_2_3_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev6 + dist_diff_abs_pair_c25_2_3_prev6
   - 240 dist_diff_abs_pair_c25_2_3_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev6_force_positive:
   dist_diff_pair_c25_2_3_prev6
   - 240 dist_diff_abs_pair_c25_2_3_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev6_force_negative:
   dist_diff_pair_c25_2_3_prev6
   - 240 dist_diff_abs_pair_c25_2_3_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev6: 240 same_dist_pair_c25_2_3_prev6
   + dist_diff_abs_pair_c25_2_3_prev6 <= 249.8
 same_dist_pair_lower_c25_2_3_prev6: 240 same_dist_pair_c25_2_3_prev6
   + dist_diff_abs_pair_c25_2_3_prev6 >= 10
 dist_diff_pair_def_c25_2_3_prev7: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev7 = -52
 dist_diff_abs_pair_c25_2_3_prev7_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev7 + dist_diff_abs_pair_c25_2_3_prev7
   - 240 dist_diff_abs_pair_c25_2_3_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev7_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev7 + dist_diff_abs_pair_c25_2_3_prev7
   + 240 dist_diff_abs_pair_c25_2_3_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev7 + dist_diff_abs_pair_c25_2_3_prev7
   + 240 dist_diff_abs_pair_c25_2_3_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev7 + dist_diff_abs_pair_c25_2_3_prev7
   - 240 dist_diff_abs_pair_c25_2_3_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev7_force_positive:
   dist_diff_pair_c25_2_3_prev7
   - 240 dist_diff_abs_pair_c25_2_3_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev7_force_negative:
   dist_diff_pair_c25_2_3_prev7
   - 240 dist_diff_abs_pair_c25_2_3_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev7: 240 same_dist_pair_c25_2_3_prev7
   + dist_diff_abs_pair_c25_2_3_prev7 <= 249.8
 same_dist_pair_lower_c25_2_3_prev7: 240 same_dist_pair_c25_2_3_prev7
   + dist_diff_abs_pair_c25_2_3_prev7 >= 10
 dist_diff_pair_def_c25_2_3_prev8: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev8 = -62
 dist_diff_abs_pair_c25_2_3_prev8_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev8 + dist_diff_abs_pair_c25_2_3_prev8
   - 240 dist_diff_abs_pair_c25_2_3_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev8_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev8 + dist_diff_abs_pair_c25_2_3_prev8
   + 240 dist_diff_abs_pair_c25_2_3_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev8 + dist_diff_abs_pair_c25_2_3_prev8
   + 240 dist_diff_abs_pair_c25_2_3_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev8 + dist_diff_abs_pair_c25_2_3_prev8
   - 240 dist_diff_abs_pair_c25_2_3_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev8_force_positive:
   dist_diff_pair_c25_2_3_prev8
   - 240 dist_diff_abs_pair_c25_2_3_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev8_force_negative:
   dist_diff_pair_c25_2_3_prev8
   - 240 dist_diff_abs_pair_c25_2_3_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev8: 240 same_dist_pair_c25_2_3_prev8
   + dist_diff_abs_pair_c25_2_3_prev8 <= 249.8
 same_dist_pair_lower_c25_2_3_prev8: 240 same_dist_pair_c25_2_3_prev8
   + dist_diff_abs_pair_c25_2_3_prev8 >= 10
 dist_diff_pair_def_c25_2_3_prev9: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev9 = -58
 dist_diff_abs_pair_c25_2_3_prev9_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev9 + dist_diff_abs_pair_c25_2_3_prev9
   - 240 dist_diff_abs_pair_c25_2_3_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev9_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev9 + dist_diff_abs_pair_c25_2_3_prev9
   + 240 dist_diff_abs_pair_c25_2_3_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev9 + dist_diff_abs_pair_c25_2_3_prev9
   + 240 dist_diff_abs_pair_c25_2_3_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev9 + dist_diff_abs_pair_c25_2_3_prev9
   - 240 dist_diff_abs_pair_c25_2_3_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev9_force_positive:
   dist_diff_pair_c25_2_3_prev9
   - 240 dist_diff_abs_pair_c25_2_3_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev9_force_negative:
   dist_diff_pair_c25_2_3_prev9
   - 240 dist_diff_abs_pair_c25_2_3_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev9: 240 same_dist_pair_c25_2_3_prev9
   + dist_diff_abs_pair_c25_2_3_prev9 <= 249.8
 same_dist_pair_lower_c25_2_3_prev9: 240 same_dist_pair_c25_2_3_prev9
   + dist_diff_abs_pair_c25_2_3_prev9 >= 10
 dist_diff_pair_def_c25_2_3_prev10: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev10 = -63
 dist_diff_abs_pair_c25_2_3_prev10_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev10 + dist_diff_abs_pair_c25_2_3_prev10
   - 240 dist_diff_abs_pair_c25_2_3_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev10_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev10 + dist_diff_abs_pair_c25_2_3_prev10
   + 240 dist_diff_abs_pair_c25_2_3_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev10 + dist_diff_abs_pair_c25_2_3_prev10
   + 240 dist_diff_abs_pair_c25_2_3_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev10 + dist_diff_abs_pair_c25_2_3_prev10
   - 240 dist_diff_abs_pair_c25_2_3_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev10_force_positive:
   dist_diff_pair_c25_2_3_prev10
   - 240 dist_diff_abs_pair_c25_2_3_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev10_force_negative:
   dist_diff_pair_c25_2_3_prev10
   - 240 dist_diff_abs_pair_c25_2_3_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev10: 240 same_dist_pair_c25_2_3_prev10
   + dist_diff_abs_pair_c25_2_3_prev10 <= 249.8
 same_dist_pair_lower_c25_2_3_prev10: 240 same_dist_pair_c25_2_3_prev10
   + dist_diff_abs_pair_c25_2_3_prev10 >= 10
 dist_diff_pair_def_c25_2_3_prev11: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev11 = -66
 dist_diff_abs_pair_c25_2_3_prev11_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev11 + dist_diff_abs_pair_c25_2_3_prev11
   - 240 dist_diff_abs_pair_c25_2_3_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev11_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev11 + dist_diff_abs_pair_c25_2_3_prev11
   + 240 dist_diff_abs_pair_c25_2_3_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev11 + dist_diff_abs_pair_c25_2_3_prev11
   + 240 dist_diff_abs_pair_c25_2_3_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev11 + dist_diff_abs_pair_c25_2_3_prev11
   - 240 dist_diff_abs_pair_c25_2_3_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev11_force_positive:
   dist_diff_pair_c25_2_3_prev11
   - 240 dist_diff_abs_pair_c25_2_3_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev11_force_negative:
   dist_diff_pair_c25_2_3_prev11
   - 240 dist_diff_abs_pair_c25_2_3_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev11: 240 same_dist_pair_c25_2_3_prev11
   + dist_diff_abs_pair_c25_2_3_prev11 <= 249.8
 same_dist_pair_lower_c25_2_3_prev11: 240 same_dist_pair_c25_2_3_prev11
   + dist_diff_abs_pair_c25_2_3_prev11 >= 10
 dist_diff_pair_def_c25_2_3_prev12: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev12 = -58
 dist_diff_abs_pair_c25_2_3_prev12_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev12 + dist_diff_abs_pair_c25_2_3_prev12
   - 240 dist_diff_abs_pair_c25_2_3_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev12_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev12 + dist_diff_abs_pair_c25_2_3_prev12
   + 240 dist_diff_abs_pair_c25_2_3_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev12 + dist_diff_abs_pair_c25_2_3_prev12
   + 240 dist_diff_abs_pair_c25_2_3_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev12 + dist_diff_abs_pair_c25_2_3_prev12
   - 240 dist_diff_abs_pair_c25_2_3_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev12_force_positive:
   dist_diff_pair_c25_2_3_prev12
   - 240 dist_diff_abs_pair_c25_2_3_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev12_force_negative:
   dist_diff_pair_c25_2_3_prev12
   - 240 dist_diff_abs_pair_c25_2_3_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev12: 240 same_dist_pair_c25_2_3_prev12
   + dist_diff_abs_pair_c25_2_3_prev12 <= 249.8
 same_dist_pair_lower_c25_2_3_prev12: 240 same_dist_pair_c25_2_3_prev12
   + dist_diff_abs_pair_c25_2_3_prev12 >= 10
 dist_diff_pair_def_c25_2_3_prev13: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev13 = -84
 dist_diff_abs_pair_c25_2_3_prev13_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev13 + dist_diff_abs_pair_c25_2_3_prev13
   - 240 dist_diff_abs_pair_c25_2_3_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev13_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev13 + dist_diff_abs_pair_c25_2_3_prev13
   + 240 dist_diff_abs_pair_c25_2_3_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev13 + dist_diff_abs_pair_c25_2_3_prev13
   + 240 dist_diff_abs_pair_c25_2_3_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev13 + dist_diff_abs_pair_c25_2_3_prev13
   - 240 dist_diff_abs_pair_c25_2_3_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev13_force_positive:
   dist_diff_pair_c25_2_3_prev13
   - 240 dist_diff_abs_pair_c25_2_3_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev13_force_negative:
   dist_diff_pair_c25_2_3_prev13
   - 240 dist_diff_abs_pair_c25_2_3_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev13: 240 same_dist_pair_c25_2_3_prev13
   + dist_diff_abs_pair_c25_2_3_prev13 <= 249.8
 same_dist_pair_lower_c25_2_3_prev13: 240 same_dist_pair_c25_2_3_prev13
   + dist_diff_abs_pair_c25_2_3_prev13 >= 10
 dist_diff_pair_def_c25_2_3_prev14: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev14 = -64
 dist_diff_abs_pair_c25_2_3_prev14_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev14 + dist_diff_abs_pair_c25_2_3_prev14
   - 240 dist_diff_abs_pair_c25_2_3_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev14_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev14 + dist_diff_abs_pair_c25_2_3_prev14
   + 240 dist_diff_abs_pair_c25_2_3_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev14 + dist_diff_abs_pair_c25_2_3_prev14
   + 240 dist_diff_abs_pair_c25_2_3_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev14 + dist_diff_abs_pair_c25_2_3_prev14
   - 240 dist_diff_abs_pair_c25_2_3_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev14_force_positive:
   dist_diff_pair_c25_2_3_prev14
   - 240 dist_diff_abs_pair_c25_2_3_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev14_force_negative:
   dist_diff_pair_c25_2_3_prev14
   - 240 dist_diff_abs_pair_c25_2_3_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev14: 240 same_dist_pair_c25_2_3_prev14
   + dist_diff_abs_pair_c25_2_3_prev14 <= 249.8
 same_dist_pair_lower_c25_2_3_prev14: 240 same_dist_pair_c25_2_3_prev14
   + dist_diff_abs_pair_c25_2_3_prev14 >= 10
 dist_diff_pair_def_c25_2_3_prev15: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev15 = -62
 dist_diff_abs_pair_c25_2_3_prev15_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev15 + dist_diff_abs_pair_c25_2_3_prev15
   - 240 dist_diff_abs_pair_c25_2_3_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev15_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev15 + dist_diff_abs_pair_c25_2_3_prev15
   + 240 dist_diff_abs_pair_c25_2_3_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev15 + dist_diff_abs_pair_c25_2_3_prev15
   + 240 dist_diff_abs_pair_c25_2_3_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev15 + dist_diff_abs_pair_c25_2_3_prev15
   - 240 dist_diff_abs_pair_c25_2_3_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev15_force_positive:
   dist_diff_pair_c25_2_3_prev15
   - 240 dist_diff_abs_pair_c25_2_3_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev15_force_negative:
   dist_diff_pair_c25_2_3_prev15
   - 240 dist_diff_abs_pair_c25_2_3_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev15: 240 same_dist_pair_c25_2_3_prev15
   + dist_diff_abs_pair_c25_2_3_prev15 <= 249.8
 same_dist_pair_lower_c25_2_3_prev15: 240 same_dist_pair_c25_2_3_prev15
   + dist_diff_abs_pair_c25_2_3_prev15 >= 10
 dist_diff_pair_def_c25_2_3_prev16: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev16 = -48
 dist_diff_abs_pair_c25_2_3_prev16_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev16 + dist_diff_abs_pair_c25_2_3_prev16
   - 240 dist_diff_abs_pair_c25_2_3_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev16_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev16 + dist_diff_abs_pair_c25_2_3_prev16
   + 240 dist_diff_abs_pair_c25_2_3_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev16 + dist_diff_abs_pair_c25_2_3_prev16
   + 240 dist_diff_abs_pair_c25_2_3_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev16 + dist_diff_abs_pair_c25_2_3_prev16
   - 240 dist_diff_abs_pair_c25_2_3_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev16_force_positive:
   dist_diff_pair_c25_2_3_prev16
   - 240 dist_diff_abs_pair_c25_2_3_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev16_force_negative:
   dist_diff_pair_c25_2_3_prev16
   - 240 dist_diff_abs_pair_c25_2_3_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev16: 240 same_dist_pair_c25_2_3_prev16
   + dist_diff_abs_pair_c25_2_3_prev16 <= 249.8
 same_dist_pair_lower_c25_2_3_prev16: 240 same_dist_pair_c25_2_3_prev16
   + dist_diff_abs_pair_c25_2_3_prev16 >= 10
 dist_diff_pair_def_c25_2_3_prev17: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev17 = -38
 dist_diff_abs_pair_c25_2_3_prev17_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev17 + dist_diff_abs_pair_c25_2_3_prev17
   - 240 dist_diff_abs_pair_c25_2_3_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev17_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev17 + dist_diff_abs_pair_c25_2_3_prev17
   + 240 dist_diff_abs_pair_c25_2_3_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev17 + dist_diff_abs_pair_c25_2_3_prev17
   + 240 dist_diff_abs_pair_c25_2_3_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev17 + dist_diff_abs_pair_c25_2_3_prev17
   - 240 dist_diff_abs_pair_c25_2_3_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev17_force_positive:
   dist_diff_pair_c25_2_3_prev17
   - 240 dist_diff_abs_pair_c25_2_3_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev17_force_negative:
   dist_diff_pair_c25_2_3_prev17
   - 240 dist_diff_abs_pair_c25_2_3_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev17: 240 same_dist_pair_c25_2_3_prev17
   + dist_diff_abs_pair_c25_2_3_prev17 <= 249.8
 same_dist_pair_lower_c25_2_3_prev17: 240 same_dist_pair_c25_2_3_prev17
   + dist_diff_abs_pair_c25_2_3_prev17 >= 10
 dist_diff_pair_def_c25_2_3_prev18: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev18 = -75
 dist_diff_abs_pair_c25_2_3_prev18_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev18 + dist_diff_abs_pair_c25_2_3_prev18
   - 240 dist_diff_abs_pair_c25_2_3_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev18_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev18 + dist_diff_abs_pair_c25_2_3_prev18
   + 240 dist_diff_abs_pair_c25_2_3_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev18 + dist_diff_abs_pair_c25_2_3_prev18
   + 240 dist_diff_abs_pair_c25_2_3_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev18 + dist_diff_abs_pair_c25_2_3_prev18
   - 240 dist_diff_abs_pair_c25_2_3_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev18_force_positive:
   dist_diff_pair_c25_2_3_prev18
   - 240 dist_diff_abs_pair_c25_2_3_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev18_force_negative:
   dist_diff_pair_c25_2_3_prev18
   - 240 dist_diff_abs_pair_c25_2_3_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev18: 240 same_dist_pair_c25_2_3_prev18
   + dist_diff_abs_pair_c25_2_3_prev18 <= 249.8
 same_dist_pair_lower_c25_2_3_prev18: 240 same_dist_pair_c25_2_3_prev18
   + dist_diff_abs_pair_c25_2_3_prev18 >= 10
 dist_diff_pair_def_c25_2_3_prev19: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev19 = -74
 dist_diff_abs_pair_c25_2_3_prev19_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev19 + dist_diff_abs_pair_c25_2_3_prev19
   - 240 dist_diff_abs_pair_c25_2_3_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev19_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev19 + dist_diff_abs_pair_c25_2_3_prev19
   + 240 dist_diff_abs_pair_c25_2_3_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev19 + dist_diff_abs_pair_c25_2_3_prev19
   + 240 dist_diff_abs_pair_c25_2_3_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev19 + dist_diff_abs_pair_c25_2_3_prev19
   - 240 dist_diff_abs_pair_c25_2_3_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev19_force_positive:
   dist_diff_pair_c25_2_3_prev19
   - 240 dist_diff_abs_pair_c25_2_3_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev19_force_negative:
   dist_diff_pair_c25_2_3_prev19
   - 240 dist_diff_abs_pair_c25_2_3_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev19: 240 same_dist_pair_c25_2_3_prev19
   + dist_diff_abs_pair_c25_2_3_prev19 <= 249.8
 same_dist_pair_lower_c25_2_3_prev19: 240 same_dist_pair_c25_2_3_prev19
   + dist_diff_abs_pair_c25_2_3_prev19 >= 10
 dist_diff_pair_def_c25_2_3_prev20: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev20 = -82
 dist_diff_abs_pair_c25_2_3_prev20_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev20 + dist_diff_abs_pair_c25_2_3_prev20
   - 240 dist_diff_abs_pair_c25_2_3_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev20_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev20 + dist_diff_abs_pair_c25_2_3_prev20
   + 240 dist_diff_abs_pair_c25_2_3_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev20 + dist_diff_abs_pair_c25_2_3_prev20
   + 240 dist_diff_abs_pair_c25_2_3_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev20 + dist_diff_abs_pair_c25_2_3_prev20
   - 240 dist_diff_abs_pair_c25_2_3_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev20_force_positive:
   dist_diff_pair_c25_2_3_prev20
   - 240 dist_diff_abs_pair_c25_2_3_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev20_force_negative:
   dist_diff_pair_c25_2_3_prev20
   - 240 dist_diff_abs_pair_c25_2_3_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev20: 240 same_dist_pair_c25_2_3_prev20
   + dist_diff_abs_pair_c25_2_3_prev20 <= 249.8
 same_dist_pair_lower_c25_2_3_prev20: 240 same_dist_pair_c25_2_3_prev20
   + dist_diff_abs_pair_c25_2_3_prev20 >= 10
 dist_diff_pair_def_c25_2_3_prev21: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev21 = -94
 dist_diff_abs_pair_c25_2_3_prev21_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev21 + dist_diff_abs_pair_c25_2_3_prev21
   - 240 dist_diff_abs_pair_c25_2_3_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev21_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev21 + dist_diff_abs_pair_c25_2_3_prev21
   + 240 dist_diff_abs_pair_c25_2_3_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev21 + dist_diff_abs_pair_c25_2_3_prev21
   + 240 dist_diff_abs_pair_c25_2_3_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev21 + dist_diff_abs_pair_c25_2_3_prev21
   - 240 dist_diff_abs_pair_c25_2_3_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev21_force_positive:
   dist_diff_pair_c25_2_3_prev21
   - 240 dist_diff_abs_pair_c25_2_3_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev21_force_negative:
   dist_diff_pair_c25_2_3_prev21
   - 240 dist_diff_abs_pair_c25_2_3_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev21: 240 same_dist_pair_c25_2_3_prev21
   + dist_diff_abs_pair_c25_2_3_prev21 <= 249.8
 same_dist_pair_lower_c25_2_3_prev21: 240 same_dist_pair_c25_2_3_prev21
   + dist_diff_abs_pair_c25_2_3_prev21 >= 10
 dist_diff_pair_def_c25_2_3_prev22: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev22 = -38
 dist_diff_abs_pair_c25_2_3_prev22_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev22 + dist_diff_abs_pair_c25_2_3_prev22
   - 240 dist_diff_abs_pair_c25_2_3_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev22_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev22 + dist_diff_abs_pair_c25_2_3_prev22
   + 240 dist_diff_abs_pair_c25_2_3_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev22 + dist_diff_abs_pair_c25_2_3_prev22
   + 240 dist_diff_abs_pair_c25_2_3_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev22 + dist_diff_abs_pair_c25_2_3_prev22
   - 240 dist_diff_abs_pair_c25_2_3_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev22_force_positive:
   dist_diff_pair_c25_2_3_prev22
   - 240 dist_diff_abs_pair_c25_2_3_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev22_force_negative:
   dist_diff_pair_c25_2_3_prev22
   - 240 dist_diff_abs_pair_c25_2_3_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev22: 240 same_dist_pair_c25_2_3_prev22
   + dist_diff_abs_pair_c25_2_3_prev22 <= 249.8
 same_dist_pair_lower_c25_2_3_prev22: 240 same_dist_pair_c25_2_3_prev22
   + dist_diff_abs_pair_c25_2_3_prev22 >= 10
 dist_diff_pair_def_c25_2_3_prev23: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev23 = -45
 dist_diff_abs_pair_c25_2_3_prev23_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev23 + dist_diff_abs_pair_c25_2_3_prev23
   - 240 dist_diff_abs_pair_c25_2_3_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev23_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev23 + dist_diff_abs_pair_c25_2_3_prev23
   + 240 dist_diff_abs_pair_c25_2_3_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev23 + dist_diff_abs_pair_c25_2_3_prev23
   + 240 dist_diff_abs_pair_c25_2_3_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev23 + dist_diff_abs_pair_c25_2_3_prev23
   - 240 dist_diff_abs_pair_c25_2_3_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev23_force_positive:
   dist_diff_pair_c25_2_3_prev23
   - 240 dist_diff_abs_pair_c25_2_3_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev23_force_negative:
   dist_diff_pair_c25_2_3_prev23
   - 240 dist_diff_abs_pair_c25_2_3_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev23: 240 same_dist_pair_c25_2_3_prev23
   + dist_diff_abs_pair_c25_2_3_prev23 <= 249.8
 same_dist_pair_lower_c25_2_3_prev23: 240 same_dist_pair_c25_2_3_prev23
   + dist_diff_abs_pair_c25_2_3_prev23 >= 10
 dist_diff_pair_def_c25_2_3_prev24: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev24 = -74
 dist_diff_abs_pair_c25_2_3_prev24_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev24 + dist_diff_abs_pair_c25_2_3_prev24
   - 240 dist_diff_abs_pair_c25_2_3_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev24_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev24 + dist_diff_abs_pair_c25_2_3_prev24
   + 240 dist_diff_abs_pair_c25_2_3_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev24 + dist_diff_abs_pair_c25_2_3_prev24
   + 240 dist_diff_abs_pair_c25_2_3_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev24 + dist_diff_abs_pair_c25_2_3_prev24
   - 240 dist_diff_abs_pair_c25_2_3_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev24_force_positive:
   dist_diff_pair_c25_2_3_prev24
   - 240 dist_diff_abs_pair_c25_2_3_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev24_force_negative:
   dist_diff_pair_c25_2_3_prev24
   - 240 dist_diff_abs_pair_c25_2_3_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev24: 240 same_dist_pair_c25_2_3_prev24
   + dist_diff_abs_pair_c25_2_3_prev24 <= 249.8
 same_dist_pair_lower_c25_2_3_prev24: 240 same_dist_pair_c25_2_3_prev24
   + dist_diff_abs_pair_c25_2_3_prev24 >= 10
 dist_diff_pair_def_c25_2_3_prev25: - dist_curr_pair_c25_2_3
   + dist_diff_pair_c25_2_3_prev25 = -38
 dist_diff_abs_pair_c25_2_3_prev25_abs_ge_orig:
   - dist_diff_pair_c25_2_3_prev25 + dist_diff_abs_pair_c25_2_3_prev25
   - 240 dist_diff_abs_pair_c25_2_3_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev25_abs_le_orig:
   - dist_diff_pair_c25_2_3_prev25 + dist_diff_abs_pair_c25_2_3_prev25
   + 240 dist_diff_abs_pair_c25_2_3_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_2_3_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_2_3_prev25 + dist_diff_abs_pair_c25_2_3_prev25
   + 240 dist_diff_abs_pair_c25_2_3_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_2_3_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_2_3_prev25 + dist_diff_abs_pair_c25_2_3_prev25
   - 240 dist_diff_abs_pair_c25_2_3_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_2_3_prev25_force_positive:
   dist_diff_pair_c25_2_3_prev25
   - 240 dist_diff_abs_pair_c25_2_3_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_2_3_prev25_force_negative:
   dist_diff_pair_c25_2_3_prev25
   - 240 dist_diff_abs_pair_c25_2_3_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_2_3_prev25: 240 same_dist_pair_c25_2_3_prev25
   + dist_diff_abs_pair_c25_2_3_prev25 <= 249.8
 same_dist_pair_lower_c25_2_3_prev25: 240 same_dist_pair_c25_2_3_prev25
   + dist_diff_abs_pair_c25_2_3_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_2_3_prev0: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev0: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev1: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev1: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev2: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev2: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev3: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev3: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev4: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev4: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev5: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev5: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev6: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev6: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev7: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev7: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev8: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev8: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev9: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev9: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev10: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev10: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev11: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev11: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev12: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev12: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev13: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev13: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev14: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev14: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev15: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev15: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev16: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev16: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev17: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev17: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev18: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev18: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev19: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev19: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev20: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev20: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev21: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev21: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev22: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev22: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev23: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev23: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev24: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev24: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_2_3_prev25: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_2_3_prev25: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_2_3: diff_dist_pair_c25_2_3
   + same_dist_pair_c25_2_3_prev0 + same_dist_pair_c25_2_3_prev1
   + same_dist_pair_c25_2_3_prev2 + same_dist_pair_c25_2_3_prev3
   + same_dist_pair_c25_2_3_prev4 + same_dist_pair_c25_2_3_prev5
   + same_dist_pair_c25_2_3_prev6 + same_dist_pair_c25_2_3_prev7
   + same_dist_pair_c25_2_3_prev8 + same_dist_pair_c25_2_3_prev9
   + same_dist_pair_c25_2_3_prev10 + same_dist_pair_c25_2_3_prev11
   + same_dist_pair_c25_2_3_prev12 + same_dist_pair_c25_2_3_prev13
   + same_dist_pair_c25_2_3_prev14 + same_dist_pair_c25_2_3_prev15
   + same_dist_pair_c25_2_3_prev16 + same_dist_pair_c25_2_3_prev17
   + same_dist_pair_c25_2_3_prev18 + same_dist_pair_c25_2_3_prev19
   + same_dist_pair_c25_2_3_prev20 + same_dist_pair_c25_2_3_prev21
   + same_dist_pair_c25_2_3_prev22 + same_dist_pair_c25_2_3_prev23
   + same_dist_pair_c25_2_3_prev24 + same_dist_pair_c25_2_3_prev25 >= 1
 dx_grid_diff_def_c25_2_4: - cx_grid_center_c25_2 + cx_grid_center_c25_4
   + dx_grid_diff_c25_2_4 = 0
 dx_grid_abs_pair_c25_2_4_abs_ge_orig: dx_grid_abs_pair_c25_2_4
   - dx_grid_diff_c25_2_4 - 130 dx_grid_abs_pair_c25_2_4_is_positive
   >= -130
 dx_grid_abs_pair_c25_2_4_abs_le_orig: dx_grid_abs_pair_c25_2_4
   - dx_grid_diff_c25_2_4 + 130 dx_grid_abs_pair_c25_2_4_is_positive
   <= 130
 dx_grid_abs_pair_c25_2_4_abs_ge_neg_orig: dx_grid_abs_pair_c25_2_4
   + dx_grid_diff_c25_2_4 + 130 dx_grid_abs_pair_c25_2_4_is_positive >= 0
 dx_grid_abs_pair_c25_2_4_abs_le_neg_orig: dx_grid_abs_pair_c25_2_4
   + dx_grid_diff_c25_2_4 - 130 dx_grid_abs_pair_c25_2_4_is_positive <= 0
 dx_grid_abs_pair_c25_2_4_force_positive: dx_grid_diff_c25_2_4
   - 130 dx_grid_abs_pair_c25_2_4_is_positive >= -130
 dx_grid_abs_pair_c25_2_4_force_negative: dx_grid_diff_c25_2_4
   - 130 dx_grid_abs_pair_c25_2_4_is_positive <= -0.001
 dy_grid_diff_def_c25_2_4: - cy_grid_center_c25_2 + cy_grid_center_c25_4
   + dy_grid_diff_c25_2_4 = 0
 dy_grid_abs_pair_c25_2_4_abs_ge_orig: dy_grid_abs_pair_c25_2_4
   - dy_grid_diff_c25_2_4 - 130 dy_grid_abs_pair_c25_2_4_is_positive
   >= -130
 dy_grid_abs_pair_c25_2_4_abs_le_orig: dy_grid_abs_pair_c25_2_4
   - dy_grid_diff_c25_2_4 + 130 dy_grid_abs_pair_c25_2_4_is_positive
   <= 130
 dy_grid_abs_pair_c25_2_4_abs_ge_neg_orig: dy_grid_abs_pair_c25_2_4
   + dy_grid_diff_c25_2_4 + 130 dy_grid_abs_pair_c25_2_4_is_positive >= 0
 dy_grid_abs_pair_c25_2_4_abs_le_neg_orig: dy_grid_abs_pair_c25_2_4
   + dy_grid_diff_c25_2_4 - 130 dy_grid_abs_pair_c25_2_4_is_positive <= 0
 dy_grid_abs_pair_c25_2_4_force_positive: dy_grid_diff_c25_2_4
   - 130 dy_grid_abs_pair_c25_2_4_is_positive >= -130
 dy_grid_abs_pair_c25_2_4_force_negative: dy_grid_diff_c25_2_4
   - 130 dy_grid_abs_pair_c25_2_4_is_positive <= -0.001
 dist_curr_pair_def_c25_2_4: - dx_grid_abs_pair_c25_2_4
   - dy_grid_abs_pair_c25_2_4 + dist_curr_pair_c25_2_4 = 0
 dist_diff_pair_def_c25_2_4_prev0: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev0 = -24
 dist_diff_abs_pair_c25_2_4_prev0_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev0 + dist_diff_abs_pair_c25_2_4_prev0
   - 240 dist_diff_abs_pair_c25_2_4_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev0_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev0 + dist_diff_abs_pair_c25_2_4_prev0
   + 240 dist_diff_abs_pair_c25_2_4_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev0 + dist_diff_abs_pair_c25_2_4_prev0
   + 240 dist_diff_abs_pair_c25_2_4_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev0 + dist_diff_abs_pair_c25_2_4_prev0
   - 240 dist_diff_abs_pair_c25_2_4_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev0_force_positive:
   dist_diff_pair_c25_2_4_prev0
   - 240 dist_diff_abs_pair_c25_2_4_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev0_force_negative:
   dist_diff_pair_c25_2_4_prev0
   - 240 dist_diff_abs_pair_c25_2_4_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev0: 240 same_dist_pair_c25_2_4_prev0
   + dist_diff_abs_pair_c25_2_4_prev0 <= 249.8
 same_dist_pair_lower_c25_2_4_prev0: 240 same_dist_pair_c25_2_4_prev0
   + dist_diff_abs_pair_c25_2_4_prev0 >= 10
 dist_diff_pair_def_c25_2_4_prev1: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev1 = -24
 dist_diff_abs_pair_c25_2_4_prev1_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev1 + dist_diff_abs_pair_c25_2_4_prev1
   - 240 dist_diff_abs_pair_c25_2_4_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev1_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev1 + dist_diff_abs_pair_c25_2_4_prev1
   + 240 dist_diff_abs_pair_c25_2_4_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev1 + dist_diff_abs_pair_c25_2_4_prev1
   + 240 dist_diff_abs_pair_c25_2_4_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev1 + dist_diff_abs_pair_c25_2_4_prev1
   - 240 dist_diff_abs_pair_c25_2_4_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev1_force_positive:
   dist_diff_pair_c25_2_4_prev1
   - 240 dist_diff_abs_pair_c25_2_4_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev1_force_negative:
   dist_diff_pair_c25_2_4_prev1
   - 240 dist_diff_abs_pair_c25_2_4_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev1: 240 same_dist_pair_c25_2_4_prev1
   + dist_diff_abs_pair_c25_2_4_prev1 <= 249.8
 same_dist_pair_lower_c25_2_4_prev1: 240 same_dist_pair_c25_2_4_prev1
   + dist_diff_abs_pair_c25_2_4_prev1 >= 10
 dist_diff_pair_def_c25_2_4_prev2: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev2 = -24
 dist_diff_abs_pair_c25_2_4_prev2_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev2 + dist_diff_abs_pair_c25_2_4_prev2
   - 240 dist_diff_abs_pair_c25_2_4_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev2_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev2 + dist_diff_abs_pair_c25_2_4_prev2
   + 240 dist_diff_abs_pair_c25_2_4_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev2 + dist_diff_abs_pair_c25_2_4_prev2
   + 240 dist_diff_abs_pair_c25_2_4_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev2 + dist_diff_abs_pair_c25_2_4_prev2
   - 240 dist_diff_abs_pair_c25_2_4_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev2_force_positive:
   dist_diff_pair_c25_2_4_prev2
   - 240 dist_diff_abs_pair_c25_2_4_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev2_force_negative:
   dist_diff_pair_c25_2_4_prev2
   - 240 dist_diff_abs_pair_c25_2_4_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev2: 240 same_dist_pair_c25_2_4_prev2
   + dist_diff_abs_pair_c25_2_4_prev2 <= 249.8
 same_dist_pair_lower_c25_2_4_prev2: 240 same_dist_pair_c25_2_4_prev2
   + dist_diff_abs_pair_c25_2_4_prev2 >= 10
 dist_diff_pair_def_c25_2_4_prev3: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev3 = -27
 dist_diff_abs_pair_c25_2_4_prev3_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev3 + dist_diff_abs_pair_c25_2_4_prev3
   - 240 dist_diff_abs_pair_c25_2_4_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev3_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev3 + dist_diff_abs_pair_c25_2_4_prev3
   + 240 dist_diff_abs_pair_c25_2_4_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev3 + dist_diff_abs_pair_c25_2_4_prev3
   + 240 dist_diff_abs_pair_c25_2_4_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev3 + dist_diff_abs_pair_c25_2_4_prev3
   - 240 dist_diff_abs_pair_c25_2_4_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev3_force_positive:
   dist_diff_pair_c25_2_4_prev3
   - 240 dist_diff_abs_pair_c25_2_4_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev3_force_negative:
   dist_diff_pair_c25_2_4_prev3
   - 240 dist_diff_abs_pair_c25_2_4_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev3: 240 same_dist_pair_c25_2_4_prev3
   + dist_diff_abs_pair_c25_2_4_prev3 <= 249.8
 same_dist_pair_lower_c25_2_4_prev3: 240 same_dist_pair_c25_2_4_prev3
   + dist_diff_abs_pair_c25_2_4_prev3 >= 10
 dist_diff_pair_def_c25_2_4_prev4: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev4 = -24
 dist_diff_abs_pair_c25_2_4_prev4_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev4 + dist_diff_abs_pair_c25_2_4_prev4
   - 240 dist_diff_abs_pair_c25_2_4_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev4_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev4 + dist_diff_abs_pair_c25_2_4_prev4
   + 240 dist_diff_abs_pair_c25_2_4_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev4 + dist_diff_abs_pair_c25_2_4_prev4
   + 240 dist_diff_abs_pair_c25_2_4_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev4 + dist_diff_abs_pair_c25_2_4_prev4
   - 240 dist_diff_abs_pair_c25_2_4_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev4_force_positive:
   dist_diff_pair_c25_2_4_prev4
   - 240 dist_diff_abs_pair_c25_2_4_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev4_force_negative:
   dist_diff_pair_c25_2_4_prev4
   - 240 dist_diff_abs_pair_c25_2_4_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev4: 240 same_dist_pair_c25_2_4_prev4
   + dist_diff_abs_pair_c25_2_4_prev4 <= 249.8
 same_dist_pair_lower_c25_2_4_prev4: 240 same_dist_pair_c25_2_4_prev4
   + dist_diff_abs_pair_c25_2_4_prev4 >= 10
 dist_diff_pair_def_c25_2_4_prev5: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev5 = -34
 dist_diff_abs_pair_c25_2_4_prev5_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev5 + dist_diff_abs_pair_c25_2_4_prev5
   - 240 dist_diff_abs_pair_c25_2_4_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev5_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev5 + dist_diff_abs_pair_c25_2_4_prev5
   + 240 dist_diff_abs_pair_c25_2_4_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev5 + dist_diff_abs_pair_c25_2_4_prev5
   + 240 dist_diff_abs_pair_c25_2_4_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev5 + dist_diff_abs_pair_c25_2_4_prev5
   - 240 dist_diff_abs_pair_c25_2_4_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev5_force_positive:
   dist_diff_pair_c25_2_4_prev5
   - 240 dist_diff_abs_pair_c25_2_4_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev5_force_negative:
   dist_diff_pair_c25_2_4_prev5
   - 240 dist_diff_abs_pair_c25_2_4_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev5: 240 same_dist_pair_c25_2_4_prev5
   + dist_diff_abs_pair_c25_2_4_prev5 <= 249.8
 same_dist_pair_lower_c25_2_4_prev5: 240 same_dist_pair_c25_2_4_prev5
   + dist_diff_abs_pair_c25_2_4_prev5 >= 10
 dist_diff_pair_def_c25_2_4_prev6: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev6 = -24
 dist_diff_abs_pair_c25_2_4_prev6_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev6 + dist_diff_abs_pair_c25_2_4_prev6
   - 240 dist_diff_abs_pair_c25_2_4_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev6_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev6 + dist_diff_abs_pair_c25_2_4_prev6
   + 240 dist_diff_abs_pair_c25_2_4_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev6 + dist_diff_abs_pair_c25_2_4_prev6
   + 240 dist_diff_abs_pair_c25_2_4_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev6 + dist_diff_abs_pair_c25_2_4_prev6
   - 240 dist_diff_abs_pair_c25_2_4_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev6_force_positive:
   dist_diff_pair_c25_2_4_prev6
   - 240 dist_diff_abs_pair_c25_2_4_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev6_force_negative:
   dist_diff_pair_c25_2_4_prev6
   - 240 dist_diff_abs_pair_c25_2_4_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev6: 240 same_dist_pair_c25_2_4_prev6
   + dist_diff_abs_pair_c25_2_4_prev6 <= 249.8
 same_dist_pair_lower_c25_2_4_prev6: 240 same_dist_pair_c25_2_4_prev6
   + dist_diff_abs_pair_c25_2_4_prev6 >= 10
 dist_diff_pair_def_c25_2_4_prev7: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev7 = -26
 dist_diff_abs_pair_c25_2_4_prev7_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev7 + dist_diff_abs_pair_c25_2_4_prev7
   - 240 dist_diff_abs_pair_c25_2_4_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev7_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev7 + dist_diff_abs_pair_c25_2_4_prev7
   + 240 dist_diff_abs_pair_c25_2_4_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev7 + dist_diff_abs_pair_c25_2_4_prev7
   + 240 dist_diff_abs_pair_c25_2_4_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev7 + dist_diff_abs_pair_c25_2_4_prev7
   - 240 dist_diff_abs_pair_c25_2_4_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev7_force_positive:
   dist_diff_pair_c25_2_4_prev7
   - 240 dist_diff_abs_pair_c25_2_4_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev7_force_negative:
   dist_diff_pair_c25_2_4_prev7
   - 240 dist_diff_abs_pair_c25_2_4_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev7: 240 same_dist_pair_c25_2_4_prev7
   + dist_diff_abs_pair_c25_2_4_prev7 <= 249.8
 same_dist_pair_lower_c25_2_4_prev7: 240 same_dist_pair_c25_2_4_prev7
   + dist_diff_abs_pair_c25_2_4_prev7 >= 10
 dist_diff_pair_def_c25_2_4_prev8: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev8 = -25
 dist_diff_abs_pair_c25_2_4_prev8_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev8 + dist_diff_abs_pair_c25_2_4_prev8
   - 240 dist_diff_abs_pair_c25_2_4_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev8_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev8 + dist_diff_abs_pair_c25_2_4_prev8
   + 240 dist_diff_abs_pair_c25_2_4_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev8 + dist_diff_abs_pair_c25_2_4_prev8
   + 240 dist_diff_abs_pair_c25_2_4_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev8 + dist_diff_abs_pair_c25_2_4_prev8
   - 240 dist_diff_abs_pair_c25_2_4_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev8_force_positive:
   dist_diff_pair_c25_2_4_prev8
   - 240 dist_diff_abs_pair_c25_2_4_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev8_force_negative:
   dist_diff_pair_c25_2_4_prev8
   - 240 dist_diff_abs_pair_c25_2_4_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev8: 240 same_dist_pair_c25_2_4_prev8
   + dist_diff_abs_pair_c25_2_4_prev8 <= 249.8
 same_dist_pair_lower_c25_2_4_prev8: 240 same_dist_pair_c25_2_4_prev8
   + dist_diff_abs_pair_c25_2_4_prev8 >= 10
 dist_diff_pair_def_c25_2_4_prev9: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev9 = -28
 dist_diff_abs_pair_c25_2_4_prev9_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev9 + dist_diff_abs_pair_c25_2_4_prev9
   - 240 dist_diff_abs_pair_c25_2_4_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev9_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev9 + dist_diff_abs_pair_c25_2_4_prev9
   + 240 dist_diff_abs_pair_c25_2_4_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev9 + dist_diff_abs_pair_c25_2_4_prev9
   + 240 dist_diff_abs_pair_c25_2_4_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev9 + dist_diff_abs_pair_c25_2_4_prev9
   - 240 dist_diff_abs_pair_c25_2_4_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev9_force_positive:
   dist_diff_pair_c25_2_4_prev9
   - 240 dist_diff_abs_pair_c25_2_4_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev9_force_negative:
   dist_diff_pair_c25_2_4_prev9
   - 240 dist_diff_abs_pair_c25_2_4_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev9: 240 same_dist_pair_c25_2_4_prev9
   + dist_diff_abs_pair_c25_2_4_prev9 <= 249.8
 same_dist_pair_lower_c25_2_4_prev9: 240 same_dist_pair_c25_2_4_prev9
   + dist_diff_abs_pair_c25_2_4_prev9 >= 10
 dist_diff_pair_def_c25_2_4_prev10: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev10 = -34
 dist_diff_abs_pair_c25_2_4_prev10_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev10 + dist_diff_abs_pair_c25_2_4_prev10
   - 240 dist_diff_abs_pair_c25_2_4_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev10_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev10 + dist_diff_abs_pair_c25_2_4_prev10
   + 240 dist_diff_abs_pair_c25_2_4_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev10 + dist_diff_abs_pair_c25_2_4_prev10
   + 240 dist_diff_abs_pair_c25_2_4_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev10 + dist_diff_abs_pair_c25_2_4_prev10
   - 240 dist_diff_abs_pair_c25_2_4_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev10_force_positive:
   dist_diff_pair_c25_2_4_prev10
   - 240 dist_diff_abs_pair_c25_2_4_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev10_force_negative:
   dist_diff_pair_c25_2_4_prev10
   - 240 dist_diff_abs_pair_c25_2_4_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev10: 240 same_dist_pair_c25_2_4_prev10
   + dist_diff_abs_pair_c25_2_4_prev10 <= 249.8
 same_dist_pair_lower_c25_2_4_prev10: 240 same_dist_pair_c25_2_4_prev10
   + dist_diff_abs_pair_c25_2_4_prev10 >= 10
 dist_diff_pair_def_c25_2_4_prev11: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev11 = -44
 dist_diff_abs_pair_c25_2_4_prev11_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev11 + dist_diff_abs_pair_c25_2_4_prev11
   - 240 dist_diff_abs_pair_c25_2_4_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev11_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev11 + dist_diff_abs_pair_c25_2_4_prev11
   + 240 dist_diff_abs_pair_c25_2_4_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev11 + dist_diff_abs_pair_c25_2_4_prev11
   + 240 dist_diff_abs_pair_c25_2_4_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev11 + dist_diff_abs_pair_c25_2_4_prev11
   - 240 dist_diff_abs_pair_c25_2_4_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev11_force_positive:
   dist_diff_pair_c25_2_4_prev11
   - 240 dist_diff_abs_pair_c25_2_4_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev11_force_negative:
   dist_diff_pair_c25_2_4_prev11
   - 240 dist_diff_abs_pair_c25_2_4_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev11: 240 same_dist_pair_c25_2_4_prev11
   + dist_diff_abs_pair_c25_2_4_prev11 <= 249.8
 same_dist_pair_lower_c25_2_4_prev11: 240 same_dist_pair_c25_2_4_prev11
   + dist_diff_abs_pair_c25_2_4_prev11 >= 10
 dist_diff_pair_def_c25_2_4_prev12: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev12 = -28
 dist_diff_abs_pair_c25_2_4_prev12_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev12 + dist_diff_abs_pair_c25_2_4_prev12
   - 240 dist_diff_abs_pair_c25_2_4_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev12_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev12 + dist_diff_abs_pair_c25_2_4_prev12
   + 240 dist_diff_abs_pair_c25_2_4_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev12 + dist_diff_abs_pair_c25_2_4_prev12
   + 240 dist_diff_abs_pair_c25_2_4_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev12 + dist_diff_abs_pair_c25_2_4_prev12
   - 240 dist_diff_abs_pair_c25_2_4_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev12_force_positive:
   dist_diff_pair_c25_2_4_prev12
   - 240 dist_diff_abs_pair_c25_2_4_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev12_force_negative:
   dist_diff_pair_c25_2_4_prev12
   - 240 dist_diff_abs_pair_c25_2_4_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev12: 240 same_dist_pair_c25_2_4_prev12
   + dist_diff_abs_pair_c25_2_4_prev12 <= 249.8
 same_dist_pair_lower_c25_2_4_prev12: 240 same_dist_pair_c25_2_4_prev12
   + dist_diff_abs_pair_c25_2_4_prev12 >= 10
 dist_diff_pair_def_c25_2_4_prev13: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev13 = -24
 dist_diff_abs_pair_c25_2_4_prev13_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev13 + dist_diff_abs_pair_c25_2_4_prev13
   - 240 dist_diff_abs_pair_c25_2_4_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev13_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev13 + dist_diff_abs_pair_c25_2_4_prev13
   + 240 dist_diff_abs_pair_c25_2_4_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev13 + dist_diff_abs_pair_c25_2_4_prev13
   + 240 dist_diff_abs_pair_c25_2_4_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev13 + dist_diff_abs_pair_c25_2_4_prev13
   - 240 dist_diff_abs_pair_c25_2_4_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev13_force_positive:
   dist_diff_pair_c25_2_4_prev13
   - 240 dist_diff_abs_pair_c25_2_4_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev13_force_negative:
   dist_diff_pair_c25_2_4_prev13
   - 240 dist_diff_abs_pair_c25_2_4_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev13: 240 same_dist_pair_c25_2_4_prev13
   + dist_diff_abs_pair_c25_2_4_prev13 <= 249.8
 same_dist_pair_lower_c25_2_4_prev13: 240 same_dist_pair_c25_2_4_prev13
   + dist_diff_abs_pair_c25_2_4_prev13 >= 10
 dist_diff_pair_def_c25_2_4_prev14: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev14 = -44
 dist_diff_abs_pair_c25_2_4_prev14_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev14 + dist_diff_abs_pair_c25_2_4_prev14
   - 240 dist_diff_abs_pair_c25_2_4_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev14_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev14 + dist_diff_abs_pair_c25_2_4_prev14
   + 240 dist_diff_abs_pair_c25_2_4_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev14 + dist_diff_abs_pair_c25_2_4_prev14
   + 240 dist_diff_abs_pair_c25_2_4_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev14 + dist_diff_abs_pair_c25_2_4_prev14
   - 240 dist_diff_abs_pair_c25_2_4_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev14_force_positive:
   dist_diff_pair_c25_2_4_prev14
   - 240 dist_diff_abs_pair_c25_2_4_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev14_force_negative:
   dist_diff_pair_c25_2_4_prev14
   - 240 dist_diff_abs_pair_c25_2_4_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev14: 240 same_dist_pair_c25_2_4_prev14
   + dist_diff_abs_pair_c25_2_4_prev14 <= 249.8
 same_dist_pair_lower_c25_2_4_prev14: 240 same_dist_pair_c25_2_4_prev14
   + dist_diff_abs_pair_c25_2_4_prev14 >= 10
 dist_diff_pair_def_c25_2_4_prev15: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev15 = -24
 dist_diff_abs_pair_c25_2_4_prev15_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev15 + dist_diff_abs_pair_c25_2_4_prev15
   - 240 dist_diff_abs_pair_c25_2_4_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev15_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev15 + dist_diff_abs_pair_c25_2_4_prev15
   + 240 dist_diff_abs_pair_c25_2_4_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev15 + dist_diff_abs_pair_c25_2_4_prev15
   + 240 dist_diff_abs_pair_c25_2_4_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev15 + dist_diff_abs_pair_c25_2_4_prev15
   - 240 dist_diff_abs_pair_c25_2_4_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev15_force_positive:
   dist_diff_pair_c25_2_4_prev15
   - 240 dist_diff_abs_pair_c25_2_4_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev15_force_negative:
   dist_diff_pair_c25_2_4_prev15
   - 240 dist_diff_abs_pair_c25_2_4_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev15: 240 same_dist_pair_c25_2_4_prev15
   + dist_diff_abs_pair_c25_2_4_prev15 <= 249.8
 same_dist_pair_lower_c25_2_4_prev15: 240 same_dist_pair_c25_2_4_prev15
   + dist_diff_abs_pair_c25_2_4_prev15 >= 10
 dist_diff_pair_def_c25_2_4_prev16: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev16 = -24
 dist_diff_abs_pair_c25_2_4_prev16_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev16 + dist_diff_abs_pair_c25_2_4_prev16
   - 240 dist_diff_abs_pair_c25_2_4_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev16_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev16 + dist_diff_abs_pair_c25_2_4_prev16
   + 240 dist_diff_abs_pair_c25_2_4_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev16 + dist_diff_abs_pair_c25_2_4_prev16
   + 240 dist_diff_abs_pair_c25_2_4_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev16 + dist_diff_abs_pair_c25_2_4_prev16
   - 240 dist_diff_abs_pair_c25_2_4_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev16_force_positive:
   dist_diff_pair_c25_2_4_prev16
   - 240 dist_diff_abs_pair_c25_2_4_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev16_force_negative:
   dist_diff_pair_c25_2_4_prev16
   - 240 dist_diff_abs_pair_c25_2_4_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev16: 240 same_dist_pair_c25_2_4_prev16
   + dist_diff_abs_pair_c25_2_4_prev16 <= 249.8
 same_dist_pair_lower_c25_2_4_prev16: 240 same_dist_pair_c25_2_4_prev16
   + dist_diff_abs_pair_c25_2_4_prev16 >= 10
 dist_diff_pair_def_c25_2_4_prev17: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev17 = -24
 dist_diff_abs_pair_c25_2_4_prev17_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev17 + dist_diff_abs_pair_c25_2_4_prev17
   - 240 dist_diff_abs_pair_c25_2_4_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev17_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev17 + dist_diff_abs_pair_c25_2_4_prev17
   + 240 dist_diff_abs_pair_c25_2_4_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev17 + dist_diff_abs_pair_c25_2_4_prev17
   + 240 dist_diff_abs_pair_c25_2_4_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev17 + dist_diff_abs_pair_c25_2_4_prev17
   - 240 dist_diff_abs_pair_c25_2_4_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev17_force_positive:
   dist_diff_pair_c25_2_4_prev17
   - 240 dist_diff_abs_pair_c25_2_4_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev17_force_negative:
   dist_diff_pair_c25_2_4_prev17
   - 240 dist_diff_abs_pair_c25_2_4_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev17: 240 same_dist_pair_c25_2_4_prev17
   + dist_diff_abs_pair_c25_2_4_prev17 <= 249.8
 same_dist_pair_lower_c25_2_4_prev17: 240 same_dist_pair_c25_2_4_prev17
   + dist_diff_abs_pair_c25_2_4_prev17 >= 10
 dist_diff_pair_def_c25_2_4_prev18: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev18 = -28
 dist_diff_abs_pair_c25_2_4_prev18_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev18 + dist_diff_abs_pair_c25_2_4_prev18
   - 240 dist_diff_abs_pair_c25_2_4_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev18_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev18 + dist_diff_abs_pair_c25_2_4_prev18
   + 240 dist_diff_abs_pair_c25_2_4_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev18 + dist_diff_abs_pair_c25_2_4_prev18
   + 240 dist_diff_abs_pair_c25_2_4_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev18 + dist_diff_abs_pair_c25_2_4_prev18
   - 240 dist_diff_abs_pair_c25_2_4_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev18_force_positive:
   dist_diff_pair_c25_2_4_prev18
   - 240 dist_diff_abs_pair_c25_2_4_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev18_force_negative:
   dist_diff_pair_c25_2_4_prev18
   - 240 dist_diff_abs_pair_c25_2_4_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev18: 240 same_dist_pair_c25_2_4_prev18
   + dist_diff_abs_pair_c25_2_4_prev18 <= 249.8
 same_dist_pair_lower_c25_2_4_prev18: 240 same_dist_pair_c25_2_4_prev18
   + dist_diff_abs_pair_c25_2_4_prev18 >= 10
 dist_diff_pair_def_c25_2_4_prev19: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev19 = -24
 dist_diff_abs_pair_c25_2_4_prev19_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev19 + dist_diff_abs_pair_c25_2_4_prev19
   - 240 dist_diff_abs_pair_c25_2_4_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev19_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev19 + dist_diff_abs_pair_c25_2_4_prev19
   + 240 dist_diff_abs_pair_c25_2_4_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev19 + dist_diff_abs_pair_c25_2_4_prev19
   + 240 dist_diff_abs_pair_c25_2_4_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev19 + dist_diff_abs_pair_c25_2_4_prev19
   - 240 dist_diff_abs_pair_c25_2_4_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev19_force_positive:
   dist_diff_pair_c25_2_4_prev19
   - 240 dist_diff_abs_pair_c25_2_4_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev19_force_negative:
   dist_diff_pair_c25_2_4_prev19
   - 240 dist_diff_abs_pair_c25_2_4_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev19: 240 same_dist_pair_c25_2_4_prev19
   + dist_diff_abs_pair_c25_2_4_prev19 <= 249.8
 same_dist_pair_lower_c25_2_4_prev19: 240 same_dist_pair_c25_2_4_prev19
   + dist_diff_abs_pair_c25_2_4_prev19 >= 10
 dist_diff_pair_def_c25_2_4_prev20: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev20 = -40
 dist_diff_abs_pair_c25_2_4_prev20_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev20 + dist_diff_abs_pair_c25_2_4_prev20
   - 240 dist_diff_abs_pair_c25_2_4_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev20_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev20 + dist_diff_abs_pair_c25_2_4_prev20
   + 240 dist_diff_abs_pair_c25_2_4_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev20 + dist_diff_abs_pair_c25_2_4_prev20
   + 240 dist_diff_abs_pair_c25_2_4_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev20 + dist_diff_abs_pair_c25_2_4_prev20
   - 240 dist_diff_abs_pair_c25_2_4_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev20_force_positive:
   dist_diff_pair_c25_2_4_prev20
   - 240 dist_diff_abs_pair_c25_2_4_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev20_force_negative:
   dist_diff_pair_c25_2_4_prev20
   - 240 dist_diff_abs_pair_c25_2_4_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev20: 240 same_dist_pair_c25_2_4_prev20
   + dist_diff_abs_pair_c25_2_4_prev20 <= 249.8
 same_dist_pair_lower_c25_2_4_prev20: 240 same_dist_pair_c25_2_4_prev20
   + dist_diff_abs_pair_c25_2_4_prev20 >= 10
 dist_diff_pair_def_c25_2_4_prev21: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev21 = -24
 dist_diff_abs_pair_c25_2_4_prev21_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev21 + dist_diff_abs_pair_c25_2_4_prev21
   - 240 dist_diff_abs_pair_c25_2_4_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev21_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev21 + dist_diff_abs_pair_c25_2_4_prev21
   + 240 dist_diff_abs_pair_c25_2_4_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev21 + dist_diff_abs_pair_c25_2_4_prev21
   + 240 dist_diff_abs_pair_c25_2_4_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev21 + dist_diff_abs_pair_c25_2_4_prev21
   - 240 dist_diff_abs_pair_c25_2_4_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev21_force_positive:
   dist_diff_pair_c25_2_4_prev21
   - 240 dist_diff_abs_pair_c25_2_4_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev21_force_negative:
   dist_diff_pair_c25_2_4_prev21
   - 240 dist_diff_abs_pair_c25_2_4_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev21: 240 same_dist_pair_c25_2_4_prev21
   + dist_diff_abs_pair_c25_2_4_prev21 <= 249.8
 same_dist_pair_lower_c25_2_4_prev21: 240 same_dist_pair_c25_2_4_prev21
   + dist_diff_abs_pair_c25_2_4_prev21 >= 10
 dist_diff_pair_def_c25_2_4_prev22: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev22 = -24
 dist_diff_abs_pair_c25_2_4_prev22_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev22 + dist_diff_abs_pair_c25_2_4_prev22
   - 240 dist_diff_abs_pair_c25_2_4_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev22_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev22 + dist_diff_abs_pair_c25_2_4_prev22
   + 240 dist_diff_abs_pair_c25_2_4_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev22 + dist_diff_abs_pair_c25_2_4_prev22
   + 240 dist_diff_abs_pair_c25_2_4_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev22 + dist_diff_abs_pair_c25_2_4_prev22
   - 240 dist_diff_abs_pair_c25_2_4_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev22_force_positive:
   dist_diff_pair_c25_2_4_prev22
   - 240 dist_diff_abs_pair_c25_2_4_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev22_force_negative:
   dist_diff_pair_c25_2_4_prev22
   - 240 dist_diff_abs_pair_c25_2_4_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev22: 240 same_dist_pair_c25_2_4_prev22
   + dist_diff_abs_pair_c25_2_4_prev22 <= 249.8
 same_dist_pair_lower_c25_2_4_prev22: 240 same_dist_pair_c25_2_4_prev22
   + dist_diff_abs_pair_c25_2_4_prev22 >= 10
 dist_diff_pair_def_c25_2_4_prev23: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev23 = -24
 dist_diff_abs_pair_c25_2_4_prev23_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev23 + dist_diff_abs_pair_c25_2_4_prev23
   - 240 dist_diff_abs_pair_c25_2_4_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev23_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev23 + dist_diff_abs_pair_c25_2_4_prev23
   + 240 dist_diff_abs_pair_c25_2_4_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev23 + dist_diff_abs_pair_c25_2_4_prev23
   + 240 dist_diff_abs_pair_c25_2_4_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev23 + dist_diff_abs_pair_c25_2_4_prev23
   - 240 dist_diff_abs_pair_c25_2_4_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev23_force_positive:
   dist_diff_pair_c25_2_4_prev23
   - 240 dist_diff_abs_pair_c25_2_4_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev23_force_negative:
   dist_diff_pair_c25_2_4_prev23
   - 240 dist_diff_abs_pair_c25_2_4_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev23: 240 same_dist_pair_c25_2_4_prev23
   + dist_diff_abs_pair_c25_2_4_prev23 <= 249.8
 same_dist_pair_lower_c25_2_4_prev23: 240 same_dist_pair_c25_2_4_prev23
   + dist_diff_abs_pair_c25_2_4_prev23 >= 10
 dist_diff_pair_def_c25_2_4_prev24: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev24 = -36
 dist_diff_abs_pair_c25_2_4_prev24_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev24 + dist_diff_abs_pair_c25_2_4_prev24
   - 240 dist_diff_abs_pair_c25_2_4_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev24_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev24 + dist_diff_abs_pair_c25_2_4_prev24
   + 240 dist_diff_abs_pair_c25_2_4_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev24 + dist_diff_abs_pair_c25_2_4_prev24
   + 240 dist_diff_abs_pair_c25_2_4_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev24 + dist_diff_abs_pair_c25_2_4_prev24
   - 240 dist_diff_abs_pair_c25_2_4_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev24_force_positive:
   dist_diff_pair_c25_2_4_prev24
   - 240 dist_diff_abs_pair_c25_2_4_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev24_force_negative:
   dist_diff_pair_c25_2_4_prev24
   - 240 dist_diff_abs_pair_c25_2_4_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev24: 240 same_dist_pair_c25_2_4_prev24
   + dist_diff_abs_pair_c25_2_4_prev24 <= 249.8
 same_dist_pair_lower_c25_2_4_prev24: 240 same_dist_pair_c25_2_4_prev24
   + dist_diff_abs_pair_c25_2_4_prev24 >= 10
 dist_diff_pair_def_c25_2_4_prev25: - dist_curr_pair_c25_2_4
   + dist_diff_pair_c25_2_4_prev25 = -41
 dist_diff_abs_pair_c25_2_4_prev25_abs_ge_orig:
   - dist_diff_pair_c25_2_4_prev25 + dist_diff_abs_pair_c25_2_4_prev25
   - 240 dist_diff_abs_pair_c25_2_4_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev25_abs_le_orig:
   - dist_diff_pair_c25_2_4_prev25 + dist_diff_abs_pair_c25_2_4_prev25
   + 240 dist_diff_abs_pair_c25_2_4_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_2_4_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_2_4_prev25 + dist_diff_abs_pair_c25_2_4_prev25
   + 240 dist_diff_abs_pair_c25_2_4_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_2_4_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_2_4_prev25 + dist_diff_abs_pair_c25_2_4_prev25
   - 240 dist_diff_abs_pair_c25_2_4_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_2_4_prev25_force_positive:
   dist_diff_pair_c25_2_4_prev25
   - 240 dist_diff_abs_pair_c25_2_4_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_2_4_prev25_force_negative:
   dist_diff_pair_c25_2_4_prev25
   - 240 dist_diff_abs_pair_c25_2_4_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_2_4_prev25: 240 same_dist_pair_c25_2_4_prev25
   + dist_diff_abs_pair_c25_2_4_prev25 <= 249.8
 same_dist_pair_lower_c25_2_4_prev25: 240 same_dist_pair_c25_2_4_prev25
   + dist_diff_abs_pair_c25_2_4_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_2_4_prev0: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev0: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev1: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev1: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev2: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev2: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev3: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev3: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev4: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev4: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev5: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev5: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev6: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev6: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev7: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev7: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev8: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev8: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev9: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev9: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev10: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev10: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev11: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev11: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev12: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev12: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev13: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev13: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev14: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev14: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev15: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev15: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev16: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev16: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev17: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev17: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev18: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev18: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev19: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev19: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev20: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev20: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev21: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev21: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev22: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev22: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev23: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev23: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev24: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev24: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_2_4_prev25: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_2_4_prev25: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_2_4: diff_dist_pair_c25_2_4
   + same_dist_pair_c25_2_4_prev0 + same_dist_pair_c25_2_4_prev1
   + same_dist_pair_c25_2_4_prev2 + same_dist_pair_c25_2_4_prev3
   + same_dist_pair_c25_2_4_prev4 + same_dist_pair_c25_2_4_prev5
   + same_dist_pair_c25_2_4_prev6 + same_dist_pair_c25_2_4_prev7
   + same_dist_pair_c25_2_4_prev8 + same_dist_pair_c25_2_4_prev9
   + same_dist_pair_c25_2_4_prev10 + same_dist_pair_c25_2_4_prev11
   + same_dist_pair_c25_2_4_prev12 + same_dist_pair_c25_2_4_prev13
   + same_dist_pair_c25_2_4_prev14 + same_dist_pair_c25_2_4_prev15
   + same_dist_pair_c25_2_4_prev16 + same_dist_pair_c25_2_4_prev17
   + same_dist_pair_c25_2_4_prev18 + same_dist_pair_c25_2_4_prev19
   + same_dist_pair_c25_2_4_prev20 + same_dist_pair_c25_2_4_prev21
   + same_dist_pair_c25_2_4_prev22 + same_dist_pair_c25_2_4_prev23
   + same_dist_pair_c25_2_4_prev24 + same_dist_pair_c25_2_4_prev25 >= 1
 dx_grid_diff_def_c25_3_4: - cx_grid_center_c25_3 + cx_grid_center_c25_4
   + dx_grid_diff_c25_3_4 = 0
 dx_grid_abs_pair_c25_3_4_abs_ge_orig: dx_grid_abs_pair_c25_3_4
   - dx_grid_diff_c25_3_4 - 130 dx_grid_abs_pair_c25_3_4_is_positive
   >= -130
 dx_grid_abs_pair_c25_3_4_abs_le_orig: dx_grid_abs_pair_c25_3_4
   - dx_grid_diff_c25_3_4 + 130 dx_grid_abs_pair_c25_3_4_is_positive
   <= 130
 dx_grid_abs_pair_c25_3_4_abs_ge_neg_orig: dx_grid_abs_pair_c25_3_4
   + dx_grid_diff_c25_3_4 + 130 dx_grid_abs_pair_c25_3_4_is_positive >= 0
 dx_grid_abs_pair_c25_3_4_abs_le_neg_orig: dx_grid_abs_pair_c25_3_4
   + dx_grid_diff_c25_3_4 - 130 dx_grid_abs_pair_c25_3_4_is_positive <= 0
 dx_grid_abs_pair_c25_3_4_force_positive: dx_grid_diff_c25_3_4
   - 130 dx_grid_abs_pair_c25_3_4_is_positive >= -130
 dx_grid_abs_pair_c25_3_4_force_negative: dx_grid_diff_c25_3_4
   - 130 dx_grid_abs_pair_c25_3_4_is_positive <= -0.001
 dy_grid_diff_def_c25_3_4: - cy_grid_center_c25_3 + cy_grid_center_c25_4
   + dy_grid_diff_c25_3_4 = 0
 dy_grid_abs_pair_c25_3_4_abs_ge_orig: dy_grid_abs_pair_c25_3_4
   - dy_grid_diff_c25_3_4 - 130 dy_grid_abs_pair_c25_3_4_is_positive
   >= -130
 dy_grid_abs_pair_c25_3_4_abs_le_orig: dy_grid_abs_pair_c25_3_4
   - dy_grid_diff_c25_3_4 + 130 dy_grid_abs_pair_c25_3_4_is_positive
   <= 130
 dy_grid_abs_pair_c25_3_4_abs_ge_neg_orig: dy_grid_abs_pair_c25_3_4
   + dy_grid_diff_c25_3_4 + 130 dy_grid_abs_pair_c25_3_4_is_positive >= 0
 dy_grid_abs_pair_c25_3_4_abs_le_neg_orig: dy_grid_abs_pair_c25_3_4
   + dy_grid_diff_c25_3_4 - 130 dy_grid_abs_pair_c25_3_4_is_positive <= 0
 dy_grid_abs_pair_c25_3_4_force_positive: dy_grid_diff_c25_3_4
   - 130 dy_grid_abs_pair_c25_3_4_is_positive >= -130
 dy_grid_abs_pair_c25_3_4_force_negative: dy_grid_diff_c25_3_4
   - 130 dy_grid_abs_pair_c25_3_4_is_positive <= -0.001
 dist_curr_pair_def_c25_3_4: - dx_grid_abs_pair_c25_3_4
   - dy_grid_abs_pair_c25_3_4 + dist_curr_pair_c25_3_4 = 0
 dist_diff_pair_def_c25_3_4_prev0: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev0 = -40
 dist_diff_abs_pair_c25_3_4_prev0_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev0 + dist_diff_abs_pair_c25_3_4_prev0
   - 240 dist_diff_abs_pair_c25_3_4_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev0_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev0 + dist_diff_abs_pair_c25_3_4_prev0
   + 240 dist_diff_abs_pair_c25_3_4_prev0_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev0_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev0 + dist_diff_abs_pair_c25_3_4_prev0
   + 240 dist_diff_abs_pair_c25_3_4_prev0_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev0_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev0 + dist_diff_abs_pair_c25_3_4_prev0
   - 240 dist_diff_abs_pair_c25_3_4_prev0_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev0_force_positive:
   dist_diff_pair_c25_3_4_prev0
   - 240 dist_diff_abs_pair_c25_3_4_prev0_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev0_force_negative:
   dist_diff_pair_c25_3_4_prev0
   - 240 dist_diff_abs_pair_c25_3_4_prev0_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev0: 240 same_dist_pair_c25_3_4_prev0
   + dist_diff_abs_pair_c25_3_4_prev0 <= 249.8
 same_dist_pair_lower_c25_3_4_prev0: 240 same_dist_pair_c25_3_4_prev0
   + dist_diff_abs_pair_c25_3_4_prev0 >= 10
 dist_diff_pair_def_c25_3_4_prev1: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev1 = -74
 dist_diff_abs_pair_c25_3_4_prev1_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev1 + dist_diff_abs_pair_c25_3_4_prev1
   - 240 dist_diff_abs_pair_c25_3_4_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev1_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev1 + dist_diff_abs_pair_c25_3_4_prev1
   + 240 dist_diff_abs_pair_c25_3_4_prev1_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev1_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev1 + dist_diff_abs_pair_c25_3_4_prev1
   + 240 dist_diff_abs_pair_c25_3_4_prev1_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev1_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev1 + dist_diff_abs_pair_c25_3_4_prev1
   - 240 dist_diff_abs_pair_c25_3_4_prev1_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev1_force_positive:
   dist_diff_pair_c25_3_4_prev1
   - 240 dist_diff_abs_pair_c25_3_4_prev1_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev1_force_negative:
   dist_diff_pair_c25_3_4_prev1
   - 240 dist_diff_abs_pair_c25_3_4_prev1_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev1: 240 same_dist_pair_c25_3_4_prev1
   + dist_diff_abs_pair_c25_3_4_prev1 <= 249.8
 same_dist_pair_lower_c25_3_4_prev1: 240 same_dist_pair_c25_3_4_prev1
   + dist_diff_abs_pair_c25_3_4_prev1 >= 10
 dist_diff_pair_def_c25_3_4_prev2: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev2 = -38
 dist_diff_abs_pair_c25_3_4_prev2_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev2 + dist_diff_abs_pair_c25_3_4_prev2
   - 240 dist_diff_abs_pair_c25_3_4_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev2_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev2 + dist_diff_abs_pair_c25_3_4_prev2
   + 240 dist_diff_abs_pair_c25_3_4_prev2_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev2_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev2 + dist_diff_abs_pair_c25_3_4_prev2
   + 240 dist_diff_abs_pair_c25_3_4_prev2_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev2_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev2 + dist_diff_abs_pair_c25_3_4_prev2
   - 240 dist_diff_abs_pair_c25_3_4_prev2_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev2_force_positive:
   dist_diff_pair_c25_3_4_prev2
   - 240 dist_diff_abs_pair_c25_3_4_prev2_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev2_force_negative:
   dist_diff_pair_c25_3_4_prev2
   - 240 dist_diff_abs_pair_c25_3_4_prev2_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev2: 240 same_dist_pair_c25_3_4_prev2
   + dist_diff_abs_pair_c25_3_4_prev2 <= 249.8
 same_dist_pair_lower_c25_3_4_prev2: 240 same_dist_pair_c25_3_4_prev2
   + dist_diff_abs_pair_c25_3_4_prev2 >= 10
 dist_diff_pair_def_c25_3_4_prev3: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev3 = -50
 dist_diff_abs_pair_c25_3_4_prev3_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev3 + dist_diff_abs_pair_c25_3_4_prev3
   - 240 dist_diff_abs_pair_c25_3_4_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev3_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev3 + dist_diff_abs_pair_c25_3_4_prev3
   + 240 dist_diff_abs_pair_c25_3_4_prev3_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev3_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev3 + dist_diff_abs_pair_c25_3_4_prev3
   + 240 dist_diff_abs_pair_c25_3_4_prev3_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev3_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev3 + dist_diff_abs_pair_c25_3_4_prev3
   - 240 dist_diff_abs_pair_c25_3_4_prev3_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev3_force_positive:
   dist_diff_pair_c25_3_4_prev3
   - 240 dist_diff_abs_pair_c25_3_4_prev3_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev3_force_negative:
   dist_diff_pair_c25_3_4_prev3
   - 240 dist_diff_abs_pair_c25_3_4_prev3_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev3: 240 same_dist_pair_c25_3_4_prev3
   + dist_diff_abs_pair_c25_3_4_prev3 <= 249.8
 same_dist_pair_lower_c25_3_4_prev3: 240 same_dist_pair_c25_3_4_prev3
   + dist_diff_abs_pair_c25_3_4_prev3 >= 10
 dist_diff_pair_def_c25_3_4_prev4: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev4 = -74
 dist_diff_abs_pair_c25_3_4_prev4_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev4 + dist_diff_abs_pair_c25_3_4_prev4
   - 240 dist_diff_abs_pair_c25_3_4_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev4_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev4 + dist_diff_abs_pair_c25_3_4_prev4
   + 240 dist_diff_abs_pair_c25_3_4_prev4_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev4_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev4 + dist_diff_abs_pair_c25_3_4_prev4
   + 240 dist_diff_abs_pair_c25_3_4_prev4_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev4_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev4 + dist_diff_abs_pair_c25_3_4_prev4
   - 240 dist_diff_abs_pair_c25_3_4_prev4_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev4_force_positive:
   dist_diff_pair_c25_3_4_prev4
   - 240 dist_diff_abs_pair_c25_3_4_prev4_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev4_force_negative:
   dist_diff_pair_c25_3_4_prev4
   - 240 dist_diff_abs_pair_c25_3_4_prev4_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev4: 240 same_dist_pair_c25_3_4_prev4
   + dist_diff_abs_pair_c25_3_4_prev4 <= 249.8
 same_dist_pair_lower_c25_3_4_prev4: 240 same_dist_pair_c25_3_4_prev4
   + dist_diff_abs_pair_c25_3_4_prev4 >= 10
 dist_diff_pair_def_c25_3_4_prev5: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev5 = -30
 dist_diff_abs_pair_c25_3_4_prev5_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev5 + dist_diff_abs_pair_c25_3_4_prev5
   - 240 dist_diff_abs_pair_c25_3_4_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev5_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev5 + dist_diff_abs_pair_c25_3_4_prev5
   + 240 dist_diff_abs_pair_c25_3_4_prev5_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev5_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev5 + dist_diff_abs_pair_c25_3_4_prev5
   + 240 dist_diff_abs_pair_c25_3_4_prev5_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev5_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev5 + dist_diff_abs_pair_c25_3_4_prev5
   - 240 dist_diff_abs_pair_c25_3_4_prev5_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev5_force_positive:
   dist_diff_pair_c25_3_4_prev5
   - 240 dist_diff_abs_pair_c25_3_4_prev5_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev5_force_negative:
   dist_diff_pair_c25_3_4_prev5
   - 240 dist_diff_abs_pair_c25_3_4_prev5_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev5: 240 same_dist_pair_c25_3_4_prev5
   + dist_diff_abs_pair_c25_3_4_prev5 <= 249.8
 same_dist_pair_lower_c25_3_4_prev5: 240 same_dist_pair_c25_3_4_prev5
   + dist_diff_abs_pair_c25_3_4_prev5 >= 10
 dist_diff_pair_def_c25_3_4_prev6: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev6 = -50
 dist_diff_abs_pair_c25_3_4_prev6_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev6 + dist_diff_abs_pair_c25_3_4_prev6
   - 240 dist_diff_abs_pair_c25_3_4_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev6_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev6 + dist_diff_abs_pair_c25_3_4_prev6
   + 240 dist_diff_abs_pair_c25_3_4_prev6_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev6_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev6 + dist_diff_abs_pair_c25_3_4_prev6
   + 240 dist_diff_abs_pair_c25_3_4_prev6_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev6_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev6 + dist_diff_abs_pair_c25_3_4_prev6
   - 240 dist_diff_abs_pair_c25_3_4_prev6_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev6_force_positive:
   dist_diff_pair_c25_3_4_prev6
   - 240 dist_diff_abs_pair_c25_3_4_prev6_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev6_force_negative:
   dist_diff_pair_c25_3_4_prev6
   - 240 dist_diff_abs_pair_c25_3_4_prev6_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev6: 240 same_dist_pair_c25_3_4_prev6
   + dist_diff_abs_pair_c25_3_4_prev6 <= 249.8
 same_dist_pair_lower_c25_3_4_prev6: 240 same_dist_pair_c25_3_4_prev6
   + dist_diff_abs_pair_c25_3_4_prev6 >= 10
 dist_diff_pair_def_c25_3_4_prev7: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev7 = -30
 dist_diff_abs_pair_c25_3_4_prev7_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev7 + dist_diff_abs_pair_c25_3_4_prev7
   - 240 dist_diff_abs_pair_c25_3_4_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev7_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev7 + dist_diff_abs_pair_c25_3_4_prev7
   + 240 dist_diff_abs_pair_c25_3_4_prev7_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev7_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev7 + dist_diff_abs_pair_c25_3_4_prev7
   + 240 dist_diff_abs_pair_c25_3_4_prev7_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev7_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev7 + dist_diff_abs_pair_c25_3_4_prev7
   - 240 dist_diff_abs_pair_c25_3_4_prev7_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev7_force_positive:
   dist_diff_pair_c25_3_4_prev7
   - 240 dist_diff_abs_pair_c25_3_4_prev7_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev7_force_negative:
   dist_diff_pair_c25_3_4_prev7
   - 240 dist_diff_abs_pair_c25_3_4_prev7_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev7: 240 same_dist_pair_c25_3_4_prev7
   + dist_diff_abs_pair_c25_3_4_prev7 <= 249.8
 same_dist_pair_lower_c25_3_4_prev7: 240 same_dist_pair_c25_3_4_prev7
   + dist_diff_abs_pair_c25_3_4_prev7 >= 10
 dist_diff_pair_def_c25_3_4_prev8: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev8 = -73
 dist_diff_abs_pair_c25_3_4_prev8_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev8 + dist_diff_abs_pair_c25_3_4_prev8
   - 240 dist_diff_abs_pair_c25_3_4_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev8_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev8 + dist_diff_abs_pair_c25_3_4_prev8
   + 240 dist_diff_abs_pair_c25_3_4_prev8_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev8_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev8 + dist_diff_abs_pair_c25_3_4_prev8
   + 240 dist_diff_abs_pair_c25_3_4_prev8_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev8_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev8 + dist_diff_abs_pair_c25_3_4_prev8
   - 240 dist_diff_abs_pair_c25_3_4_prev8_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev8_force_positive:
   dist_diff_pair_c25_3_4_prev8
   - 240 dist_diff_abs_pair_c25_3_4_prev8_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev8_force_negative:
   dist_diff_pair_c25_3_4_prev8
   - 240 dist_diff_abs_pair_c25_3_4_prev8_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev8: 240 same_dist_pair_c25_3_4_prev8
   + dist_diff_abs_pair_c25_3_4_prev8 <= 249.8
 same_dist_pair_lower_c25_3_4_prev8: 240 same_dist_pair_c25_3_4_prev8
   + dist_diff_abs_pair_c25_3_4_prev8 >= 10
 dist_diff_pair_def_c25_3_4_prev9: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev9 = -30
 dist_diff_abs_pair_c25_3_4_prev9_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev9 + dist_diff_abs_pair_c25_3_4_prev9
   - 240 dist_diff_abs_pair_c25_3_4_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev9_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev9 + dist_diff_abs_pair_c25_3_4_prev9
   + 240 dist_diff_abs_pair_c25_3_4_prev9_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev9_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev9 + dist_diff_abs_pair_c25_3_4_prev9
   + 240 dist_diff_abs_pair_c25_3_4_prev9_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev9_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev9 + dist_diff_abs_pair_c25_3_4_prev9
   - 240 dist_diff_abs_pair_c25_3_4_prev9_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev9_force_positive:
   dist_diff_pair_c25_3_4_prev9
   - 240 dist_diff_abs_pair_c25_3_4_prev9_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev9_force_negative:
   dist_diff_pair_c25_3_4_prev9
   - 240 dist_diff_abs_pair_c25_3_4_prev9_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev9: 240 same_dist_pair_c25_3_4_prev9
   + dist_diff_abs_pair_c25_3_4_prev9 <= 249.8
 same_dist_pair_lower_c25_3_4_prev9: 240 same_dist_pair_c25_3_4_prev9
   + dist_diff_abs_pair_c25_3_4_prev9 >= 10
 dist_diff_pair_def_c25_3_4_prev10: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev10 = -63
 dist_diff_abs_pair_c25_3_4_prev10_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev10 + dist_diff_abs_pair_c25_3_4_prev10
   - 240 dist_diff_abs_pair_c25_3_4_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev10_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev10 + dist_diff_abs_pair_c25_3_4_prev10
   + 240 dist_diff_abs_pair_c25_3_4_prev10_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev10_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev10 + dist_diff_abs_pair_c25_3_4_prev10
   + 240 dist_diff_abs_pair_c25_3_4_prev10_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev10_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev10 + dist_diff_abs_pair_c25_3_4_prev10
   - 240 dist_diff_abs_pair_c25_3_4_prev10_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev10_force_positive:
   dist_diff_pair_c25_3_4_prev10
   - 240 dist_diff_abs_pair_c25_3_4_prev10_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev10_force_negative:
   dist_diff_pair_c25_3_4_prev10
   - 240 dist_diff_abs_pair_c25_3_4_prev10_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev10: 240 same_dist_pair_c25_3_4_prev10
   + dist_diff_abs_pair_c25_3_4_prev10 <= 249.8
 same_dist_pair_lower_c25_3_4_prev10: 240 same_dist_pair_c25_3_4_prev10
   + dist_diff_abs_pair_c25_3_4_prev10 >= 10
 dist_diff_pair_def_c25_3_4_prev11: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev11 = -22
 dist_diff_abs_pair_c25_3_4_prev11_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev11 + dist_diff_abs_pair_c25_3_4_prev11
   - 240 dist_diff_abs_pair_c25_3_4_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev11_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev11 + dist_diff_abs_pair_c25_3_4_prev11
   + 240 dist_diff_abs_pair_c25_3_4_prev11_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev11_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev11 + dist_diff_abs_pair_c25_3_4_prev11
   + 240 dist_diff_abs_pair_c25_3_4_prev11_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev11_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev11 + dist_diff_abs_pair_c25_3_4_prev11
   - 240 dist_diff_abs_pair_c25_3_4_prev11_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev11_force_positive:
   dist_diff_pair_c25_3_4_prev11
   - 240 dist_diff_abs_pair_c25_3_4_prev11_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev11_force_negative:
   dist_diff_pair_c25_3_4_prev11
   - 240 dist_diff_abs_pair_c25_3_4_prev11_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev11: 240 same_dist_pair_c25_3_4_prev11
   + dist_diff_abs_pair_c25_3_4_prev11 <= 249.8
 same_dist_pair_lower_c25_3_4_prev11: 240 same_dist_pair_c25_3_4_prev11
   + dist_diff_abs_pair_c25_3_4_prev11 >= 10
 dist_diff_pair_def_c25_3_4_prev12: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev12 = -78
 dist_diff_abs_pair_c25_3_4_prev12_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev12 + dist_diff_abs_pair_c25_3_4_prev12
   - 240 dist_diff_abs_pair_c25_3_4_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev12_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev12 + dist_diff_abs_pair_c25_3_4_prev12
   + 240 dist_diff_abs_pair_c25_3_4_prev12_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev12_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev12 + dist_diff_abs_pair_c25_3_4_prev12
   + 240 dist_diff_abs_pair_c25_3_4_prev12_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev12_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev12 + dist_diff_abs_pair_c25_3_4_prev12
   - 240 dist_diff_abs_pair_c25_3_4_prev12_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev12_force_positive:
   dist_diff_pair_c25_3_4_prev12
   - 240 dist_diff_abs_pair_c25_3_4_prev12_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev12_force_negative:
   dist_diff_pair_c25_3_4_prev12
   - 240 dist_diff_abs_pair_c25_3_4_prev12_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev12: 240 same_dist_pair_c25_3_4_prev12
   + dist_diff_abs_pair_c25_3_4_prev12 <= 249.8
 same_dist_pair_lower_c25_3_4_prev12: 240 same_dist_pair_c25_3_4_prev12
   + dist_diff_abs_pair_c25_3_4_prev12 >= 10
 dist_diff_pair_def_c25_3_4_prev13: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev13 = -60
 dist_diff_abs_pair_c25_3_4_prev13_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev13 + dist_diff_abs_pair_c25_3_4_prev13
   - 240 dist_diff_abs_pair_c25_3_4_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev13_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev13 + dist_diff_abs_pair_c25_3_4_prev13
   + 240 dist_diff_abs_pair_c25_3_4_prev13_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev13_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev13 + dist_diff_abs_pair_c25_3_4_prev13
   + 240 dist_diff_abs_pair_c25_3_4_prev13_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev13_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev13 + dist_diff_abs_pair_c25_3_4_prev13
   - 240 dist_diff_abs_pair_c25_3_4_prev13_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev13_force_positive:
   dist_diff_pair_c25_3_4_prev13
   - 240 dist_diff_abs_pair_c25_3_4_prev13_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev13_force_negative:
   dist_diff_pair_c25_3_4_prev13
   - 240 dist_diff_abs_pair_c25_3_4_prev13_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev13: 240 same_dist_pair_c25_3_4_prev13
   + dist_diff_abs_pair_c25_3_4_prev13 <= 249.8
 same_dist_pair_lower_c25_3_4_prev13: 240 same_dist_pair_c25_3_4_prev13
   + dist_diff_abs_pair_c25_3_4_prev13 >= 10
 dist_diff_pair_def_c25_3_4_prev14: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev14 = -68
 dist_diff_abs_pair_c25_3_4_prev14_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev14 + dist_diff_abs_pair_c25_3_4_prev14
   - 240 dist_diff_abs_pair_c25_3_4_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev14_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev14 + dist_diff_abs_pair_c25_3_4_prev14
   + 240 dist_diff_abs_pair_c25_3_4_prev14_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev14_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev14 + dist_diff_abs_pair_c25_3_4_prev14
   + 240 dist_diff_abs_pair_c25_3_4_prev14_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev14_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev14 + dist_diff_abs_pair_c25_3_4_prev14
   - 240 dist_diff_abs_pair_c25_3_4_prev14_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev14_force_positive:
   dist_diff_pair_c25_3_4_prev14
   - 240 dist_diff_abs_pair_c25_3_4_prev14_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev14_force_negative:
   dist_diff_pair_c25_3_4_prev14
   - 240 dist_diff_abs_pair_c25_3_4_prev14_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev14: 240 same_dist_pair_c25_3_4_prev14
   + dist_diff_abs_pair_c25_3_4_prev14 <= 249.8
 same_dist_pair_lower_c25_3_4_prev14: 240 same_dist_pair_c25_3_4_prev14
   + dist_diff_abs_pair_c25_3_4_prev14 >= 10
 dist_diff_pair_def_c25_3_4_prev15: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev15 = -74
 dist_diff_abs_pair_c25_3_4_prev15_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev15 + dist_diff_abs_pair_c25_3_4_prev15
   - 240 dist_diff_abs_pair_c25_3_4_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev15_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev15 + dist_diff_abs_pair_c25_3_4_prev15
   + 240 dist_diff_abs_pair_c25_3_4_prev15_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev15_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev15 + dist_diff_abs_pair_c25_3_4_prev15
   + 240 dist_diff_abs_pair_c25_3_4_prev15_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev15_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev15 + dist_diff_abs_pair_c25_3_4_prev15
   - 240 dist_diff_abs_pair_c25_3_4_prev15_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev15_force_positive:
   dist_diff_pair_c25_3_4_prev15
   - 240 dist_diff_abs_pair_c25_3_4_prev15_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev15_force_negative:
   dist_diff_pair_c25_3_4_prev15
   - 240 dist_diff_abs_pair_c25_3_4_prev15_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev15: 240 same_dist_pair_c25_3_4_prev15
   + dist_diff_abs_pair_c25_3_4_prev15 <= 249.8
 same_dist_pair_lower_c25_3_4_prev15: 240 same_dist_pair_c25_3_4_prev15
   + dist_diff_abs_pair_c25_3_4_prev15 >= 10
 dist_diff_pair_def_c25_3_4_prev16: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev16 = -24
 dist_diff_abs_pair_c25_3_4_prev16_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev16 + dist_diff_abs_pair_c25_3_4_prev16
   - 240 dist_diff_abs_pair_c25_3_4_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev16_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev16 + dist_diff_abs_pair_c25_3_4_prev16
   + 240 dist_diff_abs_pair_c25_3_4_prev16_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev16_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev16 + dist_diff_abs_pair_c25_3_4_prev16
   + 240 dist_diff_abs_pair_c25_3_4_prev16_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev16_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev16 + dist_diff_abs_pair_c25_3_4_prev16
   - 240 dist_diff_abs_pair_c25_3_4_prev16_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev16_force_positive:
   dist_diff_pair_c25_3_4_prev16
   - 240 dist_diff_abs_pair_c25_3_4_prev16_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev16_force_negative:
   dist_diff_pair_c25_3_4_prev16
   - 240 dist_diff_abs_pair_c25_3_4_prev16_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev16: 240 same_dist_pair_c25_3_4_prev16
   + dist_diff_abs_pair_c25_3_4_prev16 <= 249.8
 same_dist_pair_lower_c25_3_4_prev16: 240 same_dist_pair_c25_3_4_prev16
   + dist_diff_abs_pair_c25_3_4_prev16 >= 10
 dist_diff_pair_def_c25_3_4_prev17: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev17 = -54
 dist_diff_abs_pair_c25_3_4_prev17_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev17 + dist_diff_abs_pair_c25_3_4_prev17
   - 240 dist_diff_abs_pair_c25_3_4_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev17_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev17 + dist_diff_abs_pair_c25_3_4_prev17
   + 240 dist_diff_abs_pair_c25_3_4_prev17_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev17_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev17 + dist_diff_abs_pair_c25_3_4_prev17
   + 240 dist_diff_abs_pair_c25_3_4_prev17_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev17_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev17 + dist_diff_abs_pair_c25_3_4_prev17
   - 240 dist_diff_abs_pair_c25_3_4_prev17_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev17_force_positive:
   dist_diff_pair_c25_3_4_prev17
   - 240 dist_diff_abs_pair_c25_3_4_prev17_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev17_force_negative:
   dist_diff_pair_c25_3_4_prev17
   - 240 dist_diff_abs_pair_c25_3_4_prev17_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev17: 240 same_dist_pair_c25_3_4_prev17
   + dist_diff_abs_pair_c25_3_4_prev17 <= 249.8
 same_dist_pair_lower_c25_3_4_prev17: 240 same_dist_pair_c25_3_4_prev17
   + dist_diff_abs_pair_c25_3_4_prev17 >= 10
 dist_diff_pair_def_c25_3_4_prev18: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev18 = -57
 dist_diff_abs_pair_c25_3_4_prev18_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev18 + dist_diff_abs_pair_c25_3_4_prev18
   - 240 dist_diff_abs_pair_c25_3_4_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev18_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev18 + dist_diff_abs_pair_c25_3_4_prev18
   + 240 dist_diff_abs_pair_c25_3_4_prev18_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev18_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev18 + dist_diff_abs_pair_c25_3_4_prev18
   + 240 dist_diff_abs_pair_c25_3_4_prev18_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev18_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev18 + dist_diff_abs_pair_c25_3_4_prev18
   - 240 dist_diff_abs_pair_c25_3_4_prev18_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev18_force_positive:
   dist_diff_pair_c25_3_4_prev18
   - 240 dist_diff_abs_pair_c25_3_4_prev18_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev18_force_negative:
   dist_diff_pair_c25_3_4_prev18
   - 240 dist_diff_abs_pair_c25_3_4_prev18_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev18: 240 same_dist_pair_c25_3_4_prev18
   + dist_diff_abs_pair_c25_3_4_prev18 <= 249.8
 same_dist_pair_lower_c25_3_4_prev18: 240 same_dist_pair_c25_3_4_prev18
   + dist_diff_abs_pair_c25_3_4_prev18 >= 10
 dist_diff_pair_def_c25_3_4_prev19: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev19 = -50
 dist_diff_abs_pair_c25_3_4_prev19_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev19 + dist_diff_abs_pair_c25_3_4_prev19
   - 240 dist_diff_abs_pair_c25_3_4_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev19_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev19 + dist_diff_abs_pair_c25_3_4_prev19
   + 240 dist_diff_abs_pair_c25_3_4_prev19_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev19_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev19 + dist_diff_abs_pair_c25_3_4_prev19
   + 240 dist_diff_abs_pair_c25_3_4_prev19_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev19_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev19 + dist_diff_abs_pair_c25_3_4_prev19
   - 240 dist_diff_abs_pair_c25_3_4_prev19_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev19_force_positive:
   dist_diff_pair_c25_3_4_prev19
   - 240 dist_diff_abs_pair_c25_3_4_prev19_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev19_force_negative:
   dist_diff_pair_c25_3_4_prev19
   - 240 dist_diff_abs_pair_c25_3_4_prev19_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev19: 240 same_dist_pair_c25_3_4_prev19
   + dist_diff_abs_pair_c25_3_4_prev19 <= 249.8
 same_dist_pair_lower_c25_3_4_prev19: 240 same_dist_pair_c25_3_4_prev19
   + dist_diff_abs_pair_c25_3_4_prev19 >= 10
 dist_diff_pair_def_c25_3_4_prev20: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev20 = -42
 dist_diff_abs_pair_c25_3_4_prev20_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev20 + dist_diff_abs_pair_c25_3_4_prev20
   - 240 dist_diff_abs_pair_c25_3_4_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev20_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev20 + dist_diff_abs_pair_c25_3_4_prev20
   + 240 dist_diff_abs_pair_c25_3_4_prev20_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev20_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev20 + dist_diff_abs_pair_c25_3_4_prev20
   + 240 dist_diff_abs_pair_c25_3_4_prev20_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev20_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev20 + dist_diff_abs_pair_c25_3_4_prev20
   - 240 dist_diff_abs_pair_c25_3_4_prev20_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev20_force_positive:
   dist_diff_pair_c25_3_4_prev20
   - 240 dist_diff_abs_pair_c25_3_4_prev20_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev20_force_negative:
   dist_diff_pair_c25_3_4_prev20
   - 240 dist_diff_abs_pair_c25_3_4_prev20_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev20: 240 same_dist_pair_c25_3_4_prev20
   + dist_diff_abs_pair_c25_3_4_prev20 <= 249.8
 same_dist_pair_lower_c25_3_4_prev20: 240 same_dist_pair_c25_3_4_prev20
   + dist_diff_abs_pair_c25_3_4_prev20 >= 10
 dist_diff_pair_def_c25_3_4_prev21: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev21 = -70
 dist_diff_abs_pair_c25_3_4_prev21_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev21 + dist_diff_abs_pair_c25_3_4_prev21
   - 240 dist_diff_abs_pair_c25_3_4_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev21_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev21 + dist_diff_abs_pair_c25_3_4_prev21
   + 240 dist_diff_abs_pair_c25_3_4_prev21_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev21_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev21 + dist_diff_abs_pair_c25_3_4_prev21
   + 240 dist_diff_abs_pair_c25_3_4_prev21_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev21_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev21 + dist_diff_abs_pair_c25_3_4_prev21
   - 240 dist_diff_abs_pair_c25_3_4_prev21_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev21_force_positive:
   dist_diff_pair_c25_3_4_prev21
   - 240 dist_diff_abs_pair_c25_3_4_prev21_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev21_force_negative:
   dist_diff_pair_c25_3_4_prev21
   - 240 dist_diff_abs_pair_c25_3_4_prev21_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev21: 240 same_dist_pair_c25_3_4_prev21
   + dist_diff_abs_pair_c25_3_4_prev21 <= 249.8
 same_dist_pair_lower_c25_3_4_prev21: 240 same_dist_pair_c25_3_4_prev21
   + dist_diff_abs_pair_c25_3_4_prev21 >= 10
 dist_diff_pair_def_c25_3_4_prev22: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev22 = -54
 dist_diff_abs_pair_c25_3_4_prev22_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev22 + dist_diff_abs_pair_c25_3_4_prev22
   - 240 dist_diff_abs_pair_c25_3_4_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev22_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev22 + dist_diff_abs_pair_c25_3_4_prev22
   + 240 dist_diff_abs_pair_c25_3_4_prev22_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev22_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev22 + dist_diff_abs_pair_c25_3_4_prev22
   + 240 dist_diff_abs_pair_c25_3_4_prev22_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev22_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev22 + dist_diff_abs_pair_c25_3_4_prev22
   - 240 dist_diff_abs_pair_c25_3_4_prev22_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev22_force_positive:
   dist_diff_pair_c25_3_4_prev22
   - 240 dist_diff_abs_pair_c25_3_4_prev22_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev22_force_negative:
   dist_diff_pair_c25_3_4_prev22
   - 240 dist_diff_abs_pair_c25_3_4_prev22_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev22: 240 same_dist_pair_c25_3_4_prev22
   + dist_diff_abs_pair_c25_3_4_prev22 <= 249.8
 same_dist_pair_lower_c25_3_4_prev22: 240 same_dist_pair_c25_3_4_prev22
   + dist_diff_abs_pair_c25_3_4_prev22 >= 10
 dist_diff_pair_def_c25_3_4_prev23: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev23 = -21
 dist_diff_abs_pair_c25_3_4_prev23_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev23 + dist_diff_abs_pair_c25_3_4_prev23
   - 240 dist_diff_abs_pair_c25_3_4_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev23_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev23 + dist_diff_abs_pair_c25_3_4_prev23
   + 240 dist_diff_abs_pair_c25_3_4_prev23_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev23_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev23 + dist_diff_abs_pair_c25_3_4_prev23
   + 240 dist_diff_abs_pair_c25_3_4_prev23_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev23_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev23 + dist_diff_abs_pair_c25_3_4_prev23
   - 240 dist_diff_abs_pair_c25_3_4_prev23_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev23_force_positive:
   dist_diff_pair_c25_3_4_prev23
   - 240 dist_diff_abs_pair_c25_3_4_prev23_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev23_force_negative:
   dist_diff_pair_c25_3_4_prev23
   - 240 dist_diff_abs_pair_c25_3_4_prev23_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev23: 240 same_dist_pair_c25_3_4_prev23
   + dist_diff_abs_pair_c25_3_4_prev23 <= 249.8
 same_dist_pair_lower_c25_3_4_prev23: 240 same_dist_pair_c25_3_4_prev23
   + dist_diff_abs_pair_c25_3_4_prev23 >= 10
 dist_diff_pair_def_c25_3_4_prev24: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev24 = -62
 dist_diff_abs_pair_c25_3_4_prev24_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev24 + dist_diff_abs_pair_c25_3_4_prev24
   - 240 dist_diff_abs_pair_c25_3_4_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev24_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev24 + dist_diff_abs_pair_c25_3_4_prev24
   + 240 dist_diff_abs_pair_c25_3_4_prev24_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev24_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev24 + dist_diff_abs_pair_c25_3_4_prev24
   + 240 dist_diff_abs_pair_c25_3_4_prev24_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev24_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev24 + dist_diff_abs_pair_c25_3_4_prev24
   - 240 dist_diff_abs_pair_c25_3_4_prev24_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev24_force_positive:
   dist_diff_pair_c25_3_4_prev24
   - 240 dist_diff_abs_pair_c25_3_4_prev24_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev24_force_negative:
   dist_diff_pair_c25_3_4_prev24
   - 240 dist_diff_abs_pair_c25_3_4_prev24_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev24: 240 same_dist_pair_c25_3_4_prev24
   + dist_diff_abs_pair_c25_3_4_prev24 <= 249.8
 same_dist_pair_lower_c25_3_4_prev24: 240 same_dist_pair_c25_3_4_prev24
   + dist_diff_abs_pair_c25_3_4_prev24 >= 10
 dist_diff_pair_def_c25_3_4_prev25: - dist_curr_pair_c25_3_4
   + dist_diff_pair_c25_3_4_prev25 = -75
 dist_diff_abs_pair_c25_3_4_prev25_abs_ge_orig:
   - dist_diff_pair_c25_3_4_prev25 + dist_diff_abs_pair_c25_3_4_prev25
   - 240 dist_diff_abs_pair_c25_3_4_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev25_abs_le_orig:
   - dist_diff_pair_c25_3_4_prev25 + dist_diff_abs_pair_c25_3_4_prev25
   + 240 dist_diff_abs_pair_c25_3_4_prev25_is_positive <= 240
 dist_diff_abs_pair_c25_3_4_prev25_abs_ge_neg_orig:
   dist_diff_pair_c25_3_4_prev25 + dist_diff_abs_pair_c25_3_4_prev25
   + 240 dist_diff_abs_pair_c25_3_4_prev25_is_positive >= 0
 dist_diff_abs_pair_c25_3_4_prev25_abs_le_neg_orig:
   dist_diff_pair_c25_3_4_prev25 + dist_diff_abs_pair_c25_3_4_prev25
   - 240 dist_diff_abs_pair_c25_3_4_prev25_is_positive <= 0
 dist_diff_abs_pair_c25_3_4_prev25_force_positive:
   dist_diff_pair_c25_3_4_prev25
   - 240 dist_diff_abs_pair_c25_3_4_prev25_is_positive >= -240
 dist_diff_abs_pair_c25_3_4_prev25_force_negative:
   dist_diff_pair_c25_3_4_prev25
   - 240 dist_diff_abs_pair_c25_3_4_prev25_is_positive <= -0.001
 same_dist_pair_upper_c25_3_4_prev25: 240 same_dist_pair_c25_3_4_prev25
   + dist_diff_abs_pair_c25_3_4_prev25 <= 249.8
 same_dist_pair_lower_c25_3_4_prev25: 240 same_dist_pair_c25_3_4_prev25
   + dist_diff_abs_pair_c25_3_4_prev25 >= 10
 diff_dist_pair_implies_not_same_c25_3_4_prev0: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev0 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev0: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev0 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev1: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev1 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev1: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev1 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev2: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev2 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev2: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev2 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev3: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev3 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev3: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev3 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev4: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev4 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev4: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev4 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev5: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev5 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev5: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev5 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev6: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev6 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev6: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev6 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev7: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev7 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev7: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev7 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev8: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev8 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev8: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev8 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev9: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev9 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev9: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev9 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev10: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev10 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev10: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev10 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev11: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev11 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev11: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev11 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev12: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev12 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev12: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev12 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev13: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev13 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev13: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev13 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev14: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev14 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev14: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev14 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev15: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev15 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev15: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev15 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev16: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev16 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev16: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev16 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev17: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev17 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev17: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev17 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev18: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev18 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev18: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev18 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev19: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev19 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev19: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev19 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev20: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev20 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev20: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev20 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev21: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev21 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev21: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev21 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev22: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev22 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev22: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev22 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev23: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev23 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev23: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev23 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev24: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev24 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev24: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev24 <= 1
 diff_dist_pair_implies_not_same_c25_3_4_prev25: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev25 <= 1
 not_same_implies_diff_dist_pair_c25_3_4_prev25: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev25 <= 1
 all_not_same_implies_diff_dist_pair_c25_3_4: diff_dist_pair_c25_3_4
   + same_dist_pair_c25_3_4_prev0 + same_dist_pair_c25_3_4_prev1
   + same_dist_pair_c25_3_4_prev2 + same_dist_pair_c25_3_4_prev3
   + same_dist_pair_c25_3_4_prev4 + same_dist_pair_c25_3_4_prev5
   + same_dist_pair_c25_3_4_prev6 + same_dist_pair_c25_3_4_prev7
   + same_dist_pair_c25_3_4_prev8 + same_dist_pair_c25_3_4_prev9
   + same_dist_pair_c25_3_4_prev10 + same_dist_pair_c25_3_4_prev11
   + same_dist_pair_c25_3_4_prev12 + same_dist_pair_c25_3_4_prev13
   + same_dist_pair_c25_3_4_prev14 + same_dist_pair_c25_3_4_prev15
   + same_dist_pair_c25_3_4_prev16 + same_dist_pair_c25_3_4_prev17
   + same_dist_pair_c25_3_4_prev18 + same_dist_pair_c25_3_4_prev19
   + same_dist_pair_c25_3_4_prev20 + same_dist_pair_c25_3_4_prev21
   + same_dist_pair_c25_3_4_prev22 + same_dist_pair_c25_3_4_prev23
   + same_dist_pair_c25_3_4_prev24 + same_dist_pair_c25_3_4_prev25 >= 1
 exclude_solution_dist_pair_c25: diff_dist_pair_c25_0_1
   + diff_dist_pair_c25_0_2 + diff_dist_pair_c25_0_3
   + diff_dist_pair_c25_0_4 + diff_dist_pair_c25_1_2
   + diff_dist_pair_c25_1_3 + diff_dist_pair_c25_1_4
   + diff_dist_pair_c25_2_3 + diff_dist_pair_c25_2_4
   + diff_dist_pair_c25_3_4 >= 1
Bounds
 x_grid_0 <= 119
 y_grid_0 <= 119
 x_grid_1 <= 119
 y_grid_1 <= 119
 x_grid_2 <= 119
 y_grid_2 <= 119
 x_grid_3 <= 119
 y_grid_3 <= 119
 x_grid_4 <= 119
 y_grid_4 <= 119
 x_0 <= 60
 y_0 <= 60
 x_1 <= 60
 y_1 <= 60
 x_2 <= 60
 y_2 <= 60
 x_3 <= 60
 y_3 <= 60
 x_4 <= 60
 y_4 <= 60
 w_0 = 16
 h_0 = 16
 10 <= w_1 <= 20
 10 <= h_1 <= 20
 w_2 = 20
 h_2 = 20
 14 <= w_3 <= 16
 14 <= h_3 <= 16
 w_4 = 4
 h_4 = 4
 cx_0 <= 60
 cy_0 <= 60
 cx_1 <= 60
 cy_1 <= 60
 cx_2 <= 60
 cy_2 <= 60
 cx_3 <= 60
 cy_3 <= 60
 cx_4 <= 60
 cy_4 <= 60
 shared_y_0_1 <= 20
 shared_x_0_1 <= 16
 shared_y_2_4 <= 20
 shared_x_2_4 <= 20
 shared_y_1_2 <= 20
 shared_x_1_2 <= 20
 shared_y_1_3 <= 20
 shared_x_1_3 <= 16
 bbox_min_x <= 60
 bbox_max_x <= 60
 bbox_min_y <= 60
 bbox_max_y <= 60
 bbox_w <= 60
 bbox_h <= 60
 -60 <= dx_diff_0_1 <= 60
 -60 <= dy_diff_0_1 <= 60
 -60 <= dx_diff_2_4 <= 60
 -60 <= dy_diff_2_4 <= 60
 -60 <= dx_diff_1_2 <= 60
 -60 <= dy_diff_1_2 <= 60
 -60 <= dx_diff_1_3 <= 60
 -60 <= dy_diff_1_3 <= 60
 bbox_area_proxy_t <= 120
 cx_grid_center_c25_0 <= 130
 cy_grid_center_c25_0 <= 130
 cx_grid_center_c25_1 <= 130
 cy_grid_center_c25_1 <= 130
 cx_grid_center_c25_2 <= 130
 cy_grid_center_c25_2 <= 130
 cx_grid_center_c25_3 <= 130
 cy_grid_center_c25_3 <= 130
 cx_grid_center_c25_4 <= 130
 cy_grid_center_c25_4 <= 130
 dx_grid_abs_pair_c25_0_1 <= 130
 dy_grid_abs_pair_c25_0_1 <= 130
 -130 <= dx_grid_diff_c25_0_1 <= 130
 -130 <= dy_grid_diff_c25_0_1 <= 130
 dist_curr_pair_c25_0_1 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev0 <= 240
 dist_diff_abs_pair_c25_0_1_prev0 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev1 <= 240
 dist_diff_abs_pair_c25_0_1_prev1 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev2 <= 240
 dist_diff_abs_pair_c25_0_1_prev2 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev3 <= 240
 dist_diff_abs_pair_c25_0_1_prev3 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev4 <= 240
 dist_diff_abs_pair_c25_0_1_prev4 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev5 <= 240
 dist_diff_abs_pair_c25_0_1_prev5 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev6 <= 240
 dist_diff_abs_pair_c25_0_1_prev6 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev7 <= 240
 dist_diff_abs_pair_c25_0_1_prev7 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev8 <= 240
 dist_diff_abs_pair_c25_0_1_prev8 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev9 <= 240
 dist_diff_abs_pair_c25_0_1_prev9 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev10 <= 240
 dist_diff_abs_pair_c25_0_1_prev10 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev11 <= 240
 dist_diff_abs_pair_c25_0_1_prev11 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev12 <= 240
 dist_diff_abs_pair_c25_0_1_prev12 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev13 <= 240
 dist_diff_abs_pair_c25_0_1_prev13 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev14 <= 240
 dist_diff_abs_pair_c25_0_1_prev14 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev15 <= 240
 dist_diff_abs_pair_c25_0_1_prev15 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev16 <= 240
 dist_diff_abs_pair_c25_0_1_prev16 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev17 <= 240
 dist_diff_abs_pair_c25_0_1_prev17 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev18 <= 240
 dist_diff_abs_pair_c25_0_1_prev18 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev19 <= 240
 dist_diff_abs_pair_c25_0_1_prev19 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev20 <= 240
 dist_diff_abs_pair_c25_0_1_prev20 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev21 <= 240
 dist_diff_abs_pair_c25_0_1_prev21 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev22 <= 240
 dist_diff_abs_pair_c25_0_1_prev22 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev23 <= 240
 dist_diff_abs_pair_c25_0_1_prev23 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev24 <= 240
 dist_diff_abs_pair_c25_0_1_prev24 <= 240
 -240 <= dist_diff_pair_c25_0_1_prev25 <= 240
 dist_diff_abs_pair_c25_0_1_prev25 <= 240
 dx_grid_abs_pair_c25_0_2 <= 130
 dy_grid_abs_pair_c25_0_2 <= 130
 -130 <= dx_grid_diff_c25_0_2 <= 130
 -130 <= dy_grid_diff_c25_0_2 <= 130
 dist_curr_pair_c25_0_2 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev0 <= 240
 dist_diff_abs_pair_c25_0_2_prev0 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev1 <= 240
 dist_diff_abs_pair_c25_0_2_prev1 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev2 <= 240
 dist_diff_abs_pair_c25_0_2_prev2 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev3 <= 240
 dist_diff_abs_pair_c25_0_2_prev3 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev4 <= 240
 dist_diff_abs_pair_c25_0_2_prev4 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev5 <= 240
 dist_diff_abs_pair_c25_0_2_prev5 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev6 <= 240
 dist_diff_abs_pair_c25_0_2_prev6 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev7 <= 240
 dist_diff_abs_pair_c25_0_2_prev7 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev8 <= 240
 dist_diff_abs_pair_c25_0_2_prev8 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev9 <= 240
 dist_diff_abs_pair_c25_0_2_prev9 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev10 <= 240
 dist_diff_abs_pair_c25_0_2_prev10 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev11 <= 240
 dist_diff_abs_pair_c25_0_2_prev11 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev12 <= 240
 dist_diff_abs_pair_c25_0_2_prev12 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev13 <= 240
 dist_diff_abs_pair_c25_0_2_prev13 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev14 <= 240
 dist_diff_abs_pair_c25_0_2_prev14 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev15 <= 240
 dist_diff_abs_pair_c25_0_2_prev15 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev16 <= 240
 dist_diff_abs_pair_c25_0_2_prev16 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev17 <= 240
 dist_diff_abs_pair_c25_0_2_prev17 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev18 <= 240
 dist_diff_abs_pair_c25_0_2_prev18 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev19 <= 240
 dist_diff_abs_pair_c25_0_2_prev19 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev20 <= 240
 dist_diff_abs_pair_c25_0_2_prev20 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev21 <= 240
 dist_diff_abs_pair_c25_0_2_prev21 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev22 <= 240
 dist_diff_abs_pair_c25_0_2_prev22 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev23 <= 240
 dist_diff_abs_pair_c25_0_2_prev23 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev24 <= 240
 dist_diff_abs_pair_c25_0_2_prev24 <= 240
 -240 <= dist_diff_pair_c25_0_2_prev25 <= 240
 dist_diff_abs_pair_c25_0_2_prev25 <= 240
 dx_grid_abs_pair_c25_0_3 <= 130
 dy_grid_abs_pair_c25_0_3 <= 130
 -130 <= dx_grid_diff_c25_0_3 <= 130
 -130 <= dy_grid_diff_c25_0_3 <= 130
 dist_curr_pair_c25_0_3 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev0 <= 240
 dist_diff_abs_pair_c25_0_3_prev0 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev1 <= 240
 dist_diff_abs_pair_c25_0_3_prev1 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev2 <= 240
 dist_diff_abs_pair_c25_0_3_prev2 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev3 <= 240
 dist_diff_abs_pair_c25_0_3_prev3 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev4 <= 240
 dist_diff_abs_pair_c25_0_3_prev4 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev5 <= 240
 dist_diff_abs_pair_c25_0_3_prev5 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev6 <= 240
 dist_diff_abs_pair_c25_0_3_prev6 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev7 <= 240
 dist_diff_abs_pair_c25_0_3_prev7 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev8 <= 240
 dist_diff_abs_pair_c25_0_3_prev8 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev9 <= 240
 dist_diff_abs_pair_c25_0_3_prev9 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev10 <= 240
 dist_diff_abs_pair_c25_0_3_prev10 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev11 <= 240
 dist_diff_abs_pair_c25_0_3_prev11 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev12 <= 240
 dist_diff_abs_pair_c25_0_3_prev12 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev13 <= 240
 dist_diff_abs_pair_c25_0_3_prev13 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev14 <= 240
 dist_diff_abs_pair_c25_0_3_prev14 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev15 <= 240
 dist_diff_abs_pair_c25_0_3_prev15 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev16 <= 240
 dist_diff_abs_pair_c25_0_3_prev16 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev17 <= 240
 dist_diff_abs_pair_c25_0_3_prev17 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev18 <= 240
 dist_diff_abs_pair_c25_0_3_prev18 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev19 <= 240
 dist_diff_abs_pair_c25_0_3_prev19 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev20 <= 240
 dist_diff_abs_pair_c25_0_3_prev20 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev21 <= 240
 dist_diff_abs_pair_c25_0_3_prev21 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev22 <= 240
 dist_diff_abs_pair_c25_0_3_prev22 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev23 <= 240
 dist_diff_abs_pair_c25_0_3_prev23 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev24 <= 240
 dist_diff_abs_pair_c25_0_3_prev24 <= 240
 -240 <= dist_diff_pair_c25_0_3_prev25 <= 240
 dist_diff_abs_pair_c25_0_3_prev25 <= 240
 dx_grid_abs_pair_c25_0_4 <= 130
 dy_grid_abs_pair_c25_0_4 <= 130
 -130 <= dx_grid_diff_c25_0_4 <= 130
 -130 <= dy_grid_diff_c25_0_4 <= 130
 dist_curr_pair_c25_0_4 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev0 <= 240
 dist_diff_abs_pair_c25_0_4_prev0 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev1 <= 240
 dist_diff_abs_pair_c25_0_4_prev1 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev2 <= 240
 dist_diff_abs_pair_c25_0_4_prev2 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev3 <= 240
 dist_diff_abs_pair_c25_0_4_prev3 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev4 <= 240
 dist_diff_abs_pair_c25_0_4_prev4 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev5 <= 240
 dist_diff_abs_pair_c25_0_4_prev5 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev6 <= 240
 dist_diff_abs_pair_c25_0_4_prev6 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev7 <= 240
 dist_diff_abs_pair_c25_0_4_prev7 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev8 <= 240
 dist_diff_abs_pair_c25_0_4_prev8 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev9 <= 240
 dist_diff_abs_pair_c25_0_4_prev9 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev10 <= 240
 dist_diff_abs_pair_c25_0_4_prev10 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev11 <= 240
 dist_diff_abs_pair_c25_0_4_prev11 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev12 <= 240
 dist_diff_abs_pair_c25_0_4_prev12 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev13 <= 240
 dist_diff_abs_pair_c25_0_4_prev13 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev14 <= 240
 dist_diff_abs_pair_c25_0_4_prev14 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev15 <= 240
 dist_diff_abs_pair_c25_0_4_prev15 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev16 <= 240
 dist_diff_abs_pair_c25_0_4_prev16 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev17 <= 240
 dist_diff_abs_pair_c25_0_4_prev17 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev18 <= 240
 dist_diff_abs_pair_c25_0_4_prev18 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev19 <= 240
 dist_diff_abs_pair_c25_0_4_prev19 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev20 <= 240
 dist_diff_abs_pair_c25_0_4_prev20 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev21 <= 240
 dist_diff_abs_pair_c25_0_4_prev21 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev22 <= 240
 dist_diff_abs_pair_c25_0_4_prev22 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev23 <= 240
 dist_diff_abs_pair_c25_0_4_prev23 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev24 <= 240
 dist_diff_abs_pair_c25_0_4_prev24 <= 240
 -240 <= dist_diff_pair_c25_0_4_prev25 <= 240
 dist_diff_abs_pair_c25_0_4_prev25 <= 240
 dx_grid_abs_pair_c25_1_2 <= 130
 dy_grid_abs_pair_c25_1_2 <= 130
 -130 <= dx_grid_diff_c25_1_2 <= 130
 -130 <= dy_grid_diff_c25_1_2 <= 130
 dist_curr_pair_c25_1_2 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev0 <= 240
 dist_diff_abs_pair_c25_1_2_prev0 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev1 <= 240
 dist_diff_abs_pair_c25_1_2_prev1 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev2 <= 240
 dist_diff_abs_pair_c25_1_2_prev2 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev3 <= 240
 dist_diff_abs_pair_c25_1_2_prev3 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev4 <= 240
 dist_diff_abs_pair_c25_1_2_prev4 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev5 <= 240
 dist_diff_abs_pair_c25_1_2_prev5 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev6 <= 240
 dist_diff_abs_pair_c25_1_2_prev6 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev7 <= 240
 dist_diff_abs_pair_c25_1_2_prev7 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev8 <= 240
 dist_diff_abs_pair_c25_1_2_prev8 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev9 <= 240
 dist_diff_abs_pair_c25_1_2_prev9 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev10 <= 240
 dist_diff_abs_pair_c25_1_2_prev10 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev11 <= 240
 dist_diff_abs_pair_c25_1_2_prev11 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev12 <= 240
 dist_diff_abs_pair_c25_1_2_prev12 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev13 <= 240
 dist_diff_abs_pair_c25_1_2_prev13 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev14 <= 240
 dist_diff_abs_pair_c25_1_2_prev14 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev15 <= 240
 dist_diff_abs_pair_c25_1_2_prev15 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev16 <= 240
 dist_diff_abs_pair_c25_1_2_prev16 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev17 <= 240
 dist_diff_abs_pair_c25_1_2_prev17 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev18 <= 240
 dist_diff_abs_pair_c25_1_2_prev18 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev19 <= 240
 dist_diff_abs_pair_c25_1_2_prev19 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev20 <= 240
 dist_diff_abs_pair_c25_1_2_prev20 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev21 <= 240
 dist_diff_abs_pair_c25_1_2_prev21 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev22 <= 240
 dist_diff_abs_pair_c25_1_2_prev22 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev23 <= 240
 dist_diff_abs_pair_c25_1_2_prev23 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev24 <= 240
 dist_diff_abs_pair_c25_1_2_prev24 <= 240
 -240 <= dist_diff_pair_c25_1_2_prev25 <= 240
 dist_diff_abs_pair_c25_1_2_prev25 <= 240
 dx_grid_abs_pair_c25_1_3 <= 130
 dy_grid_abs_pair_c25_1_3 <= 130
 -130 <= dx_grid_diff_c25_1_3 <= 130
 -130 <= dy_grid_diff_c25_1_3 <= 130
 dist_curr_pair_c25_1_3 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev0 <= 240
 dist_diff_abs_pair_c25_1_3_prev0 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev1 <= 240
 dist_diff_abs_pair_c25_1_3_prev1 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev2 <= 240
 dist_diff_abs_pair_c25_1_3_prev2 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev3 <= 240
 dist_diff_abs_pair_c25_1_3_prev3 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev4 <= 240
 dist_diff_abs_pair_c25_1_3_prev4 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev5 <= 240
 dist_diff_abs_pair_c25_1_3_prev5 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev6 <= 240
 dist_diff_abs_pair_c25_1_3_prev6 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev7 <= 240
 dist_diff_abs_pair_c25_1_3_prev7 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev8 <= 240
 dist_diff_abs_pair_c25_1_3_prev8 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev9 <= 240
 dist_diff_abs_pair_c25_1_3_prev9 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev10 <= 240
 dist_diff_abs_pair_c25_1_3_prev10 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev11 <= 240
 dist_diff_abs_pair_c25_1_3_prev11 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev12 <= 240
 dist_diff_abs_pair_c25_1_3_prev12 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev13 <= 240
 dist_diff_abs_pair_c25_1_3_prev13 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev14 <= 240
 dist_diff_abs_pair_c25_1_3_prev14 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev15 <= 240
 dist_diff_abs_pair_c25_1_3_prev15 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev16 <= 240
 dist_diff_abs_pair_c25_1_3_prev16 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev17 <= 240
 dist_diff_abs_pair_c25_1_3_prev17 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev18 <= 240
 dist_diff_abs_pair_c25_1_3_prev18 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev19 <= 240
 dist_diff_abs_pair_c25_1_3_prev19 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev20 <= 240
 dist_diff_abs_pair_c25_1_3_prev20 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev21 <= 240
 dist_diff_abs_pair_c25_1_3_prev21 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev22 <= 240
 dist_diff_abs_pair_c25_1_3_prev22 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev23 <= 240
 dist_diff_abs_pair_c25_1_3_prev23 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev24 <= 240
 dist_diff_abs_pair_c25_1_3_prev24 <= 240
 -240 <= dist_diff_pair_c25_1_3_prev25 <= 240
 dist_diff_abs_pair_c25_1_3_prev25 <= 240
 dx_grid_abs_pair_c25_1_4 <= 130
 dy_grid_abs_pair_c25_1_4 <= 130
 -130 <= dx_grid_diff_c25_1_4 <= 130
 -130 <= dy_grid_diff_c25_1_4 <= 130
 dist_curr_pair_c25_1_4 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev0 <= 240
 dist_diff_abs_pair_c25_1_4_prev0 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev1 <= 240
 dist_diff_abs_pair_c25_1_4_prev1 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev2 <= 240
 dist_diff_abs_pair_c25_1_4_prev2 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev3 <= 240
 dist_diff_abs_pair_c25_1_4_prev3 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev4 <= 240
 dist_diff_abs_pair_c25_1_4_prev4 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev5 <= 240
 dist_diff_abs_pair_c25_1_4_prev5 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev6 <= 240
 dist_diff_abs_pair_c25_1_4_prev6 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev7 <= 240
 dist_diff_abs_pair_c25_1_4_prev7 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev8 <= 240
 dist_diff_abs_pair_c25_1_4_prev8 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev9 <= 240
 dist_diff_abs_pair_c25_1_4_prev9 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev10 <= 240
 dist_diff_abs_pair_c25_1_4_prev10 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev11 <= 240
 dist_diff_abs_pair_c25_1_4_prev11 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev12 <= 240
 dist_diff_abs_pair_c25_1_4_prev12 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev13 <= 240
 dist_diff_abs_pair_c25_1_4_prev13 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev14 <= 240
 dist_diff_abs_pair_c25_1_4_prev14 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev15 <= 240
 dist_diff_abs_pair_c25_1_4_prev15 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev16 <= 240
 dist_diff_abs_pair_c25_1_4_prev16 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev17 <= 240
 dist_diff_abs_pair_c25_1_4_prev17 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev18 <= 240
 dist_diff_abs_pair_c25_1_4_prev18 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev19 <= 240
 dist_diff_abs_pair_c25_1_4_prev19 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev20 <= 240
 dist_diff_abs_pair_c25_1_4_prev20 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev21 <= 240
 dist_diff_abs_pair_c25_1_4_prev21 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev22 <= 240
 dist_diff_abs_pair_c25_1_4_prev22 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev23 <= 240
 dist_diff_abs_pair_c25_1_4_prev23 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev24 <= 240
 dist_diff_abs_pair_c25_1_4_prev24 <= 240
 -240 <= dist_diff_pair_c25_1_4_prev25 <= 240
 dist_diff_abs_pair_c25_1_4_prev25 <= 240
 dx_grid_abs_pair_c25_2_3 <= 130
 dy_grid_abs_pair_c25_2_3 <= 130
 -130 <= dx_grid_diff_c25_2_3 <= 130
 -130 <= dy_grid_diff_c25_2_3 <= 130
 dist_curr_pair_c25_2_3 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev0 <= 240
 dist_diff_abs_pair_c25_2_3_prev0 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev1 <= 240
 dist_diff_abs_pair_c25_2_3_prev1 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev2 <= 240
 dist_diff_abs_pair_c25_2_3_prev2 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev3 <= 240
 dist_diff_abs_pair_c25_2_3_prev3 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev4 <= 240
 dist_diff_abs_pair_c25_2_3_prev4 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev5 <= 240
 dist_diff_abs_pair_c25_2_3_prev5 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev6 <= 240
 dist_diff_abs_pair_c25_2_3_prev6 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev7 <= 240
 dist_diff_abs_pair_c25_2_3_prev7 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev8 <= 240
 dist_diff_abs_pair_c25_2_3_prev8 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev9 <= 240
 dist_diff_abs_pair_c25_2_3_prev9 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev10 <= 240
 dist_diff_abs_pair_c25_2_3_prev10 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev11 <= 240
 dist_diff_abs_pair_c25_2_3_prev11 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev12 <= 240
 dist_diff_abs_pair_c25_2_3_prev12 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev13 <= 240
 dist_diff_abs_pair_c25_2_3_prev13 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev14 <= 240
 dist_diff_abs_pair_c25_2_3_prev14 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev15 <= 240
 dist_diff_abs_pair_c25_2_3_prev15 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev16 <= 240
 dist_diff_abs_pair_c25_2_3_prev16 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev17 <= 240
 dist_diff_abs_pair_c25_2_3_prev17 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev18 <= 240
 dist_diff_abs_pair_c25_2_3_prev18 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev19 <= 240
 dist_diff_abs_pair_c25_2_3_prev19 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev20 <= 240
 dist_diff_abs_pair_c25_2_3_prev20 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev21 <= 240
 dist_diff_abs_pair_c25_2_3_prev21 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev22 <= 240
 dist_diff_abs_pair_c25_2_3_prev22 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev23 <= 240
 dist_diff_abs_pair_c25_2_3_prev23 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev24 <= 240
 dist_diff_abs_pair_c25_2_3_prev24 <= 240
 -240 <= dist_diff_pair_c25_2_3_prev25 <= 240
 dist_diff_abs_pair_c25_2_3_prev25 <= 240
 dx_grid_abs_pair_c25_2_4 <= 130
 dy_grid_abs_pair_c25_2_4 <= 130
 -130 <= dx_grid_diff_c25_2_4 <= 130
 -130 <= dy_grid_diff_c25_2_4 <= 130
 dist_curr_pair_c25_2_4 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev0 <= 240
 dist_diff_abs_pair_c25_2_4_prev0 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev1 <= 240
 dist_diff_abs_pair_c25_2_4_prev1 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev2 <= 240
 dist_diff_abs_pair_c25_2_4_prev2 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev3 <= 240
 dist_diff_abs_pair_c25_2_4_prev3 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev4 <= 240
 dist_diff_abs_pair_c25_2_4_prev4 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev5 <= 240
 dist_diff_abs_pair_c25_2_4_prev5 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev6 <= 240
 dist_diff_abs_pair_c25_2_4_prev6 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev7 <= 240
 dist_diff_abs_pair_c25_2_4_prev7 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev8 <= 240
 dist_diff_abs_pair_c25_2_4_prev8 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev9 <= 240
 dist_diff_abs_pair_c25_2_4_prev9 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev10 <= 240
 dist_diff_abs_pair_c25_2_4_prev10 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev11 <= 240
 dist_diff_abs_pair_c25_2_4_prev11 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev12 <= 240
 dist_diff_abs_pair_c25_2_4_prev12 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev13 <= 240
 dist_diff_abs_pair_c25_2_4_prev13 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev14 <= 240
 dist_diff_abs_pair_c25_2_4_prev14 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev15 <= 240
 dist_diff_abs_pair_c25_2_4_prev15 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev16 <= 240
 dist_diff_abs_pair_c25_2_4_prev16 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev17 <= 240
 dist_diff_abs_pair_c25_2_4_prev17 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev18 <= 240
 dist_diff_abs_pair_c25_2_4_prev18 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev19 <= 240
 dist_diff_abs_pair_c25_2_4_prev19 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev20 <= 240
 dist_diff_abs_pair_c25_2_4_prev20 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev21 <= 240
 dist_diff_abs_pair_c25_2_4_prev21 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev22 <= 240
 dist_diff_abs_pair_c25_2_4_prev22 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev23 <= 240
 dist_diff_abs_pair_c25_2_4_prev23 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev24 <= 240
 dist_diff_abs_pair_c25_2_4_prev24 <= 240
 -240 <= dist_diff_pair_c25_2_4_prev25 <= 240
 dist_diff_abs_pair_c25_2_4_prev25 <= 240
 dx_grid_abs_pair_c25_3_4 <= 130
 dy_grid_abs_pair_c25_3_4 <= 130
 -130 <= dx_grid_diff_c25_3_4 <= 130
 -130 <= dy_grid_diff_c25_3_4 <= 130
 dist_curr_pair_c25_3_4 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev0 <= 240
 dist_diff_abs_pair_c25_3_4_prev0 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev1 <= 240
 dist_diff_abs_pair_c25_3_4_prev1 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev2 <= 240
 dist_diff_abs_pair_c25_3_4_prev2 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev3 <= 240
 dist_diff_abs_pair_c25_3_4_prev3 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev4 <= 240
 dist_diff_abs_pair_c25_3_4_prev4 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev5 <= 240
 dist_diff_abs_pair_c25_3_4_prev5 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev6 <= 240
 dist_diff_abs_pair_c25_3_4_prev6 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev7 <= 240
 dist_diff_abs_pair_c25_3_4_prev7 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev8 <= 240
 dist_diff_abs_pair_c25_3_4_prev8 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev9 <= 240
 dist_diff_abs_pair_c25_3_4_prev9 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev10 <= 240
 dist_diff_abs_pair_c25_3_4_prev10 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev11 <= 240
 dist_diff_abs_pair_c25_3_4_prev11 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev12 <= 240
 dist_diff_abs_pair_c25_3_4_prev12 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev13 <= 240
 dist_diff_abs_pair_c25_3_4_prev13 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev14 <= 240
 dist_diff_abs_pair_c25_3_4_prev14 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev15 <= 240
 dist_diff_abs_pair_c25_3_4_prev15 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev16 <= 240
 dist_diff_abs_pair_c25_3_4_prev16 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev17 <= 240
 dist_diff_abs_pair_c25_3_4_prev17 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev18 <= 240
 dist_diff_abs_pair_c25_3_4_prev18 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev19 <= 240
 dist_diff_abs_pair_c25_3_4_prev19 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev20 <= 240
 dist_diff_abs_pair_c25_3_4_prev20 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev21 <= 240
 dist_diff_abs_pair_c25_3_4_prev21 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev22 <= 240
 dist_diff_abs_pair_c25_3_4_prev22 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev23 <= 240
 dist_diff_abs_pair_c25_3_4_prev23 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev24 <= 240
 dist_diff_abs_pair_c25_3_4_prev24 <= 240
 -240 <= dist_diff_pair_c25_3_4_prev25 <= 240
 dist_diff_abs_pair_c25_3_4_prev25 <= 240
Binaries
 r_0 r_1 r_2 r_3 r_4 z1_0_1 z2_0_1 z1L_0_1 z1R_0_1 z2D_0_1 z2U_0_1 z1_2_4
 z2_2_4 z1L_2_4 z1R_2_4 z2D_2_4 z2U_2_4 z1_1_2 z2_1_2 z1L_1_2 z1R_1_2
 z2D_1_2 z2U_1_2 z1_1_3 z2_1_3 z1L_1_3 z1R_1_3 z2D_1_3 z2U_1_3 p_left_0_1
 p_right_0_1 p_down_0_1 p_up_0_1 p_left_0_2 p_right_0_2 p_down_0_2 p_up_0_2
 p_left_0_3 p_right_0_3 p_down_0_3 p_up_0_3 p_left_0_4 p_right_0_4
 p_down_0_4 p_up_0_4 p_left_1_2 p_right_1_2 p_down_1_2 p_up_1_2 p_left_1_3
 p_right_1_3 p_down_1_3 p_up_1_3 p_left_1_4 p_right_1_4 p_down_1_4 p_up_1_4
 p_left_2_3 p_right_2_3 p_down_2_3 p_up_2_3 p_left_2_4 p_right_2_4
 p_down_2_4 p_up_2_4 p_left_3_4 p_right_3_4 p_down_3_4 p_up_3_4
 dx_abs_0_1_is_positive dy_abs_0_1_is_positive dx_abs_2_4_is_positive
 dy_abs_2_4_is_positive dx_abs_1_2_is_positive dy_abs_1_2_is_positive
 dx_abs_1_3_is_positive dy_abs_1_3_is_positive
 dx_grid_abs_pair_c25_0_1_is_positive dy_grid_abs_pair_c25_0_1_is_positive
 diff_dist_pair_c25_0_1 same_dist_pair_c25_0_1_prev0
 dist_diff_abs_pair_c25_0_1_prev0_is_positive same_dist_pair_c25_0_1_prev1
 dist_diff_abs_pair_c25_0_1_prev1_is_positive same_dist_pair_c25_0_1_prev2
 dist_diff_abs_pair_c25_0_1_prev2_is_positive same_dist_pair_c25_0_1_prev3
 dist_diff_abs_pair_c25_0_1_prev3_is_positive same_dist_pair_c25_0_1_prev4
 dist_diff_abs_pair_c25_0_1_prev4_is_positive same_dist_pair_c25_0_1_prev5
 dist_diff_abs_pair_c25_0_1_prev5_is_positive same_dist_pair_c25_0_1_prev6
 dist_diff_abs_pair_c25_0_1_prev6_is_positive same_dist_pair_c25_0_1_prev7
 dist_diff_abs_pair_c25_0_1_prev7_is_positive same_dist_pair_c25_0_1_prev8
 dist_diff_abs_pair_c25_0_1_prev8_is_positive same_dist_pair_c25_0_1_prev9
 dist_diff_abs_pair_c25_0_1_prev9_is_positive same_dist_pair_c25_0_1_prev10
 dist_diff_abs_pair_c25_0_1_prev10_is_positive
 same_dist_pair_c25_0_1_prev11
 dist_diff_abs_pair_c25_0_1_prev11_is_positive
 same_dist_pair_c25_0_1_prev12
 dist_diff_abs_pair_c25_0_1_prev12_is_positive
 same_dist_pair_c25_0_1_prev13
 dist_diff_abs_pair_c25_0_1_prev13_is_positive
 same_dist_pair_c25_0_1_prev14
 dist_diff_abs_pair_c25_0_1_prev14_is_positive
 same_dist_pair_c25_0_1_prev15
 dist_diff_abs_pair_c25_0_1_prev15_is_positive
 same_dist_pair_c25_0_1_prev16
 dist_diff_abs_pair_c25_0_1_prev16_is_positive
 same_dist_pair_c25_0_1_prev17
 dist_diff_abs_pair_c25_0_1_prev17_is_positive
 same_dist_pair_c25_0_1_prev18
 dist_diff_abs_pair_c25_0_1_prev18_is_positive
 same_dist_pair_c25_0_1_prev19
 dist_diff_abs_pair_c25_0_1_prev19_is_positive
 same_dist_pair_c25_0_1_prev20
 dist_diff_abs_pair_c25_0_1_prev20_is_positive
 same_dist_pair_c25_0_1_prev21
 dist_diff_abs_pair_c25_0_1_prev21_is_positive
 same_dist_pair_c25_0_1_prev22
 dist_diff_abs_pair_c25_0_1_prev22_is_positive
 same_dist_pair_c25_0_1_prev23
 dist_diff_abs_pair_c25_0_1_prev23_is_positive
 same_dist_pair_c25_0_1_prev24
 dist_diff_abs_pair_c25_0_1_prev24_is_positive
 same_dist_pair_c25_0_1_prev25
 dist_diff_abs_pair_c25_0_1_prev25_is_positive
 dx_grid_abs_pair_c25_0_2_is_positive dy_grid_abs_pair_c25_0_2_is_positive
 diff_dist_pair_c25_0_2 same_dist_pair_c25_0_2_prev0
 dist_diff_abs_pair_c25_0_2_prev0_is_positive same_dist_pair_c25_0_2_prev1
 dist_diff_abs_pair_c25_0_2_prev1_is_positive same_dist_pair_c25_0_2_prev2
 dist_diff_abs_pair_c25_0_2_prev2_is_positive same_dist_pair_c25_0_2_prev3
 dist_diff_abs_pair_c25_0_2_prev3_is_positive same_dist_pair_c25_0_2_prev4
 dist_diff_abs_pair_c25_0_2_prev4_is_positive same_dist_pair_c25_0_2_prev5
 dist_diff_abs_pair_c25_0_2_prev5_is_positive same_dist_pair_c25_0_2_prev6
 dist_diff_abs_pair_c25_0_2_prev6_is_positive same_dist_pair_c25_0_2_prev7
 dist_diff_abs_pair_c25_0_2_prev7_is_positive same_dist_pair_c25_0_2_prev8
 dist_diff_abs_pair_c25_0_2_prev8_is_positive same_dist_pair_c25_0_2_prev9
 dist_diff_abs_pair_c25_0_2_prev9_is_positive same_dist_pair_c25_0_2_prev10
 dist_diff_abs_pair_c25_0_2_prev10_is_positive
 same_dist_pair_c25_0_2_prev11
 dist_diff_abs_pair_c25_0_2_prev11_is_positive
 same_dist_pair_c25_0_2_prev12
 dist_diff_abs_pair_c25_0_2_prev12_is_positive
 same_dist_pair_c25_0_2_prev13
 dist_diff_abs_pair_c25_0_2_prev13_is_positive
 same_dist_pair_c25_0_2_prev14
 dist_diff_abs_pair_c25_0_2_prev14_is_positive
 same_dist_pair_c25_0_2_prev15
 dist_diff_abs_pair_c25_0_2_prev15_is_positive
 same_dist_pair_c25_0_2_prev16
 dist_diff_abs_pair_c25_0_2_prev16_is_positive
 same_dist_pair_c25_0_2_prev17
 dist_diff_abs_pair_c25_0_2_prev17_is_positive
 same_dist_pair_c25_0_2_prev18
 dist_diff_abs_pair_c25_0_2_prev18_is_positive
 same_dist_pair_c25_0_2_prev19
 dist_diff_abs_pair_c25_0_2_prev19_is_positive
 same_dist_pair_c25_0_2_prev20
 dist_diff_abs_pair_c25_0_2_prev20_is_positive
 same_dist_pair_c25_0_2_prev21
 dist_diff_abs_pair_c25_0_2_prev21_is_positive
 same_dist_pair_c25_0_2_prev22
 dist_diff_abs_pair_c25_0_2_prev22_is_positive
 same_dist_pair_c25_0_2_prev23
 dist_diff_abs_pair_c25_0_2_prev23_is_positive
 same_dist_pair_c25_0_2_prev24
 dist_diff_abs_pair_c25_0_2_prev24_is_positive
 same_dist_pair_c25_0_2_prev25
 dist_diff_abs_pair_c25_0_2_prev25_is_positive
 dx_grid_abs_pair_c25_0_3_is_positive dy_grid_abs_pair_c25_0_3_is_positive
 diff_dist_pair_c25_0_3 same_dist_pair_c25_0_3_prev0
 dist_diff_abs_pair_c25_0_3_prev0_is_positive same_dist_pair_c25_0_3_prev1
 dist_diff_abs_pair_c25_0_3_prev1_is_positive same_dist_pair_c25_0_3_prev2
 dist_diff_abs_pair_c25_0_3_prev2_is_positive same_dist_pair_c25_0_3_prev3
 dist_diff_abs_pair_c25_0_3_prev3_is_positive same_dist_pair_c25_0_3_prev4
 dist_diff_abs_pair_c25_0_3_prev4_is_positive same_dist_pair_c25_0_3_prev5
 dist_diff_abs_pair_c25_0_3_prev5_is_positive same_dist_pair_c25_0_3_prev6
 dist_diff_abs_pair_c25_0_3_prev6_is_positive same_dist_pair_c25_0_3_prev7
 dist_diff_abs_pair_c25_0_3_prev7_is_positive same_dist_pair_c25_0_3_prev8
 dist_diff_abs_pair_c25_0_3_prev8_is_positive same_dist_pair_c25_0_3_prev9
 dist_diff_abs_pair_c25_0_3_prev9_is_positive same_dist_pair_c25_0_3_prev10
 dist_diff_abs_pair_c25_0_3_prev10_is_positive
 same_dist_pair_c25_0_3_prev11
 dist_diff_abs_pair_c25_0_3_prev11_is_positive
 same_dist_pair_c25_0_3_prev12
 dist_diff_abs_pair_c25_0_3_prev12_is_positive
 same_dist_pair_c25_0_3_prev13
 dist_diff_abs_pair_c25_0_3_prev13_is_positive
 same_dist_pair_c25_0_3_prev14
 dist_diff_abs_pair_c25_0_3_prev14_is_positive
 same_dist_pair_c25_0_3_prev15
 dist_diff_abs_pair_c25_0_3_prev15_is_positive
 same_dist_pair_c25_0_3_prev16
 dist_diff_abs_pair_c25_0_3_prev16_is_positive
 same_dist_pair_c25_0_3_prev17
 dist_diff_abs_pair_c25_0_3_prev17_is_positive
 same_dist_pair_c25_0_3_prev18
 dist_diff_abs_pair_c25_0_3_prev18_is_positive
 same_dist_pair_c25_0_3_prev19
 dist_diff_abs_pair_c25_0_3_prev19_is_positive
 same_dist_pair_c25_0_3_prev20
 dist_diff_abs_pair_c25_0_3_prev20_is_positive
 same_dist_pair_c25_0_3_prev21
 dist_diff_abs_pair_c25_0_3_prev21_is_positive
 same_dist_pair_c25_0_3_prev22
 dist_diff_abs_pair_c25_0_3_prev22_is_positive
 same_dist_pair_c25_0_3_prev23
 dist_diff_abs_pair_c25_0_3_prev23_is_positive
 same_dist_pair_c25_0_3_prev24
 dist_diff_abs_pair_c25_0_3_prev24_is_positive
 same_dist_pair_c25_0_3_prev25
 dist_diff_abs_pair_c25_0_3_prev25_is_positive
 dx_grid_abs_pair_c25_0_4_is_positive dy_grid_abs_pair_c25_0_4_is_positive
 diff_dist_pair_c25_0_4 same_dist_pair_c25_0_4_prev0
 dist_diff_abs_pair_c25_0_4_prev0_is_positive same_dist_pair_c25_0_4_prev1
 dist_diff_abs_pair_c25_0_4_prev1_is_positive same_dist_pair_c25_0_4_prev2
 dist_diff_abs_pair_c25_0_4_prev2_is_positive same_dist_pair_c25_0_4_prev3
 dist_diff_abs_pair_c25_0_4_prev3_is_positive same_dist_pair_c25_0_4_prev4
 dist_diff_abs_pair_c25_0_4_prev4_is_positive same_dist_pair_c25_0_4_prev5
 dist_diff_abs_pair_c25_0_4_prev5_is_positive same_dist_pair_c25_0_4_prev6
 dist_diff_abs_pair_c25_0_4_prev6_is_positive same_dist_pair_c25_0_4_prev7
 dist_diff_abs_pair_c25_0_4_prev7_is_positive same_dist_pair_c25_0_4_prev8
 dist_diff_abs_pair_c25_0_4_prev8_is_positive same_dist_pair_c25_0_4_prev9
 dist_diff_abs_pair_c25_0_4_prev9_is_positive same_dist_pair_c25_0_4_prev10
 dist_diff_abs_pair_c25_0_4_prev10_is_positive
 same_dist_pair_c25_0_4_prev11
 dist_diff_abs_pair_c25_0_4_prev11_is_positive
 same_dist_pair_c25_0_4_prev12
 dist_diff_abs_pair_c25_0_4_prev12_is_positive
 same_dist_pair_c25_0_4_prev13
 dist_diff_abs_pair_c25_0_4_prev13_is_positive
 same_dist_pair_c25_0_4_prev14
 dist_diff_abs_pair_c25_0_4_prev14_is_positive
 same_dist_pair_c25_0_4_prev15
 dist_diff_abs_pair_c25_0_4_prev15_is_positive
 same_dist_pair_c25_0_4_prev16
 dist_diff_abs_pair_c25_0_4_prev16_is_positive
 same_dist_pair_c25_0_4_prev17
 dist_diff_abs_pair_c25_0_4_prev17_is_positive
 same_dist_pair_c25_0_4_prev18
 dist_diff_abs_pair_c25_0_4_prev18_is_positive
 same_dist_pair_c25_0_4_prev19
 dist_diff_abs_pair_c25_0_4_prev19_is_positive
 same_dist_pair_c25_0_4_prev20
 dist_diff_abs_pair_c25_0_4_prev20_is_positive
 same_dist_pair_c25_0_4_prev21
 dist_diff_abs_pair_c25_0_4_prev21_is_positive
 same_dist_pair_c25_0_4_prev22
 dist_diff_abs_pair_c25_0_4_prev22_is_positive
 same_dist_pair_c25_0_4_prev23
 dist_diff_abs_pair_c25_0_4_prev23_is_positive
 same_dist_pair_c25_0_4_prev24
 dist_diff_abs_pair_c25_0_4_prev24_is_positive
 same_dist_pair_c25_0_4_prev25
 dist_diff_abs_pair_c25_0_4_prev25_is_positive
 dx_grid_abs_pair_c25_1_2_is_positive dy_grid_abs_pair_c25_1_2_is_positive
 diff_dist_pair_c25_1_2 same_dist_pair_c25_1_2_prev0
 dist_diff_abs_pair_c25_1_2_prev0_is_positive same_dist_pair_c25_1_2_prev1
 dist_diff_abs_pair_c25_1_2_prev1_is_positive same_dist_pair_c25_1_2_prev2
 dist_diff_abs_pair_c25_1_2_prev2_is_positive same_dist_pair_c25_1_2_prev3
 dist_diff_abs_pair_c25_1_2_prev3_is_positive same_dist_pair_c25_1_2_prev4
 dist_diff_abs_pair_c25_1_2_prev4_is_positive same_dist_pair_c25_1_2_prev5
 dist_diff_abs_pair_c25_1_2_prev5_is_positive same_dist_pair_c25_1_2_prev6
 dist_diff_abs_pair_c25_1_2_prev6_is_positive same_dist_pair_c25_1_2_prev7
 dist_diff_abs_pair_c25_1_2_prev7_is_positive same_dist_pair_c25_1_2_prev8
 dist_diff_abs_pair_c25_1_2_prev8_is_positive same_dist_pair_c25_1_2_prev9
 dist_diff_abs_pair_c25_1_2_prev9_is_positive same_dist_pair_c25_1_2_prev10
 dist_diff_abs_pair_c25_1_2_prev10_is_positive
 same_dist_pair_c25_1_2_prev11
 dist_diff_abs_pair_c25_1_2_prev11_is_positive
 same_dist_pair_c25_1_2_prev12
 dist_diff_abs_pair_c25_1_2_prev12_is_positive
 same_dist_pair_c25_1_2_prev13
 dist_diff_abs_pair_c25_1_2_prev13_is_positive
 same_dist_pair_c25_1_2_prev14
 dist_diff_abs_pair_c25_1_2_prev14_is_positive
 same_dist_pair_c25_1_2_prev15
 dist_diff_abs_pair_c25_1_2_prev15_is_positive
 same_dist_pair_c25_1_2_prev16
 dist_diff_abs_pair_c25_1_2_prev16_is_positive
 same_dist_pair_c25_1_2_prev17
 dist_diff_abs_pair_c25_1_2_prev17_is_positive
 same_dist_pair_c25_1_2_prev18
 dist_diff_abs_pair_c25_1_2_prev18_is_positive
 same_dist_pair_c25_1_2_prev19
 dist_diff_abs_pair_c25_1_2_prev19_is_positive
 same_dist_pair_c25_1_2_prev20
 dist_diff_abs_pair_c25_1_2_prev20_is_positive
 same_dist_pair_c25_1_2_prev21
 dist_diff_abs_pair_c25_1_2_prev21_is_positive
 same_dist_pair_c25_1_2_prev22
 dist_diff_abs_pair_c25_1_2_prev22_is_positive
 same_dist_pair_c25_1_2_prev23
 dist_diff_abs_pair_c25_1_2_prev23_is_positive
 same_dist_pair_c25_1_2_prev24
 dist_diff_abs_pair_c25_1_2_prev24_is_positive
 same_dist_pair_c25_1_2_prev25
 dist_diff_abs_pair_c25_1_2_prev25_is_positive
 dx_grid_abs_pair_c25_1_3_is_positive dy_grid_abs_pair_c25_1_3_is_positive
 diff_dist_pair_c25_1_3 same_dist_pair_c25_1_3_prev0
 dist_diff_abs_pair_c25_1_3_prev0_is_positive same_dist_pair_c25_1_3_prev1
 dist_diff_abs_pair_c25_1_3_prev1_is_positive same_dist_pair_c25_1_3_prev2
 dist_diff_abs_pair_c25_1_3_prev2_is_positive same_dist_pair_c25_1_3_prev3
 dist_diff_abs_pair_c25_1_3_prev3_is_positive same_dist_pair_c25_1_3_prev4
 dist_diff_abs_pair_c25_1_3_prev4_is_positive same_dist_pair_c25_1_3_prev5
 dist_diff_abs_pair_c25_1_3_prev5_is_positive same_dist_pair_c25_1_3_prev6
 dist_diff_abs_pair_c25_1_3_prev6_is_positive same_dist_pair_c25_1_3_prev7
 dist_diff_abs_pair_c25_1_3_prev7_is_positive same_dist_pair_c25_1_3_prev8
 dist_diff_abs_pair_c25_1_3_prev8_is_positive same_dist_pair_c25_1_3_prev9
 dist_diff_abs_pair_c25_1_3_prev9_is_positive same_dist_pair_c25_1_3_prev10
 dist_diff_abs_pair_c25_1_3_prev10_is_positive
 same_dist_pair_c25_1_3_prev11
 dist_diff_abs_pair_c25_1_3_prev11_is_positive
 same_dist_pair_c25_1_3_prev12
 dist_diff_abs_pair_c25_1_3_prev12_is_positive
 same_dist_pair_c25_1_3_prev13
 dist_diff_abs_pair_c25_1_3_prev13_is_positive
 same_dist_pair_c25_1_3_prev14
 dist_diff_abs_pair_c25_1_3_prev14_is_positive
 same_dist_pair_c25_1_3_prev15
 dist_diff_abs_pair_c25_1_3_prev15_is_positive
 same_dist_pair_c25_1_3_prev16
 dist_diff_abs_pair_c25_1_3_prev16_is_positive
 same_dist_pair_c25_1_3_prev17
 dist_diff_abs_pair_c25_1_3_prev17_is_positive
 same_dist_pair_c25_1_3_prev18
 dist_diff_abs_pair_c25_1_3_prev18_is_positive
 same_dist_pair_c25_1_3_prev19
 dist_diff_abs_pair_c25_1_3_prev19_is_positive
 same_dist_pair_c25_1_3_prev20
 dist_diff_abs_pair_c25_1_3_prev20_is_positive
 same_dist_pair_c25_1_3_prev21
 dist_diff_abs_pair_c25_1_3_prev21_is_positive
 same_dist_pair_c25_1_3_prev22
 dist_diff_abs_pair_c25_1_3_prev22_is_positive
 same_dist_pair_c25_1_3_prev23
 dist_diff_abs_pair_c25_1_3_prev23_is_positive
 same_dist_pair_c25_1_3_prev24
 dist_diff_abs_pair_c25_1_3_prev24_is_positive
 same_dist_pair_c25_1_3_prev25
 dist_diff_abs_pair_c25_1_3_prev25_is_positive
 dx_grid_abs_pair_c25_1_4_is_positive dy_grid_abs_pair_c25_1_4_is_positive
 diff_dist_pair_c25_1_4 same_dist_pair_c25_1_4_prev0
 dist_diff_abs_pair_c25_1_4_prev0_is_positive same_dist_pair_c25_1_4_prev1
 dist_diff_abs_pair_c25_1_4_prev1_is_positive same_dist_pair_c25_1_4_prev2
 dist_diff_abs_pair_c25_1_4_prev2_is_positive same_dist_pair_c25_1_4_prev3
 dist_diff_abs_pair_c25_1_4_prev3_is_positive same_dist_pair_c25_1_4_prev4
 dist_diff_abs_pair_c25_1_4_prev4_is_positive same_dist_pair_c25_1_4_prev5
 dist_diff_abs_pair_c25_1_4_prev5_is_positive same_dist_pair_c25_1_4_prev6
 dist_diff_abs_pair_c25_1_4_prev6_is_positive same_dist_pair_c25_1_4_prev7
 dist_diff_abs_pair_c25_1_4_prev7_is_positive same_dist_pair_c25_1_4_prev8
 dist_diff_abs_pair_c25_1_4_prev8_is_positive same_dist_pair_c25_1_4_prev9
 dist_diff_abs_pair_c25_1_4_prev9_is_positive same_dist_pair_c25_1_4_prev10
 dist_diff_abs_pair_c25_1_4_prev10_is_positive
 same_dist_pair_c25_1_4_prev11
 dist_diff_abs_pair_c25_1_4_prev11_is_positive
 same_dist_pair_c25_1_4_prev12
 dist_diff_abs_pair_c25_1_4_prev12_is_positive
 same_dist_pair_c25_1_4_prev13
 dist_diff_abs_pair_c25_1_4_prev13_is_positive
 same_dist_pair_c25_1_4_prev14
 dist_diff_abs_pair_c25_1_4_prev14_is_positive
 same_dist_pair_c25_1_4_prev15
 dist_diff_abs_pair_c25_1_4_prev15_is_positive
 same_dist_pair_c25_1_4_prev16
 dist_diff_abs_pair_c25_1_4_prev16_is_positive
 same_dist_pair_c25_1_4_prev17
 dist_diff_abs_pair_c25_1_4_prev17_is_positive
 same_dist_pair_c25_1_4_prev18
 dist_diff_abs_pair_c25_1_4_prev18_is_positive
 same_dist_pair_c25_1_4_prev19
 dist_diff_abs_pair_c25_1_4_prev19_is_positive
 same_dist_pair_c25_1_4_prev20
 dist_diff_abs_pair_c25_1_4_prev20_is_positive
 same_dist_pair_c25_1_4_prev21
 dist_diff_abs_pair_c25_1_4_prev21_is_positive
 same_dist_pair_c25_1_4_prev22
 dist_diff_abs_pair_c25_1_4_prev22_is_positive
 same_dist_pair_c25_1_4_prev23
 dist_diff_abs_pair_c25_1_4_prev23_is_positive
 same_dist_pair_c25_1_4_prev24
 dist_diff_abs_pair_c25_1_4_prev24_is_positive
 same_dist_pair_c25_1_4_prev25
 dist_diff_abs_pair_c25_1_4_prev25_is_positive
 dx_grid_abs_pair_c25_2_3_is_positive dy_grid_abs_pair_c25_2_3_is_positive
 diff_dist_pair_c25_2_3 same_dist_pair_c25_2_3_prev0
 dist_diff_abs_pair_c25_2_3_prev0_is_positive same_dist_pair_c25_2_3_prev1
 dist_diff_abs_pair_c25_2_3_prev1_is_positive same_dist_pair_c25_2_3_prev2
 dist_diff_abs_pair_c25_2_3_prev2_is_positive same_dist_pair_c25_2_3_prev3
 dist_diff_abs_pair_c25_2_3_prev3_is_positive same_dist_pair_c25_2_3_prev4
 dist_diff_abs_pair_c25_2_3_prev4_is_positive same_dist_pair_c25_2_3_prev5
 dist_diff_abs_pair_c25_2_3_prev5_is_positive same_dist_pair_c25_2_3_prev6
 dist_diff_abs_pair_c25_2_3_prev6_is_positive same_dist_pair_c25_2_3_prev7
 dist_diff_abs_pair_c25_2_3_prev7_is_positive same_dist_pair_c25_2_3_prev8
 dist_diff_abs_pair_c25_2_3_prev8_is_positive same_dist_pair_c25_2_3_prev9
 dist_diff_abs_pair_c25_2_3_prev9_is_positive same_dist_pair_c25_2_3_prev10
 dist_diff_abs_pair_c25_2_3_prev10_is_positive
 same_dist_pair_c25_2_3_prev11
 dist_diff_abs_pair_c25_2_3_prev11_is_positive
 same_dist_pair_c25_2_3_prev12
 dist_diff_abs_pair_c25_2_3_prev12_is_positive
 same_dist_pair_c25_2_3_prev13
 dist_diff_abs_pair_c25_2_3_prev13_is_positive
 same_dist_pair_c25_2_3_prev14
 dist_diff_abs_pair_c25_2_3_prev14_is_positive
 same_dist_pair_c25_2_3_prev15
 dist_diff_abs_pair_c25_2_3_prev15_is_positive
 same_dist_pair_c25_2_3_prev16
 dist_diff_abs_pair_c25_2_3_prev16_is_positive
 same_dist_pair_c25_2_3_prev17
 dist_diff_abs_pair_c25_2_3_prev17_is_positive
 same_dist_pair_c25_2_3_prev18
 dist_diff_abs_pair_c25_2_3_prev18_is_positive
 same_dist_pair_c25_2_3_prev19
 dist_diff_abs_pair_c25_2_3_prev19_is_positive
 same_dist_pair_c25_2_3_prev20
 dist_diff_abs_pair_c25_2_3_prev20_is_positive
 same_dist_pair_c25_2_3_prev21
 dist_diff_abs_pair_c25_2_3_prev21_is_positive
 same_dist_pair_c25_2_3_prev22
 dist_diff_abs_pair_c25_2_3_prev22_is_positive
 same_dist_pair_c25_2_3_prev23
 dist_diff_abs_pair_c25_2_3_prev23_is_positive
 same_dist_pair_c25_2_3_prev24
 dist_diff_abs_pair_c25_2_3_prev24_is_positive
 same_dist_pair_c25_2_3_prev25
 dist_diff_abs_pair_c25_2_3_prev25_is_positive
 dx_grid_abs_pair_c25_2_4_is_positive dy_grid_abs_pair_c25_2_4_is_positive
 diff_dist_pair_c25_2_4 same_dist_pair_c25_2_4_prev0
 dist_diff_abs_pair_c25_2_4_prev0_is_positive same_dist_pair_c25_2_4_prev1
 dist_diff_abs_pair_c25_2_4_prev1_is_positive same_dist_pair_c25_2_4_prev2
 dist_diff_abs_pair_c25_2_4_prev2_is_positive same_dist_pair_c25_2_4_prev3
 dist_diff_abs_pair_c25_2_4_prev3_is_positive same_dist_pair_c25_2_4_prev4
 dist_diff_abs_pair_c25_2_4_prev4_is_positive same_dist_pair_c25_2_4_prev5
 dist_diff_abs_pair_c25_2_4_prev5_is_positive same_dist_pair_c25_2_4_prev6
 dist_diff_abs_pair_c25_2_4_prev6_is_positive same_dist_pair_c25_2_4_prev7
 dist_diff_abs_pair_c25_2_4_prev7_is_positive same_dist_pair_c25_2_4_prev8
 dist_diff_abs_pair_c25_2_4_prev8_is_positive same_dist_pair_c25_2_4_prev9
 dist_diff_abs_pair_c25_2_4_prev9_is_positive same_dist_pair_c25_2_4_prev10
 dist_diff_abs_pair_c25_2_4_prev10_is_positive
 same_dist_pair_c25_2_4_prev11
 dist_diff_abs_pair_c25_2_4_prev11_is_positive
 same_dist_pair_c25_2_4_prev12
 dist_diff_abs_pair_c25_2_4_prev12_is_positive
 same_dist_pair_c25_2_4_prev13
 dist_diff_abs_pair_c25_2_4_prev13_is_positive
 same_dist_pair_c25_2_4_prev14
 dist_diff_abs_pair_c25_2_4_prev14_is_positive
 same_dist_pair_c25_2_4_prev15
 dist_diff_abs_pair_c25_2_4_prev15_is_positive
 same_dist_pair_c25_2_4_prev16
 dist_diff_abs_pair_c25_2_4_prev16_is_positive
 same_dist_pair_c25_2_4_prev17
 dist_diff_abs_pair_c25_2_4_prev17_is_positive
 same_dist_pair_c25_2_4_prev18
 dist_diff_abs_pair_c25_2_4_prev18_is_positive
 same_dist_pair_c25_2_4_prev19
 dist_diff_abs_pair_c25_2_4_prev19_is_positive
 same_dist_pair_c25_2_4_prev20
 dist_diff_abs_pair_c25_2_4_prev20_is_positive
 same_dist_pair_c25_2_4_prev21
 dist_diff_abs_pair_c25_2_4_prev21_is_positive
 same_dist_pair_c25_2_4_prev22
 dist_diff_abs_pair_c25_2_4_prev22_is_positive
 same_dist_pair_c25_2_4_prev23
 dist_diff_abs_pair_c25_2_4_prev23_is_positive
 same_dist_pair_c25_2_4_prev24
 dist_diff_abs_pair_c25_2_4_prev24_is_positive
 same_dist_pair_c25_2_4_prev25
 dist_diff_abs_pair_c25_2_4_prev25_is_positive
 dx_grid_abs_pair_c25_3_4_is_positive dy_grid_abs_pair_c25_3_4_is_positive
 diff_dist_pair_c25_3_4 same_dist_pair_c25_3_4_prev0
 dist_diff_abs_pair_c25_3_4_prev0_is_positive same_dist_pair_c25_3_4_prev1
 dist_diff_abs_pair_c25_3_4_prev1_is_positive same_dist_pair_c25_3_4_prev2
 dist_diff_abs_pair_c25_3_4_prev2_is_positive same_dist_pair_c25_3_4_prev3
 dist_diff_abs_pair_c25_3_4_prev3_is_positive same_dist_pair_c25_3_4_prev4
 dist_diff_abs_pair_c25_3_4_prev4_is_positive same_dist_pair_c25_3_4_prev5
 dist_diff_abs_pair_c25_3_4_prev5_is_positive same_dist_pair_c25_3_4_prev6
 dist_diff_abs_pair_c25_3_4_prev6_is_positive same_dist_pair_c25_3_4_prev7
 dist_diff_abs_pair_c25_3_4_prev7_is_positive same_dist_pair_c25_3_4_prev8
 dist_diff_abs_pair_c25_3_4_prev8_is_positive same_dist_pair_c25_3_4_prev9
 dist_diff_abs_pair_c25_3_4_prev9_is_positive same_dist_pair_c25_3_4_prev10
 dist_diff_abs_pair_c25_3_4_prev10_is_positive
 same_dist_pair_c25_3_4_prev11
 dist_diff_abs_pair_c25_3_4_prev11_is_positive
 same_dist_pair_c25_3_4_prev12
 dist_diff_abs_pair_c25_3_4_prev12_is_positive
 same_dist_pair_c25_3_4_prev13
 dist_diff_abs_pair_c25_3_4_prev13_is_positive
 same_dist_pair_c25_3_4_prev14
 dist_diff_abs_pair_c25_3_4_prev14_is_positive
 same_dist_pair_c25_3_4_prev15
 dist_diff_abs_pair_c25_3_4_prev15_is_positive
 same_dist_pair_c25_3_4_prev16
 dist_diff_abs_pair_c25_3_4_prev16_is_positive
 same_dist_pair_c25_3_4_prev17
 dist_diff_abs_pair_c25_3_4_prev17_is_positive
 same_dist_pair_c25_3_4_prev18
 dist_diff_abs_pair_c25_3_4_prev18_is_positive
 same_dist_pair_c25_3_4_prev19
 dist_diff_abs_pair_c25_3_4_prev19_is_positive
 same_dist_pair_c25_3_4_prev20
 dist_diff_abs_pair_c25_3_4_prev20_is_positive
 same_dist_pair_c25_3_4_prev21
 dist_diff_abs_pair_c25_3_4_prev21_is_positive
 same_dist_pair_c25_3_4_prev22
 dist_diff_abs_pair_c25_3_4_prev22_is_positive
 same_dist_pair_c25_3_4_prev23
 dist_diff_abs_pair_c25_3_4_prev23_is_positive
 same_dist_pair_c25_3_4_prev24
 dist_diff_abs_pair_c25_3_4_prev24_is_positive
 same_dist_pair_c25_3_4_prev25
 dist_diff_abs_pair_c25_3_4_prev25_is_positive
Generals
 x_grid_0 y_grid_0 x_grid_1 y_grid_1 x_grid_2 y_grid_2 x_grid_3 y_grid_3
 x_grid_4 y_grid_4
End
