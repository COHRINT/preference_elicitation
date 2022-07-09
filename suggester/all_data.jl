include("data_read.jl")

#Load point data
random_data = read_data("./data/combined_features/sampled_300_airstrip_hand_segmented_combined.csv")
random_data_300 = read_data("./data/combined_features/sampled_300_airstrip_hand_segmented_combined.csv")
#neighborhood_data = read_data("./data/combined_features/sampled_300_neighborhood_image_segmented_combined.csv")
neighborhood_data = read_data("./data/combined_features/final_features/all_final_41_neighborhood_image_segmented_r_combined.csv")

#User Data
user_frontdoor = read_data("./data/combined_features/user_features/user_frontdoor_5_airstrip_hand_segmented_combined.csv")
user_backdoor = read_data("./data/combined_features/user_features/user_backdoor_5_airstrip_hand_segmented_combined.csv")
user_building = read_data("./data/combined_features/user_features/user_building_5_airstrip_hand_segmented_combined.csv")
user_road = read_data("./data/combined_features/user_features/user_road_5_airstrip_hand_segmented_combined.csv")
user_other = read_data("./data/combined_features/user_features/user_other_5_airstrip_hand_segmented_combined.csv")
user_road_intersection = read_data("./data/combined_features/user_features/user_roadintersection_5_airstrip_hand_segmented_combined.csv")
user_corner = read_data("./data/combined_features/user_features/user_corners_5_airstrip_hand_segmented_combined.csv")
#test_points = read_data("./data/user_test_5_airstrip_hand_segmented_ae16.csv")
user_road_edges = read_data("./data/combined_features/user_features/user_roadedges_5_airstrip_hand_segmented_combined.csv")
points_data = random_data_300 #* (100/30)
final_points_data = neighborhood_data #