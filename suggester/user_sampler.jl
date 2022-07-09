using Distributions

include("user_model.jl")

#User Data
user_frontdoor = read_data("./data/point_sampling_reference/user_frontdoor.csv")
user_backdoor = read_data("./data/point_sampling_reference/user_backdoor.csv")
user_building = read_data("./data/point_sampling_reference/user_building.csv")
user_road = read_data("./data/point_sampling_reference/user_road.csv")
test_points = read_data("./data/point_sampling_reference/user_test.csv")
user_road_edges = read_data("./data/point_sampling_reference/user_roadedges.csv")
user_corners = read_data("./data/point_sampling_reference/user_corners.csv")
points_data = random_data_300 #* (100/30)

user_ideal = [0.05,0.5,0.5]
user_data = [user_road_edges[i][4:end] for i in 1:length(user_road_edges)]
user = user_oracle
n_points = 50

point_set = [rand(Dirichlet([1,1,1])) for i in 1:n_points]
answer = ones(n_points)
for (p_idx,p) in enumerate(point_set)
    r = sample_user_response(p,user_ideal,false,user)
    if r =="accept"
        answer[p_idx] = 0
    else
        answer[p_idx] = 1
    end 
end



user_features = [user_data[i][4:end] for i in 1:length(user_data)]
matwrite("road_basic_output.mat", Dict(
	"step" => vcat(zeros(5),range(1,n_points)),
    "point_vectors" => vcat(user_data,point_set),
    "response" => vcat(ones(5)*-1,answer),
))