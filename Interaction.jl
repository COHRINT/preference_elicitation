#Interaction Script
# This script iterates through possible points to suggest. 
# Input:
    #User data
    #Random data for suggestions
    #Data to apply suggestions on
using MAT
using CSV

#Include other functions
include("data_read.jl")
include("plot_image.jl")
include("user_model.jl")
include("PE_POMDP_Def.jl")
include("ParticleFilter_Def.jl")

#Load point data
random_data = read_data("./data/point_sampling_reference/random_data.csv")
random_data_300 = read_data("./data/point_sampling_reference/random_data_300.csv")
neighborhood_data = read_data("./data/point_sampling_reference/neighborhood_350.csv")

#User Data
user_frontdoor = read_data("./data/point_sampling_reference/user_frontdoor.csv")
user_backdoor = read_data("./data/point_sampling_reference/user_backdoor.csv")
user_building = read_data("./data/point_sampling_reference/user_building.csv")
user_road = read_data("./data/point_sampling_reference/user_road.csv")
test_points = read_data("./data/point_sampling_reference/user_test.csv")
user_road_edges = read_data("./data/point_sampling_reference/user_roadedges.csv")
user_corners = read_data("./data/point_sampling_reference/user_corners.csv")
points_data = random_data_300 #* (100/30)
final_points_data = neighborhood_data #

# Points operator has chosen: 
### ---  MODIFY TEST CASE HERE  --- ###
# user_data = user_frontdoor
filename = "./data/point_sampling_reference/out_images/testimage.png" #Final image for saving
filename_final = "./data/point_sampling_reference/out_images/test_final_image.png" #Final image for saving

#Choose a user model
user = user_oracle
# user_ideal = [0.05,0.9,0.05] #[%building,%road,%other] # road
# user_ideal = [0.9,0.05,0.05] # building
# user_ideal = [0.85,0.05,0.8] # backdoor
# user_ideal = [0.8,0.5,0.05] # frontdoor
user_ideal = [0.05,0.8,0.5]  # road edges
# user_ideal = [0.05,0.5,0.5] # road edges, exact
# user_ideal = [0.25, 0.05, 0.75] # corners
user_data = user_road_edges
#Number of steps before making selection
num_guess = 50

function _run(user_data,user_ideal_seg,user_ideal_nn,guess_points,final_points,choice_points,user_mode,guess_steps)
    #Input:
    #   user_data = [p_x,p_y,radius,%building,%road,%other] Full data vector    #Initial set of user points
    #   user_ideal = [%building,%road,%other] Desired feature vector from segmentation           # True ideal input from segmentation
    #   user_ideal_seg = [%f1,...,%fn] Desired feature vector from nn components  # True ideal input from neural network
    #   guess_points = [p_x,p_y,radius,%building,%road,%other] Full data vector # Points to be guessed/suggested by algorithm
    #   final_points = [p_x,p_y,radius,%building,%road,%other] Full data vector #Final points to be propagated
    #   choice_points = [p_x,p_y,radius,%building,%road,%other] Full data vector# Points that the user may suggest
    #   user_mode = user type (i.e. Novice, or expert)
    #   guess_steps = number of guesses by algorithm
    #   
    #Output:
    #   p_belief = Particle Filter object --> See ParticleFilter_Def.jl
    #   
    #   Following outputs are all in consistent format: ["idx1","idx2",...]
    #       user_points: Set of points chosen by the user as part of wait action
    #       accepted_points: Set of points suggested and accepted by user
    #       denied_points: Set of points suggested and denied by user
    ####

    #Extract beta Values
    beta_values = [guess_points[i][4:end] for i in 1:length(guess_points)]
    final_beta_values = [final_points[i][4:end] for i in 1:length(final_points)]
    choice_beta_values = [choice_points[i][4:end] for i in 1:length(choice_points)]

    u_points = user_data            #Initial set of user points
    s_points = beta_values          #Points that can be suggested by algorithm
    f_points = final_beta_values    #Final points to be propagated
    o_points = choice_beta_values   #Points that the user can randomly select
    #Solver Definition
    randomMDP = FORollout(RandomSolver())
    solver = POMCPSolver(tree_queries=100, c=100.0, rng=MersenneTwister(1), tree_in_info=true,estimate_value = randomMDP)

    #Get statistics on initial set of user points
    phi = []
    cov = []
    for i in 4:length(u_points[1])
        ave_i = mean([u_points[a][i] for a in 1:length(u_points)])
        push!(phi, ave_i)
        cov_i = std([u_points[a][i] for a in 1:length(u_points)])
        push!(cov, cov_i)
    end

    phi = phi/norm(phi)  # Normalize
    cov = cov/norm(cov)
    #Any zero values must be made non-zero to make phi a positive vector in Dirichlet Dist.
    for a in 1:length(phi)
        if phi[a] == 0.0
            phi[a] = 1e-5
        end
    end

    #Initilize Belief with Particle Filter
    #Create Gaussian Distribution
    p = 10000 #Number of particles
    p_sample = 20 #Number of user actions to consider --> Size of action space
    p_belief = init_PF(user_ideal_seg,user_ideal_nn,p)

    point_history = []
    accepted_points = []
    user_points = []
    denied_points = []
    suggested_points = []
    p_belief_history = Array{InjectionParticleFilter}(undef,guess_steps+1)
    best_points_idx,best_points_phi = find_similar_points(s_points,phi,p_sample,[])
    for step in 1:guess_steps+1
        #Save particle belief
        p_belief_history[step] = p_belief
        #Initialize POMDP with new action space: Figure out best action
        #   Input into POMDP is only the beta values
        #   Output is the index of the suggested value or "wait"
        model_step = guess_steps+1-step  # Lets solver know how many steps are left
        PE_fun =  PE_POMDP(u_points,best_points_phi,o_points,f_points,user_mode,0.99,model_step)  # Define POMDP
        planner = solve(solver, PE_fun)
        a, info = action_info(planner, initialstate(PE_fun), tree_in_info=false)
        # inchrome(D3Tree(info[:tree], init_expand=3))
        
        # Action response 
        if a == "wait"
            #Randomly sample point based on user model
            new_user_idx,new_user_point = sample_new_point(choice_beta_values,user_ideal_seg,user_ideal_nn,user_mode,user_points)
            #Update Particle Belief
            p_belief = update_PF(p_belief,PE_fun,a,new_user_point)
            push!(user_points,new_user_idx[1]) #Record keeping
        else
            #Find global point from suggested point
            suggested_idx = best_points_idx[parse(Int64,a)]
            #Randomly sample user's response based on user model
            response = sample_user_response(s_points[parse(Int64,suggested_idx)],user_ideal_seg,user_ideal_nn,user_mode)
            #Update Particle Belief
            p_belief = update_PF(p_belief,PE_fun,a,response)
            
            #Record Keeping
            if response=="accept"
                push!(point_history,(string(suggested_idx),0))
                push!(accepted_points,string(suggested_idx))
            else
                push!(point_history,(string(suggested_idx),1))
                push!(denied_points,string(suggested_idx))
            end
            push!(suggested_points,string(suggested_idx))
        end

        #Update set of points to iterate through
        # #Sample from Particle filter
        # particles = mean(p_belief.states)
        # #Replace sampled points
        # for sample in 1:p_sample
        #     idx,phi = find_similar_points(s_points,particles,1,suggested_points)
        #     best_points_idx[sample] = idx[1]
        #     best_points_phi[sample] = phi[1]
        # end
        # Semi random sampling
        particles = rand(p_belief.states,p_sample)
        #Replace sampled points
        for sample in 1:length(particles)
            idx,phi = find_similar_points(s_points,particles[sample],1,suggested_points)
            best_points_idx[sample] = idx[1]
            best_points_phi[sample] = phi[1]
        end
    end
    return p_belief,user_points,accepted_points,denied_points,p_belief_history, point_history
end

belief,user_points,accepted_points,denied_points,belief_hist,point_history = _run(user_data,user_ideal,false,points_data,final_points_data,random_data,user,num_guess)


# #Propagate belief onto new image
#  chosen = final_guess(final_points_data,belief,10)


# #Visualization and image plotting
# #Initial Image extraction
# u_x,u_y = extract_xy(user_points,points_data)
# a_x,a_y = extract_xy(accepted_points,points_data)
# d_x,d_y = extract_xy(denied_points,points_data)
# i_x = [user_data[i][1] for i in 1:length(user_data)]
# i_y = [user_data[i][2] for i in 1:length(user_data)]

# #Final values extraction
# p_x,p_y = extract_xy(chosen,final_points_data)

# guess_image = "./images/Image1_raw.png"
# final_image = "./images/neighborhood_image.jpg"
# plot_image(guess_image,[i_x,i_y],[u_x,u_y], [a_x,a_y], [d_x,d_y], filename)
# plot_image(final_image,[],[],[p_y,p_x],[],filename_final)

# Extract point feature vectors
# user_data = vcat(user_road_edges[1:2],final_points_data[1:5:16])

## MATLAB Data Extraction and Saving
accept_points_f = extract_vector(accepted_points,points_data)
denied_points_f = extract_vector(denied_points,points_data)
all_points_i = [point_history[i][1] for i in 1:length(point_history)]
all_points_responses = [point_history[i][2] for i in 1:length(point_history)]
all_points_f = extract_vector(all_points_i,points_data)

# # Save interaction to CSV
# matwrite("road_edges_interaction_output_oracle.mat", Dict(
# 	"step" => vcat(zeros(5),range(1,num_guess)),
#     "point_vectors" => vcat(user_data,accept_points_f,denied_points_f),
#     "response" => vcat(ones(5)*-1,zeros(length(accept_points_f))*0,ones(length(denied_points))),
# ))
matwrite("road_edges_bad_prior.mat", Dict(
	"step" => vcat(zeros(5),range(1,num_guess,length=num_guess)),
    "point_vectors" => vcat(user_data,all_points_f),
    "response" => vcat(ones(5)*-1,all_points_responses),
))
