#This script creates and runs Monte_Carlo testing 
using HypothesisTests
using Plots
using LaTeXStrings

include("data_read.jl")
include("plot_image.jl")
include("user_model.jl")
include("PE_POMDP_Def.jl")
include("ParticleFilter_Def.jl")
include("Interaction.jl")
include("stats_fun.jl")
include("all_data.jl")
#Load point data
#random_data = read_data("./data/random_data.csv")
#random_data_300 = read_data("./data/random_data_300.csv")
#neighborhood_data = read_data("./data/neighborhood_350.csv")

#User Data
#user_frontdoor = read_data("./data/user_frontdoor.csv")
#user_backdoor = read_data("./data/user_backdoor.csv")
#user_building = read_data("./data/user_building.csv")
#user_road = read_data("./data/user_road.csv")
#test_points = read_data("./data/user_test.csv")
#user_road_edges = read_data("./data/user_roadedges.csv")
#points_data = random_data_300 #* (100/30)
#final_points_data = neighborhood_data #

# Points operator has chosen: 
### ---  MODIFY TEST CASE HERE  --- ###
case = "Road_Edge.png"
filename = "./data/out_images/testimage.png" #Final image for saving
filename_final = "./data/out_images/Propagation_"*case
brier_filename = "./data/out_images/Brier_Score_"*case #Final image for saving
acc_filename = "./data/out_images/Accuracy_"*case
combined_filename = "./data/out_images/combined_"*case
plot_title = L"\textrm{Road \; Edge\; Case}"
save_image = false # should images be saved
#Choose a user model
#user = user_expert
user_label = "Expert"
# If nn segmentation is required, calculate statistics in the following format
# Upper bounds could also be added by expanding feature n vector with max bound. Requires modification of user_dist in user_model.jl
    # user_ideal_nn = [[f1_mean,f1_std],[f2_mean,f2_std],....]
#Number of steps before making selection

# Other
# user_ideal_seg = [1.0e-5, 0.003008, 0.9970000000000001]
# # user_ideal_seg = false
# user_ideal_nn = [[0.07263907233762741, 0.02165527830410004], [0.41146644516086583, 0.12480474872684479], [0.3319278836250305, 0.1724170058965683], [0.0006032396819591521, 0.0006032396819591521], [0.009920102754116058, 0.0032006491212844845], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [0.06819520387125017, 0.06819520387125017], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [0.11896820905876161, 0.11896820905876161], [0.22682479843497277, 0.17731990069150924], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [0.4476867571473122, 0.21345937997102737], [0.31375013291835785, 0.24240114092826845]]
# # user_ideal_nn = false
# # user_data = user_other
# user_data = vcat(user_other[1:2],random_data[10:12])
# user_type = "other"
# truth_data = [37, 38, 39, 40, 41]
# secondary_data = [6,7,8,9,10,22,23,24,25,26]

# Building
# user_ideal_seg = [0.938, 0.051008, 0.011007]
# user_ideal_seg = [0.99,0.01,0.01]
# # user_ideal_seg = false
# user_ideal_nn = [[5.264311169219971, 4.29630260684877], [43.97087006433105, 33.96614423331778], [42.15649070739746, 28.097835829635844], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [39.572313690185545, 25.068681618215294], [1.0e-5, 1.0e-5], [1.5778259859809876, 1.2738855601378307], [36.16738414764404, 20.966888473719315], [11.714532208442687, 10.576620850911347], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [41.75664463043213, 26.140954152374192], [58.43567428588867, 34.762936927278616]]
# # user_ideal_nn = false
# user_data = vcat(user_building[1:2],random_data[5:7])
# # user_data = user_building
# user_type = "building"
# truth_data = [32, 33, 34, 35, 36]
# secondary_data = [22,23,24,25,26,1,2,3,4,5,16,17,18,19,20,21]

# Road
user_ideal_seg = [1.0e-5, 0.965, 0.035006999999999996]
user_ideal_seg = [1e-05,0.99,0.01]
# user_ideal_seg = false
user_ideal_nn = [[1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [5.906597518920899, 3.2692086383050594], [1.2925521731376648, 0.89715294504248], [1.0e-5, 1.0e-5], [0.3755006941814423, 0.28498842936894064], [1.0e-5, 1.0e-5], [2.3031199336051937, 1.6053436483921901], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [15.200850009918213, 9.044101738626972], [31.499194335937503, 16.395874084182477], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [6.879333925247193, 4.162503932599528], [34.41471214294434, 17.23839556758868]]
# user_ideal_nn = false
# user_data = user_road
user_data = vcat(user_road[1:2],random_data[5:7])
# user_data = vcat([user_road[1]],random_data[5:6])
user_type = "road"
truth_data = [27, 28, 29, 30, 31]
secondary_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# Back
# user_ideal_seg = [0.42700000000000005, 0.028006000000000007, 0.542]
# user_ideal_seg = [0.5,0.001,0.5]
# # user_ideal_seg = true
# user_ideal_nn = [[5.664534752731322, 5.52978103866133], [30.118653392791746, 27.337810540139664], [27.76417036056519, 25.697659788668897], [1.3806003264236448, 1.226070801624417], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [16.23710560798645, 12.259506138521743], [1.0e-5, 1.0e-5], [0.016992027756214143, 0.03396905551242828], [14.294671915099025, 12.833800031936985], [12.129328127601624, 11.10831372971263], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [31.535446453094483, 27.930144286335643], [32.1404447555542, 27.206077283819766]]
# # user_ideal_nn = false
# user_data = user_backdoor
# user_type = "backdoor"
# truth_data = [22,23,24,25,26]
# secondary_data = [1,2,3,4,5,32, 33, 34, 35, 36]

# Front
# user_ideal_seg = [0.508, 0.425, 0.06700700000000001] # 10 points mean
# user_ideal_seg = [0.5,0.5,0.001]
# # user_ideal_seg = false
# user_ideal_nn = [[13.23763856072998, 15.239379933873611], [27.95487188749695, 33.195982076716746], [19.781485843658444, 14.846136717543558], [3.18339701781559, 3.2680729276842237], [1.0e-5, 1.0e-5], [0.5918124805526733, 0.5918124805526733], [1.0e-5, 1.0e-5], [19.195171928405763, 11.22429294062719], [1.0e-5, 1.0e-5], [0.24409540861511234, 0.4881758172302245], [28.695131874084474, 15.678388500608618], [23.136948080757143, 21.00518717801964], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [33.05136442184448, 28.620755150999152], [36.4965124130249, 22.144324881279104]]
# # user_ideal_nn = false
# user_data = vcat(user_frontdoor[1:2],random_data[5:7])
# user_type = "frontdoor"
# truth_data = [1,2,3,4,5]
# secondary_data = [1,2,3,4,5,32, 33, 34, 35, 36]

# Corner
# user_ideal_seg = [0.3373333333333334, 0.14133633333333334, 0.526]
# user_ideal_seg = [0.25,0.75,0.15]
# # user_ideal_seg = false
# user_ideal_nn = [[5.587942456319173, 7.695231683369954], [24.80763174818548, 24.373808615623272], [17.051923394203186, 15.349956170636485], [1.3579950174830755, 1.701846639629682], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [18.957929150263467, 14.087000374884578], [1.0e-5, 1.0e-5], [0.6319231958332061, 0.8437448554401397], [26.728793799380462, 21.679275325952162], [9.980989354698181, 9.040870465319346], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [20.175454576810203, 17.79307487611532], [29.23522793451945, 17.649390878791138]]
# # user_ideal_nn = false
# # user_data = user_corner
# user_data = vcat(user_backdoor[1:2],random_data[5:7])
# user_type = "corner"
# truth_data = [16,17,18,19,20,21]
# secondary_data = [32, 33, 34, 35, 36,1,2,3,4,5,22,23,24,25,26]

# Road edges
# user_ideal_seg =[1.0e-5, 0.402, 0.598]
# user_ideal_seg = [1e-05,0.5,0.5]
# # user_ideal_seg = true
# user_ideal_nn = [[0.014648630913734436, 0.02928226182746887], [0.02007299929523468, 0.04013099859046936], [3.952627662088394, 4.1230016144205806], [3.259864567352295, 2.336608401626127], [1.0e-5, 1.0e-5], [1.3095373801040648, 0.8749611425484967], [1.0e-5, 1.0e-5], [2.3160428588905333, 3.163564261034497], [1.0e-5, 1.0e-5], [0.2970242630081177, 0.3667072078948005], [11.789165332913399, 10.788755789237065], [16.840385460853575, 14.518254473298343], [1.0e-5, 1.0e-5], [0.04930475622081757, 0.09859451244163514], [4.516840573051453, 4.926783939602433], [18.030269956588747, 15.76251229118354]]
# user_ideal_nn = false
# user_data = vcat(user_road_edges[1:2],random_data[7:10])
# # user_data = user_road_edges[1:2]
# user_type = "road_edge"
# truth_data = [6,7,8,9,10]
# secondary_data = [11,12,13,14,15,27, 28, 29, 30, 31]

# Road intersections
# user_ideal_seg = [1.0e-5, 0.601, 0.399]
# user_ideal_seg = [0.001,0.75,0.25]
# # user_ideal_seg = false
# user_ideal_nn = [[1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [4.4143859395215514, 4.63824825165725], [3.388052255443573, 3.0480931655319807], [1.0e-5, 1.0e-5], [1.3562076500701905, 1.4684666641597086], [1.0e-5, 1.0e-5], [3.1946378965167996, 3.352817576223992], [1.0e-5, 1.0e-5], [0.0054849338498115545, 0.010954867699623108], [13.293785285949706, 10.526544075643134], [21.16497669219971, 15.539728994988035], [1.0e-5, 1.0e-5], [1.0e-5, 1.0e-5], [4.920911428596496, 5.247877544992313], [23.436019802093504, 16.736848160976326]]
# # user_ideal_nn = false
# # user_data = user_road_intersection
# user_data = vcat(user_road_intersection[1:2],random_data[3:5])
# user_type = "road_intersection"
# truth_data = [11,12,13,14,15]
# secondary_data = [11,12,13,14,15,27, 28, 29, 30, 31, 6,7,8,9,10]

num_guess = 10
MC_runs = 500

chosen_set = []
chosen_idx = []
avg_belief = []
std_belief = []


#Consider nn feature vector case
if typeof(user_ideal_nn) == Bool
    user_ideal = user_ideal_seg
elseif typeof(user_ideal_seg) == Bool
    user_ideal = [a[1] for a in user_ideal_nn]
else
    user_ideal = vcat(user_ideal_seg,[a[1] for a in user_ideal_nn])
end

for u in 1:1
# for u in 1:2
    println(u)
    brier_diff = Array{Float64}(undef,MC_runs,num_guess+1)
    global acc_score_hist = Array{Float64}(undef,MC_runs,num_guess)
    if u == 2
        local user = user_novice
        local user_label = "Novice POMDP"
    else
        local user = user_expert
        local user_label = "Expert POMDP"
    end
    for i in 1:MC_runs
        #Run Planner to find belief
        global belief,user_points,accepted_points,denied_points,hist =
            _run(user_data,user_ideal_seg,user_ideal_nn,points_data,final_points_data,random_data,user,num_guess)
        #Propagate belief onto new image
        local chosen = final_guess(final_points_data,belief,5)

        #Add chosen points to list
        for I in chosen
            vals = final_points_data[parse(Int64,I)][4:end]
            push!(chosen_set,vals)
            push!(chosen_idx,I)
        end
        push!(avg_belief,mean(belief.states))
        push!(std_belief,std(belief.states))
        
        #Parse and save belief data 
        for g in 1:length(hist.p_belief)
            brier_diff[i,g] = norm(mean(hist.p_belief[g].states)-user_ideal)
        end

        #Parse and save accuracy data
        r,c = size(hist.final_points)
        for g in 1:num_guess
            # Calculate accuracy at every guess point
            global acc_score_hist[i,g] = accuracy_score(hist.final_points[:,g],truth_data,secondary_data)
            # print(hist.final_points[:,g])
        end

    end
    brier_plot = brier_crunch(brier_diff,MC_runs,num_guess)
    acc_plot = acc_crunch(acc_score_hist,MC_runs,num_guess)
    local x = range(1,num_guess,length = num_guess)
    acc_score = accuracy_score(chosen_idx,truth_data,secondary_data)
    println(user_type)
    println("Accuracy Score",acc_score)
    if (u == 1) 
        local x = range(0,num_guess+1,length = num_guess+1)
        global p_MSE = plot(x,brier_plot[1,:],ribbon = brier_plot[2,:],label = user_label,
        title= plot_title, xlabel= L"\textrm{User\;  Suggestions}",ylabel = L"\textrm{Mean\;  Error}", legend = false)
        local x = range(1,num_guess,length = num_guess)
        global p_acc = plot(x,acc_plot[1,:],ribbon = acc_plot[2,:],label = user_label,
        title = plot_title, xlabel = L"\textrm{User\;  Suggestions}",ylabel = L"\textrm{Accuracy}",legend = false)
    else
        # plot!(x,brier_plot[1,:],ribbon = brier_plot[2,:],label = user_label)
        plot!(x,acc_plot[1,:],ribbon = acc_plot[2,:],label = user_label)
    end
end

#user_set_mean = [mean([user_data[a][4] for a in 1:length(user_data)]),
#                mean([user_data[a][5] for a in 1:length(user_data)]),
#                mean([user_data[a][6] for a in 1:length(user_data)])]
user_set_betas = [user_data[i][4:end] for i in 1:length(user_data)]
#Final values extraction
p_x,p_y = extract_xy(chosen_idx,final_points_data)


# #Initiate User Selection for EXPERT
# # user = user_expert
# # user_avg_belief_exp = user_select_MC(user_set_betas,points_data,user,user_ideal_seg,user_ideal_nn,MC_runs,num_guess)
# #Sum briers
# # user_brier_plot_exp = brier_crunch(user_avg_belief_exp,MC_runs,num_guess+length(user_set_betas))
# #Plot user data
# # x = range(0,num_guess+1,length = num_guess+1)
# #p = plot!(x,user_brier_plot_exp[1,length(user_set_betas)+1:end],ribbon = user_brier_plot_exp[2,length(user_set_betas)+1:end],label =:"Expert Average")

# #Initiate User Selection for NOVICE
# # user = user_novice
# # user_avg_belief_nov = user_select_MC(user_set_betas,points_data,user,user_ideal_seg,user_ideal_nn,MC_runs,num_guess)
# #Sum brier 
# # user_brier_plot_nov = brier_crunch(user_avg_belief_nov,MC_runs,num_guess+length(user_set_betas))
# #Plot user data
# # x = range(0,num_guess+1,length = num_guess+1)
# #p = plot!(x,user_brier_plot_nov[1,length(user_set_betas)+1:end],ribbon = user_brier_plot_nov[2,length(user_set_betas)+1:end],label =:"Novice Average")

# #Greedy Policy

# title!(plot_title)
# xlabel!(L"\textrm{User Suggestions}")
# ylabel!(L"\textrm{Accuracy}")
# display(p)
if save_image
    # savefig(p_acc, brier_filename)
    savefig(p_acc, acc_filename)
end
#Plot all points on image
final_image = "./images/neighborhood_image.jpg"
p_neighborhood = plot_image(final_image,[],[],[p_x,p_y],[],filename_final,save_image)
#Initial Image extraction
u_x,u_y = extract_xy(user_points,points_data)
a_x,a_y = extract_xy(accepted_points,points_data)
d_x,d_y = extract_xy(denied_points,points_data)
i_x = [user_data[i][1] for i in 1:length(user_data)]
i_y = [user_data[i][2] for i in 1:length(user_data)]
guess_image = "./images/Image1_raw.png"
p_guess = plot_image(guess_image,[i_x,i_y],[u_x,u_y], [a_x,a_y], [d_x,d_y], filename,save_image)


# p_hist = plot(histogram([wrong_data,correct_data], bins=0:1:42), xticks=0:5:40)
# title!("Accuracy: "*string(length(correct_data)/(length(correct_data)+length(wrong_data))))
acc_score = accuracy_score(chosen_idx,truth_data,secondary_data)
println(user_type)
println("Accuracy Score",acc_score)
cat_score = find_categories(chosen_idx)
names = [L"\textrm{Frontdoor}",L"\textrm{Edges}",L"\textrm{Intersections}",L"\textrm{Corners}",L"\textrm{Backdoor}",
L"\textrm{Road}",L"\textrm{Building}",L"\textrm{Other}"]
p_bar = bar(names,
            [cat_score[1],cat_score[2],cat_score[3],cat_score[4],cat_score[5],cat_score[6],cat_score[7],cat_score[8]],
            xrotation=45,labels = names,
            title = L"\textrm{Accuracy: %$acc_score}",
            ylabel = L"\textrm{Number\; of\; Guesses}",
            legend = false)

l = @layout [a b; c d] 
full = plot(p_acc,p_bar,p_MSE,p_guess, layout = l)
display(full)
if save_image
    savefig(full, combined_filename)
end

# out_array = []
# for i in user_ideal_nn
#     push!(out_array, i[1])
# end
# # println(out_array)

# mean_bl = mean(avg_belief)
println("Monte Carlo Testing Finished")
# print(mean_bl)
# #Finding means
# #Compare chosen points with sampled ideal
# sample_avg_b = mean([chosen_set[a][1] for a in 1:length(chosen_set)])
# sample_avg_r = mean([chosen_set[a][2] for a in 1:length(chosen_set)])
# sample_avg_n = mean([chosen_set[a][3] for a in 1:length(chosen_set)])
# sample_avg_set = [sample_avg_b,sample_avg_r,sample_avg_n]
# # println(sample_avg_b," ",sample_avg_r," ",sample_avg_n)


# #Sample user dist
# user_sample = rand(user_dist,MC_runs*num_guess)
# user_b = mean([user_sample[:,a][1] for a in 1:length(user_sample[1,:])])
# user_r = mean([user_sample[:,a][2] for a in 1:length(user_sample[1,:])])
# user_n = mean([user_sample[:,a][3] for a in 1:length(user_sample[1,:])])
# user_set = [user_b,user_r,user_n]
# # println(user_set)

# #TTesting
# mean_bl = mean(avg_belief)
# std_bl = mean(std_belief)
# ttest_b = OneSampleTTest(mean_bl[1],std_bl[1],MC_runs,user_ideal[1])
# ttest_r = OneSampleTTest(mean_bl[2],std_bl[2],MC_runs,user_ideal[2])
# ttest_n = OneSampleTTest(mean_bl[3],std_bl[3],MC_runs,user_ideal[3])
