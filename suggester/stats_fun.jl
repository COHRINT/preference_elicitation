#Statistics Helper Function

#Brier Score Helper function
function brier_crunch(brier_diff,MC_runs,num_obs)
    #Function takes in an array with row for each MC_run and column is the number of observations
    #Returns an array with mean and std deviation for each observation
    brier_score = Array{Float64}(undef,MC_runs,num_obs+1)
    brier_plot = Array{Float64}(undef,2,num_obs+1)
    for r in 1:MC_runs
        for g in 1:num_obs+1
            #Apply summation
            brier_score[r,g] = sum(brier_diff[r,1:g])/g 
        end
    end
    #Extract statistics for plotting
    for g in 1:num_obs+1
        brier_plot[1,g] = mean(brier_diff[:,g])
        brier_plot[2,g] = std(brier_diff[:,g])
    end
    return brier_plot
end

function acc_crunch(acc_hist,MC_runs,num_obs)
    """Function will parse and calculate sequential accuracy and standard deviation over time"""
    acc_plot = Array{Float64}(undef,2,num_obs)
    for g in 1:num_obs
        acc_plot[1,g] = mean(acc_hist[:,g])
        acc_plot[2,g] = std(acc_hist[:,g])
    end
    return acc_plot
end

function user_select_MC(user_set_betas,guessing_points,user,user_ideal_seg,user_ideal_nn,MC_runs,num_guess)
    #Function to simulate the user automatically selecting points. 
    # Phi_estimated is calculated as the mean of all selected points including the initial set
    #Generate Average Set of Points for User as comparison
    user_avg_belief = Array{Float64}(undef,MC_runs,num_guess+1+length(user_set_betas))
    beta_values_select = [guessing_points[i][4:end] for i in 1:length(guessing_points)]
    #Code for taking average of initial set of points
    # mean_user_vals = [mean([user_set_betas[a][1] for a in 1:length(user_set_betas)]),
    #                 mean([user_set_betas[a][2] for a in 1:length(user_set_betas)]),
    #                 mean([user_set_betas[a][3] for a in 1:length(user_set_betas)])]
    # #Any zero values must be made non-zero to make phi a positive vector in Dirichlet Dist.
    # for a in 1:length(mean_user_vals)
    #     if mean_user_vals[a] == 0.0
    #         mean_user_vals[a] = 0.001
    #     end
    # end
    # all_points = Array{Vector{Float64}}(undef,length(user_data)+num_guess)
    all_user_points_phi = []
    if typeof(user_ideal_nn) == Bool
        user_ideal_vec = user_ideal_seg
    else
        user_ideal_vec = vcat(user_ideal_seg,[a[1] for a in user_ideal_nn])
    end
    # Account for initial set of points
    for p in 1:length(user_set_betas)
        point = user_set_betas[p]
        push!(all_user_points_phi,point)
        user_avg_belief[:,p] .= norm(mean(all_user_points_phi)-user_ideal_vec)
    end
    # Generate and add random observations
    for i in 1:MC_runs  # Generate independent set of interactions
        new_user_points = []
        for g in 1:num_guess+1  # Generate individual guesses
            new_point_idx,new_point_phi = sample_new_point(beta_values_select,user_ideal_seg,user_ideal_nn,user,new_user_points)
            push!(new_user_points,new_point_idx[1])        
            push!(all_user_points_phi,new_point_phi[1])
            user_avg_belief[i,g+length(user_set_betas)] = norm(mean(all_user_points_phi)-user_ideal_vec)
        end
    end

    return user_avg_belief
end

function find_categories(chosen_idx)
    """Function will sort through the chosen truth data and categorize it into necessary bins"""
    chosen_data = [parse(Int64,i) for i in chosen_idx]
    score = zeros(8) # Init array
    for i in chosen_data
        if i<= 5 # frontdoor case
            score[1] += 1
        elseif i<=10 # road edges
            score[2] += 1
        elseif i<=15 # road_intersection
            score[3] += 1
        elseif i<=21 # corners
            score[4] += 1
        elseif i<=26 # backdoor
            score[5] += 1
        elseif i<= 31 # road
            score[6] += 1
        elseif i<= 36 # building
            score[7] += 1
        else          # other
            score[8] += 1
        end
    end
    return score

end

function accuracy_score(chosen_idx,truth_data,secondary_data)
    """Function will take in a set of chosen points, truth vector, and secondary vector.
    Will output a single value"""
    chosen_data = [parse(Int64,i) for i in chosen_idx]
    correct = 0.0
    wrong_data = 0.0
    # Calculate accuracy score
    for i in chosen_data
        if i in truth_data # Full credit
            correct += 1
        elseif i in secondary_data # Provide partial credit
            correct += 0.5
        else
            wrong_data += 1
        end
    end
    score = correct/(length(chosen_data))

    return score

end