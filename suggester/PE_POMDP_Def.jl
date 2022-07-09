using POMDPs, POMDPModelTools
using BasicPOMCP
using Random
using ParticleFilters, Distributions
# using Plots
using LinearAlgebra
using Distances
using Statistics
using POMDPPolicies

# #Include other functions
include("user_model.jl")


function make_observations()
    # Creates a list of potential observations in an array of strings
    total_act = length(suggest_points)
    a = Array{String}(undef, total_act+3)
    for i in 1:total_act
        a[i] = string(i)
    end
    a[end-1] = "accept"
    a[end] = "deny"
    a[end-2] = "no response"
    return a
end

function make_actions() # TODO move into actions function for final map guess
    # Creates a list of potential actions in an array of strings
    total_act = length(best_points) # TODO points_data -> final map guess
    a = Array{String}(undef, total_act+1)
    for i in 1:total_act
        a[i] = string(i)
    end
    a[end] = "wait"
    return a
    # deleteat!(a, findall(x->x=="1", a))
end

function similarity(x, y)
    # Similarity metric calculator
    if length(x)!= length(y)
        if (length(x)==3) | (length(y) == 3)
            return 1-cosine_dist(x[1:3],y[1:3])
        elseif length(y) == 16  # If nn only is being used
            return 1-cosine_dist(x[4:end],y)
        elseif length(x) == 16  # nn only being used
            return 1-cosine_dist(x,y[4:end])
        else
            println("Vector Length ",length(x)," ",length(y))
            error("Similarity vector inconsistency")
        end
    else
        return 1-cosine_dist(x, y)
    end
end

function sample_initial_state(m)
    # Determine the starting state. Calls the global state variable and puts it in a Vector
    # Can be modified to add noise
    POI = m.user_points  
    phi = []
    cov = []
    for i in 4:length(POI[1])
        ave_i = mean([POI[a][i] for a in 1:length(POI)])
        push!(phi, ave_i)
        cov_i = std([POI[a][i] for a in 1:length(POI)])
        push!(cov, cov_i)
    end
    phi = phi/norm(phi)  # Normalize
    init_cov = zeros(length(phi), length(phi))
    for i in 1:length(phi)
        init_cov[i,i] = cov[i]^2
    end
    # Create Dirichlet distribution
    phi = phi*m.user.certainty
    for (idx,i) in enumerate(phi)
        if i==0
            phi[idx] = 0.001
        end
    end
    phi_dist = rand(Dirichlet(phi))
    return State(phi_dist,[],0)
end

#--- POMDP Definition ---#
struct State
    phi::Vector{Float64}
    # cov::Array{Float64,2}    
    acts::Array{String,1}
    step::Int64
end

struct PE_POMDP <: POMDP{State,String,String} 
    user_points::Array{Any,1}       # Points provided by the user
    suggest_points::Array{Any,1}    # Points that can be suggested to the user
    observe_points::Array{Any,1}    # Points that can be added by user without suggestion
    final_points::Array{Any,1}      # Final set of points to be propagated over
    user::User_Model                # User type
    discount_factor::Float64        # Discount 
    guess_steps::Int64              # Number of steps for guessing 
end

function POMDPs.initialstate(m::PE_POMDP)
    function init_state(rng)
        POI = m.user_points  
        means = []
        cov = []
        for i in 4:length(POI[1])
            ave_i = mean([POI[a][i] for a in 1:length(POI)])
            push!(means, ave_i)
            cov_i = std([POI[a][i] for a in 1:length(POI)])
            push!(cov, cov_i)
        end
        means = means/norm(means)  # Normalize
        # Make sure nothing is equal to zero
        for (idx,i) in enumerate(means)
            if i==0
                means[idx] = 0.0001
            end
        end
        # Sample the set of means to find the initial state
        state_vec = zeros(length(means))
        state_vec[1:3] = [rand(Dirichlet(means))[i] for i in 1:3]
        for (idx,i) in enumerate(means)
            if idx>3
                state_vec[idx] = rand(TruncatedNormal(i,cov[idx],0,100))
            end
        end
        # init_cov = zeros(length(phi), length(phi))
        # for i in 1:length(phi)
        #     init_cov[i,i] = cov[i]^2
        # end
        new_state = State(state_vec,[],0)
        return new_state
    end
    r = ImplicitDistribution(init_state)
    return r
end

POMDPs.discount(pomdp::PE_POMDP) = pomdp.discount_factor

function POMDPs.transition(::PE_POMDP,s,a)
    new_step =s.step+ 1  # Increment step counter
    new_acts = push!(s.acts,a)
    # println(a)
    #Make a new state
    new_state = State(s.phi,new_acts,new_step)
    return Deterministic(new_state) # The state never changes
end

function POMDPs.reward(pomdp::PE_POMDP,s,a)
    r = 0.0
    if s.step <= pomdp.guess_steps+1
        if a == "wait"
            r = -1.1  #Small negative for suggesting?
        else 
            r = -1.0
        end            
    else
        vec = final_points[parse(Int64,a)] # Calculate similarity with final_beta_values
        r = similarity(s.phi, vec)
    end
    return r
end

# Observation Definition

POMDPs.observations(::PE_POMDP) = make_observations()  #How to tie in history?

function POMDPs.observation(m::PE_POMDP,s,a,sp)
    # if typeof(s) != State
    #     s = State(s,[],1)
    # end
    if a == "wait"
        beta_values = m.observe_points
        # points already accepted should get zero probability mass
        # [p1, p2,... pn] = (p(p1), p(p2)) = normalized(sim_metric(p1, s.phi), sim_metric(p2, s.phi), ...)
        p_a = []
        acts = []
        # Look through all actions and see which one the user is most likely to add
        for (i, act) in enumerate(actions(m,s))  # list out indexes and specific actions
            if act != "wait"
                b = beta_values[parse(Int64,act)] # (x,y,z)
                out = sample_initial_state(m)
                sim_metric = observation_similarity_wait(out.phi,b)
                push!(p_a,sim_metric)
                push!(acts, act)
            end
        end
        p_a = length(p_a)>0 ? p_a/norm(p_a) : p_a  #Catch cases where A is empty
        #available_actions = actions(m)[1:end-1]
        return SparseCat(acts, p_a)
        # distribution over all operator add points
    else
        beta_values = m.suggest_points
        # agent is suggesting 'a'
        # operator likes s.phi
        # points already accepted should get denied
        # recompute ParticleCollection
        b = beta_values[parse(Int64,a)]#(x,y,z)
        sim_metric = similarity(s.phi, b)
        sim_metric = sim_metric*0.8 #Semi-arbitrary weighting
        random_guess = 1/length(m.suggest_points)
        # random_guess = 1-sim_metric
        acc = m.user.accuracy
        av = m.user.availability
        # sim_metric = sim_metric/norm(sim_metric)
        
        # Belief update:
        #p(accept) = p(accurate)p(accept|accurate) + p(not accurate)p(accept|not accurate)
        #p(deny) = p(accurate)p(deny|accurate) + p(not accurate)p(deny|not accurate)
        percentage = [av*(acc*sim_metric + (1-acc)*(random_guess)),av*(acc*(random_guess)+(1-acc)*sim_metric)]
        # println(p)
        return SparseCat(["accept", "deny"], percentage)
    end

end

# POMDPs.actions(::PE_POMDP) = make_actions()

function POMDPs.actions(m::PE_POMDP,b)
    # Creates a list of potential actions in an array of strings
    if typeof(b) == LeafNodeBelief{Tuple{NamedTuple{(:a, :o),Tuple{String,String}}},State}
        prev_a = b.sp.acts
        step = b.sp.step
    elseif typeof(b) == State
        prev_a = b.acts
        step = b.step
    elseif typeof(b) == ParticleCollection{State} # For particle distribution
        # print(typeof(b))
        prev_a = b.particles[1].acts
        step = b.particles[1].step
    # elseif typeof(b) == ImplicitDistribution{var"#init_state#21"{PE_POMDP},Tuple{}}
    #     prev_a = []
    #     step = 0
    else
    # elseif typeof(b) == ImplicitDistribution{var"#init_state#54"{PE_POMDP},Tuple{}}
        # println("Else case triggered")
        prev_a = []
        step = 0
    end
    if step <= m.guess_steps
        points_data = m.suggest_points
        prev_a_nums = 0 
        taken_acts = []
        #Count number of acts that are not "wait"
        for pa in 1:length(prev_a)
            if prev_a[pa] != "wait"
                #Check if any actions are taken multiple times. Occurs due to random rollout
                if ~any(i -> prev_a[pa] == i,taken_acts)
                    prev_a_nums += 1
                    push!(taken_acts,prev_a[pa])
                else
                    println("Double  action found")
                end
            end
        end
        total_act = length(points_data)-length(taken_acts)
        acts = Array{String}(undef, total_act+1)

        # Add actions to list if they have not been taken
        min_count = 0
        # println(prev_a)
        for a in 1:length(points_data) # 1->len(suggest_points)
            if ~any(i -> string(a) == i,prev_a) #Iterate through prev_a and check if a is in old steps
                acts[a-min_count] = string(a)
            else # action a has been taken
                #min_count += sum(i-> string(a) == i,prev_a) #Counter to keep index properly matched.
                min_count += 1
                #Note: Multiple actions can be taken during tree search due to random rollout not updating the belief
            end
        end
        acts[end] = "wait"
    else  # Currently guess best point on the list. Need to have it guess on final image.
        
        total_act = length(m.final_points)
        acts = Array{String}(undef, total_act)
       
        # Add actions to list if they have not been taken
        # TODO recompute actions for new map
        for a in 1:total_act
            acts[a] = string(a)
        end
    end
    # print(acts)
    return acts
end

function POMDPs.isterminal(m::PE_POMDP,s) 
    if s.step == m.guess_steps+1
        return true
    else
        return false
    end
end

function find_similar_points(points,phi,n,bad_points)
    #Function takes in a phi value and finds the top n similar points
    #Find if any points are taken multiple times
    double_acts = 0
    points_copy = deepcopy(bad_points)
    for a in 1:length(bad_points)
        popfirst!(points_copy)
        if any(i -> bad_points[a] == i,points_copy)
            double_acts += 1
        end
    end
    #Find top similar points
    similar_points = Array{Vector{Any}}(undef,length(points)-length(bad_points)+double_acts)
    min_count = 0
    for p in 1:length(points)
        #Remove any previously suggested points
        if ~any(i -> string(p) == i,bad_points)
            similar_points[p-min_count] = [similarity(phi,points[p]),p]
        else
            min_count += 1
        end
    end

    #Sort likeliest points
    likely_points = sort!(similar_points,rev = true)
    #Extract best points for input into solver
    #Idx should be in string form
    best_point_idx = [Int(likely_points[ind][2]) for ind in 1:n]
    best_point_phi = [points[i] for i in best_point_idx]
    return string.(best_point_idx), best_point_phi
end

function observation_similarity_wait(state,point)
    #This function is used to obtain a probability for an observation received at a particular point
    #Used in the POMDP observation definition and particle filter update
    sim_metric = similarity(state, point)
    if sim_metric < 0.5
        sim_metric = 0.0001
    else
        sim_metric = sim_metric*sim_metric
    end
    return sim_metric
end

function observation_suggest(m::PE_POMDP,s,obs,act)
    #Function used to obtain a probability for an accept or deny observation
    ar = observation(m,State(s,[],1),act,State(s,[],1))
    #Parse the response array to output specific probability
    if obs == "accept"
        val = ar.probs[1]
    else 
        val = ar.probs[2]
    end
    return val
end

function final_guess(final_points_data,belief,num_points)
    #Propagate belief onto new image by sampling particle filter
    #Outputs a vector of string values
    #Extract features
    final_beta_values = [final_points_data[i][4:end] for i in 1:length(final_points_data)]
    #Sample particles
    avg_belief = mean(belief.states)
    chosen = []
    #Find most similar point to belief
    for sample in 1:num_points
        idx,phi = find_similar_points(final_beta_values,avg_belief,1,chosen)
        push!(chosen,idx[1]) 
    end
    return chosen 
end

function find_next_action(u_points, best_points_phi, o_points, f_points, user_mode,model_step)
    #Wrapper function to find the next action.
    solver = POMCPSolver(tree_queries=100, c=100.0, rng=MersenneTwister(1), tree_in_info=true, estimate_value = FORollout(RandomSolver()))
    PE_fun =  PE_POMDP(u_points,best_points_phi,o_points,f_points,user_mode,0.99,model_step+1)  # Define POMDP
    planner = solve(solver, PE_fun)
    a, info = action_info(planner, initialstate(PE_fun), tree_in_info=false)
    return a
end