using Distributions
novice_accuracy = 0.7
expert_accuracy = 0.98

novice_certainty = 7
expert_certainty = 40
novice_availability = 1.0
expert_availability = 1.0

test_accuracy = 0.5
test_certainty = 1
test_availability = 1.0
struct User_Model
    accuracy::Float64
    certainty::Float64
    availability::Float64
end

user_novice = User_Model(novice_accuracy, novice_certainty,novice_availability)
user_expert = User_Model(expert_accuracy, expert_certainty,expert_availability)
user_test = User_Model(test_accuracy, test_certainty,test_availability)

function sample_new_point(points,ideal_seg,ideal_nn,user,bad_points)
    #Determine point accuracy
    if rand(Float64,1)[1]<=user.accuracy #Accurate guess
        #Create distribution
        if typeof(ideal_nn) == Bool # If nn is not being used
            choice_dist = Dirichlet(ideal_seg*user.certainty)
            #Sample from distribution
            choice_phi = rand(choice_dist,1)
        elseif typeof(ideal_seg) == Bool # If segmentation is not being used
            #Create distribution     
            choice_dist = user_dist(ideal_seg,ideal_nn)
            #Sample from distribution
            choice_phi = rand_user_dist(choice_dist,1)[1]
        else # If nn features are being used
             #Create distribution     
            choice_dist = user_dist(ideal_seg*user.certainty,ideal_nn)
            #Sample from distribution
            choice_phi = rand_user_dist(choice_dist,1)[1]
        end
        #Find closest point
        new_point_idx,new_point_phi = find_similar_points(points,choice_phi,1,bad_points)
    else #Not accurate. Random sample
        new_point_idx = rand(range(1,length(points),step=1))
        #Make sure that randomly picked point is not the same as previously chosen
        if any(i -> string(new_point_idx) == i,bad_points)
            new_pick = true
            while new_pick # Iterate until new picked point is unique
                new_point_idx = rand(range(1,length(points),step=1))
                if ~any(i -> new_point_idx == i,bad_points)
                    new_pick = false
                end
            end
        end
        new_point_phi = [points[new_point_idx]]
        new_point_idx = [string.(new_point_idx)]
    end
    return new_point_idx,new_point_phi
end

function sample_user_response(point,ideal_seg,ideal_nn,user)
    #This function samples a user's response to a suggested point
    if typeof(ideal_nn) == Bool # If nn is not being used   
        choice_dist = Dirichlet(ideal_seg*user.certainty)
        #Sample from distribution
        choice_phi = rand(choice_dist,1)
    elseif typeof(ideal_seg) == Bool # If segmentation is not being used
        # Create distribution
        choice_dist = user_dist(ideal_seg,ideal_nn)
        # Sample from distribution
        choice_phi = rand_user_dist(choice_dist,1)[1]
    else # If nn features are being used
            #Create distribution     
        choice_dist = user_dist(ideal_seg*user.certainty,ideal_nn)
        #Sample from distribution
        choice_phi = rand_user_dist(choice_dist,1)[1]
    end
    #Compare point similarity
    sim = similarity(point,choice_phi)

    #Generate Probabilities
    acc = user.accuracy
     # Temporary stand-in for points that are estimated to be accepted
        #p(accept) = p(accurate)p(accept|accurate) + p(not accurate)p(accept|not accurate)
        #p(deny) = p(accurate)p(deny|accurate) + p(not accurate)p(deny|not accurate)
    percentage = [acc*sim + (1-acc)*(1-sim),acc*(1-sim)+(1-acc)*sim]

    #Setup distribution
    answer_dist = SparseCat(["accept","deny"],percentage)
    return rand(answer_dist)
end

function user_dist(seg_ideal::Vector{Float64},nn_ideal)
    """This function takes in the ideal user vectors, creates a distribution and outputs a vector of distributions.
    Used at Particle filter initialization and for user response generation"""
        # println(seg_ideal)
    seg_dist = Dirichlet(seg_ideal)
    # Create initial array
    distributions = Array{Any}(undef,length(nn_ideal)+1)
    distributions[1] = seg_dist
    #Generate required set of samples
    for f in 1:length(nn_ideal)
        distributions[f+1] = TruncatedNormal(nn_ideal[f][1],nn_ideal[f][2],0,Inf)
    end
    return distributions
end

function user_dist(seg_ideal::Vector{Float64},nn_ideal::Bool)
    """Companion method if only segmentation is used, creates a distribution and outputs a single Dirichlet distribution"""
    seg_dist = Dirichlet(seg_ideal)
    return seg_dist
end


function user_dist(seg_ideal::Bool,nn_ideal)
    """Companion method if only neural network is used, creates a distribution and outputs a series of normal distributions"""
    # Create initial array
    distributions = Array{Any}(undef,length(nn_ideal))
    #Generate required set of samples
    for f in 1:length(nn_ideal)
        distributions[f] = TruncatedNormal(nn_ideal[f][1],nn_ideal[f][2],0,Inf)
    end
    return distributions
end

function rand_user_dist(u_dist,n)
    """Function takes in a vector of distributions and generates n respective samples"""
    samples = zeros(n)

    for i in 1:length(u_dist)
        d = u_dist[i] # Extract distribution
        # Handle Dirichlet distrubtion
        if typeof(d) == Dirichlet{Float64}
            samples = [rand(d) for s in 1:n]
        # Handle other normal distrubtions
        else
            # Check if Dirichlet/segmentation not being used. Requires different method of combining
            if samples[1] == 0.0 # If the first element in the array is being added
                samples = [rand(d) for s in 1:n] 
            else 
                val = [rand(d) for s in 1:n]
                # Combine all features into single vector array
                samples = [vcat(samples[i],val[i]) for i in 1:n]
            end
        end
    end
    return samples
end

function rand_user_dist(u_dist::Dirichlet{Float64},n)
    """Function takes in a Dirichlet distributions and generates n respective samples.
        Companion method for when nn_architecture is not used"""
    samples = [rand(u_dist) for s in 1:n]
    return samples
end

#Testing
# dist = user_dist([0.5,0.5,0.001],[[1.0,2.0],[2.0,3.2],[3.2,4.2]],user_novice)
# vec = rand_user_dist(dist,5)
