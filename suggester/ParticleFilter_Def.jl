#Particle Filter Definition
using CategoricalArrays

struct InjectionParticleFilter 
    states # vector of state samples
    P_inject # number of samples to inject
    D_inject # injection distribution
end

function update_PF(b::InjectionParticleFilter,m::PE_POMDP,a,o)
    #Updating function for injection particle filter. Returns a new particle filter structure
    P_inject,D_inject = b.P_inject,b.D_inject
    states = b.states #Assumes constant state/phi distribution. Could include dynamics 
    if a == "wait"
        weights = [observation_similarity_wait(s_p,o[1]) for s_p in states] #Find relative weighting of observation
    else
        weights = [observation_suggest(m,s_p,o,a) for s_p in states] #Find relative probability for given state
    end
    P = length(states)
    #Create array to feed into Categorical Framework
    acts = [string(i) for i in 1:P]
    D = SetCategorical(acts,weights) #Create associated set of elements
    sampled_states = rand(D,P-P_inject) #Sample from Categorical distribution
    actual_states = [states[parse(Int64,a)] for a in sampled_states] #Extract vector values from sampled states
    new_states = rand_user_dist(D_inject,P_inject)
    # new_states = [rand(D_inject) for a in 1:P_inject] #Sample new states to inject
    states = vcat(actual_states,new_states) # Concatenate 
    # states = vcat(rand(D,P-P_inject),rand(D_inject,P_inject))  #Sample new set of particles
    return InjectionParticleFilter(states,P_inject,D_inject)
end

function init_PF(seg_truth,nn_truth,particles)
    # initial_belief = Dirichlet(phi) #Initialize belief. TODO: How to take variance into account?
    # initial_p_set = [rand(initial_belief) for a in 1:particles]
    initial_belief = user_dist(seg_truth,nn_truth)
    initial_p_set = rand_user_dist(initial_belief,particles)
    p_belief = InjectionParticleFilter(initial_p_set,Int(round(particles*0.05)),initial_belief)
    return p_belief
end

#Set Categorical Array [Taken from Algorithms For Decision Making Textbook]
struct SetCategorical{S}
    elements::Vector{S} # Set elements (could be repeated)
    distr::Categorical # Categorical distribution over set elements
    function SetCategorical(elements::Vector{Vector{S}}) where S
        weights = ones(length(elements))
        return new{S}(elements, Categorical(normalize(weights, 1)))
    end
    function SetCategorical(elements::Vector{S}, weights::AbstractVector{Float64}) where S
        ℓ₁ = norm(weights,1)
        if ℓ₁ < 1e-6 || isinf(ℓ₁)
            return SetCategorical(elements)
        end
        distr = Categorical(normalize(weights, 1))
        return new{S}(elements, distr)
    end
end

Distributions.rand(D::SetCategorical) = D.elements[rand(D.distr)]
Distributions.rand(D::SetCategorical, n::Int) = D.elements[rand(D.distr, n)]