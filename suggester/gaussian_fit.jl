# File creates functions that apply GaussianMixtures onto a particle filter object
using GaussianMixtures
using Distributions



function fit_gmm(PF::InjectionParticleFilter,components::Int64)
    """This function takes in a particle filter and returns a fitten Gaussian mixture model"""
    # Convert data to matrix
    states = PF.states
    data = mapreduce(permutedims,vcat,states)
    # Generate a GMM
    g = GMM(components, data; method=:kmeans, nInit=100, nIter=10, nFinal=10)

    return g

end

# front_door_ideal = [0.5 0.45 0.05]
# value = Distributions.pdf(model,front_door_ideal')
