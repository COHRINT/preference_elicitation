using Pkg
Pkg.add(["LaTeXStrings","ImageIO","CategoricalArrays","CUDA","Flux","BSON","DrWatson","Images","Logging","Parameters","ProgressMeter","TensorBoardLogger","Random","MosaicViews","BasicPOMCP","POMDPs","ParticleFilters","Distributions","LinearAlgebra","Distances","Statistics","POMDPPolicies","POMDPModelTools","MAT","CSV"])
using CUDA
CUDA.versioninfo()