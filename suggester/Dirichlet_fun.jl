using GLMakie: xlabel!, ylabel!, zlabel!,Axis, meshscatter, meshscatter!, scatter
using Distributions

test_d = Dirichlet([2,2.01,1])
s = rand(test_d,1000)
co = LinRange(0,3,1000)
# meshscatter(s[1,:],s[2,:],s[3,:],markersize = 0.03,color = co)

# Make a matrix
particle_data = zeros(length(belief.states),3)
for r in 1:length(belief.states)
    particle_data[r,:] = belief.states[r]
end
# println(particle_data)
mmse_point = mean(belief.states)
map_point = mode(belief.states)
mmae_point = [median(particle_data[:,1]),median(particle_data[:,2]),median(particle_data[:,3])]
data_points = [mmse_point map_point mmae_point]
# pdf_plot =Figure()
# Axis3(pdf_plot[1, 1, 1])
pdf_plot = meshscatter(particle_data[:,1],particle_data[:,2],particle_data[:,3], markersize = 0.003, color = co)
meshscatter!(data_points[1,:],data_points[2,:],data_points[3,:], markersize = 0.03, color = LinRange(0,4,3))
# meshscatter!(map_point[1],map_point[2],map_point[3], markersize = 0.01, color =:blue)
# meshscatter!(mmae_point[1],mmae_point[2],mmae_point[3], markersize = 0.01, color =:green)


# xlabel!(pdf_plot,"Feature 1")
# ylabel!(pdf_plot,"Feature 2")
# zlabel!(pdf_plot,"Feature 3")
title!("Particle filter distribution")
display(pdf_plot)
