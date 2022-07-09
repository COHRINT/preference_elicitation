#This file reads in the user supplied data and creates a data set that is used by the POMDP
using CSV

function read_data(file)
    # file = "random_data.csv"
    # random_data= Array{Float64,1}
    random_data = []
    ffile = CSV.File(file)
    columns = length(ffile.names)
    #Read through the different rows
    for row in ffile
        #Convert to Float64
        x = row[1]
        y = row[2]
        r = row[3]
        features = Array{Float64}([x,y,r])
        for feature in 4:columns
            push!(features, row[feature])
        end
        push!(random_data,features) #Add to list
    end
    return random_data
end
# read_data("random_data.csv")
# println(random_data)

# data = read_data("/home/hunter/Documents/Research/Preference_Elicitation/dmu_final_project/data/combined_features/final_features/all_final_41_neighborhood_image_segmented_r_combined.csv")
function find_mean(mean_seg,mean_nn,data,index_start,index_stop)
    local_data = data[index_start:index_stop]
    mean_f = Array{Float64}(undef,length(data[1])-3)
    var_f = Array{Float64}(undef,length(data[1])-3)
    data_vec = Array{Array{Float64}}(undef,length(data[1])-3)
    # Find means for each specific vector
    for e in 4:length(data[1])
        mean_f[e-3] = mean([local_data[a][e] for a in 1:length(local_data)])
        var_f[e-3] = mean([local_data[a][e] for a in 1:length(local_data)])
        data_vec[e-3] = [mean_f[e-3],var_f[e-3]]
    end
    # Find mean with old vector
    seg_vec = mean([mean_seg,mean_f[1:3]])
    nn_vec = mean([mean_nn,data_vec[4:end]])
    return seg_vec,nn_vec
end

