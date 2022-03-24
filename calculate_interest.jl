using Images, FileIO
using MeshViz, Meshes
import GLMakie: surface
using Plots


# Include necessary files
include("Interaction.jl")
include("gaussian_fit.jl")

function process_pixels(img_mat,rows,cols)
    """This function sythesizes the pixels within the range specified by rows and cols.
    Specific implementation is used for synthesizing the red, blue and black pixels drawing from 
        Virginia Tech's image segmentation.
    Inputs:
        img_mat = channelview(image)
        rows = [row_index_1, row_index_2]
        cols = [column_index_1, column_index2]
    Output:
        A vector synthesizing the values of the processed image location
    """
    sample_size = rows[2]-rows[1]
    # Sum the red channel
    s_r = sum(img_mat[1,rows[1]:rows[2],cols[1]:cols[2]])
    # Sum the blue channel
    s_b = sum(img_mat[3,rows[1]:rows[2],cols[1]:cols[2]])
    # Calculate black channel
    s_none = sample_size^2-s_r-s_b
    # Combine into vector and normalize
    sample_vec = [s_r,s_b,s_none]/sample_size^2
    return sample_vec
end

# Specify Image pathimg_path = "./images/airstrip_hand_segmented.png"
function discretize_image(img_path,d_size::Int64)
    """Function takes in an image path, and discretization size to compress an image. 
    Input:
        img_path = "./images/airstrip_hand_segmented.png"
        s_dize = 10
    Output:
        new_image_array = Array{RGB{Float64}}
        discrete_data = Array{Vector{Float64}}
    """
    img = load(img_path)
    img_ch = channelview(img)
    # Define descritization
    # d_size = 10
    rows = size(img)[1]
    cols = size(img)[2]
    row_bins = Int64(round(rows/d_size))
    col_bins = Int64(round(cols/d_size))
    new_size = round.(size(img)./d_size)

    row_block = Int.(round.(range(1,rows,row_bins)))
    col_block = Int.(round.(range(1,cols,col_bins)))
    # Create new arrays
    new_image_array = Array{RGB{Float64}}(undef,row_bins-1,col_bins-1)
    discrete_data = Array{Vector{Float64}}(undef,row_bins-1,col_bins-1)
    # Loop through all rows
    for (idx_r,r) in enumerate(row_block[1:end-1])
        # Find next index
        # Check that we don't overshoot
        if idx_r+1>length(row_block)
            r_n = row_block[end]
        else
            r_n = row_block[idx_r+1]
        end
        # Loop through columns
        for (idx_c,c) in enumerate(col_block[1:end-1])
            # Find next index and check that we don't overshoot
            if idx_c+1>length(row_block) 
                c_n = col_block[end]
            else
                c_n = col_block[idx_c+1]
            end
            # Process Pixels and put in arrays
            new_array = process_pixels(img_ch,[r,r_n],[c,c_n])
            new_image_array[idx_r,idx_c] = RGB(new_array[1],0,new_array[2])
            discrete_data[row_bins-idx_r,idx_c] = new_array
        end
    end
    return new_image_array, discrete_data
end

new_img, data_array = discretize_image("./images/airstrip_hand_segmented.png",60)

# Fit GMM
model = fit_gmm(belief,5)
model = MixtureModel(model)
# Make new array
interest_array = zeros(size(data_array))
# Evaluate PDF at each point
evaluated = pdf(model,data_array)
# evaluated= normalize(evaluated)
grid = CartesianGrid(size(data_array))
# viz(grid, color = log.(evaluated[:]))


x = range(1,size(evaluated)[1])
y = range(1,size(evaluated)[2])
a = heatmap(y,x,log.(evaluated),label = "Log-likelihood")
title!("User Belief")
# savefig(a,"./data/interest_data/road_edges_60.png")
# a = heatmap(y,x,log.(evaluated))
# s = surface(x,y,evaluated)
# s_l = surface(x,y,log.(evaluated))
# display(s_l)
