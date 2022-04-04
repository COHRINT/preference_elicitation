using Images, ImageIO
using MosaicViews

function get_image_samples(img_path,input_dim,n,batch_size)
    """Function will open an image and create n samples of size sample_pix"""
    
    # Load image
    img_main = load(img_path)
    row, col = size(img_main)

    #Create distribution to sample from
    sample_pix = Int(sqrt(input_dim))
    row_dist = range(1,row-sample_pix)
    col_dist = range(1,col-sample_pix)

    #Pre-allocate matrix
    # img_set = rand(RGB,sample_pix,sample_pix,n)
    img_set = Array{Float32}(undef,3,sample_pix,sample_pix,n)
    for s in 1:n
        # Choose a random choice
        s_x = rand(row_dist)
        s_y = rand(col_dist)
        img_set[:,:,:,s] = channelview(img_main[s_x:s_x+sample_pix-1,s_y:s_y+sample_pix-1])
    end  
    DataLoader(img_set,batchsize=batch_size, shuffle=true)
end

function convert_to_mosaic(x,y_size)
    # Convert to RGB matrix
    RGB_mat = colorview(RGB,x)
    mosaicview(RGB_mat,ncol=y_size)
end

# base_path = "./images/Boulder_flatirons_topoV1_cropped.jfif"
# sample_size = 64^2 # Size of sampled image
# number_samples = 100
# batch_size = 10
# loader = get_image_samples(base_path,sample_size,number_samples,batch_size)