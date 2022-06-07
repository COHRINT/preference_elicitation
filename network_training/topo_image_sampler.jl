using Images, ImageIO
using MosaicViews
using Flux.Data: DataLoader

function get_image_samples(img_path,input_dim,n,batch_size,OG_size)
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
    img_set = Array{Float32}(undef,sample_pix,sample_pix,3,n)
    for s in 1:n
        # Choose a random choice
        s_x = rand(row_dist)
        s_y = rand(col_dist)
        sample = channelview(img_main[s_x:s_x+sample_pix-1,s_y:s_y+sample_pix-1])
        # Reshaping required to fit in WHCN order
        img_set[:,:,:,s] = reshape(sample,sample_pix,sample_pix,3)
    end  
    OG_image = deepcopy(img_set[:,:,:,1:OG_size^2])

    return DataLoader(img_set,batchsize=batch_size, shuffle=true), OG_image
end

function convert_to_mosaic(x,y_size,input_size)
    # Check image size
    if size(x)[1]!= 3
        image_size = Int(sqrt(input_size))
        x = reshape(x,3,image_size,image_size,:)
    end
    x_cpu = cpu(x)
    # Convert to RGB matrix
    RGB_mat = colorview(RGB,x_cpu)
    mosaicview(RGB_mat,ncol=y_size)
end

# base_path = "./images/Boulder_flatirons_topoV1_cropped.jfif"
# sample_size = 64^2 # Size of sampled image
# number_samples = 100
# batch_size = 10
# loader = get_image_samples(base_path,sample_size,number_samples,batch_size)