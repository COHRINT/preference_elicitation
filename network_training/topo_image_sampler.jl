using Flux
using Images, ImageIO

function get_image_samples(img_path,sample_pix,n,batch_size)
    """Function will open an image and create n samples of size sample_pix"""
    
    img_main = load(img_path)
    img_main_ch = channelview(img_main)
    depth, row, col = size(img_main_ch)

    #Create distribution to sample from
    row_dist = range(0,row-sample_pix)
    col_dist = range(0,col-sample_pix)

    #Pre-allocate matrix
    img_set = rand(RGB,sample_pix,sample_pix,n)
    for s in 1:n
        # Choose a random choice
        s_x = rand(row_dist)
        s_y = rand(col_dist)
        img_set[:,:,s] = img_main[s_x:s_x+sample_pix-1,s_y:s_y+sample_pix-1]
    end  
    DataLoader(img_set,batchsize=batch_size, shuffle=true)
end

base_path = "./images/Boulder_flatirons_topoV1_cropped.jfif"
sample_size = 60 # Size of sampled image
number_samples = 100
batch_size = 10
loader = get_image_samples(base_path,sample_size,number_samples,batch_size)
mosaicview(loader.data,ncol=10)