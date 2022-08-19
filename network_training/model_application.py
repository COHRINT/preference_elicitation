import copy
import os.path
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as transform
from torchvision.utils import save_image
import torch
from atlas_vae import VAE
from util import normalize_image
from matplotlib import image
from PIL import Image
from scipy.spatial import distance
import numpy as np

input_size = 64


def extract_image_data(pil_image, batch_size, model, gpu):
    """Function will load and process an image using the relevant model.
    Outputs reconstructed vectors, mean, and variance
    Inputs:
        PIL_image -> PIL image object
        batch_size -> Processing size
        model       -> atlas_vae object
        gpu         -> Boolean to use gpu
    Outputs:
        recon_set
        mu_set
        var_set
        """
    # Load portion of map image
    data = transform.to_tensor(pil_image)
    img_size = np.shape(data)
    # print(img_size)
    total_samples = int(pil_image.width * pil_image.height / (input_size ** 2))
    # print("Total Samples", total_samples)
    recon_set = torch.randn(total_samples, 3, input_size, input_size)
    mu_set = np.zeros((total_samples, model.latent_variable_size))
    var_set = np.zeros((total_samples, model.latent_variable_size))

    x = 0
    # Create set of images
    for r in range(0, pil_image.width, input_size):
        for c in range(0, pil_image.height, input_size):
            # print(r, c)
            recon_set[x, :, :, :] = normalize_image(data[:, c:c + input_size, r:r + input_size])
            # print(np.shape(sample))
            x += 1
    # gr = torchvision.utils.make_grid(recon_set[0:93, :, :, :], nrow=1, padding=2)
    # torchvision.utils.save_image(gr, '../network_training/saved_models/image_sample.jpg')

    # Run through batches
    for b in range(0, total_samples, batch_size):
        batch = recon_set[b:min(b + batch_size, total_samples), :, :, :]
        # Run network on sampled image
        if gpu:
            recon_x, var, = model.apply_model(batch.to(device='cuda'))
        else:
            recon_x, var, = model.apply_model(batch)
        recon_set[b:min(b + batch_size, total_samples), :, :, :] = recon_x
        mu_set[b:min(b+batch_size, total_samples), :] = var.cpu().detach().numpy()

    return recon_set, mu_set, var_set


def reconstruct_image(img_path, saved_reference, gpu):
    """Function will take in an image and apply the saved reference model to subsets of the image. Returns a saved
    image that represents to VAE's reconstruction.
    Inputs:
        img_path        -> reference image location
        saved_reference -> model location
        gpu             -> Boolean for whether a gpu is used
    Outputs:
        saved image
    """
    save_loc = "saved_models/"  # Location for image save
    batch_size = 64
    # Load model parameters
    model = VAE(nc=3, ngf=32, ndf=32, latent_variable_size=150, batch_size=128, image_size=input_size, gpu=gpu)
    state_dict = torch.load(saved_reference)
    model.load_state_dict(state_dict)
    if gpu:
        model.cuda()
    model.eval()

    # Extract data
    # Load portion of map image
    PIL_image = Image.open(img_path)
    # Crop to input specification
    new_width = int(np.floor(PIL_image.width/input_size)*input_size)
    new_height = int(np.floor(PIL_image.height/input_size)*input_size)
    PIL_image = PIL_image.crop((0, 0, new_width, new_height))
    data = transform.to_tensor(PIL_image)
    new_image = copy.deepcopy(data)
    data_set, _, _ = extract_image_data(PIL_image, batch_size, model, gpu)

    # Build back image
    x = 0
    for c in range(0, PIL_image.width, input_size):
        for r in range(0, PIL_image.height, input_size):
            # print(r, c)
            new_image[:, r:r+input_size, c:c+input_size] = data_set[x, :, :, :]
            x += 1

    # Save image
    save_path = str(save_loc+os.path.basename(img_path)[0:-5]+os.path.basename(saved_reference)[0:-4]+'.jpeg')
    save_image(new_image, save_path)


def extract_feature_vec(user_points, img_path, reference_model, gpu):
    """This function extracts the features from an image using the saved reference model.
    Input:
        user_points     -> list of x,y points
        img_path        -> path for img to be analyzed
        reference_model -> saved pickle module, or model object
        gpu             -> boolean for saved model parameters
    Output:
        A numpy array
        """
    # Load model
    lvs = 150
    # Take variable input for when we want to pre-load the model
    if type(reference_model) == str:
        model = VAE(nc=3, ngf=32, ndf=32, latent_variable_size=lvs, batch_size=128, image_size=input_size, gpu=gpu)
        state_dict = torch.load(reference_model)
        model.load_state_dict(state_dict)
        if gpu:
            model.cuda()
        model.eval()
    else:
        model = reference_model

    # Load image
    PIL_image = Image.open(img_path)
    data = transform.to_tensor(PIL_image)
    img_size = np.shape(data)

    # Create matrices
    tot_points = len(user_points)
    data_set = torch.randn(tot_points, 3, input_size, input_size)
    # Loop through user_points
    for count, p in enumerate(user_points):
        # Find image bounds
        x = int(p[0] - input_size / 2)
        y = int(p[1] - input_size / 2)
        if x < 0:
            x = 0
        elif (x+input_size) > img_size[2]:
            x = img_size[2]-input_size
        if y < 0:
            y = 0
        elif (y+input_size) > img_size[1]:
            y = img_size[1]-input_size
        data_set[count, :, :, :] = normalize_image(data[:, y:y+input_size, x:x+input_size])
    print("Total Analyzed Points", tot_points)
    gr = torchvision.utils.make_grid(data_set, padding=2)
    torchvision.utils.save_image(gr, '../network_training/saved_models/image_sample.jpg')

    # Feed into model
    if gpu:
        z = model.get_latent_var(data_set.to(device='cuda'))
    else:
        z = model.get_latent_var(data_set)
    # print(mu)
    return z.cpu().detach().numpy()


def show_user_interest(user_points, img_path, reference_model, gpu):
    """This function will take in a set of user points and plot the relevant user interest on a graph.
    """
    # Function parameters
    lvs = 150
    batch_size = 64

    # Load model
    model = VAE(nc=3, ngf=32, ndf=32, latent_variable_size=lvs, batch_size=128, image_size=input_size, gpu=gpu)
    state_dict = torch.load(reference_model)
    model.load_state_dict(state_dict)
    if gpu:
        model.cuda()
    model.eval()

    # Extract user points
    user_points_vec = extract_feature_vec(user_points, img_path, reference_model, gpu)

    # Find most likely point
    best_vec = np.mean(user_points_vec, axis=0)

    # Extract relevant features from image
    PIL_image = Image.open(img_path)
    new_width = int(np.floor(PIL_image.width/input_size)*input_size)
    new_height = int(np.floor(PIL_image.height/input_size)*input_size)
    PIL_image = PIL_image.crop((0, 0, new_width, new_height))
    recon_set, mu_set, var_set = extract_image_data(PIL_image, batch_size, model, gpu)

    total_cols = int(new_width/input_size)
    total_rows = int(new_height/input_size)
    opportunity_vec = np.zeros((total_rows, total_cols))
    # Calculate likelihood
    for c in range(total_cols):
        for r in range(total_rows):
            # Extract appropriate vector
            target_vec = mu_set[c*total_rows+r]
            # Compute similarity
            opportunity_vec[r, c] = distance.cosine(target_vec, best_vec)

    # Flip max and min
    opportunity_vec = np.max(opportunity_vec)-opportunity_vec
    # opportunity_vec = fudge_factor(opportunity_vec)
    # Plotting
    x = np.arange(0, PIL_image.width+input_size, input_size)
    y = np.arange(0, PIL_image.height+input_size, input_size)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    im1 = plt.imshow(PIL_image, extent=extent)
    im2 = plt.imshow(opportunity_vec, cmap=plt.cm.plasma, alpha=.4, interpolation='spline36', extent=extent)  # spline36

    # im2 = plt.imshow(opportunity_vec, cmap=plt.cm.plasma, alpha=.4, extent=extent)  # spline36
    plt.colorbar(im2)
    u_x = [user_points[i][0] for i in range(len(user_points))]
    u_y = [new_height-user_points[i][1] for i in range(len(user_points))]
    im3 = plt.scatter(u_x, u_y, marker='x', c='purple')
    plt.show()


def fudge_factor(vec):
    # Res to junction
    set = [[15,8],[16,8],[17, 8], [17,9], [18, 9], [19,9], [19,10], [19, 11], [20, 11], [21,11], [21, 12], [20, 12],[21, 13], [22, 13], [22, 14],
    # South to junction
           [30,5], [29, 6], [28, 6],[27,6], [26,6],[26,7], [25,7],[24,7], [24,8],[24,9],[24,10],[24,11],[24,12],[23,12],[23,11],[22,12], [22,13],
    # Junction south boulder
            [22,15], [21, 15], [21, 16], [22,16], [22,17], [22, 18], [21, 18], [20, 18], [20, 19], [19, 19], [19, 20], [18, 21],
    # Final junction south boulder
            [19, 21], [19, 22], [20,22], [22, 22],[21, 21],[22, 22], [22,23], [23,23],[23,24],[23,24], [22,25],[21,25],[20,26],[21,22],[20,27],[21,27],[22,27],
    # south spit 1
           [28,12], [27, 12], [26, 12], [26,13], [25,13], [24,13],
    # south spit 2
           [23,17],[24, 17], [24, 16], [25, 17], [25, 18], [26, 18],[25, 16], [25, 15], [26, 15],
    # North Final Junction
           [18, 20], [17, 20], [17, 19], [16, 19], [16, 18],[16,17], [15, 17], [15, 16], [14, 16]]

    max_val = 0.65
    for grid_point in set:
        vec[grid_point[0], grid_point[1]] = max_val
    return vec


# Reference user points
u_res_stream = [[928, 1454], [1002, 1454], [1043, 1389], [1178, 1439], [1387, 1206], [565, 1093]]
u_res_trails = [[1282, 659], [1195, 711], [1319, 778], [1174, 893], [1329, 1072], [1274, 1513], [1756, 802]]
u_res_shore = [[115, 572], [308, 582], [447, 772], [561, 826], [406, 946]]
u_res_roads = [[139, 95], [452, 267], [680, 356], [950, 432], [1287, 280], [1661, 328]]
u_res_water = [[254, 695], [180, 1022]]

u_res_shore_8x = [[468, 1199], [141, 1148], [382, 1754]]

u_eldo_trails = [[1532, 402], [1443, 176], [1202, 215], [1001, 271], [914, 291], [829, 60], [443, 136]]
u_eldo_treeline = [[544, 422], [588, 516], [747, 563], [733, 440], [1102, 468], [1189, 582], [1307, 517]]

u_eldo_treeline_8x = [[1724, 1698], [1653, 1437], [1973, 931], [2286, 1148], [1970, 425]]
u_eldo_trails_8x = [[2816, 341], [2195, 500], [3113, 452], [2602, 132], [1641, 171]]
u_eldo_cliffs_8x = [[292, 365], [458, 745], [672, 1386], [858, 1671], [593, 939]]

if __name__ == '__main__':
    # reference_img = "../images/Eldorado_town.jpeg"
    # reference_img = "../images/Reservoir@8.jpeg"
    reference_img = "../images/Reservoir.jpeg"
    # reference_img = "../images/Eldorado_town@8.jpeg"
    model_save = "saved_models/_ExpLR0.8_0.0001_LVS150_nf32_b0.01.pth"
    # model_save = "saved_models/8x_res_ExpLR0.8_0.0001_LVS150_nf32_b0.1.pth"
    run_with_gpu = False
    # reconstruct_image(reference_img, model_save, False)

    # res = image.imread(reference_img)
    # plt.imshow(res)
    # plt.show()

    test_run = u_res_stream
    # u_points = extract_feature_vec(test_run, reference_img, model_save, False)
    # print(np.shape(u_points))
    show_user_interest(test_run, reference_img, model_save, run_with_gpu)

