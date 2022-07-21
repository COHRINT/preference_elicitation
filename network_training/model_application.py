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
    for r in range(0, pil_image.width - input_size-1, input_size):
        for c in range(0, pil_image.height - input_size-1, input_size):
            # print(r, c)
            recon_set[x, :, :, :] = normalize_image(data[:, c:c + input_size, r:r + input_size])
            # print(np.shape(sample))
            x += 1

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
    data = transform.to_tensor(PIL_image)
    new_image = copy.deepcopy(data)
    data_set, _, _ = extract_image_data(PIL_image, batch_size, model, gpu)

    # Build back image
    x = 0
    for c in range(0, PIL_image.width-input_size-1, input_size):
        for r in range(0, PIL_image.height-input_size-1, input_size):
            # print(r, c)
            new_image[:, c:c+input_size,  r:r+input_size] = data_set[x, :, :, :]
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
    feature_vec = np.zeros((tot_points, model.latent_variable_size))
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
        data_set[count, :, :, :] = data[:, y:y+input_size, x:x+input_size]

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
    model = VAE(nc=3, ngf=32, ndf=32, latent_variable_size=lvs, batch_size=128, image_size=input_size)
    state_dict = torch.load(reference_model)
    model.load_state_dict(state_dict)
    if gpu:
        model.cuda()
    model.eval()

    # Extract user points
    user_points_vec = extract_feature_vec(user_points, img_path, reference_model, gpu)

    # Find most likely point
    best_vec = np.mean(user_points_vec)

    # Extract relevant features from image
    PIL_image = Image.open(img_path)
    new_width = int(np.floor(PIL_image.width/input_size)*input_size)
    new_height = int(np.floor(PIL_image.height/input_size)*input_size)
    PIL_image.crop((0, 0, new_width, new_height))
    recon_set, mu_set, var_set = extract_image_data(PIL_image, batch_size, model, gpu)

    opportunity_vec = np.zeros(PIL_image.height, PIL_image.width)
    # Calculate likelihood
    for c in range(0, PIL_image.width-input_size, input_size):
        for r in range(0, PIL_image.height-input_size, input_size):
            # Extract appropriate vector
            target_vec = mu_set[c*r]
            # Compute similarity
            opportunity_vec[r, c] = distance.cosine(target_vec, best_vec)

    # Plotting
    x = np.arange(0, PIL_image.width, input_size)
    y = np.arange(0, PIL_image.height, input_size)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    im1 = plt.imshow(PIL_image, extent=extent)
    im2 = plt.imshow(opportunity_vec, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)


# Reference user points
u_res_stream = [[928, 1454], [1002, 1454], [1043, 1389], [1178, 1439], [1387, 1206], [565, 1093]]
u_res_trails = [[1282, 659], [1195, 711], [1319, 778], [1174, 893], [1329, 1072], [1274, 1513], [1756, 802]]
u_res_shore = [[115, 572], [308, 582], [447, 772], [561, 826], [406, 946]]
u_res_roads = [[139, 95], [452, 267], [680, 356], [950, 432], [1287, 280], [1661, 328]]

u_eldo_trails = [[1532, 402], [1443, 176], [1202, 215], [1001, 271], [914, 291], [829, 60], [443, 136]]
u_eldo_treeline = [[544, 422], [588, 516], [747, 563], [733, 440], [1102, 468], [1189, 582], [1307, 517]]

if __name__ == '__main__':
    reference_img = "../images/Eldorado_town.jpeg"
    model_save = "saved_models/_ExpLR0.8_0.0001_LVS150_nf32_b0.1.pth"
    run_with_gpu = True
    # reconstruct_image(reference_img, model_save, False)

    # res = image.imread(reference_img)
    # matplotlib.pyplot.imshow(res)
    # matplotlib.pyplot.show()

    test_run = u_eldo_treeline
    u_points = extract_feature_vec(test_run, reference_img, model_save, False)
    print(np.shape(u_points))
    # show_user_interest(test_run, reference_img, model_save, False)


