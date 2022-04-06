# Variational Autoencoder(VAE)
#
# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling
# https://arxiv.org/abs/1312.6114

using BSON
using CUDA
using DrWatson: struct2dict
using Flux
using Flux: @functor, chunk,unsqueeze
using Flux.Losses: logitbinarycrossentropy
using Images
using Logging: with_logger
using MLDatasets
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random
using MosaicViews

include("topo_image_sampler.jl")

# # load MNIST images and return loader
# function get_data(batch_size)
#     xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
#     xtrain = reshape(xtrain, 28^2, :)
#     DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
# end

struct Encoder
    conv_1
    conv_2
    linear_1
    linear_2
    linear_3
    linear_4
    linear_5
    linear_6
    μ
    logσ
end
@functor Encoder
    
Encoder(layer_dims::Vector{Int64}, CNN_input,CNN_depth::Vector{Int64}) = Encoder(
    Conv(CNN_input[1], 3 => CNN_depth[1], relu; stride = 1, pad = SamePad()), # conv_1
    Conv(CNN_input[2], CNN_depth[1] => CNN_depth[2], relu; stride = 1, pad = SamePad()),# conv_2
    Dense(layer_dims[1],layer_dims[2]),     #  linear_1
    Dense(layer_dims[2],layer_dims[3]),     #  linear_2
    Dense(layer_dims[3],layer_dims[4]),     #  linear_3
    Dense(layer_dims[4],layer_dims[5]),     #  linear_4
    Dense(layer_dims[5],layer_dims[6]),     # linear 5
    Dense(layer_dims[6],layer_dims[7]),     # linear 6
    Dense(layer_dims[7],layer_dims[8]),        # μ
    Dense(layer_dims[7],layer_dims[8])        # logσ
)

function (encoder::Encoder)(x)
    h = encoder.linear_6(
        encoder.linear_5(
        encoder.linear_4(
        encoder.linear_3(
        encoder.linear_2(
        encoder.linear_1(
        flatten(
        encoder.conv_2(
        MaxPool((2,2))(
        encoder.conv_1(x))))))))))
    encoder.μ(h), encoder.logσ(h)
end

Decoder(layer_dims::Vector{Int64},CNN_input,CNN_depth::Vector{Int64}) = Chain(
    Dense(layer_dims[8], layer_dims[7], tanh),
    Dense(layer_dims[7], layer_dims[6]),
    Dense(layer_dims[6], layer_dims[5]),
    Dense(layer_dims[5], layer_dims[4]),
    Dense(layer_dims[4], layer_dims[3]),
    Dense(layer_dims[3], layer_dims[2]),
    Dense(layer_dims[2], layer_dims[1]),
    softmax,
    unsqueeze(2),
    x->reshape(x,32,32,CNN_depth[2],:),
    Conv(CNN_input[2], CNN_depth[2] => CNN_depth[1], relu; stride = 1, pad=SamePad()),
    Upsample((2,2)),
    Conv(CNN_input[1], CNN_depth[1] => 3, relu; stride = 1, pad=SamePad())
)

function reconstuct(encoder, decoder, x, device)
    μ, logσ = encoder(x)
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    # out = unsqueeze(decoder(z),2)
    μ, logσ, decoder(z)
end

function model_loss(encoder, decoder, λ, x, device)
    # println("Calculating Loss")
    μ, logσ, decoder_z = reconstuct(encoder, decoder, x, device)
    len = size(x)[end]
    # KL-divergence
    # kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len
    kl_q_p = 0.5 * sum(@. (exp(2 * logσ) + μ^2 -1 - 2 * logσ)) / len

    # println("Logitbinary",size(decoder_z),size(x))
    logp_x_z = -logitbinarycrossentropy(decoder_z, x, agg=sum) / len
    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(decoder))
    # println("regularization")
    # println("Loss",logp_x_z," ",kl_q_p," ",reg)
    -logp_x_z + kl_q_p + reg
end



# arguments for the `train` function 
@with_kw struct Args
    η::Float64          = 1e-4     # learning rate
    λ::Float32          = 0.001f0   # regularization paramater
    batch_size::Int64   = 1000      # batch size
    mosaic_size::Int64  = 10       # sampling size for output    
    epochs::Int64       = 20       # number of epochs
    samples::Int64      = 100000    # number of image samples
    seed::Int64         = 0        # random seed
    cuda::Bool          = true     # use GPU
    input_dim::Int64    = 64^2     # image size
    input_channels::Int64 = 3      # Number of channels on the input image
    layer_dims::Vector{Int64} = [18432,4096,2058,1024,512,256,128,64]
    CNN_dims::Vector{Tuple{Int64,Int64}} = [(3,3),(3,3)] # Size of Convolutional layers
    CNN_depth::Vector{Int64} = [9,18] # Depth of convolutional layers
    # latent_dim::Int64   = 24        # latent dimension
    # hidden_dim::Int64   = 500      # hidden dimension
    verbose_freq::Int64 = 10       # logging for every verbose_freq iterations
    tblogger::Bool      = false    # log training with tensorboard
    save_path           = "output" # results path
end

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end


    # load image samples
    base_path = "./images/Boulder_flatirons_topoV1_cropped.jfif"
    loader = get_image_samples(base_path,args.input_dim,args.samples,args.batch_size)
    
    # initialize encoder and decoder
    encoder = Encoder(args.layer_dims,args.CNN_dims,args.CNN_depth) |> device
    decoder = Decoder(args.layer_dims,args.CNN_dims,args.CNN_depth) |> device

    # ADAM optimizer
    opt = ADAM(args.η)
    
    # parameters
    ps = Flux.params(encoder.conv_1,
                    encoder.conv_2,
                    encoder.linear_1,
                    encoder.linear_2,
                    encoder.linear_3,
                    encoder.linear_4,
                    encoder.linear_5,
                    encoder.linear_6,
                    encoder.μ, encoder.logσ, decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # fixed input
    # original, _ = first(get_data(args.mosaic_size^2))
    # original = original |> device
    # image = convert_to_image(original, args.mosaic_size)
    # image_path = joinpath(args.save_path, "original.png")
    # save(image_path, image)
    original_image = loader.data[:,:,:,1:args.mosaic_size^2] |> device
    # original_data = original_image |> device
    # original = original |> device
    image = convert_to_mosaic(original_image,args.mosaic_size,args.input_dim)
    image_path = joinpath(args.save_path, "original.png")
    save(image_path, image)

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for x in loader 
            # x_flat = flatten(x)
            # println(typeof(x),size(x))
            loss, back = Flux.pullback(ps) do
                model_loss(encoder, decoder, args.λ, x |> device, device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)]) 

            # logging with TensorBoard
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss=loss
                end
            end

            train_steps += 1
        end
        # save image
        _, _, rec_original = reconstuct(encoder, decoder, original_image, device)
        # rec_original = sigmoid.(rec_original)
        # Convert back to image size
        # rec_original_image = reshape(unsqueeze(rec_original,2),size(original_image))
        image = convert_to_mosaic(rec_original,args.mosaic_size,args.input_dim)
        image_path = joinpath(args.save_path, "epoch_$(epoch).png")
        save(image_path, image)
        @info "Image saved: $(image_path)"
    end

    # save model
    model_path = joinpath(args.save_path, "model.bson") 
    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        BSON.@save model_path encoder decoder args
        @info "Model saved: $(model_path)"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__ 
    train()
end
