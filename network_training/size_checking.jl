using Flux

CNN_dims =  [(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)] # Size of Convolutional layers
CNN_depth = [6,6,12,12,24,24,32,32]

# Encoder Example
r1 = rand(Float32,64,64,3,10)
layer = Conv(CNN_dims[1], 3 => CNN_depth[1], relu; stride = 1,pad=SamePad())
layer2 = Conv(CNN_dims[2], CNN_depth[1] => CNN_depth[2], relu; stride = 1,pad=SamePad())
layer_batch = Chain(Conv(CNN_dims[1], 3 => CNN_depth[1], relu; stride = 1,pad=SamePad()),BatchNorm(CNN_depth[1]))
m = Chain(
    Conv(CNN_dims[1], 3 => CNN_depth[1], relu; stride = 1,pad=SamePad()), # conv_1
    Conv(CNN_dims[2], CNN_depth[1] => CNN_depth[2], relu; stride = 1,pad=SamePad()),# conv_2
    BatchNorm(CNN_depth[2], relu),
    MaxPool((2,2)),
    Conv(CNN_dims[3], CNN_depth[2] => CNN_depth[3], relu; stride = 1,pad=SamePad()),# conv_3
    Conv(CNN_dims[4], CNN_depth[3] => CNN_depth[4], relu; stride = 1,pad=SamePad()),# conv_4
    BatchNorm(CNN_depth[4], relu),
    MaxPool((2,2)),
    Conv(CNN_dims[5], CNN_depth[4] => CNN_depth[5], relu; stride = 1, pad = SamePad()), #conv_5
    Conv(CNN_dims[6], CNN_depth[5] => CNN_depth[6], relu; stride = 1, pad = SamePad()),# conv_6
    BatchNorm(CNN_depth[6], relu),
    MaxPool((2,2)),
    Conv(CNN_dims[7], CNN_depth[6] => CNN_depth[7], relu; stride = 1, pad = SamePad()), #conv_7
    Conv(CNN_dims[8], CNN_depth[7] => CNN_depth[8], relu; stride = 1, pad = SamePad()),# conv_8
    BatchNorm(CNN_depth[8], relu),
    MaxPool((2,2))
    )

m = Chain(
    Conv(CNN_dims[1], 3 => CNN_depth[1], relu; stride = 1), # conv_1
    Conv(CNN_dims[2], CNN_depth[1] => CNN_depth[2], relu; stride = 1),# conv_2
    BatchNorm(CNN_depth[2], relu),
    MaxPool((2,2)),
    Conv(CNN_dims[3], CNN_depth[2] => CNN_depth[3], relu; stride = 1),# conv_3
    Conv(CNN_dims[4], CNN_depth[3] => CNN_depth[4], relu; stride = 1),# conv_4
    BatchNorm(CNN_depth[4], relu),
    Conv(CNN_dims[5], CNN_depth[4] => CNN_depth[5], relu; stride = 1), #conv_5
    Conv(CNN_dims[6], CNN_depth[5] => CNN_depth[6], relu; stride = 1),# conv_6
    BatchNorm(CNN_depth[6], relu),
    Conv(CNN_dims[7], CNN_depth[6] => CNN_depth[7], relu; stride = 1), #conv_7
    Conv(CNN_dims[8], CNN_depth[7] => CNN_depth[8], relu; stride = 1),# conv_8
    BatchNorm(CNN_depth[8], relu),
    MaxPool((2,2))
    )

d = Chain(
    Conv(CNN_dims[2], CNN_depth[2] => CNN_depth[1],relu; stride = 1, pad=SamePad()),
    Conv(CNN_dims[1],CNN_depth[1] => 3, relu; stride = 1,pad=SamePad()))

  
mod = r1 |> m
println("Pre-Flattened Size)", size(mod))
mod_flat = Flux.flatten(mod)
println("Flattened Size", size(mod_flat))

# encode = r1 |> m
# decode = encode |> d
# println("Size-encoded", size(encode))
# println("Size-decoded",size(decode))

# Decoder Example
d = Chain(Dense(1024,1024),
    x->reshape(x,8,8,16,size(x)[end]),
    Conv(CNN_dims[5],CNN_depth[5] =>CNN_depth[4], relu; stride = 1, pad = SamePad()),
    MaxPool((2,2))
    )

function apply_chain(x,c)
    x |> gpu
    c |> gpu
    x |> c
end
# r2 = Float32.(rand(1024,10))
# d |> gpu
# r2 |> gpu
# mod_d = r2 |> d

# println("Decoded Size", size(apply_chain(r2,d)))