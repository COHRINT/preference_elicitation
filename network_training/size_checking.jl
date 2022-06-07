using Flux

CNN_input = CNN_input = [(3,3),(3,3),(3,3),(3,3),(3,3)]
CNN_depth = [4,4,8,8,16]

# Encoder Example
r1 = rand(Float32,64,64,3,10)
m = Chain(Conv(CNN_input[1], 3 => CNN_depth[1], relu; stride = 1, pad = SamePad()), # conv_1
Conv(CNN_input[2], CNN_depth[1] => CNN_depth[2], relu; stride = 1, pad = SamePad()),# conv_2
MaxPool((2,2)),
Conv(CNN_input[3], CNN_depth[2] => CNN_depth[3], relu; stride = 1, pad = SamePad()),# conv_2
Conv(CNN_input[4], CNN_depth[3] => CNN_depth[4], relu; stride = 1, pad = SamePad()),# conv_2
MaxPool((2,2)),
Conv(CNN_input[5], CNN_depth[4] => CNN_depth[5], relu; stride = 1, pad = SamePad()),# conv_2
MaxPool((2,2))
)

mod = r1 |> m 
println("Pre-Flattened Size)", size(mod))
mod_flat = flatten(mod)
println("Flattened Size", size(mod_flat))

# Decoder Example
d = Chain(Dense(1024,1024),
    x->reshape(x,8,8,16,size(x)[end]),
    Conv(CNN_input[5],CNN_depth[5] =>CNN_depth[4], relu; stride = 1, pad = SamePad()),
    MaxPool((2,2))
    )

function apply_chain(x,c)
    x |> gpu
    c |> gpu
    x |> c
end
r2 = Float32.(rand(1024,10))
# d |> gpu
# r2 |> gpu
# mod_d = r2 |> d

println("Decoded Size", size(apply_chain(r2,d)))