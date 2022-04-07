

CNN_input = CNN_input = [(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)]
CNN_depth = [64,128,256,512,512,512,512,512]

r1 = rand(Float32,64,64,3,10)
m = Chain(Conv(CNN_input[1], 3 => CNN_depth[1], relu; stride = 1, pad = SamePad()), # conv_1
MaxPool((2,2)),
Conv(CNN_input[2], CNN_depth[1] => CNN_depth[2], relu; stride = 1, pad = SamePad()),# conv_2
MaxPool((2,2)),
Conv(CNN_input[3], CNN_depth[2] => CNN_depth[3], relu; stride = 1, pad = SamePad()),# conv_2
Conv(CNN_input[4], CNN_depth[3] => CNN_depth[4], relu; stride = 1, pad = SamePad()),# conv_2
MaxPool((2,2)),
Conv(CNN_input[5], CNN_depth[4] => CNN_depth[5], relu; stride = 1, pad = SamePad()),# conv_2
Conv(CNN_input[6], CNN_depth[5] => CNN_depth[6], relu; stride = 1, pad = SamePad()),# conv_2
MaxPool((2,2)),
Conv(CNN_input[7], CNN_depth[6] => CNN_depth[7], relu; stride = 1, pad = SamePad()),# conv_2
Conv(CNN_input[8], CNN_depth[7] => CNN_depth[8], relu; stride = 1, pad = SamePad()))

mod = r1 |> m 
mod_flat = flatten(mod)
println("Flattened Size", mod_flat)
