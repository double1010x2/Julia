module MLModel
using Flux, Metalhead, Statistics

export vgg16, Conv2d_bn, Conv2d_bn_res, inception_resnet_block 

function Conv2d_bn(channel_in, channel_out, kernel, pad=(0,0), stride=(1,1), act=identity)
    #pad = kernel == (3,3) ? (1,1) : (0,0)

    return Chain(
             Conv(kernel, channel_in=>channel_out, act, pad=pad, stride=stride),
             BatchNorm(channel_out)
             )
end

function Conv2d_bn_res(channel_in, channel_out, downsample::Bool = true)
    stride = downsample ? (2,2) : (1,1)
    shortcut   = Chain(Conv((1,1), channel_in=>channel_out, pad = (0,0), stride = stride, BatchNorm(channel_out)))
    return Metalhead.ResidualBlock(
                    [channel_in, channel_in, channel_in, channel_out], 
                    [1,3,1], 
                    [0,1,0], 
                    [1,1,stride[1]], 
                    shortcut) 
end


function inception_resnet_block(block, channel_in, channel_out)
    if block=="A"
        net0    = Conv2d_bn(channel_in, channel_out, (1,1), (0,0), (1,1), identity)           
        net1    = Chain(
                    Conv2d_bn(channel_in, channel_out, (1,1), (0,0), (1,1), identity),
                    Conv2d_bn(channel_out, channel_out, (3,3), (1,1), (1,1), identity)
                    )
        net2    = Chain(
                    Conv2d_bn(channel_in, channel_out, (1,1), (0,0), (1,1), identity),
                    Conv2d_bn(channel_out, channel_out, (3,3), (1,1), (1,1), relu),
                    Conv2d_bn(channel_out, channel_out, (3,3), (1,1), (1,1), relu)
                    )
        net     = Chain(cat(net0, net1, net2; dims=3),
                        Conv2d_bn(channel_out*3, channel_out, (1,1), (0,0),(1,1),relu)
                       )
        
    elseif block == "B"
        net0    = Conv2d_bn(channel_in, channel_out, (1,1), (0,0), (1,1),identity)           
        net1    = Chain(
                    Conv2d_bn(channel_in, channel_out, (1,1), (0,0), (1,1), identity),
                    Conv2d_bn(channel_out, channel_out, (1,7), (0, 3), (1,1),relu),
                    Conv2d_bn(channel_out, channel_out, (7,1), (3, 0), (1,1), relu)
                    )
        net     = Chain(cat(net0, net1; dims=3),
                        Conv2d_bn(channel_out*2, channel_out, (1,1), (0,0),(1,1),relu)
                       )
    elseif block == "C"
        net0    = Conv2d_bn(channel_in, channel_out, (1,1), (0,0),(1,1), identity)           
        net1    = Chain(
                    Conv2d_bn(channel_in, channel_out, (1,1), (0,0), (1,1),identity),
                    Conv2d_bn(channel_out, channel_out, (1,3), (0, 1), (1,1),relu),
                    Conv2d_bn(channel_out, channel_out, (3,1), (1, 0), (1,1),relu)
                    )
        net     = Chain(cat(net0, net1; dims=3),
                        Conv2d_bn(channel_out*2, channel_out, (1,1), (0,0),(1,1),relu)
                       )
    end
    return net 
end

function vgg16()
    return Chain(
            Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            MaxPool((2,2)),
            Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            MaxPool((2,2)),
            Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            MaxPool((2,2)),
            Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            MaxPool((2,2)),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            MaxPool((2,2)),
            flatten,
            Dense(512, 4096, relu),
            Dropout(0.5),
            Dense(4096, 4096, relu),
            Dropout(0.5),
            Dense(4096, 10)) |> gpu
end # end of vgg16
end # end of module 
