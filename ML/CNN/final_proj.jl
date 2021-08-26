import Images;          const im    = Images
import ProgressMeter;   const pm    = ProgressMeter
import Augmentor;       const aug   = Augmentor
import ImageTransformations; const imTran = ImageTransformations
using PyCall
using Augmentor
using Flux, Metalhead
using Flux.Data: DataLoader
using Flux: @epochs, onecold, onehotbatch, throttle, logitcrossentropy, outdims
using Statistics
using CuArrays
push!(LOAD_PATH, pwd())
using Revise
using MLModel
pk = pyimport("pickle")
np = pyimport("numpy")

using Debugger

#========== controller ===========#
switch_image_extract    = false
dir_training            = "/Users/vincentwu/Documents/GitHub/1st-JuliaMarathon/homework/final_project/data/training/"
dir_verify              = "/Users/vincentwu/Documents/GitHub/1st-JuliaMarathon/homework/final_project/data/verify/"
image_training          = "/Users/vincentwu/Documents/GitHub/1st-JuliaMarathon/homework/final_project/data/10-monkey-species/training/training/" 
image_verify            = "/Users/vincentwu/Documents/GitHub/1st-JuliaMarathon/homework/final_project/data/10-monkey-species/validation/validation/" 
dim_train               = 300
image_data_file         = string("image_dict_dim$dim_train.pk")
image_path              = Array([image_training, image_verify])

#========== training parameter ===========#
n_class     = 10
batchsize   = [8, 32, 64]
batchsize2  = 64
learn_rate  = 0.005
epochs      = [10, 10, 10]
epochs2     = 10
#========== funciton ===========#

function test_loss()
    l = 0f0
    for (x, y) in test
        l += loss(x, y)
    end
    l/length(test)
end

function mypickle(filename, obj)
    out = open(filename,"w")
    pk.dump(obj, out)
    close(out)
end

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pk.load(f)
    end
    return r
end

function loadImageInDir(filedir, image_dim)
    img_dir     = readdir(filedir)
    dir_dim     = length(img_dir)
    p           = pm.Progress(dir_dim)
    image_all   = zeros(Float32, (dir_dim, image_dim[1], image_dim[2], image_dim[end]))
    img_i       = 0
    Threads.@threads  for img_file in img_dir
        img_file = string(filedir, img_file)
        img_i  += 1 
        image   = im.load(img_file);
        image   = im.imresize(image, image_dim[2], image_dim[end])     # resize image 
        image   = im.channelview(image) 
        image_all[img_i, :, :, :] = image
        pm.next!(p)
    end 
    return image_all
end

function getImageLength(filepath)
    n_data  = 0
    for fi in filepath
        img_dir = readdir(fi)
        n_data += length(img_dir)
    end
    return n_data
end

function getClassPath(filepath)
    train_img_path  = Array{String}([])
    for ii in range(0, stop=n_class-1)
        img_path = string(filepath, "n$ii/")
        push!(train_img_path, img_path)
    end
    return train_img_path
end

#========== read Image ===========#
image_dict  = Dict()
image_all   = [] 
if switch_image_extract
    global image_dict  = Dict() 
    img_dict = Dict()
    for pi in image_path
        if match(r"training", pi) != nothing
            println("[Training image]") 
        else
            println("[Verify image]") 
        end
        data_count  = getImageLength(getClassPath(pi))
        image_mat   = zeros(Float32, (data_count, 3, dim_train, dim_train))
        image_tar   = ones(Int64,    data_count)
        i0          = 0
        for ii in range(0,stop=n_class-1) 
            println(" - <Image extraction at folder #$ii>")
            img_path        = string(pi, "n$ii/")
            idim            = Array{Int64}([3, dim_train, dim_train]) 
            class_count     = getImageLength([img_path])
            image_mat[(i0+1):(i0+class_count),:,:,:] = loadImageInDir(img_path, idim)
            image_tar[(i0+1):(i0+class_count)] .= ii
            i0             += class_count
        end
        
        image_mat           = np.transpose(image_mat, (2,3,1,0))
        #========== save Image ===========#
        path_tmp = ""
        img_key = match(r"training", pi) != nothing ? "training" : "verify"
        tar_key = match(r"training", pi) != nothing ? "training_tar" : "verify_tar"
        image_dict[img_key] = image_mat
        image_dict[tar_key] = image_tar
    end
    path_tmp  = string(image_data_file)
    println("[Dump image dict to $path_tmp]") 
    mypickle(path_tmp, image_dict)
    println("[Done !!!]") 
else
    #========== read Image ===========#
    global image_dict   = Dict()
    path_tmp = ""
    path_tmp  = string(image_data_file)
    println("[Load image dict from $path_tmp]") 
    image_dict = myunpickle(path_tmp)
    println("[Done !!!]")
    
end

#============= traning =============#
train_x     = image_dict["training"]
train_y     = image_dict["training_tar"]
test_x      = image_dict["verify"]
test_y      = image_dict["verify_tar"]
train_y     = onehotbatch(train_y, 0:9)
test_y      = onehotbatch(test_y, 0:9)
#train       = DataLoader(train_x, train_y, batchsize=batchsize, shuffle=true)
#test        = DataLoader(test_x, test_y, batchsize=batchsize)

#model       = vgg16()

layer_arr   = []
#========== level 1 (down to 150x150)==========#
filters     = (3, 16)
sstep       = 2
step_b      = 1
shortcut0   = Chain(Conv((1,1), filters[1]=>filters[2], pad = (0,0), stride = (sstep,sstep)), BatchNorm(filters[2]))
push!(layer_arr, Metalhead.ResidualBlock([filters[1], step_b, step_b, filters[2]], [1,3,1], [0,1,0], [1,1,sstep], shortcut0))
push!(layer_arr, Conv((1,1), filters[2]=>filters[2], leakyrelu, pad=(0,0), stride=(1,1)))
#========== level 2 (down to 75x75)==========#
filters     = (16, 32)
sstep       = 2
step_b      = 2
shortcut0   = Chain(Conv((1,1), filters[1]=>filters[2], pad = (0,0), stride = (sstep,sstep)), BatchNorm(filters[2]))
push!(layer_arr, Metalhead.ResidualBlock([filters[1], step_b, step_b, filters[2]], [1,3,1], [0,1,0], [1,1,sstep], shortcut0))
push!(layer_arr, Conv((1,1), filters[2]=>filters[2], leakyrelu, pad=(0,0), stride=(1,1)))
#========== level 3 (down to 38x38)==========#
filters     = (32, 32)
sstep       = 2
step_b      = 2
shortcut0   = Chain(Conv((1,1), filters[1]=>filters[2], pad = (0,0), stride = (sstep,sstep)), BatchNorm(filters[2]))
push!(layer_arr, Metalhead.ResidualBlock([filters[1], step_b, step_b, filters[2]], [1,3,1], [0,1,0], [1,1,sstep], shortcut0))
push!(layer_arr, Conv((1,1), filters[2]=>filters[2], leakyrelu, pad=(0,0), stride=(1,1)))
#========== level 4 (down to 19x19) ==========#
filters     = (32, 64)
sstep       = 2
step_b      = 4
shortcut0   = Chain(Conv((1,1), filters[1]=>filters[2], pad = (0,0), stride = (sstep,sstep)), BatchNorm(filters[2]))
push!(layer_arr, Metalhead.ResidualBlock([filters[1], step_b, step_b, filters[2]], [1,3,1], [0,1,0], [1,1,sstep], shortcut0))
push!(layer_arr, Conv((1,1), filters[2]=>filters[2], leakyrelu, pad=(0,0), stride=(1,1)))
#========== level 4 (down to 10x10) ==========#
filters     = (64, 128)
sstep       = 2
step_b      = 16
shortcut0   = Chain(Conv((1,1), filters[1]=>filters[2], pad = (0,0), stride = (sstep,sstep)), BatchNorm(filters[2]))
push!(layer_arr, Metalhead.ResidualBlock([filters[1], step_b, step_b, filters[2]], [1,3,1], [0,1,0], [1,1,sstep], shortcut0))
push!(layer_arr, Conv((1,1), filters[2]=>filters[2], leakyrelu, pad=(0,0), stride=(1,1)))
#========== level 5 (down to 5x5) ==========#
filters     = (128, 256)
sstep       = 2
step_b      = 32
shortcut0   = Chain(Conv((1,1), filters[1]=>filters[2], pad = (0,0), stride = (sstep,sstep)), BatchNorm(filters[2]))
push!(layer_arr, Metalhead.ResidualBlock([filters[1], step_b, step_b, filters[2]], [1,3,1], [0,1,0], [1,1,sstep], shortcut0))
push!(layer_arr, Conv((1,1), filters[2]=>filters[2], leakyrelu, pad=(0,0), stride=(1,1)))
#========== level 6 (MeanPooling to 19x19) ==========#
push!(layer_arr, MeanPool((5,5)))

push!(layer_arr, x->reshape(x, :, size(x,4)))
push!(layer_arr, (Dense(256, 10)))
push!(layer_arr, softmax)
model       = Chain(layer_arr...)

#layer_arr    = []
#push!(layer_arr, Conv2d_bn(3,  8,  (3,3),  (1,1), (2,2), leakyrelu))  #150x150
#push!(layer_arr, Conv2d_bn(8,  16, (3,3),  (1,1), (2,2), leakyrelu))  #75x75
#push!(layer_arr, Conv2d_bn(16, 32, (3,3),  (1,1), (2,2), leakyrelu))  #38x38
#push!(layer_arr, Conv2d_bn(32, 64, (3,3),  (1,1), (2,2), leakyrelu))  #19x19
#push!(layer_arr, Conv2d_bn(64, 128, (3,3), (1,1), (2,2), leakyrelu)) #10x10
#push!(layer_arr, MaxPool((3,3))) 
#
#push!(layer_arr, x->reshape(x, :, size(x,4)))
#push!(layer_arr, (Dense(1152, 10)))
#push!(layer_arr, softmax)
#model       = Chain(layer_arr...)
model   = model     |> cpu
train_x = train_x   |> cpu
train_y = train_y   |> cpu
test_x  = test_x    |> cpu
test_y  = test_y    |> cpu

loss(x, y) = logitcrossentropy(model(x), y)


evalcb() = @show(test_loss())

train       = DataLoader(train_x, train_y, batchsize=batchsize[1], shuffle=true)
test        = DataLoader(test_x, test_y, batchsize=batchsize[1])
for ii in 1:length(epochs)
    global train       = DataLoader(train_x, train_y, batchsize=batchsize[ii], shuffle=true)
    global test        = DataLoader(test_x, test_y, batchsize=batchsize[ii])
    println("[Training Start !!!]")
    println("\tTraining size: $(size(train_x)[end])")
    println("\tTest     size: $(size(test_x)[end])")
    println("\tBatch    size: $(batchsize[ii])")
    println("\tThreads count: $(Threads.nthreads())")
    @epochs epochs[ii] Flux.train!(loss, params(model), train, ADAM(learn_rate), cb=throttle(evalcb, 10))
    accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
end
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

println(accuracy(test_x, test_y))

