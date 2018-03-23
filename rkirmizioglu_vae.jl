
for p in ("Knet","ArgParse","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, FileIO, ImageCore, Images, ImageView, ArgParse

const F = Float32

Atype = gpu() >= 0 ? KnetArray{F} : Array{F}

BINARIZE = false

binarize(x) = convert(Atype, rand(F, size(x))) .< x

function encode(ϕ, x)
    x = reshape(x, (64,64,3,:))

    x = conv4(ϕ[1], x, stride=2, padding=1)
    x = relu.(x .+ ϕ[2])

    x = conv4(ϕ[3], x, stride=2, padding=1)
    x = relu.(x .+ ϕ[4])

    x = conv4(ϕ[5], x, stride=2, padding=1)
    x = relu.(x .+ ϕ[6])

    x = conv4(ϕ[7], x, stride=2, padding=1)
    x = relu.(x .+ ϕ[8])

    x = mat(x)
    #x = relu.(ϕ[9]*x .+ ϕ[10])

    μ = ϕ[end-3]*x .+ ϕ[end-2]
    logσ² = ϕ[end-1]*x .+ ϕ[end]

    return μ, logσ²
end

function decode(θ, z)
    z = relu.(θ[1]*z .+ θ[2])
    filters = size(θ[3], 3)

    width = Int(sqrt(size(z,1) ÷ filters))
    z = reshape(z, (width, width, filters, :))

    z = nearest_neighbor(z)
    z = conv4(θ[3], z, padding=1)
    z = relu.(z .+ θ[4])

    z = nearest_neighbor(z)
    z = conv4(θ[5], z, padding=1)
    z = relu.(z .+ θ[6])

    z = nearest_neighbor(z)
    z = conv4(θ[7], z, padding=1)
    z = relu.(z .+ θ[8])

    z = nearest_neighbor(z)
    z = conv4(θ[9], z, padding=1)
    z = sigm.(z .+ θ[10])
    return z
end

function loss(w, x; samples=1)
    θ, ϕ = w[:decoder], w[:encoder]
    μ, logσ² = encode(ϕ, x)
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)

    KL =  - sum(@. 1 + logσ² - μ*μ - σ²) / 2
    # Normalise by same number of elements as in reconstruction
    KL /= length(x)
    BCE = F(0)
    x̂ = Any[]
    for s=1:samples
        # ϵ = randn!(similar(μ))
        ϵ = convert(Atype, randn(F, size(μ)))
        z = @. μ + ϵ * σ
        x̂ = decode(θ, z)
        BCE += binary_cross_entropy(x, x̂)
    end
    BCE /= samples

    return BCE + KL
end

lossgradient = grad(loss)

function binary_cross_entropy(x, x̂)
    x = reshape(x, size(x̂))
    s = @. x * log(x̂ + F(1e-10)) + (1-x) * log(1 - x̂ + F(1e-10))
    return -sum(s) / length(x)
end

function aveloss(w, xtrn; samples=1, batchsize=100)
    θ, ϕ = w[:decoder], w[:encoder]
    ls = F(0)
    count = 0
    for x in minibatch(xtrn, batchsize; xtype=Atype)
        BINARIZE && (x = binarize(x))
        ls += loss(w, x; samples=samples)
        count += 1
    end
    N = length(xtrn) ÷ size(xtrn, ndims(xtrn))
    return (ls / count) * N
end

function nearest_neighbor(z)
    interp = KnetArray(zeros(size(z,1)*2, size(z,2)*2, size(z,3), size(z,4)))
    interp = convert(KnetArray{Float32,4}, interp)
    for channel = 1:size(interp,3)
        for i = 2:2:size(interp,1)
            for j = 2:2:size(interp,2)
                interp[i,j,channel,1] = z[Int(i/2),Int(j/2),channel,1]
                interp[i-1,j-1,channel,1] = z[Int(i/2),Int(j/2),channel,1]
                interp[i,j-1,channel,1] = z[Int(i/2),Int(j/2),channel,1]
                interp[i-1,j,channel,1] = z[Int(i/2),Int(j/2),channel,1]
            end
        end
    end
    return interp
end

function weights(nz, nh)
    θ = [] # z->x

    push!(θ, xavier(nh, nz))
    push!(θ, zeros(nh))

    push!(θ, xavier(3, 3, 256, 128))
    push!(θ, zeros(1,1,128,1))

    push!(θ, xavier(3, 3, 128, 64))
    push!(θ, zeros(1,1,64,1))

    push!(θ, xavier(3, 3, 64, 32))
    push!(θ, zeros(1, 1, 32, 1))

    push!(θ, xavier(3, 3, 32, 3))
    push!(θ, zeros(1,1,3,1))


    θ = map(a->convert(Atype,a), θ)

    ϕ = [] # x->z

    push!(ϕ, xavier(4, 4, 3, 32))
    push!(ϕ, zeros(1, 1, 32, 1))

    push!(ϕ, xavier(4, 4,  32, 64))
    push!(ϕ, zeros(1, 1, 64, 1))

    push!(ϕ, xavier(4, 4, 64, 128))
    push!(ϕ, zeros(1,1, 128, 1))

    push!(ϕ, xavier(4, 4, 128, 256))
    push!(ϕ, zeros(1, 1, 256, 1))

    #push!(ϕ, xavier(nh, 256*4^2))
    #push!(ϕ, zeros(nh))

    push!(ϕ, xavier(nz, nh)) # μ
    push!(ϕ, zeros(nz))
    push!(ϕ, xavier(nz, nh)) # logσ^2
    push!(ϕ, zeros(nz))

    ϕ = map(a->convert(Atype,a), ϕ)

    return θ, ϕ
end

function main(args="")
    s = ArgParseSettings()
    s.description="Variational Auto Encoder on MNIST dataset."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=64; help="minibatch size")
        ("--epochs"; arg_type=Int; default=5; help="number of epochs for training")
        ("--nh"; arg_type=Int; default=256*(4^2); help="hidden layer dimension")
        ("--nz"; arg_type=Int; default=100; help="encoding dimention")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{F}" : "Array{F}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--infotime"; arg_type=Int; default=2; help="report every infotime epochs")
        ("--binarize"; arg_type=Bool; default=false; help="dinamically binarize during training")
        ("--optim"; default="Adam()"; help="optimizer")
    end
    isa(args, String) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end

    o = parse_args(args, s; as_symbols=true)

    global Atype = eval(parse(o[:atype]))
    global BINARIZE = o[:binarize]
    info("using ", Atype)
    # gc(); knetgc();
    o[:seed] > 0 && setseed(o[:seed])
    path = "/home/rizzy/Downloads/DeepLearning/pp_dataset/"
    files = readdir(path)
    dataset = Any[]
    for i = 1:128
    	image = load(path * files[i])
    	image = Float32.(rawview(channelview(image)[1:3, :, :]))
    	image = permutedims(image, [2 3 1])
        image = KnetArray(image)
    	push!(dataset, image)
    end
    wdec, wenc = weights(o[:nz], o[:nh])
    w = Dict(
        :encoder => wenc,
        :decoder => wdec)
    opt = Dict(
        :encoder => map(wi->eval(parse(o[:optim])), w[:encoder]),
        :decoder => map(wi->eval(parse(o[:optim])), w[:decoder]),
    )

    println(loss(w, dataset[1]; samples=1))
end

main("--infotime 10 --seed 1 --epochs 5")
