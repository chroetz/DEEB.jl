using DEEB: get_mlp, get_scheduler, train!
using Flux, OrdinaryDiffEq, Optimisers, DelimitedFiles, Random, MLUtils, CUDA

steps = 3
train_frac = 0.7
batchsize = 512
device = gpu

# Set up the data
data_path = "/p/projects/ou/labs/ai/DEEB/DeebDbLorenzTune/lorenz63std/observation/truth0001obs0001.csv"
raw_data, headers = readdlm(data_path, ',', Float32; header=true)
times = raw_data[1:steps, 2]
trajectory = copy(raw_data[:, 3:5]')

# data = Data{Float32}(time_series; steps, split_at=train_frac)

chunks = MLUtils.chunk(trajectory; size=steps)
if size(chunks[end]) != size(chunks[1])
    pop!(chunks)  # If there's a short chunk, get rid of it
end

train_chunks, valid_chunks = splitobs(chunks; at=train_frac, shuffle=true)
train_data = BatchView(device(stack(train_chunks)); batchsize)
valid_data = BatchView(device(stack(valid_chunks)); batchsize)
data = (; train_data, valid_data)

# Set up the neural network
Random.seed!(1)
hidden_layers = 3
hidden_width = 64
activation = gelu
θ, re = get_mlp(3 => 3, hidden_layers, hidden_width, activation) |> device

# Set up the ODEProblem
function f(u, θ, t)
    return re(θ)(u)
end
u0 = zeros(Float32, 3)  # Arbitrary
tspan = (0.0f0, 1.0f0)  # Arbitrary
prob = ODEProblem{false,SciMLBase.FullSpecialize}(f, u0, tspan, θ)

# Set up the optimiser and the learning rate schedule
optimiser = Optimisers.AdamW(1.0f-3, (9.0f-1, 9.99f-1), 1.0f-4)
scheduler = get_scheduler("schedule_const_medium.toml")
epochs = 10000

# Train
@time training_duration, learning_curve, min_val_epoch, min_val_loss =
    train!(θ, prob, data, times, epochs, optimiser, scheduler; manual_gc=false, verbose=false)
