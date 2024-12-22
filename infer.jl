using InteractiveUtils
@info sprint(versioninfo)

using DEEB: parse_command_line, get_mlp, load_model
using OrdinaryDiffEq, Flux, CUDA, Parameters, Random, DelimitedFiles, MLUtils

function generate_header_traj(dim::Int)
    header = ["trajId", "time"]
    states = ["state$i" for i = 1:dim]
    x = vcat(header, states)
    return reshape(x, 1, length(x))
end

function main(args)
    #! format: off
    # Data args
    @unpack data_path, dim, device= args
    # Neural net args
    @unpack hidden_layers, hidden_width, activation = args
    # Solver args
    @unpack reltol, abstol, solver = args
    # Timing
    @unpack t0, t1, dt = args
    # I/0
    @unpack pred_path = args
    #! format: on
    

    if device == :cuda
        device = gpu
    else
        device = cpu
    end

    # Set up the data
    raw_data, headers = readdlm(data_path, ',', Float32; header = true)
    # times = raw_data[:, 2]  # Can we just do this instead of passing t0,t1,dt?
    trajectory = raw_data[:, 3:(2+dim)]'

    # Load the model weights
    model_data = load_model("model_weights.jld2")
    θ = model_data["weights"] |> device

    # Set up the neural ODE
    _, re = get_mlp(dim => dim, hidden_layers, hidden_width, activation) |> device
    rhs(u, θ, t) = re(θ)(u)
    u0 = device(trajectory[:, 1:1])
    saveat = t0:dt:t1
    tspan = (t0, t1)
    esti_prob = ODEProblem{false,SciMLBase.FullSpecialize}(rhs, u0, tspan, θ)

    # Solve the neural ODE
    esti_sol = solve(esti_prob, solver; p = θ, reltol, abstol, saveat)
    esti_traj = dropdims(cpu(stack(esti_sol.u)), dims = 2)
    esti_times = esti_sol.t
    traj_id = ones(Int, length(esti_times))

    open(pred_path, "w") do io
        writedlm(io, generate_header_traj(dim), ',')
        writedlm(io, Any[traj_id esti_times esti_traj'], ',')
    end
end

args = parse_command_line(; log = true)
main(args)
