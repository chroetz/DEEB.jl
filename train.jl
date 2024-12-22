using InteractiveUtils
@info sprint(versioninfo)

using DEEB:
    parse_command_line,
    get_data,
    get_mlp,
    get_optimiser,
    get_scheduler,
    get_adjoint,
    train!,
    save_model
using OrdinaryDiffEq,
    Flux, SciMLSensitivity, CUDA, Parameters, Random, DelimitedFiles, MLUtils

function main(args)
    #! format: off
    # Unpack command line args into local scope
    @unpack job_id, rng_seed, device = args
    # Data args
    @unpack data_path, dim, steps, batchsize, train_frac = args
    # Solver args
    @unpack reltol, abstol, solver, sensealg, vjp, checkpointing = args
    # Neural net args
    @unpack hidden_layers, hidden_width, activation = args
    # Training args
    @unpack epochs, schedule_file, optimiser_rule, optimiser_hyperparams, manual_gc = args
    # I/0
    @unpack verbose = args
    #! format: on

    if device == :cuda
        device = gpu
    else
        device = cpu
    end

    Random.seed!(rng_seed)
    if CUDA.functional()
        CUDA.seed!(rng_seed)
    end

    # Set up the data
    data, times = get_data(data_path, dim, steps, batchsize, train_frac, device)

    # Set up the neural ODE
    θ, re = get_mlp(dim => dim, hidden_layers, hidden_width, activation) |> device
    rhs(u, θ, t) = re(θ)(u)
    u0 = zeros(Float32, dim)  # Arbitrary
    tspan = (0.0f0, 1.0f0)    # Arbitrary
    prob = ODEProblem{false,SciMLBase.FullSpecialize}(rhs, u0, tspan, θ)

    # Set up the optimiser and the learning schedule
    optimiser = get_optimiser(optimiser_rule, optimiser_hyperparams)
    scheduler = get_scheduler(schedule_file)

    # Set up the adjoint
    adjoint = BacksolveAdjoint(; autojacvec = ZygoteVJP(), checkpointing = false)

    # Train the model
    train!(
        θ,
        prob,
        data,
        times,
        epochs,
        optimiser,
        scheduler;
        solver,
        adjoint,
        reltol,
        abstol,
        verbose,
        manual_gc,
    )
    save_model(θ, "model_weights.jld2")
end

args = parse_command_line(; log = true)
main(args)
