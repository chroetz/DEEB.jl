function parse_command_line(; log = false)
    deeb_settings = ArgParseSettings(autofix_names = true)

    @add_arg_table deeb_settings begin
        # Experiment args
        "--job-id"
            arg_type = String
            default = get(ENV, "SLURM_JOB_ID", "1")
        "--rng-seed", "--seed"
            arg_type = Int
            default = 1
        "--device"
            help = "Whether to train on CPU or CUDA device"
            arg_type = Symbol
            default = :cpu
        "--data-path"
            help = "Path to data"
            arg_type = String
            required = true
        "--dim"
            help = "Dimension of the data"
            arg_type = Int
            required = true

        # Neural net args
        "--hidden-layers", "--layers"
            arg_type = Int
            required = true
        "--hidden-width", "--width"
            arg_type = Int
            required = true
        "--activation"
            arg_type = Function
            default = relu

        # Training args
        "--steps"
            help = "Number of timesteps in each chunk of training data"
            arg_type = Int
        "--batchsize"
            help = "Batch size for training"
            arg_type = Int
        "--train-frac"
            help = "Fraction of data chunks for training"
            arg_type = Float64
        "--epochs"
            arg_type = Int
            required = true
        "--schedule-file", "--schedule"
            arg_type = String
            required = true
        "--optimiser-rule", "--opt"
            arg_type = Symbol
            default = :Adam
        "--optimiser-hyperparams", "--opt-params"
            arg_type = NamedTuple
            default = (;)
        "--manual-gc"
            action = :store_true

        # Solver args
        "--reltol"
            arg_type = Float64
            default = 1e-5
        "--abstol"
            arg_type = Float64
            default = 1e-5
        "--solver"
            arg_type = SciMLBase.AbstractODEAlgorithm
            default = Tsit5()
        "--sensealg"
            arg_type = Symbol
            default = :BacksolveAdjoint
        "--vjp"
            arg_type = Symbol
            default = :ZygoteVJP
        "--checkpointing"
            action = :store_true

        # I/0
        "--verbose"
            action = :store_true

        # Inference-specific settings
        "--pred-traj"
            help = "Use the trained model to predict beyond the last observation"
            action = :store_true
        "--pred-path"
            help = "Where to store prediction"
            arg_type = String
        "--t0"
            help = "Strart time"
            arg_type = Float32
        "--t1"
            help = "End time"
            arg_type = Float32
        "--dt"
            help = "Time step"
            arg_type = Float32
    end

    args = parse_args(deeb_settings)
    
    if log
        log_args(args)
    end

    return args
end

function log_args(args)
    ordered_args = sort(collect(args); by = x -> x[1])
    for (arg_name, arg_value) in ordered_args
        @info "$arg_name = $arg_value"
    end
end

function eval_string(s)
    return eval(Meta.parse(s))
end

function ArgParse.parse_item(::Type{Function}, function_name::AbstractString)
    return eval_string(function_name)
end

function ArgParse.parse_item(::Type{SciMLBase.AbstractODEAlgorithm}, solver_name::AbstractString)
    return eval_string(solver_name * "()")
end

function ArgParse.parse_item(::Type{NamedTuple}, arg_string::AbstractString)
    return eval_string(arg_string)
end

# function ArgParse.parse_item(::Type{NamedTuple}, arg_string::AbstractString)
#     items = eachsplit(arg_string, ",")
#     names = [Symbol(strip.(split(item, "="))[1]) for item in items]
#     args = [parse(Float32, (split(item, "=")[2])) for item in items]
#     return (; zip(names, args)...)
# end
