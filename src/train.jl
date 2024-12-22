function train!(
    θ::AbstractVector{T},
    prob::SciMLBase.AbstractDEProblem,
    (; train_data, valid_data)::NamedTuple,
    times,
    epochs::Int,
    optimiser, # ::Optimisers.AbstractRule,
    scheduler; #::ParameterSchedulers.AbstractSchedule;
    loss::Function=MSE,
    solver::SciMLBase.AbstractDEAlgorithm=Tsit5(),
    adjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm=BacksolveAdjoint(;
        autojacvec=ZygoteVJP(),
    ),
    reltol=1.0e-5,
    abstol=1.0e-5,
    verbose=false,
    manual_gc=false,
) where {T<:AbstractFloat}
    @info "Beginning training..."

    # Initial setup
    opt_state = Optimisers.setup(optimiser, θ)
    θ_min = copy(θ)
    min_val_loss = Inf32
    min_val_epoch = 0

    tspan = (times[1], times[end])

    training_start_time = time()
    for (epoch, learning_rate) in zip(1:epochs, scheduler)
        Optimisers.adjust!(opt_state, learning_rate)

        iter = 0
        training_losses = Float32[]
        epoch_start_time = time()
        for target_trajectory in train_data # shuffle(train_data)
            iter += 1
            u0 = target_trajectory[:, 1, :]
            prob = remake(prob; u0, tspan)

            training_loss, gradients = Zygote.withgradient(θ) do θ
                predicted_trajectory = stack(
                    solve(
                        prob,
                        solver;
                        p=θ,
                        saveat=times,
                        sensealg=adjoint,
                        reltol,
                        abstol,
                    ).u,
                    dims=2,
                )
                return loss(predicted_trajectory, target_trajectory)
            end

            push!(training_losses, training_loss)

            opt_state, θ = Optimisers.update!(opt_state, θ, gradients[1])

            if verbose
                @info @sprintf "[epoch = %04i] [iter = %04i] [tspan = (%05.2f, %05.2f)] Loss = %.2e\n" epoch iter tspan[1] tspan[2] training_loss
            end
        end

        val_loss = evaluate(prob, θ, valid_data, times, loss, solver, reltol, abstol)
        epoch_duration = time() - epoch_start_time

        @info @sprintf "[epoch = %04i] Learning rate = %.1e" epoch learning_rate
        @info @sprintf "[epoch = %04i] Train loss = %.2e\n" epoch mean(training_losses)
        @info @sprintf "[epoch = %04i] Valid loss = %.2e\n" epoch val_loss
        @info @sprintf "[epoch = %04i] Duration = %.1f seconds\n" epoch epoch_duration

        if val_loss < min_val_loss
            θ_min = copy(θ)
            min_val_epoch = epoch
            min_val_loss = val_loss
        end

        # Manually call the GC to (hopefully) avoid OOM errors
        if manual_gc
            GC.gc(true)
            ccall(:malloc_trim, Cvoid, (Cint,), 0)
        end

        flush(stderr)  # So we can watch log files on the cluster
    end
    training_duration = time() - training_start_time

    # Use optimal args
    θ .= θ_min

    @info "Training complete."
    @info @sprintf "Minimum validation loss = %.2e\n" min_val_loss
    @info @sprintf "Training duration = %.1f seconds\n" training_duration

    return training_duration, min_val_epoch, min_val_loss
end
