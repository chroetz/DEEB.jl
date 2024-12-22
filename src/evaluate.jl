function evaluate(prob, θ, data, times, loss, solver, reltol, abstol)
    losses = Float32[]
    tspan = (times[1], times[end])
    for target_trajectory in data
        u0 = target_trajectory[:, 1, :]
        prob = remake(prob; u0, tspan)
        predicted_trajectory = stack(
            solve(
                prob,
                solver;
                p=θ,
                saveat=times,
                reltol,
                abstol,
            ).u,
            dims=2,
        )
        push!(losses, loss(predicted_trajectory, target_trajectory))
    end
    return mean(losses)
end
