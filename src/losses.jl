"""
    MSE(predicted_trajectory, target_trajectory)

Compute the mean-squared error between the predicted trajectory and the target trajectory.
"""
function MSE(predicted_trajectory, target_trajectory)
    return mean(abs2, predicted_trajectory .- target_trajectory)  # Do not include u0
end

"""
    MAE(predicted_trajectory, target_trajectory)

Compute the mean absolute error between the predicted trajectory and the target trajectory.
"""
function MAE(predicted_trajectory, target_trajectory)
    return mean(abs, predicted_trajectory[:, :, 2:end] .- target_trajectory[:, :, 2:end])  # Do not include u0
end
