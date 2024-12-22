function get_data(data_path, dim, steps, batchsize, train_frac, device, shuffle=true, precision=Float32)
    raw_data, headers = readdlm(data_path, ',', precision; header=true)
    times = raw_data[1:steps, 2]  # System is autonomous so we just always use these times
    trajectory = raw_data[:, 3:(2+dim)]'

    chunks = MLUtils.chunk(trajectory; size=steps)
    if size(chunks[end]) != size(chunks[1])
        pop!(chunks)  # If there's a short chunk, get rid of it
    end

    train_chunks, valid_chunks = splitobs(chunks; at=train_frac, shuffle)
    train_data = MLUtils.BatchView(device(stack(train_chunks)); batchsize)
    valid_data = MLUtils.BatchView(device(stack(valid_chunks)); batchsize)
    return (; train_data, valid_data), times
end
