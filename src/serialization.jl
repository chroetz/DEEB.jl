function save_model(weights, path)
    jldsave(path; weights=cpu(weights))
    return nothing
end

function load_model(path)
    return load(path)
end
