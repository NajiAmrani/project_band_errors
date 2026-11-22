using JLD2


function run_or_load(heavy_func::Function, args...; name::Union{Nothing, AbstractString}=nothing, cache_key=nothing, kwargs...)
    
    project_root = dirname(@__DIR__)
    results_dir = joinpath(project_root, "results")

    mkpath(results_dir)
    func_name = string(heavy_func)

    if !isnothing(name)
        # Use custom name directly (strip .jld2 if already included so we don't get .jld2.jld2)
        clean_name = endswith(name, ".jld2") ? name[1:end-5] : name
        filename = joinpath(results_dir, "$(clean_name).jld2")
    else
        key_to_hash = isnothing(cache_key) ? (args, kwargs) : cache_key
        param_hash = string(hash(key_to_hash), base=16)
        filename = joinpath(results_dir, "$(func_name)_$(param_hash).jld2")
    end


    if isfile(filename)
        println("Loading cached result from: $filename")
        return JLD2.load(filename, "result")
    else
        println("Cache not found in results/. Running calculation...")
        result = heavy_func(args...; kwargs...)
        JLD2.jldsave(filename; result)
        println("Saved new result to: $filename")
        return result
    end
end
