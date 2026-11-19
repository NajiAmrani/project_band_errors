using JLD2


function run_or_load(heavy_func::Function, args...; kwargs...)
    
    project_root = dirname(@__DIR__)
    results_dir = joinpath(project_root, "results")

    mkpath(results_dir)

    param_hash = string(hash((args,kwargs)), base=16)
    func_name = string(heavy_func)

    filename = joinpath(results_dir, "$(func_name)_$(param_hash).jld2")

    if isfile(filename)
        println("Loading cached result from: $filename")
        @load filename result
        return result
    else
        println("Cache not found in results/. Running calculation...")

        result = heavy_func(args...; kwargs...)

        @save filename result
        println("Saved new result to: $filename")
        return result
    end
end
