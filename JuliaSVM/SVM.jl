##############################
## by Qin Yu, Apr 2019
## using Julia 1.1.0
##############################

using Revise, BenchmarkTools                           # Development
using JLD2, FileIO, MLDatasets                         # Data & IO
using LinearAlgebra, Distances, Random, Distributions  # Maths
using CUDAdrv, CUDAnative, CuArrays                    # GPU

MAX_MINI_BATCH_ID = 50

include("SVMhelper.jl")

#################### Compute Adatron SGD in GPU: ####################
@inline sync_threads_and(predicate::Int32) = ccall("llvm.nvvm.barrier0.and", llvmcall, Int32, (Int32,), predicate)
@inline sync_threads_and(predicate::Bool) = ifelse(sync_threads_and(Int32(predicate)) !== Int32(0), true, false)

function kernel_soft_SGD_SVM(𝜶, 𝝈, K, 𝒚, l::Int32, 𝐶::Int32)
    j = (blockIdx().x-1) * blockDim().x + threadIdx().x
    can_stop = false
    δᵢ = @cuStaticSharedMem(Float32, 1)
    while !sync_threads_and(can_stop)
        last_α_j = 𝜶[j]
        # Adatron:
        for i = 1:l
            if j == i  # Online
                last_α = 𝜶[i]
                𝜇ᵢ = 1 / K[i,i]
                δᵢ[1] = 𝜇ᵢ * (1 - 𝒚[i] * 𝝈[i])
                𝜶[i] = 𝜶[i] + δᵢ[1]
                𝜶[i] < 0 && (𝜶[i] = 0; δᵢ[1] = 0 - last_α)
                𝜶[i] > 𝐶 && (𝜶[i] = 𝐶; δᵢ[1] = 𝐶 - last_α)
            end
            sync_threads()
            𝝈[j] += δᵢ[1] * 𝒚[i] * K[i,j]  # Parallel update
        end
        # Stopping criterion:
        can_stop = false
        isapprox(𝜶[j], last_α_j; atol=1e-4) && (can_stop = true)
    end
    return nothing
end

function optimise_working_set(𝜶, 𝝈, K, 𝒚; 𝐶::Int32=Int32(1))
    l = Int32(length(𝜶))
    cu_𝜶 = CuArray{Float32}(𝜶)
    cu_𝝈 = CuArray{Float32}(𝝈)
    cu_K = CuArray{Float32}(K)
    cu_𝒚 = CuArray{Float32}(𝒚)

    @cuda threads=l kernel_soft_SGD_SVM(cu_𝜶, cu_𝝈, cu_K, cu_𝒚, l, 𝐶)

    𝜶 = Array{Float32}(cu_𝜶)
    return 𝜶
end

function optimise_working_set_CPU(𝜶, 𝝈, K, 𝒚; 𝐶=1)
    l = length(𝜶)
    while true
        last_𝜶 = copy(𝜶)
        for i = 1:l
            𝜇ᵢ = 1 / K[i,i]
            δᵢ = 𝜇ᵢ * (1 - 𝒚[i] * 𝝈[i])
            𝜶[i] = 𝜶[i] + δᵢ
            𝜶[i] < 0 && (𝜶[i] = 0; δᵢ = 0 - last_𝜶[i])
            𝜶[i] > 𝐶 && (𝜶[i] = 𝐶; δᵢ = 𝐶 - last_𝜶[i])
            𝝈 .+= δᵢ * 𝒚[i] * K[i,:]
        end
        all(isapprox.(last_𝜶, 𝜶; atol=1e-4)) && break
    end
    return 𝜶
end

function stochastic_decomposition_test(𝐶, 𝑀, ACCURACY, l, 𝒚, K, l_test, 𝒚_test, K_test; usecpu=false)
    𝑀_safe = div(𝑀, 4) * 3
    𝑀_rand = 𝑀 - 𝑀_safe

    𝜶 = zeros(Float32, l)
    𝝈 = zeros(Float32, l)

    selected_new_idx = []
    support_vector_idx, alpha0_vector_idx = collect(1:𝑀), collect(𝑀+1:l)
    working_set_counter = zeros(Int32, l)

    batch_id = 0
    while !monitor_kkt_condition(𝜶, 𝝈, 𝒚; 𝐶=𝐶) && batch_id < MAX_MINI_BATCH_ID
        batch_id += 1

        current_working_set_counter = working_set_counter[support_vector_idx]
        sorted_support_vector_idx = support_vector_idx[sortperm(current_working_set_counter)]

        if length(support_vector_idx) >= 𝑀_safe
            no_of_new_idx = min(length(selected_new_idx), 𝑀_rand)
        else
            no_of_new_idx = min(length(selected_new_idx), 𝑀-length(support_vector_idx))
        end

        no_of_old_idx = length(support_vector_idx)
        no_of_exceeds = no_of_new_idx + no_of_old_idx - 𝑀
        if no_of_exceeds > 0
            abandoned_support_vector_idx = sorted_support_vector_idx[end+1-no_of_exceeds:end]
            setdiff!(support_vector_idx, abandoned_support_vector_idx)
            union!(alpha0_vector_idx, abandoned_support_vector_idx)
        end
        selected_new_idx = selected_new_idx[1:no_of_new_idx]
        union!(support_vector_idx, selected_new_idx)
        setdiff!(alpha0_vector_idx, selected_new_idx)

        working_set_counter[support_vector_idx] .+= 1
        𝜶_subset, 𝝈_subset = 𝜶[support_vector_idx], 𝝈[support_vector_idx]
        𝒚_subset, K_subset = 𝒚[support_vector_idx], K[support_vector_idx,support_vector_idx]
        if usecpu
            𝜶[support_vector_idx] = optimise_working_set_CPU(𝜶_subset, 𝝈_subset, K_subset, 𝒚_subset; 𝐶=𝐶)
        else
            𝜶[support_vector_idx] = optimise_working_set(𝜶_subset, 𝝈_subset, K_subset, 𝒚_subset; 𝐶=𝐶)
        end

        𝝈 = K * (𝜶 .* 𝒚)

        for (i_index,i) in enumerate(support_vector_idx)
            if 𝜶[i] == 0
                deleteat!(support_vector_idx, i_index)
                push!(alpha0_vector_idx, i)
            end
        end

        selected_new_idx = select_new_points(𝜶, 𝝈, 𝒚, 𝑀, support_vector_idx; accuracy=ACCURACY, 𝐶=𝐶)
    end

    𝒚̂      = sign.(𝝈)
    𝒚̂_test = sign.(K_test * (𝜶 .* 𝒚))
    error_rate      = count(𝒚̂ .!= 𝒚) / l * 100
    error_rate_test = count(𝒚̂_test .!= 𝒚_test) / l_test * 100
    return 𝜶, error_rate, error_rate_test, monitor_kkt_condition(𝜶, 𝝈, 𝒚; 𝐶=𝐶)
end

function stochastic_decomposition(𝐶, 𝑀, ACCURACY, l, 𝒚, K; usecpu=false)
    𝑀_safe = div(𝑀, 4) * 3
    𝑀_rand = 𝑀 - 𝑀_safe

    𝜶 = zeros(Float32, l)
    𝝈 = zeros(Float32, l)

    selected_new_idx = []
    support_vector_idx, alpha0_vector_idx = collect(1:𝑀), collect(𝑀+1:l)
    working_set_counter = zeros(Int32, l)

    batch_id = 0
    while !monitor_kkt_condition(𝜶, 𝝈, 𝒚; accuracy=ACCURACY, 𝐶=𝐶) && batch_id < MAX_MINI_BATCH_ID
        batch_id += 1

        current_working_set_counter = working_set_counter[support_vector_idx]
        sorted_support_vector_idx = support_vector_idx[sortperm(current_working_set_counter)]

        if length(support_vector_idx) >= 𝑀_safe
            no_of_new_idx = min(length(selected_new_idx), 𝑀_rand)
        else
            no_of_new_idx = min(length(selected_new_idx), 𝑀-length(support_vector_idx))
        end

        no_of_old_idx = length(support_vector_idx)
        no_of_exceeds = no_of_new_idx + no_of_old_idx - 𝑀
        if no_of_exceeds > 0
            abandoned_support_vector_idx = sorted_support_vector_idx[end+1-no_of_exceeds:end]
            setdiff!(support_vector_idx, abandoned_support_vector_idx)
            union!(alpha0_vector_idx, abandoned_support_vector_idx)
        end
        selected_new_idx = selected_new_idx[1:no_of_new_idx]
        union!(support_vector_idx, selected_new_idx)
        setdiff!(alpha0_vector_idx, selected_new_idx)

        working_set_counter[support_vector_idx] .+= 1
        𝜶_subset, 𝝈_subset = 𝜶[support_vector_idx], 𝝈[support_vector_idx]
        𝒚_subset, K_subset = 𝒚[support_vector_idx], K[support_vector_idx,support_vector_idx]
        if usecpu
            println("using CPU")
            𝜶[support_vector_idx] = optimise_working_set_CPU(𝜶_subset, 𝝈_subset, K_subset, 𝒚_subset; 𝐶=𝐶)
        else
            𝜶[support_vector_idx] = optimise_working_set(𝜶_subset, 𝝈_subset, K_subset, 𝒚_subset; 𝐶=𝐶)
        end
        𝝈 = K * (𝜶 .* 𝒚)

        for (i_index,i) in enumerate(support_vector_idx)
            if 𝜶[i] == 0
                deleteat!(support_vector_idx, i_index)
                push!(alpha0_vector_idx, i)
            end
        end

        selected_new_idx = select_new_points(𝜶, 𝝈, 𝒚, 𝑀, support_vector_idx; accuracy=ACCURACY, 𝐶=𝐶)
    end

    𝒚̂ = sign.(𝝈)
    error_rate = count(𝒚̂ .!= 𝒚) / l * 100
    return 𝜶, error_rate, monitor_kkt_condition(𝜶, 𝝈, 𝒚; accuracy=ACCURACY, 𝐶=𝐶)
end
