#################### MNIST One-vs-One Multi-class ####################
function MNIST_train(L, L_test, 𝑋s, 𝒚s, 𝐶, 𝑀, ACCURACY, γ)
    time_K_spent = time_GPU_spent = time_total_spent = 0
    time_total_start = time()

    randomisation_list = Array{Int32,1}[]
    𝜶_list = Array{Float32,1}[]
    error_list = Float64[]
    error_list_test = Float64[]
    finished_list = Bool[]
    time_ivj_list = Float64[]

    for ci = 1:10, cj = 1:10
        ci >= cj && continue
        print("$(ci-1)-vs-$(cj-1) - ")

        time_K_start = time()
        l = L[ci] + L[cj]
        randomisation = randperm(l)
        𝒚 = vcat(ones(Int32, size(𝒚s[ci])), -ones(Int32, size(𝒚s[cj])))[randomisation,:]
        𝑋 = vcat(𝑋s[ci], 𝑋s[cj])[randomisation,:]
        K = gaussian_kernel_matrix(𝑋, γ)
        time_K_spent += time() - time_K_start

        # l_test = L_test[ci] + L_test[cj]
        # randomisation_test = randperm(l_test)
        # 𝒚_test = vcat(ones(Int32, size(𝒚s_test[ci])), -ones(Int32, size(𝒚s_test[cj])))[randomisation_test,:]
        # 𝑋_test = vcat(𝑋s_test[ci], 𝑋s_test[cj])[randomisation_test,:]
        # K_test = gaussian_kernel_matrix(𝑋, 𝑋_test, γ)

        time_GPU_start = time()
        𝜶, error_rate, finished = stochastic_decomposition(𝐶, 𝑀, ACCURACY, l, 𝒚, K)
        # 𝜶, error_rate, error_rate_test, finished = stochastic_decomposition(𝐶, 𝑀, l, 𝒚, K, l_test, 𝒚_test, K_test)
        time_GPU_spent_this = time() - time_GPU_start
        push!(time_ivj_list, time_GPU_spent_this)
        time_GPU_spent += time_GPU_spent_this

        println("error rate = $error_rate% ; finished = $finished ; time = $time_GPU_spent_this")
        # println("error rate = $error_rate% ; test error rate = $error_rate_test% ; finished = $finished ; time = $time_GPU_spent_this")

        push!(randomisation_list, randomisation)
        push!(𝜶_list, 𝜶)
        push!(error_list, error_rate)
        # push!(error_list_test, error_rate_test)
        push!(finished_list, finished)
    end

    time_total_spent = time() - time_total_start
    return [time_K_spent, time_GPU_spent, time_total_spent], randomisation_list, 𝜶_list, error_list, error_list_test, finished_list, time_ivj_list
end

function MNIST_test_slave(𝑋_ALL, 𝒚_ALL, 𝑋s, 𝒚s, 𝜶_list, randomisation_list, γ)
    i = 0
    𝒚̂_test_list = Array{Int}[]
    for ci = 1:10, cj = 1:10
        ci >= cj && continue
        print("$(ci-1)-vs-$(cj-1) - ")
        i += 1

        𝑋_test, 𝒚_test = extract_all_test(𝑋_ALL, 𝒚_ALL)
        K_test = gaussian_kernel_matrix(vcat(𝑋s[ci], 𝑋s[cj])[randomisation_list[i],:], 𝑋_test, γ)
        𝝈_test = K_test * (𝜶_list[i] .* vcat(ones(Int32, size(𝒚s[ci])), -ones(Int32, size(𝒚s[cj])))[randomisation_list[i],:])
        𝒚̂_test = (y -> y > 0 ? ci-1 : cj-1).(𝝈_test)

        push!(𝒚̂_test_list, 𝒚̂_test)
    end
    return 𝒚̂_test_list
end

function MNIST_test(𝒚_test, 𝒚̂_test_list)
    𝒚̂_test_matrix = hcat(𝒚̂_test_list...)
    𝒚̂_test = mapslices(mode, 𝒚̂_test_matrix; dims=2)
    error_rate_test = count(𝒚̂_test .!= 𝒚_test) / length(𝒚_test) * 100
    correct_rate_test = 100 - error_rate_test  # First run: 99.1%
    return 𝒚̂_test, error_rate_test, correct_rate_test
end

function main(𝐶::Int32=Int32(1),    # Penalty
              𝑀::Int=512,    # Max size of minibatch
              ACCURACY=1e-2, # GPU accuracy = 0.01 * ACCURACY
              γ=0.015)       # Gaussian kernel parameter

    𝑋_ALL, 𝒚_ALL = MNIST.traindata(Float32)
    𝑋_ALL_test, 𝒚_ALL_test = MNIST.testdata(Float32)
    L, 𝑋s, 𝒚s = extract_each_classes(𝑋_ALL, 𝒚_ALL)
    L_test, 𝑋s_test, 𝒚s_test = extract_each_classes(𝑋_ALL_test,  𝒚_ALL_test)

    @time time_list, randomisation_list, 𝜶_list, error_list, error_list_test, finished_list, time_ivj_list =
        MNIST_train(L, L_test, 𝑋s, 𝒚s, 𝐶, 𝑀, ACCURACY, γ)
    @save "one-vs-one.jld2" time_list, randomisation_list, 𝜶_list, error_list, error_list_test, finished_list, time_ivj_list
    @time 𝒚̂_test_list = MNIST_test_slave(𝑋_ALL, 𝒚_ALL, 𝑋s, 𝒚s, 𝜶_list, randomisation_list, γ)
    @time 𝒚̂_test, error_rate_test = MNIST_test(𝒚_ALL, 𝒚̂_test_list)
    println(error_rate_test)
end
