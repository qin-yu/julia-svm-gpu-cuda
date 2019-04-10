##############################
## by Qin Yu, Feb 2019
## using Julia 1.0.3
##############################

gaussian_kernel_matrix(𝑋, γ) =  # 𝑋's rows are input tensors.
    exp.(- γ * pairwise(SqEuclidean(), Matrix(𝑋), dims=1))
gaussian_kernel_matrix(𝑋, 𝑋_test, γ) =  # 𝑋's rows are input tensors.
    exp.(- γ * pairwise(SqEuclidean(), Matrix(𝑋_test), Matrix(𝑋), dims=1))

function extract_each_classes(𝑋_ALL, 𝒚_ALL)
    CLASSES = sort(unique(𝒚_ALL))
    NO_OF_CLASSES = length(CLASSES)
    L = [sum(𝒚_ALL .== class) for class in CLASSES]
    𝑋_EACH_train = [𝑋_ALL[:,:,𝒚_ALL .== class] for class in CLASSES]
    𝑋s = [hcat(transpose(reshape(𝑋_EACH_train[i], 28*28, L[i])), ones(L[i])) for i = 1:NO_OF_CLASSES]
    𝒚s = ones.(Int, L) .* CLASSES
    return L, 𝑋s, 𝒚s
end

function extract_all_test(𝑋_ALL, 𝒚_ALL)
    𝑋s = hcat(transpose(reshape(𝑋_ALL, 28*28, :)), ones(size(𝑋_ALL, 3)))
    return 𝑋s, 𝒚_ALL
end

function prepare_one_vs_rest(𝑋_ALL, 𝒚_ALL)
    𝑋 = hcat(transpose(reshape(𝑋_ALL, 28*28, :)), ones(size(𝑋_ALL, 3)))
    𝒚s = [(y -> y == pos_label ? 1 : -1).(𝒚_ALL) for pos_label = 0:9]
    return 𝑋, 𝒚s
end

function prepare_MNIST_data(class_pos, class_neg, γ)
    # Load train data
    𝑋_ALL_train, 𝒚_ALL_train = MNIST.traindata(Float32)

    NO_OF_NEGATIVE = sum(𝒚_ALL_train .== class_neg)
    NO_OF_POSITIVE = sum(𝒚_ALL_train .== class_pos)
    l = NO_OF_NEGATIVE + NO_OF_POSITIVE

    𝑋_neg_train = 𝑋_ALL_train[:,:,𝒚_ALL_train .== class_neg]
    𝑋_pos_train = 𝑋_ALL_train[:,:,𝒚_ALL_train .== class_pos]
    𝒚_neg_train = - ones(Int, NO_OF_NEGATIVE)
    𝒚_pos_train =   ones(Int, NO_OF_POSITIVE)

    randomisation = randperm(l)
    𝑋 = hcat(transpose(reshape(cat(𝑋_neg_train, 𝑋_pos_train; dims=3), 28*28, l)), ones(l))[randomisation,:]
    𝒚 = vcat(𝒚_neg_train, 𝒚_pos_train)[randomisation,:]

    gaussian_kernel_matrix(𝑋, γ) = exp.(- γ * pairwise(SqEuclidean(), Matrix(𝑋), dims=1))  # 𝑋's rows are input tensors.
    K = gaussian_kernel_matrix(𝑋, γ)

    # Load test data
    𝑋_ALL_test,  𝒚_ALL_test  = MNIST.testdata(Float32)

    NO_OF_NEGATIVE_test = sum(𝒚_ALL_test .== class_neg)
    NO_OF_POSITIVE_test = sum(𝒚_ALL_test .== class_pos)
    l_test = NO_OF_NEGATIVE_test + NO_OF_POSITIVE_test

    𝑋_neg_test = 𝑋_ALL_test[:,:,𝒚_ALL_test .== class_neg]
    𝑋_pos_test = 𝑋_ALL_test[:,:,𝒚_ALL_test .== class_pos]
    𝒚_neg_test = - ones(Int, NO_OF_NEGATIVE_test)
    𝒚_pos_test =   ones(Int, NO_OF_POSITIVE_test)

    𝑋_test = hcat(transpose(reshape(cat(𝑋_neg_test, 𝑋_pos_test; dims=3), 28*28, l_test)), ones(l_test))
    𝒚_test = vcat(𝒚_neg_test, 𝒚_pos_test)

    gaussian_kernel_matrix(𝑋, 𝑋_test, γ) = exp.(- γ * pairwise(SqEuclidean(), Matrix(𝑋_test), Matrix(𝑋), dims=1))  # 𝑋's rows are input tensors.
    K_test = gaussian_kernel_matrix(𝑋, 𝑋_test, γ)

    return (𝑋, 𝒚, K, l), (𝑋_test, 𝒚_test, K_test, l_test)
end

# Only as termination criterion for the whole process.
function monitor_kkt_condition(𝜶, 𝝈, 𝒚; accuracy=1e-2, 𝐶::Int32=Int32(1))
    𝑦𝑓𝒙 = 𝒚 .* 𝝈
    for (idx,α) in enumerate(𝜶)
        if α == 0 && 𝑦𝑓𝒙[idx] < 1
            return false
        elseif α == 𝐶 && 𝑦𝑓𝒙[idx] > 1
            return false
        elseif 0 < α < 𝐶 && !isapprox(𝑦𝑓𝒙[idx], 1; atol=accuracy)
            return false
        end
    end
    return true
end

function select_new_points(𝜶, 𝝈, 𝒚, 𝑀, support_vector_idx; accuracy=1e-2, 𝐶::Int32=Int32(1))
    𝑦𝑓𝒙 = 𝒚 .* 𝝈
    list_of_new_ids = Int[]
    list_of_new_min = Float64[]
    for (idx,α) in enumerate(𝜶)
        if (α == 0 && 𝑦𝑓𝒙[idx] < 1) || (α == 𝐶 && 𝑦𝑓𝒙[idx] > 1) || (0 < α < 𝐶 && !isapprox(𝑦𝑓𝒙[idx], 1; atol=accuracy))
            push!(list_of_new_ids, idx)
            push!(list_of_new_min, 𝑦𝑓𝒙[idx])
        end
    end
    list_of_new_ids = list_of_new_ids[sortperm(list_of_new_min)]
    setdiff!(list_of_new_ids, support_vector_idx)
    return list_of_new_ids
end
