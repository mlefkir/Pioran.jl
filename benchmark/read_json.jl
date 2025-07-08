using Pkg
using AirspeedVelocity
using HDF5

function read_dictfromJson(package_name, rev)

    results = AirspeedVelocity.Utils.load_results([Pkg.PackageSpec(name = package_name, rev = rev)], input_dir = "results/")
    keys = results[rev].keys
    deleteat!(keys, keys .== "time_to_load")

    filename = "plots/bench_celerite_$(rev).h5"
    if isfile(filename)
        rm(filename)
    end
    fid = h5open(filename, "w")
    subkeys = unique([i[1:2] for i in split.(keys, "/")])
    subkeys_mat = mapreduce(permutedims, vcat, subkeys)
    # unique([i[x``] for i in split.(keys,"/")]
    mask = isnothing.(tryparse.(Int64, subkeys_mat[:, 2]))
    no_subcat = unique(subkeys_mat[.!mask, 1])
    has_subcat = subkeys_mat[mask, 1]
    categ = [no_subcat..., has_subcat...]
    second_cat = [fill("", length(no_subcat))..., subkeys_mat[mask, 2]...]
    for i in 1:length(categ)
        first_label = categ[i]
        second_label = ""
        if second_cat[i] == ""
            m = occursin.(first_label, keys) # mask the results
        else
            second_label = second_cat[i]
            m = occursin.(first_label * "/" * second_label, keys)
        end
        println(first_label, " ", second_label)
        current_keys = keys[m]

        if second_label == "" # check if the second value after the / is an integer

            J_list, N_list = eachcol(parse.(Int64, mapreduce(permutedims, vcat, split.(current_keys, "/"))[:, 2:end]))
            J_list = sort(unique(J_list))
            N_list = sort(unique(N_list))

            ArrMean = zeros(length(J_list), length(N_list))
            ArrMedian = zeros(length(J_list), length(N_list))
            ArrMemory = zeros(length(J_list), length(N_list))
            ArrPerc25 = zeros(length(J_list), length(N_list))
            ArrPerc75 = zeros(length(J_list), length(N_list))

            for (j, J) in enumerate(J_list)
                for (n, N) in enumerate(N_list)
                    key = "$(first_label)/$J/$N"
                    ArrMean[j, n] = results[rev][key]["mean"] / 1.0e9 # convert to seconds
                    ArrMedian[j, n] = results[rev][key]["median"] / 1.0e9 # convert to seconds
                    ArrPerc25[j, n] = results[rev][key]["25"] / 1.0e9 # convert to seconds
                    ArrPerc75[j, n] = results[rev][key]["75"] / 1.0e9 # convert to seconds
                    ArrMemory[j, n] = results[rev][key]["memory"] / 1024^2 # convert to MegaBytes
                end
            end
            data = reshape(hcat(ArrMedian, ArrMean, ArrPerc25, ArrPerc75, ArrMemory), (length(J_list), length(N_list), 5))
            fid[first_label] = data
            fid[first_label * "_J"] = J_list
            fid[first_label * "_N"] = N_list
        else
            J_list, N_list = eachcol(parse.(Int64, mapreduce(permutedims, vcat, split.(current_keys, "/"))[:, 3:end]))
            J_list = sort(unique(J_list))
            N_list = sort(unique(N_list))

            ArrMean = zeros(length(J_list), length(N_list))
            ArrMedian = zeros(length(J_list), length(N_list))
            ArrMemory = zeros(length(J_list), length(N_list))
            ArrPerc25 = zeros(length(J_list), length(N_list))
            ArrPerc75 = zeros(length(J_list), length(N_list))

            for (j, J) in enumerate(J_list)
                for (n, N) in enumerate(N_list)
                    key = "$(first_label)/$(second_label)/$J/$N"
                    ArrMean[j, n] = results[rev][key]["mean"] / 1.0e9 # convert to seconds
                    ArrMedian[j, n] = results[rev][key]["median"] / 1.0e9 # convert to seconds
                    ArrPerc25[j, n] = results[rev][key]["25"] / 1.0e9 # convert to seconds
                    ArrPerc75[j, n] = results[rev][key]["75"] / 1.0e9 # convert to seconds
                    ArrMemory[j, n] = results[rev][key]["memory"] / 1024^2 # convert to MegaBytes
                end
            end
            data = reshape(hcat(ArrMedian, ArrMean, ArrPerc25, ArrPerc75, ArrMemory), (length(J_list), length(N_list), 5))
            fid[second_label] = data
            fid[second_label * "_J"] = J_list
            fid[second_label * "_N"] = N_list
        end
    end
    return close(fid)
end

read_dictfromJson(ARGS[1], ARGS[2])
