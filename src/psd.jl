using Pioran: SHO

abstract type Model end
abstract type PowerSpectralDensity <: Model end

abstract type BendingPowerLaw <: PowerSpectralDensity end

struct SimpleBendingPowerLaw{T<:Real} <: BendingPowerLaw
    α_1::T
    f_b::T
    α_2::T
end

struct DoubleBendingPowerLaw{T<:Real} <: BendingPowerLaw
    α_1::T
    f_b_1::T
    α_2::T
    f_b_2::T
    α_3::T
end

function calculate_psd(f, psd::DoubleBendingPowerLaw)
    return (f / psd.f_b_1)^(psd.α_1) / (1 + (f / psd.f_b_1)^(psd.α_1 - psd.α_2)) / (1 + (f / psd.f_b_2)^(psd.α_2 - psd.α_3))

end

function calculate_psd(f, psd::SimpleBendingPowerLaw)
    return (f / psd.f_b)^(psd.α_1) / (1 + (f / psd.f_b)^(psd.α_1 - psd.α_2))
end


function get_normalised_psd(psd_model::PowerSpectralDensity, spectral_points::AbstractVector{<:Real})
    psd_zero = calculate_psd(spectral_points[1], psd_model)
    # create the normalised psd
    psd_normalised = calculate_psd.(spectral_points, Ref(psd_model)) / psd_zero
    psd_normalised
end


function build_approx(J::Int64, f0::Real, fM::Real)
    spectral_points = zeros(J)
    spectral_matrix = zeros(J, J)
    return init_psd_decomp(spectral_points, spectral_matrix, J, f0, fM)
end

function init_psd_decomp(spectral_points::AbstractVector{<:Real}, spectral_matrix::AbstractMatrix{<:Real}, J::Int64, f0::Real, fM::Real, basis_function::String="SHO")
    """
    Initialise the spectral points and the spectral matrix
    """

    # create the spectral_points
    for j in 0:J-1
        spectral_points[j+1] = f0 * (fM / f0)^(j / (J - 1))
    end

    # fill the spectral matrix
    if basis_function == "SHO"
        for j in 1:J
            for k in 1:J
                spectral_matrix[j, k] = 1 / (1 + (spectral_points[j] / spectral_points[k])^4)
            end
        end
    elseif basis_function == "DRWSHO"
        for j in 1:J
            for k in 1:J
                spectral_matrix[j, k] = 1 / (1 + (spectral_points[j] / spectral_points[k])^6)
            end
        end
    else
        error("Basis function" * basis_function * "not implemented")
    end
    return spectral_points, spectral_matrix
end

function psd_decomp(psd_normalised::AbstractVector{<:Real}, spectral_matrix::AbstractMatrix{<:Real})
    """
    Decompose the psd into the components
    """
    # amplitudes of the components
    amplitudes = spectral_matrix \ psd_normalised
    return amplitudes
end


"""
Approximate the PSD with a sum of SHO functions
This is essentially to check that the model and the approximation are consistent.

    f: frequency vector
    psd_model: the model of the PSD
    f0: the lowest frequency
    fM: the highest frequency
    n_components: the number of components to use
    var: the variance of the process
    basis_function: the basis function to use

    
"""
function approximated_psd(f, psd_model::PowerSpectralDensity, f0::Real, fM::Real; n_components::Int64=20, var::Real=1.0, basis_function::String="SHO")
    spectral_points, spectral_matrix = build_approx(n_components, f0, fM)

    psd_normalised = get_normalised_psd(psd_model, spectral_points)
    amplitudes = psd_decomp(psd_normalised, spectral_matrix)

    psd = zeros(length(f))
    if basis_function == "SHO"
        for i in 1:n_components
            psd += amplitudes[i] * var ./ (1 .+ (f ./ spectral_points[i]) .^ 4)
        end
    elseif basis_function == "DRWSHO"
        for i in 1:n_components
            psd += amplitudes[i] * var ./ (1 .+ (f ./ spectral_points[i]) .^ 6)
        end
    else
        error("Basis function" * basis_function * "not implemented")
    end

    return psd
end

function approx(psd_model::PowerSpectralDensity, f0::Real, fM::Real, n_components::Int64=20, var::Real=1.0, basis_function::String="SHO")

    spectral_points, spectral_matrix = build_approx(n_components, f0, fM)

    psd_normalised = get_normalised_psd(psd_model, spectral_points)
    amplitudes = psd_decomp(psd_normalised, spectral_matrix)

    for i in 1:n_components
        amplitudes[i] *= spectral_points[i]
    end
    variance = sum(amplitudes)

    if basis_function == "SHO"
        covariance = SHO(var * amplitudes[1] / variance, 2π * spectral_points[1], 1 / √2)
        for i in 2:n_components
            covariance += SHO(var * amplitudes[i] / variance, 2π * spectral_points[i], 1 / √2)
        end
        # else if basis_function == "DRWSHO"
        #     covariance = SHO(var * amplitudes[1] / variance, 2π * spectral_points[1], 1 / √2)
        #     for i in 2:n_components
        #         covariance += SHO(var * amplitudes[i] / variance, 2π * spectral_points[i], 1 / √2)
        #     end
    else
        error("Basis function" * basis_function * "not implemented")
    end

    return covariance
end