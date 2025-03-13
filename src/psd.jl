using Pioran: SHO, Exp, Celerite

abstract type Model end
abstract type PowerSpectralDensity <: Model end
abstract type BendingPowerLaw <: PowerSpectralDensity end

@doc raw"""
     PowerLaw(α)

Power law model for the power spectral density

- `α`: the power law index

```math
\mathcal{P}(f) =  (f)^{-α}
```

"""
struct PowerLaw{T <: Real} <: PowerSpectralDensity
    α::T
end

function calculate(f, psd::PowerLaw)
    return (f)^(-psd.α)
end

@doc raw"""
     SingleBendingPowerLaw(α₁, f₁, α₂)

Single bending power law model for the power spectral density

- `α₁`: the first power law index
- `f₁`: the first bend frequency
- `α₂`: the second power law index

```math
\mathcal{P}(f) =  \frac{(f/f₁)^{-α₁}}{1 + (f / f₁)^{α₂ - α₁}}
```

"""
struct SingleBendingPowerLaw{T <: Real} <: BendingPowerLaw
    α₁::T
    f₁::T
    α₂::T
end

@doc raw"""
     DoubleBendingPowerLaw(α₁, f₁, α₂, f₂, α₃)

Double bending power law model for the power spectral density

- `α₁`: the first power law index
- `f₁`: the first bend frequency
- `α₂`: the second power law index
- `f₂`: the second bend frequency
- `α₃`: the third power law index

```math
\mathcal{P}(f) =  \frac{(f/f₁)^{-α₁}}{1 + (f / f₁)^{α₂ - α₁}}\frac{1}{1 + (f / f₂)^{α₃ - α₂}}
```
"""
struct DoubleBendingPowerLaw{T <: Real} <: BendingPowerLaw
    α₁::T
    f₁::T
    α₂::T
    f₂::T
    α₃::T
end

@doc raw"""
     DoubleBendingPowerLaw_Bis(α₀, f₁, Δα₁, Δf, Δα₂)

    Double bending power law model for the power spectral density

- `α₀`: the first power law index
- `f₁`: the first bend frequency
- `Δα₁`: the first difference in power law index
- `Δf`: scale for the second bend frequency, `f₂ = f₁ * Δf`
- `Δα₂`: the second difference in power law index

```math
\mathcal{P}(f) =  \frac{(f/f₁)^{-α_0}}{1 + (f / f₁)^{α_0+\Delta α₁}}\frac{1}{1 + (f / f₁ \Delta f)^{\Delta α₁ + \Delta α₂}}
```
"""
struct DoubleBendingPowerLaw_Bis{T <: Real} <: BendingPowerLaw
    α₀::T
    f₁::T
    Δα₁::T
    Δf::T
    Δα₂::T
end

struct TripleBendingPowerLaw{T <: Real} <: BendingPowerLaw
    α₁::T
    f₁::T
    α₂::T
    f₂::T
    α₃::T
    f₃::T
    α₄::T
end

""" calculate(f, psd::PowerSpectralDensity)

    Calculate the power spectral density at frequency f
"""
function calculate(f, psd::DoubleBendingPowerLaw_Bis)
    return (f / psd.f₁)^(-psd.α₀) / (1 + (f / psd.f₁)^(psd.α₀ + psd.Δα₁)) / (1 + (f / (psd.f₁ * psd.Δf))^(psd.Δα₁ + psd.Δα₂))

end

function calculate(f, psd::DoubleBendingPowerLaw)
    return (f / psd.f₁)^(-psd.α₁) / (1 + (f / psd.f₁)^(psd.α₂ - psd.α₁)) / (1 + (f / (psd.f₂))^(psd.α₃ - psd.α₂))

end

function calculate(f, psd::SingleBendingPowerLaw)
    return (f / psd.f₁)^(-psd.α₁) / (1 + (f / psd.f₁)^(psd.α₂ - psd.α₁))
end

function calculate(f, psd::TripleBendingPowerLaw)
    return (f / psd.f₁)^(-psd.α₁) / (1 + (f / psd.f₁)^(psd.α₂ - psd.α₁)) / (1 + (f / psd.f₂)^(psd.α₃ - psd.α₂)) / (1 + (f / psd.f₃)^(psd.α₄ - psd.α₃))
end

(psd::PowerSpectralDensity)(f) = calculate.(f, Ref(psd))

"""
     get_normalised_psd(psd_model::PowerSpectralDensity, spectral_points::AbstractVector{<:Real})

Get the PSD normalised at the lowest frequency
"""
function get_normalised_psd(psd_model::PowerSpectralDensity, spectral_points::AbstractVector{<:Real})
    psd_zero = calculate(spectral_points[1], psd_model)
    psd_normalised = calculate.(spectral_points, Ref(psd_model)) / psd_zero
    return psd_normalised
end

"""
     build_approx(J, f0, fM; basis_function="SHO")

Prepare the approximation of a PSD

# Arguments
- `J::Integer`: the number of basis functions
- `f0::Real`: the lowest frequency
- `fM::Real`: the highest frequency
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"

# Return
- `spectral_points::Vector{Real}`: the spectral points
- `spectral_matrix::Matrix{Real}`: the spectral matrix
"""
function build_approx(J::Int64, f0::Real, fM::Real; basis_function::String = "SHO")
    spectral_points = zeros(J)
    spectral_matrix = zeros(J, J)
    return init_psd_decomp!(spectral_points, spectral_matrix, J, f0, fM, basis_function = basis_function)
end

function init_psd_decomp!(spectral_points::AbstractVector{<:Real}, spectral_matrix::AbstractMatrix{<:Real}, J::Int64, f0::Real, fM::Real; basis_function::String = "SHO")
    # create the spectral_points
    for j in 0:(J - 1)
        spectral_points[j + 1] = f0 * (fM / f0)^(j / (J - 1))
    end

    # fill the spectral matrix
    if basis_function == "SHO"
        for j in 1:J
            for k in 1:J
                spectral_matrix[j, k] = 1 / (1 + (spectral_points[j] / spectral_points[k])^4)
            end
        end
    elseif basis_function == "DRWCelerite"
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

"""
     psd_decomp(psd_normalised, spectral_matrix)

Get amplitudes of the basis functions by solving the linear system of the approximation
"""
function psd_decomp(psd_normalised::AbstractVector{<:Real}, spectral_matrix::AbstractMatrix{<:Real})
    amplitudes = spectral_matrix \ psd_normalised
    return amplitudes
end

"""
     get_approx_coefficients(psd_model, f0, fM; n_components=20, basis_function="SHO")

Get the coefficients of the approximated PSD

# Arguments
- `psd_model::PowerSpectralDensity`: model of the PSD
- `f0::Real`: the lowest frequency
- `fM::Real`: the highest frequency
- `n_components::Integer=20`: the number of basis functions to use
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"

# Return
- `amplitudes::Vector{Real}`: the amplitudes of the basis functions
"""
function get_approx_coefficients(psd_model::PowerSpectralDensity, f0::Real, fM::Real; n_components::Int64 = 20, basis_function::String = "SHO")
    spectral_points, spectral_matrix = build_approx(n_components, f0, fM, basis_function = basis_function)

    psd_normalised = get_normalised_psd(psd_model, spectral_points)
    amplitudes = psd_decomp(psd_normalised, spectral_matrix)
    return amplitudes
end

"""
     approximated_psd(f, psd_model, f0, fM; n_components=20, var=1.0, basis_function="SHO")

Return the approximated PSD. This is essentially to check that the model and the approximation are consistent.

# Arguments
- `f::AbstractVector{<:Real}`: the frequencies at which to calculate the PSD
- `psd_model::PowerSpectralDensity`: model of the PSD
- `f0::Real`: the lowest frequency
- `fM::Real`: the highest frequency
- `n_components::Integer=20`: the number of basis functions to use
- `var::Real=1.0`: the variance of the process, integral of the PSD
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"
- `individual::Bool=false`: return the individual components
"""
function approximated_psd(f, psd_model::PowerSpectralDensity, f0::Real, fM::Real; n_components::Int64 = 20, var::Real = 1.0, basis_function::String = "SHO", individual = false)
    spectral_points, spectral_matrix = build_approx(n_components, f0, fM, basis_function = basis_function)
    psd_normalised = get_normalised_psd(psd_model, spectral_points)
    amplitudes = psd_decomp(psd_normalised, spectral_matrix)

    if individual
        psd = zeros(length(f), n_components)
        if basis_function == "SHO"
            for i in 1:n_components
                psd[:, i] = amplitudes[i] * var ./ (1 .+ (f ./ spectral_points[i]) .^ 4)
            end
        elseif basis_function == "DRWCelerite"
            for i in 1:n_components
                psd[:, i] = amplitudes[i] * var ./ (1 .+ (f ./ spectral_points[i]) .^ 6)
            end
        else
            error("Basis function" * basis_function * "not implemented")
        end
    else
        psd = zeros(length(f))
        if basis_function == "SHO"
            for i in 1:n_components
                psd += amplitudes[i] * var ./ (1 .+ (f ./ spectral_points[i]) .^ 4)
            end
        elseif basis_function == "DRWCelerite"
            for i in 1:n_components
                psd += amplitudes[i] * var ./ (1 .+ (f ./ spectral_points[i]) .^ 6)
            end
        else
            error("Basis function" * basis_function * "not implemented")
        end
    end
    return psd
end

"""
     approx(psd_model, f0, fM, n_components=20, var=1.0; basis_function="SHO")

Approximate the PSD with a sum of basis functions to form a covariance function

# Arguments
- `psd_model::PowerSpectralDensity`: model of the PSD
- `f0::Real`: the lowest frequency
- `fM::Real`: the highest frequency
- `n_components::Integer=20`: the number of basis functions to use
- `var::Real=1.0`: the variance of the process, integral of the PSD
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"

# Return
- `covariance::SumOfSemiSeparable`: the covariance function

# Example
```julia
using Pioran
𝓟 = SingleBendingPowerLaw(1.0, 1.0, 2.0)
𝓡 = approx(𝓟, 1e-4, 1e-1, 30, 2.31,basis_function="SHO")
```

"""
function approx(psd_model::PowerSpectralDensity, f0::Real, fM::Real, n_components::Int64 = 20, var::Real = 1.0; sample_var = nothing, f_min = nothing, f_max = nothing, basis_function::String = "SHO")

    spectral_points, spectral_matrix = build_approx(n_components, f0, fM, basis_function = basis_function)

    psd_normalised = get_normalised_psd(psd_model, spectral_points)
    amplitudes = psd_decomp(psd_normalised, spectral_matrix)

    if !isnothing(sample_var)
        var = sample_var
        if basis_function == "SHO"
            variance = integral(amplitudes, spectral_points, f_min, f_max)
        elseif basis_function == "DRWCelerite"
            variance = integral2(amplitudes, spectral_points, f_min, f_max) * 2
        end
    else
        if basis_function == "SHO"
            variance = sum(amplitudes .* spectral_points) #* π / √2 removed as it is also in the expression of a
        elseif basis_function == "DRWCelerite"
            variance = sum(amplitudes .* spectral_points) * 2π / 3
        end
    end

    if basis_function == "SHO"

        a = amplitudes .* spectral_points * var / variance #  * π / √2 removed as it is also in the expression of a
        c = √2 * π .* spectral_points

        covariance = SumOfCelerite(a, a, c, c)

    elseif basis_function == "DRWCelerite"

        a = amplitudes .* spectral_points * π / 3 * var / variance
        b = √3 * a
        c = π * spectral_points
        d = √3 * c

        aa = [a; a]
        bb = [b; zeros(n_components)]
        cc = [c; 2 * c]
        dd = [d; zeros(n_components)]
        covariance = SumOfCelerite(aa, bb, cc, dd)
        # covariance = Celerite(a[1], b[1], c[1], d[1]) + Exp(a[1], 2 * c[1])
        # for i in 2:n_components
        #     covariance += Celerite(a[i], b[i], c[i], d[i]) + Exp(a[i], 2 * c[i])
        # end
    else
        error("Basis function" * basis_function * "not implemented")
    end

    return covariance
end

function integral(a, c, ap, cp)
    return sum(
        (1 / (4sqrt(2))) * a .* c .* (
            2 * atan.(1 .- (sqrt(2) * ap) ./ c) .-
                2atan.(1 .+ (sqrt(2) * ap) ./ c) .- 2atan.(1 .- (sqrt(2) .* cp) ./ c) .+
                2atan.(1 .+ (sqrt(2) * cp) ./ c) .+ log.(ap^2 .- sqrt(2) .* ap .* c .+ c .^ 2) .-
                log.(ap^2 .+ sqrt(2) * ap * c .+ c .^ 2) .- log.(c .^ 2 .- sqrt(2) .* c .* cp .+ cp^2) .+
                log.(c .^ 2 .+ sqrt(2) .* c * cp .+ cp^2)
        )
    )
end

function integral2(a, c, ap, cp)
    return sum(
        a .* c .* (
            2atan.(sqrt(3) .- 2ap ./ c) .-
                2atan.(sqrt(3) .+ 2ap ./ c) .- 4atan.(ap ./ c) .+ 4atan.(cp ./ c) .-
                2atan.(sqrt(3) .- 2cp ./ c) .+ 2atan.(sqrt(3) .+ 2cp ./ c) .+
                sqrt(3) * log.(ap^2 .- sqrt(3) * ap * c .+ c .^ 2) .-
                sqrt(3) * log.(ap^2 .+ sqrt(3) * ap * c .+ c .^ 2) .-
                sqrt(3) * log.(c .^ 2 .- sqrt(3) * c * cp .+ cp^2) .+
                sqrt(3) * log.(c .^ 2 .+ sqrt(3) * c * cp .+ cp^2)
        ) ./ 12
    )
end
