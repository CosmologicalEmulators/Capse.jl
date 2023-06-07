module Capse

using Base: @kwdef
using AbstractEmulator
using FastChebInterp
#using LoopVectorization #investigate wether it is better to use Tullio

abstract type AbstractCℓEmulators end

@kwdef mutable struct CℓEmulator <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    ℓgrid::Array
    InMinMax::Matrix
    OutMinMax::Array
    PolyGrid::Matrix
    ChebDegree::Int=256
end
#TODO: consider an SMatrix for polygrid

function get_Cℓ(input_params::Array{T}, Cℓemu::CℓEmulator) where {T}
    chebcoefs = get_chebcoefs(input_params, Cℓemu)
    #Cls = zeros(T, length(Cℓemu.ℓgrid))
    #matvecmul!(Cls, Cℓemu.PolyGrid, chebcoefs)
    #return Cls
    return Cℓemu.PolyGrid * chebcoefs
end

#function matvecmul!(C::Array{T}, A::Matrix, B::Array) where {T}
#    @turbo for m ∈ axes(A,1)
#        Cm = zero(eltype(B))
#        for k ∈ axes(A,2)
#            Cm += A[m,k] * B[k]
#        end
#        C[m] = Cm
#    end
#end

function get_chebcoefs(input_params, Cℓemu::CℓEmulator)
    input = deepcopy(input_params)
    maximin_input!(input, Cℓemu.InMinMax)
    chebcoefs = Vector(run_emulator(input, Cℓemu.TrainedEmulator))
    inv_maximin_output!(chebcoefs, Cℓemu.OutMinMax)
    return chebcoefs .* exp(input_params[1]-3.)
end

function get_ℓgrid(Cℓemu::CℓEmulator)
    return Cℓemu.ℓgrid
end

function eval_polygrid!(Cl::CℓEmulator)
    cb = FastChebInterp.ChebPoly(zeros(Cl.ChebDegree),
                                 FastChebInterp.SVector{1}([Cl.ℓgrid[begin]]),
                                 FastChebInterp.SVector{1}([Cl.ℓgrid[end]]))
    Cl.PolyGrid = eval_polygrid(cb, Cl.ℓgrid)

    return nothing
end

function eval_polygrid(cb::FastChebInterp.ChebPoly, l)
    n = length(cb.coefs)
    grid = zeros(length(l), n)

    for i in 1:n
        base_coefs = zero(cb)
        base_coefs.coefs[i] = 1
        grid[:,i] = base_coefs.(l)
    end
    return grid
end

end # module
