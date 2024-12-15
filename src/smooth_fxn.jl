

using Zygote  # for automatic differentiation. 

"""
‖b - Ax‖^2, where b is specifically, a vector. 
"""
mutable struct SquareNormResidual <: SmoothFxn
    A::AbstractMatrix
    b::AbstractVector
    ATA::AbstractMatrix # <-- For faster derivative evaluation. 

    function SquareNormResidual(A::AbstractMatrix, b::AbstractVector)
        this = new(A, b)
        this.ATA = A'*A
        return this
    end

end


"""
Returns ‖Ax - b‖^2/2
"""
function (this::SquareNormResidual)(x::AbstractVector{T}) where {T <: Number}
    return dot(this.A*x - this.b, this.A*x - this.b)/2
end


"""
Returns the gradient of the 2 norm residual function. 

"""
function grad(this::SquareNormResidual, x::AbstractVector{T}) where {T <: Number}
    A = this.A; 
    b = this.b
    ATA = this.ATA
    # A'*(A*x - b)
    return ATA*x - A'*b
end





#### ===================================================================================================================

"""
Function evaluates to `dot(x, A*x)/2 + dot(b, x)+ c`
"""
mutable struct Quadratic <: SmoothFxn
    "Squared matrix. "
    A::AbstractMatrix
    "A vector. "
    b::AbstractVector
    "A constant offset for the quadratic function. "
    c::Number

    function Quadratic(A::AbstractMatrix, b::AbstractVector, c::Number)::Quadratic
        @assert size(A, 1) == size(A, 2) "Type `Quadratic` smooth function requires a squared matrix `A`, but instead we got "*
        "size(A) = $(size(A)). "
        @assert size(A, 1) == size(b, 1) "Type `Quadratic has unmathced dimension between matrix `A` and vector constant `b`. "
        this = new(A, b, c)
        return this 
    end
end


function (this::Quadratic)(x::AbstractVector)
    A, b, c = this.A, this.b, this.c
    return 0.5*dot(x, A*x) + dot(b, x) + c
end


function grad(this::Quadratic, x::AbstractVector)
    A, b, _ = this.A, this.b, this.c
    @assert length(x) == length(b) "`x` passed to Grad of `::Quadratic` has the wrong dimension"    
    return 0.5*(A + A')*x + b
end


### Logistic Loss

mutable struct LogitLoss <: SmoothFxn
    
    A::AbstractMatrix
    b::AbstractVector
    fxn::Function

    function LogitLoss(
        A::AbstractMatrix, 
        b::AbstractVector, 
        lambda::Number=0
    )
        this = new()
        this.A = A 
        this.b = b
        # this.fxn = (x) ->  sum(log.(1 .+ exp.(A*x))) - dot(b, A*x) + lambda*dot(x, x)/2
        function _fxn(x)
            Ax = A*x
            _part1 = @. log(1 + exp(Ax)) - b*Ax
            _part2 = lambda*dot(x, x)/2
            return sum(_part1) + _part2
        end
        this.fxn = _fxn
        return this
    end
end

function (this::LogitLoss)(x::AbstractArray{T}) where T<: Number
    return this.fxn(x)
end


function grad(this::LogitLoss, x::AbstractVector)::AbstractVector
    return gradient(this.fxn, x)[1]
end


