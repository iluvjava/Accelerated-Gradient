### ============================================================================
### The elementwise absolute value with a multiplier
### ============================================================================
"""
## It is
A struct that refers to the function x -> m*abs(x). It is also convex so m 
cannot be negative. 
### Fields
- `multiplier`: `m` the multiplier on every entries of the one norm, has to be 
non-negative. 
"""
struct MAbs <: NonsmoothFxn
    multiplier::Real

    function MAbs(multiplier::Real=1)
        @assert multiplier >= 0 "The multiplier for the elementwise nonsmooth"*
            "function must be non-negative so it's convex for the prox operator. "
        return new(multiplier)
    end
end


"""
The output of the elementwise absolute value on the function. 
"""
function (this::MAbs)(arg::AbstractArray{T}) where {T<:Number}
    return this.multiplier*abs.(arg)|>sum
end


"""
returns Argmin_u(this(u) + (1/(2t))‖x - u‖^2), soft thresholding here. 
"""
function (this::MAbs)(t::T1, x::AbstractArray{T2}) where {T1 <: Number, T2 <: Number}
    @assert t > 0 "The prox constant for the prox operator has to be a strictly positive real, "*
    "however we get t = $(t)"
    λ = this.multiplier
    T(z) = sign(z)*max(abs(z) - t*λ, 0)    
    return T.(x)
end



"""
Evalue the prox of the type AbsValue with a constaint t at the point x. 

### Arguments
- `this::OneNorm`: The type the function acts on. 
- `t::T1`: The scalar for the proximal operator. 
- `x::AbstractArray{T2}`: The point we are querying the prox of the one norm of. 
### Argument Type Parameters
- `T1<:Number`
- `T2<:Number`
"""
function prox(this::MAbs, t::T1, x::AbstractArray{T2}) where {T1 <: Number, T2 <: Number}
    return this(t, x)
end


"""
A multplier function for the abslute value type multiplied with a strictly positive number. 
"""
function Base.:*(m::Real, this::MAbs)
    return MAbs(m*this.multiplier)
end

