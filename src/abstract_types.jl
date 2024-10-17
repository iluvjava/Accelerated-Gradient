## =====================================================================================================================
## SMOOTH AND NON-SMOOTH FUNCTION
## =====================================================================================================================

"""
## It is: 
An abstract type. A function mapping from R^n to R. Infinity allowed but Nan is not allowed. 
## It has: 
1. Gradient at its domain. 
2. Function value at its domain. 
"""
abstract type Fxn


end

function (this::Fxn)(::AbstractArray{T})::Number where {T<:Number}
    throw(
        ErrorException(
            "Abstract type function get overloaded, method not yet implemented for $(type(this)). "+
            "If this is unexpected, please imeplement this function for your type. "
            )
    )
end


"""
## It is: 
An abstract type, a super type of `Fxn`. 
A type the models non-smooth functions. 
## It has: 
* 1. Can be evaluated at spoint poitn in R^n to obtain extended real value of the function at the point. 
* 2. Has a proximal operator for all points in R^n. 
"""
abstract type NonsmoothFxn <: Fxn

end


function prox(this::NonsmoothFxn, arg::AbstractArray{T}):: AbstractArray where {T <: Number} 
    throw(
        ErrorException(
            "Abstract type function get overloaded, method not yet implemented for $(type(this)). "+
            "If this is unexpected, please imeplement this function for your type. "
        )
    )
end




"""
## It is: 
An abstract type, a super type of Fxn. 
It represent a differentiable function with a gradient oracle. 
## It has: 
* Ask for the value at some point.
* Has a gradient. 
* Or it has a prox operator. 
"""
abstract type SmoothFxn <: Fxn

end


function grad(this::NonsmoothFxn, arg::AbstractArray{T}) :: AbstractArray where {T <: Number}
    throw(
        ErrorException(
            "Abstract type function get overloaded, method not yet implemented for $(type(this)). "+
            "If this is unexpected, please imeplement this function for your type. "
        )
    )
end



function prox(this::NonsmoothFxn, arg::AbstractArray{T}):: AbstractArray where {T <: Number} 
    throw(
        ErrorException(
            "Abstract type function get overloaded, method not yet implemented for $(type(this)). "+
            "If this is unexpected, please imeplement this function for your type. "
        )
    )
end


## =====================================================================================================================
## RESULT COLLECTOR INTERFACE
## =====================================================================================================================