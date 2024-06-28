"""
Can be evaluated at some point and returns a fixed value for each evaluated point. 
"""
abstract type Fxn

end

"""
A type the models non smooth functions, it has a simple prox operator to it. 
* Query the value at some point. 
* Has a prox operator 
"""
abstract type NonsmoothFxn <: Fxn

end


"""
A type the models smooth functions. 
* Ask for the value at some point.
* Has a gradient. 
* Has a prox operator (optional)
"""
abstract type SmoothFxn <: Fxn

end


