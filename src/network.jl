#=
network.jl
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits many desirable features.
=#
#=
Original Python version: Copyright © 2012-2018 Michael Nielsen
Rewritten Julia version: Copyright © 2018 Derik Kauffman
Licensed under MIT license
=#

#= In Julia, functions do not "belong" to any particular class.
Therefore, the functions implemented here take a Network struct as
input labeled as `self` rather than as a field of `self` (e.g. `self.SGD`)
=#

struct Network
	num_layers::Int
	sizes::Vector{Int}
	biases::Vector{Vector{Real}}
	weights::Vector{Matrix{Real}}

	# Original docstrings and comments have been left in place.
	"""The list `sizes` contains the number of neurons in the
	respective layers of the network. For example, if the list
	was [2, 3, 1] then it would be a three-layer network, with the
	first layer containing 2 neurons, the second layer 3 neurons,
	and the third layer 1 neuron. The biases and weights for the
	network are initialized randomly, using a Gaussian
	distribution with mean 0, and variance 1. Note that the first
	layer is assumed to be an input layer, and by convention we
	won't set any biases for those neurons, since biases are only
	ever used in computing the outputs from later layers."""
	function Network(sizes) # Do I need to make this take self as input?
		num_layers = length(sizes)
		#sizes = sizes
		biases = [randn(y) for y in sizes[2:end]]
		weights = [randn(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
		new(num_layers, sizes, biases, weights)
	end
end

"""Return the output of the network if `a` is input."""
function feedforward(self::Network, a)
	for (b, w) = zip(self.biases, self.weights)
		a = σ.(w * a + b)
	end
	return a
end


"""Train the neural network using mini-batch stochastic gradient
descent. The `training_data` is a list of tuples `(x, y)`
representing the training inputs and the desired outputs.
The other non-optional parameters are self-explanatory.
If `test_data` is provided then the network will be evaluated
against the test data after each epoch, and partial progress
printed out. This is useful for tracking progress, but slows things
down substantially."""
function SGD(self::Network, training_data, epochs::Int,
	  mini_batch_size::Int, η::Real, test_data=nothing)
	if test_data != nothing n_test = length(test_data) end
	n = length(training_data)
	for j in 1:epochs
		shuffle!(training_data) # Shuffle an index vector instead?
		mini_batches = [training_data[k:k+mini_batch_size-1]
		  for k in 1:mini_batch_size:n-mini_batch_size]

		for mini_batch in mini_batches
			update_mini_batch!(self, mini_batch, η)
		end

		if test_data != nothing
			println("Epoch $j: $(evaluate(self, test_data)) / $n_test")
		else
			println("Epoch $j complete")
		end
	end
end

"""Update the network's weights and biases by applying
gradient descent using backpropagation to a single mini batch.
The `mini_batch` is a list of tuples `(x, y)`, and `η`
is the learning rate."""
function update_mini_batch!(self::Network, mini_batch, η::Real)
	∇b = [zeros(b) for b in self.biases]
	∇w = [zeros(w) for w in self.weights]
	for (x, y) in mini_batch
		Δ∇b, Δ∇w = backprop(self, x, y)
		∇b .+= Δ∇b
		∇w .+= Δ∇w
		# (∇b, ∇w) .+= backprop(self, x, y)
		# (this doesn't work since you can't add tuples)
	end
	self.weights .-= [η * ∇w/length(mini_batch)] # Eq. 20
	self.biases  .-= [η * ∇b/length(mini_batch)] # Eq. 21
end

"""Return a tuple `(∇b, ∇w)` representing the gradient for the
cost function Cₓ. `∇b` and `∇w` are layer-by-layer lists of arrays,
similar to `self.biases` and `self.weights`."""
function backprop(self::Network, x, y)
	∇b = [zeros(b) for b in self.biases]
	∇w = [zeros(w) for w in self.weights]
	println(∇b)
	#feedforward
	activation = x
	activations = [x] # list to store all the activations, layer by layer
	zs = [] # list to store all the z vectors, layer by layer
	for (b, w) = zip(self.biases, self.weights)
		z = w * a + b
		append!(zs, z)
		activation = σ(z)
		append!(activations, activation)
	end
	# backward pass
	Δ = cost_derivative(activations[end], y) * σ′(zs[end])
	∇b[end] = Δ
	∇w[end] = Δ * activations[end-1]'
	#println(∇b)

	# Note that the variable l in the loop below is used a little
	# differently to the notation in Chapter 2 of the book. Here,
	# l = 1 means the last layer of neurons, l = 2 is the
	# second-last layer, and so on.
	# There is almost certainly a bug with zero/one indexing here.
	for l in 1:self.num_layers
		z = zs[end-l+1]
		sp = σ′(z)
		Δ = (self.weights[end-l+1]' * Δ) * sp
		∇b[end-l+1] = Δ
		∇w[end-l] = Δ * activations[end-l]'
	end
	return (∇b, ∇w)
end

"""Return the number of test inputs for which the neural
network outputs the correct result. Note that the neural
network's output is assumed to be the index of whichever
neuron in the final layer has the highest activation."""
function evaluate(self::Network, test_data)
	test_results = [(indmax(feedforward(self, x)), y) for (x, y) in test_data]
	return count((x==y) for (x,y) in test_results)
end

"""Return the vector of partial derivatives ∂Cₓ/∂a for the output activations."""
function cost_derivative(self::Network, output_activations, y)
	return output_activations .- y
end

#### Miscellaneous functions
#The sigmoid function.
σ(z) = 1/(1 + exp(-z))

#Derivative of the sigmoid function.
σ′(z) = σ(z) * (1-σ(z))

mynetwork = Network([2,3,2])
#println(mynetwork)

println("Weights: ", mynetwork.weights)
println("Biases: ", mynetwork.biases)
#println("Zip: ", first(zip(mynetwork.biases, mynetwork.weights)))
println("Feedforward: ", feedforward(mynetwork, [0.2, 0.4]))
