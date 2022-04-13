#=
network.jl
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits many desirable features.

Original Python code: Copyright © 2012-2022 Michael Nielsen
Julia reimplementation: Copyright © 2018-2022 Derik Kauffman
Licensed under MIT license
=#

using Random

struct Network
	sizes::Vector{Int}
	biases::Vector{Vector{Float32}}
	weights::Vector{Matrix{Float32}}
end

"""
	Network(sizes::Vector{<:Integer})

The list `sizes` contains the number of neurons in the
respective layers of the network. For example, if the list
was [2, 3, 1] then it would be a three-layer network, with the
first layer containing 2 neurons, the second layer 3 neurons,
and the third layer 1 neuron. The biases and weights for the
network are initialized randomly, using a Gaussian
distribution with mean 0 and variance 1. Note that the first
layer is assumed to be an input layer, and by convention we
won't set any biases for those neurons, since biases are only
ever used in computing the outputs from later layers.
"""
function Network(sizes::Vector{<:Integer})
	biases = randn.(sizes[2:end])
	weights = randn.(zip(sizes[2:end], sizes[1:end-1]))
	Network(sizes, biases, weights)
end

"""
	feedforward(net::Network, a::AbstractVector{<:Real})

Return the output of the network if `a` is input.
"""
function feedforward(net::Network, a::AbstractVector{<:Real})
	for (b, w) = zip(net.biases, net.weights)
		a = σ.(w * a .+ b)
	end
	return a
end

"""
The neural network's output is assumed to be the index of whichever
neuron in the final layer has the highest activation.
"""
classify(net::Network, a::AbstractVector{<:Real}) = argmax(feedforward(net, a)) - 1

"""
Train the neural network using mini-batch stochastic gradient
descent. The `training_data` is a list of tuples `(x, y)`
representing the training inputs and the desired outputs.
The other non-optional parameters are self-explanatory.
If `test_data` is provided then the network will be evaluated
against the test data after each epoch, and partial progress
printed out. This is useful for tracking progress, but slows things
down substantially.
"""
function SGD!(net::Network, training_data, epochs::Integer,
	  mini_batch_size::Integer, η::Real; test_data=nothing)
	if test_data !== nothing n_test = length(test_data) end
	for j in 1:epochs
		shuffle!(training_data)
		mini_batches = reshape(training_data, :, mini_batch_size)
		for mini_batch in eachrow(mini_batches)
			update_mini_batch!(net, mini_batch, η)
		end

		if test_data !== nothing
			println("Epoch $j: $(evaluate(net, test_data)) / $n_test")
		else
			println("Epoch $j complete")
		end
	end
end

"""
	update_mini_batch!(net::Network, mini_batch, η::Real)

Update the network's weights and biases by applying
gradient descent using backpropagation to a single mini batch.
The `mini_batch` is a list of tuples `(x, y)`, and `η`
is the learning rate.
"""
function update_mini_batch!(net::Network, mini_batch, η::Real)
	∇b = zero.(net.biases)
	∇w = zero.(net.weights)
	for (x, y) in mini_batch
		Δ∇b, Δ∇w = backprop(net, x, y)
		∇b .+= Δ∇b
		∇w .+= Δ∇w
	end
	net.biases  .-= Float32(η / length(mini_batch)) .* ∇b # Eq. 21
	net.weights .-= Float32(η / length(mini_batch)) .* ∇w # Eq. 20
end

"""
	backprop(net::Network, x, y)

Return a tuple `(∇b, ∇w)` representing the gradient for the
cost function Cₓ. `∇b` and `∇w` are layer-by-layer lists of arrays,
similar to `net.biases` and `net.weights`.
"""
function backprop(net::Network, x, y)
	∇b = zero.(net.biases)
	∇w = zero.(net.weights)

	# feedforward
	activation = x
	activations = [x] # list to store all the activations, layer by layer
	zs = typeof(x)[]  # list to store all the z vectors, layer by layer
	for (b, w) = zip(net.biases, net.weights)
		z = w * activation .+ b
		push!(zs, z)
		activation = σ.(z)
		push!(activations, activation)
	end

	# backward pass
	δ = cost_derivative(activations[end], y) .* σ′.(zs[end]) # Eq. BP1
	∇b[end] = δ
	∇w[end] = δ * activations[end-1]'

	for l in length(net.sizes)-1:2
		z = zs[l]
		δ = (net.weights[l+1]' * δ) .* σ′.(z) # Eq. BP2
		∇b[l] = δ # Eq. BP3
		∇w[l] = δ * activations[l-1]' # Eq. BP4
	end
	return ∇b, ∇w
end

"""
	evaluate(net::Network, test_data)

Return the number of test inputs for which the neural
network outputs the correct result.
"""
function evaluate(net::Network, test_data)
	return count((classify(net, x) == y) for (x, y) in test_data)
end

"""
	cost_derivative(output_activations, y)

Return the vector of partial derivatives ∂Cₓ/∂a for the output activations.
"""
cost_derivative(output_activations, y) = output_activations .- onehot(y)

#### Miscellaneous functions
"""The sigmoid function."""
σ(z) = 1/(1 + exp(-z))

"""Derivative of the sigmoid function."""
σ′(z) = σ(z) * (1 - σ(z))

"""
	onehot(j::Integer)

Return a 10-dimensional unit vector with a 1 in the jth
position and zeros elsewhere. This is used to convert a digit (0-9)
into a corresponding desired output from the neural network.
"""
function onehot(j::Integer)
	e = falses(10)
	e[j+1] = true
	return e
end

# MNIST loading code
using MLDatasets

function load_data()
	training_inputs = reshape(MNIST.traintensor(Float32), 784, :)
	training_labels = MNIST.trainlabels()
	training_data = zip(eachcol(training_inputs), training_labels)

	test_inputs = reshape(MNIST.testtensor(Float32), 784, :)
	test_labels = MNIST.testlabels()
	test_data = zip(eachcol(test_inputs), test_labels)
	println("Loaded data")
	return training_data, test_data
end

function test_mnist()
	training_data, test_data = load_data()
	net = Network([784, 30, 10])
	SGD!(net, training_data, 30, 10, 0.1, test_data=test_data)
	return net
end

function test_mnist_quick()
	training_data, test_data = load_data()
	net = Network([784, 10])
	SGD!(net, training_data, 5, 10, 3.0, test_data=test_data)
	return net
end
