#=
network.jl
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits many desirable features.

Original Python version: Copyright © 2012-2020 Michael Nielsen
Rewritten Julia version: Copyright © 2018-2020 Derik Kauffman
Licensed under MIT license
=#

#= In Julia, functions do not "belong" to any particular class.
Therefore, the functions implemented here take a Network struct
as input labeled as `net` rather than as a field of `self`
(e.g. `SGD!(net, …)` vs. `self.SGD!(…)`)
=#

using Random

struct Network
	sizes::Vector{Int}
	biases::Vector{Vector{Float32}}
	weights::Vector{Matrix{Float32}}
end

"""
    Network(sizes::Integer)

The list `sizes` contains the number of neurons in the
respective layers of the network. For example, if the list
was [2, 3, 1] then it would be a three-layer network, with the
first layer containing 2 neurons, the second layer 3 neurons,
and the third layer 1 neuron. The biases and weights for the
network are initialized randomly, using a Gaussian
distribution with mean 0, and variance 1. Note that the first
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
	feedforward(net::Network, a::Vector{<:Number})

Return the output of the network if `a` is input.
"""
function feedforward(net::Network, a::Vector{<:Number})
	for (b, w) = zip(net.biases, net.weights)
		a = σ.(w * a .+ b)
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
function SGD!(net::Network, training_data, epochs::Integer,
		mini_batch_size::Integer, η::Number; test_data=nothing)
	if test_data != nothing n_test = length(test_data) end
	n = length(training_data)
	for j in 1:epochs
		shuffle!(training_data)
		mini_batches = [training_data[k:k+mini_batch_size-1]
		  for k in 1:mini_batch_size:n-mini_batch_size]
		for mini_batch in mini_batches
			update_mini_batch!(net, mini_batch, η)
		end
		if test_data != nothing
			println("Epoch $j: $(evaluate(net, test_data)) / $n_test")
		else
			println("Epoch $j complete")
		end
	end
end

"""
    update_mini_batch!(net::Network, mini_batch, η::Number)

Update the network's weights and biases by applying
gradient descent using backpropagation to a single mini batch.
The `mini_batch` is a list of tuples `(x, y)`, and `η`
is the learning rate.
"""
function update_mini_batch!(net::Network, mini_batch, η::Number)
	∇b = zero.(net.biases)
	∇w = zero.(net.weights)
	for (x, y) in mini_batch
		Δ∇b, Δ∇w = backprop(net, x, y)
		∇b .+= Δ∇b
		∇w .+= Δ∇w
	end
	net.weights .-= η * ∇w/length(mini_batch) # Eq. 20
	net.biases  .-= η * ∇b/length(mini_batch) # Eq. 21
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
	δ = cost_derivative(activations[end], y) .* σ′.(zs[end])
	∇b[end] = δ
	∇w[end] = δ * activations[end-1]'
	# Note that the variable l in the loop below is used a little
	# differently to the notation in Chapter 2 of the book. Here,
	# l = 1 means the last layer of neurons, l = 2 is the
	# second-last layer, and so on.
	for l in 2:length(net.sizes)-1
		z = zs[end-l+1]
		sp = σ′.(z)
		δ = (net.weights[end-l+2]' * δ) .* sp
		∇b[end-l+1] = δ
		∇w[end-l+1] = δ * activations[end-l]'
	end
	return (∇b, ∇w)
end

"""
	evaluate(net::Network, test_data)

Return the number of test inputs for which the neural
network outputs the correct result. Note that the neural
network's output is assumed to be the index of whichever
neuron in the final layer has the highest activation.
"""
function evaluate(net::Network, test_data)
	test_results = [(argmax(feedforward(net, x)) - 1, y) for (x, y) in test_data]
	return count((x==y) for (x,y) in test_results)
end

"""
	cost_derivative(output_activations, y)

Return the vector of partial derivatives ∂Cₓ/∂a for the output activations.
"""
function cost_derivative(output_activations, y)
	return output_activations .- onehot(y)
end

#### Miscellaneous functions
# The sigmoid function.
σ(z) = 1/(1 + exp(-z))

# Derivative of the sigmoid function.
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
	training_inputs = reshape(MNIST.traintensor(Float32), (784, :))
	training_results = MNIST.trainlabels()
	training_data = [(training_inputs[:, i], training_results[i]) for i in 1:50000]
	validation_data = [(training_inputs[:, i], training_results[i]) for i in 50001:60000]
	test_inputs = reshape(MNIST.testtensor(Float32), (784, :))
	test_results = MNIST.testlabels()
	test_data = [(test_vectors[:, i], test_labels[i]) for i in 1:10000]
	println("Loaded data")
    return (training_data, validation_data, test_data)
end

function test_mnist()
    training_data, validation_data, test_data = load_data()
    net = Network([784,30,10])
    SGD!(net, training_data, 30, 10, 0.1, test_data=test_data)
    return net
end

function test_mnist_quick()
    training_data, validation_data, test_data = load_data()
    net = Network([784,10])
    SGD!(net, training_data, 5, 10, 3.0, test_data=test_data)
    return net
end
