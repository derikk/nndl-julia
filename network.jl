#=
network.jl
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits many desirable features.

Original Python version: Copyright © 2012-2018 Michael Nielsen
Rewritten Julia version: Copyright © 2018 Derik Kauffman
Licensed under MIT license
=#

#= In Julia, functions do not "belong" to any particular class.
Therefore, the functions implemented here take a Network struct
as input labeled as `net` rather than as a field of `self`
(e.g. `SGD!(net, …)` vs. `self.SGD!(…)`)
=#

using Random

struct Network{sType, bType, wType}
	num_layers::Int
	sizes::sType   # Vector of sizes
	biases::bType  # Vector of bias vectors
	weights::wType # Vector of weight matrices
end

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
function Network(sizes)
	num_layers = length(sizes)
	biases = [randn(y) for y in sizes[2:end]]
	weights = [randn(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
	Network(num_layers, sizes, biases, weights)
end

"""Return the output of the network if `a` is input."""
function feedforward(net::Network, a::Vector{Float64})
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
function SGD!(net::Network, training_data, epochs::Int,
	  mini_batch_size::Int, η::Float64; test_data=nothing)
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

"""Update the network's weights and biases by applying
gradient descent using backpropagation to a single mini batch.
The `mini_batch` is a list of tuples `(x, y)`, and `η`
is the learning rate."""
function update_mini_batch!(net::Network, mini_batch, η::Float64)
	∇b = [zeros(size(b)) for b in net.biases]
	∇w = [zeros(size(w)) for w in net.weights]
	for (x, y) in mini_batch
		Δ∇b, Δ∇w = backprop(net, x, y)
		∇b .+= Δ∇b
		∇w .+= Δ∇w
	end
	net.weights .-= η * ∇w/length(mini_batch) # Eq. 20
	net.biases  .-= η * ∇b/length(mini_batch) # Eq. 21
end

"""Return a tuple `(∇b, ∇w)` representing the gradient for the
cost function Cₓ. `∇b` and `∇w` are layer-by-layer lists of arrays,
similar to `net.biases` and `net.weights`."""
function backprop(net::Network, x, y)
	∇b = [zeros(size(b)) for b in net.biases]
	∇w = [zeros(size(w)) for w in net.weights]

	#feedforward
	activation = x
	activations = [x] # list to store all the activations, layer by layer
	zs = typeof(x)[] # list to store all the z vectors, layer by layer
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
	for l in 2:net.num_layers-1
		z = zs[end-l+1]
		sp = σ′.(z)
		δ = (net.weights[end-l+2]' * δ) .* sp
		∇b[end-l+1] = δ
		∇w[end-l+1] = δ * activations[end-l]'
	end
	return (∇b, ∇w)
end

"""Return the number of test inputs for which the neural
network outputs the correct result. Note that the neural
network's output is assumed to be the index of whichever
neuron in the final layer has the highest activation."""
function evaluate(net::Network, test_data)
	test_results = [(argmax(feedforward(net, x)) - 1, y) for (x, y) in test_data]
	return count((x==y) for (x,y) in test_results)
end

"""Return the vector of partial derivatives ∂Cₓ/∂a for the output activations."""
function cost_derivative(output_activations, y)
	return output_activations .- y
end

#### Miscellaneous functions
# The sigmoid function.
σ(z) = 1/(1 + exp(-z))

# Derivative of the sigmoid function.
σ′(z) = σ(z)*(1 - σ(z))

function onehot(j)
	"""Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeros elsewhere. This is used to convert a digit (0-9)
	into a corresponding desired output from the neural network."""
	e = zeros(10)
	e[Int(j)+1] = 1.0
	return e
end

# MNIST loading code
using MNIST
# The MNIST package has not yet been updated to Julia 1.0, so you will need to
# make a minor modification to it first. See pull requests.
# Alternatively, everything should work under Julia 0.7.

# Functions from rhezab/nielsen.jl
function load_data()
    train_data = [(trainfeatures(i) / 255.0, onehot(trainlabel(i)))
	  for i in 1:50000]
    validation_data = [(trainfeatures(i) / 255.0, trainlabel(i))
      for i in 50001:60000]
    test_data = [(testfeatures(i) / 255.0, testlabel(i)) for i in 1:10000]
    return (train_data, validation_data, test_data)
end

function test_mnist()
    train_data, validation_data, test_data = load_data()
    net = Network([784,30,10])
    SGD!(net, train_data, 30, 10, 0.1, test_data=test_data)
    return net
end

function test_mnist_quick()
    train_data, validation_data, test_data = load_data()
    net = Network([784,10])
    SGD!(net, train_data, 5, 10, 3.0, test_data=test_data)
    return net
end
