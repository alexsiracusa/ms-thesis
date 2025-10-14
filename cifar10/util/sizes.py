
num_iterations = 4

input_sizes = [300] * 25
hidden_sizes = [300] * 10
output_sizes = [10] * 1
sizes = input_sizes + hidden_sizes + output_sizes

num_input = len(input_sizes)
num_hidden = len(hidden_sizes)
num_output = len(output_sizes)
num_blocks = len(input_sizes + hidden_sizes + output_sizes)
