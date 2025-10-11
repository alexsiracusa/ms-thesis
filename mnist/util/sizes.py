
input_sizes = [100] * 25
hidden_sizes = [100] * 10
output_sizes = [10]

# input_sizes = [500] * 5
# hidden_sizes = [250] * 4
# output_sizes = [10]

sizes = input_sizes + hidden_sizes + output_sizes

num_blocks = len(input_sizes + hidden_sizes + output_sizes)
