# # How to generate a literate example file

# To create an example using CLIMA, please use [Literate.jl](https://github.com/fredrikekre/Literate.jl), and consult the [Literate documentation](https://fredrikekre.github.io/Literate.jl/stable/) for questions.
#
# For now, all literate examples are held in the `literate_examples` directory, and generated with the following code in `docs/make.jl`:

using Literate
const examples_directory = pwd() * "/literate_examples/"
const output_directory = pwd() * "/generated/"

examples = readdir(examples_directory)

for example in examples
    example_filepath = examples_directory * example
    Literate.markdown(example_filepath, output_directory, documenter = true)
end

# With Literate, all comments turn into markdown text and any Julia code is read and run *as if it is in the Julia REPL*.
# As a small caveat to this, you might need to suppress the output of certain commands.
# For example, if you define and run the following function

function f()
    return x = [i * i for i in 1:10]
end

x = f()

# The entire list will be output, while

f();

# does not (because of the `;`).
#
# To show plots, you may do something like the following:
using Plots
plot(x)

# Please consider writing the comments in your example as if they are meant to be read as an *article explaining the topic the example is meant to explain.*
# If there are any specific nuances to writing Literate documentation for CLIMA, please let us know!
