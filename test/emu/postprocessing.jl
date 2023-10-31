function postprocessing(input, output, Cℓemu)
    return output .* exp(input[1]-3.)
end
