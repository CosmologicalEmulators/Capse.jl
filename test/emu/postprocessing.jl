function postprocessing(input, output)
    return output .* exp(input[1]-3.)
end
