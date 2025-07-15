module TwistedStates


"""
`construct_q_twisted`: Construct a q-twisted state on n sites

### Fields
* `n` - Number of sites
* `q` - Twist
### Optional Fields
* `c=0.0` - Translation
* `Π=1` - Period
* `invert=false` - Invert the state
"""
function construct_q_twisted(n, q; c=0.0, Π=1, invert=false)

      u = mod.(q * ((0:n-1) / n) .+ c, Π)

      if invert
            @. u = Π - u
      end

      return u

end

"""
`get_q_twisted_spectrum`: Determine if the q twisted state on n sites with 2k+1
nearest neighbors is stable or not.

### Fields
* `n` - Number of sites
* `q` - Twist
* `k` - 2k+1 Nearest neighbors
* `α` - 0 for attractive and 1 for repulsive
"""
function get_q_twisted_spectrum(n, q, k, α)

      λ = zeros(n)
      for i in 0:n-1
            for j in -k:k
                  λ[i+1] += 2 * π / (2 * k + 1) * (-1)^α * (cos(2 * π * (q + i) * j / n) - 2 * cos(2 * π * q * j / n) + cos(2 * π * (q - i) * j / n))
            end
      end
      return λ
end

"""
`construct_l_saddle`: Construct a l-saddle state on n sites

### Fields
* `n` - Number of sites
* `l` - Saddle parameter
### Optional Fields
* `c=0.0` - Translation
* `Π=1` - Period
* `invert=false` - Invert the state
"""
function construct_l_saddle(n, l; c=0.0, Π=1, invert=false)

      q = (2 * l + 1) * n / (2 * (n - 2))

      u = mod.(q * ((0:n-1) / n) .+ c, Π)

      if invert
            @. u = Π - u
      end

      return u

end

"""
`line_wrap`: Wrap a solution and insert `NaN`s for plotting as a line

### Fields
* `u` - The solution
### Optional Fields
* `Π=1` - Period
"""
function line_wrap(u; Π=1)
      u_wrap = copy(mod.(u, Π))
      idx = findall(diff(u_wrap) .< 0)
      @. u_wrap[idx] = NaN

      return u_wrap

end

export construct_q_twisted, get_q_twisted_spectrum,
      construct_l_saddle, line_wrap

end
