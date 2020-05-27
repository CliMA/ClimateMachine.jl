# TODO: Add documentation
function lamb_smooth_minimum(
    l::AbstractArray,
    lower_bound::FT,
    upper_bound::FT
) where {FT}

  leng = size(l)
  xmin = minimum(l)
  lambda0 = max(xmin*lower_bound/real(LambertW.lambertw(FT(2)/MathConstants.e)), upper_bound)

  i = 1
  num = 0
  den = 0
  while(tuple(i)<leng)
    num += l[i]*exp(-(l[i]-xmin)/lambda0)
    den += exp(-(l[i]-xmin)/lambda0)
    i += 1
  end
  smin = num/den

  return smin
end