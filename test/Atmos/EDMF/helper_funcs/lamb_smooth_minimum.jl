# TODO: Add documentation
function lamb_smooth_minimum(
    l::AbstractArray,
    lower_bound::FT,
    frac_upper_bound::FT
) where {FT}

  leng = size(l)
  xmin = minimum(l)
  # try to find outhow to use dz from the model grid instead of lower_bound
  lambda0 = max(xmin*frac_upper_bound/real(LambertW.lambertw(FT(2)/MathConstants.e)), lower_bound)

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