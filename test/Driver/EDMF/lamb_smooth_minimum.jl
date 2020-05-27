function lamb_smooth_minimum(l, lower_bound, upper_bound)
  leng = size(l)
  xmin = minimum(l)
  lambda0 = max(xmin*lower_bound/real(LambertW.lambertw(2.0/MathConstants.e)), upper_bound)

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