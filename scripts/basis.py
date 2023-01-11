import numpy as np
from sympy import *

def lagrange1d(p, bfn, x):
  node_locs = np.linspace(0., 1., p + 1)
  poly      = sympify('1')
  for m in range(p + 1):
    if m is not bfn:
      poly *= (x - node_locs[m]) / (node_locs[bfn] - node_locs[m])
  return poly

def lagrange3d(p, bfx, bfy, bfz, x, y, z):
  return lagrange1d(p, bfx, x) * lagrange1d(p, bfy, y) * lagrange1d(p, bfz, z)

p    = 2
x    = symbols('x')
y    = symbols('y')
z    = symbols('z')
print(integrate(x * x + y * y + z * z, (x, 0., 1.), (y, 0., 1.), (z, 0., 1.)))
poly = lagrange3d(p, 2, 2, 2, x, y, z)
dphi_dx = diff(poly, x)
dphi_dy = diff(poly, y)
dphi_dz = diff(poly, z)
g   = 1.4;
rho = 1.;
u   = 0.2;
v   = 0.;
w   = 0.;
P   = rho / g;
vol_integrand = dphi_dx * rho * u + dphi_dy * rho * v + dphi_dz * rho * w
print(expand(vol_integrand))
qp = (np.array([-1. / np.sqrt(3.), 1. / np.sqrt(3.)]) + 1.) / 2.
print(qp)
print(dphi_dx.subs([(x, qp[0]), (y, qp[1]), (z, qp[0])]))
print(dphi_dy.subs([(x, qp[0]), (y, qp[1]), (z, qp[0])]))
print(dphi_dz.subs([(x, qp[0]), (y, qp[1]), (z, qp[0])]))
print(integrate(vol_integrand, (x, 0., 1.), (y, 0., 1.), (z, 0., 1.)))
