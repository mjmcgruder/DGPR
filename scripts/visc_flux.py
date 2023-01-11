import numpy as np

def flux(mu, u, v, w, ux, uy, uz, vx, vy, vz, wx, wy, wz):
  lam = -(2. / 3.) * mu

  dv = ux + vy + wz

  txx = lam * dv + 2. * mu * ux
  tyy = lam * dv + 2. * mu * vy
  tzz = lam * dv + 2. * mu * wz

  txy = mu * (vx + uy)
  txz = mu * (uz + wx)  # this is weird...
  tyz = mu * (wy + vz)  # this is weird...

  Q = np.empty((5, 3))

  Q[0, 0] = 0.
  Q[1, 0] = -txx
  Q[2, 0] = -txy
  Q[3, 0] = -txz
  Q[4, 0] = -u * txx - v * txy - w * txz

  Q[0, 1] = 0.
  Q[1, 1] = -txy
  Q[2, 1] = -tyy
  Q[3, 1] = -tyz
  Q[4, 1] = -u * txy - v * tyy - w * tyz

  Q[0, 2] = 0.
  Q[1, 2] = -txz
  Q[2, 2] = -tyz
  Q[3, 2] = -tzz
  Q[4, 2] = -u * txz - v * tyz - w * tzz

  return Q

Q = flux(1., 0.5, 0., 0., 0.05, 0., 0., 0., 0.15, 0., 0., 0., 0.12)
print(Q)
