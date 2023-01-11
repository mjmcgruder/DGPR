import numpy as np

# too much googling to print a full array with numpy so here we are...
def print_array(A):
  for r in range(A.shape[0]):
    for c in range(A.shape[1]):
      print(f'{A[r, c]:+.16e},', end=' ')
    print('')

# let's make some simple matrices to test with

A = np.array([[ 4.20, -3.60,  3.14],
              [ 2.71,  8.33, -4.44],
              [ 5.22,  1.01,  5.55]])
B = np.array([[ 9.88, -5.50, -1.22],
              [ 5.22,  9.52, -0.22],
              [ 8.22,  4.73, -3.22]])

print('A:')
print_array(A)
print('B:')
print_array(B)

Ainv = np.linalg.inv(A)
Binv = np.linalg.inv(B)
print('inv(A):')
print_array(Ainv)
print('inv(B):')
print_array(Binv)

Adet = np.linalg.det(A)
Bdet = np.linalg.det(B)
print(f'det(A): {Adet:+.16e}')
print(f'det(B): {Bdet:+.16e}')

print('A x B:')
print_array(np.matmul(A, B))

# let's make a random matrix to test inversion with
# I think I'll convert floats to double temporarily to remove some round-off
# error when computing mass matrix inversions so I'm mimicking that process here
# to generate the test matrices

rng     = np.random.Generator(np.random.MT19937(42))

C    = rng.random((10, 10))
Cinv = np.linalg.inv(C)

print('C:')
print_array(C)
print('inv(C):')
print_array(Cinv)

D    = rng.random((100, 100))
Dinv = np.linalg.inv(D)

print('D:')
print_array(D)
print('inv(D):')
print_array(Dinv)
