import sympy as sy

# 1d

def ord2_1d(x):
  return 1.2785 * (x ** 2.0) + 4.382 * x + 0.3854

def ord5_1d(x):
  return 2.394 * (x ** 5.0) + 8.371 * (x ** 3.0) + 0.23444 * x + 3.223

def ord16_1d(x):
  return 5.76594 * (x ** 16.0) + 3.7694 * (x ** 9.0) + 1.7674 * (x ** 2.0) + \
         4.11854

def ord32_1d(x):
  return 7.584 * (x ** 32.0) + 2.69 * (x ** 17.0) + 10.9473;

# 2d

def ord2_2d(x, y):
  return 1.2785 * x * x + 4.1854 * y * y + 9.43 * x * y + 4.382 * x + \
         2.333 * y + 0.3854;

def ord16_2d(x, y):
  return 5.76594 * (x ** 16.0) - 0.5542 * (y ** 16.0) + \
         4.2833 * (x ** 10.0) * (y ** 8.0) + \
         3.7694 * (x ** 9.0) * (y ** 3.0) + 1.7674 * (x ** 2.0) + \
         4.11854;

def ord32_2d(x, y):
  return 10.3874 * (x ** 32.0) * (y ** 32.0) + \
         3.452 * (x ** 4.0) * (y ** 20.0) + 3.6654;

# 3d

def ord2_3d(x, y, z):
  return 5.2038 * x ** 2.0 * y ** 2.0 * z ** 2.0 - 2.3385 * y * z + 1.224;

def ord16_3d(x, y, z):
  return 5.76594 * x **  16.0 - 0.5542 * y ** 16.0 * z ** 16.0 +\
         4.2833 * x ** 10.0 * y ** 8.0 * z ** 6.0 +\
         3.7694 * x ** 9.0 * y ** 3.0 +\
         1.7674 * x ** 2.0 * z ** 2.0 + 4.11854;

x = sy.Symbol("x")
y = sy.Symbol("y")
z = sy.Symbol("z")
print(sy.integrate(ord2_1d(x),  (x, -1, 1)))
print(sy.integrate(ord2_1d(x),  (x, 0, 1)))
print(sy.integrate(ord5_1d(x),  (x, -1, 1)))
print(sy.integrate(ord5_1d(x),  (x, 0.68834, 1)))
print(sy.integrate(ord16_1d(x), (x, -1, 1)))
print(sy.integrate(ord16_1d(x), (x, -0.22954, 1)))
print(sy.integrate(ord32_1d(x), (x, -1, 1)))
print(sy.integrate(ord32_1d(x), (x, 0.78452, 1)))
print()
print(sy.integrate(ord2_2d(x, y), (x, -1, 1), (y, -1, 1)))
print(sy.integrate(ord2_2d(x, y), (x, 0, 1), (y, 0, 1)))
print(sy.integrate(ord16_2d(x, y), (x, -1, 1), (y, -1, 1)))
print(sy.integrate(ord16_2d(x, y), (x, 0, 1), (y, 0, 1)))
print(sy.integrate(ord32_2d(x, y), (x, -1, 1), (y, -1, 1)))
print(sy.integrate(ord32_2d(x, y), (x, 0, 1), (y, 0, 1)))
print()
print(sy.integrate(ord2_3d(x, y, z), (x, -1, 1), (y, -1, 1), (z, -1, 1)))
print(sy.integrate(ord2_3d(x, y, z), (x, 0, 1), (y, 0, 1), (z, 0, 1)))
print(sy.integrate(ord16_3d(x, y, z), (x, -1, 1), (y, -1, 1), (z, -1, 1)))
print(sy.integrate(ord16_3d(x, y, z), (x, 0, 1), (y, 0, 1), (z, 0, 1)))
