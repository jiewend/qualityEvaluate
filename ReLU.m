function g = ReLU(z)

temp = z >= 0;
g = temp .* z;
