function [norm_y plus time] = my_norm(y)

time = max(y) - min(y);
temp = y / time;
plus = min(min(temp));
norm_y = temp - plus;
