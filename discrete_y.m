function [discrete_y grad min_y] = discrete_y(y, num);

discrete_y = ones(size(y));

max_y = max(y);
min_y = min(y);
range_y = max_y - min_y;
grad = range_y / num;
half_grad = grad / 2;
for i = 1:num,
	temp1 = y > (min_y - half_grad + i * grad); 
	temp2 = y < (min_y + half_grad + i * grad);
	temp3 = temp1 .* temp2;
	index = find((temp1 .* temp2) == 1);
	discrete_y(index) = i;
end
end

