function concreteY = concrete_y(discrete_y, grad, min_y)

concreteY = zeros(size(discrete_y));
for i = 1:length(discrete_y),
	concreteY(i) = min_y + discrete_y(i) * grad - grad / 2;
end
end


