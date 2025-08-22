data = load('sin.dat'); % Load sine data file
positions = data(:, 1); % Extract positions (first column)
dimensions = data(:, 2); % Extract dimensions (second column)
values = data(:, 3); % Extract sine values (third column)
% Find unique dimensions
unique_dims = unique(dimensions);
% Plot sine values for each dimension
figure;
hold on;
for i = 1:length(unique_dims)
    dim = unique_dims(i);
    mask = dimensions == dim; % Select data for current dimension
    plot(positions(mask), values(mask), 'DisplayName', ['Dim ', num2str(dim)]);
end
hold off;
title('Sine Positional Encoding');
xlabel('Position');
ylabel('Sine Value');
legend('show');
grid on;
