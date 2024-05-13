% MATLAB Script to Analyze and Plot Data from CSV

% Read data from CSV file
filename = 'performance_data.csv';
opts = detectImportOptions(filename);
opts.VariableNamesLine = 1;
data = readtable(filename, opts);

% Separate data by filters
gaussianData = data(strcmp(data.Filter, 'Gaussian'), :);
medianData = data(strcmp(data.Filter, 'Median'), :);
sobelData = data(strcmp(data.Filter, 'Sobel'), :);

% Filter out entries where SSIM or PSNR is -1 for plots involving these metrics
gaussianFiltered = gaussianData(gaussianData.SSIM ~= -1 & gaussianData.PSNR ~= -1, :);
medianFiltered = medianData(medianData.SSIM ~= -1 & medianData.PSNR ~= -1, :);
sobelFiltered = sobelData(sobelData.SSIM ~= -1 & sobelData.PSNR ~= -1, :);

% Line graph for execution time by filter type
figure;
hold on;
plot(gaussianData.ExecutionTime, 'b-o', 'DisplayName', 'Gaussian');
plot(medianData.ExecutionTime, 'r-o', 'DisplayName', 'Median');
plot(sobelData.ExecutionTime, 'g-o', 'DisplayName', 'Sobel');
xlabel('Sample Index');
ylabel('Execution Time (microseconds)');
title('Execution Time for Each Filter');
legend show;
hold off;

% Bar chart comparing memory usage across filters
figure;
bar([mean(gaussianData.MemoryUsage) mean(medianData.MemoryUsage) mean(sobelData.MemoryUsage)]);
set(gca, 'xticklabel', {'Gaussian', 'Median', 'Sobel'});
ylabel('Average Memory Usage (bytes)');
title('Comparison of Memory Usage by Filter');

% Scatter plot for SSIM vs. PSNR, excluding invalid entries
figure;
scatter(gaussianFiltered.SSIM, gaussianFiltered.PSNR, 'b', 'DisplayName', 'Gaussian');
hold on;
scatter(medianFiltered.SSIM, medianFiltered.PSNR, 'r', 'DisplayName', 'Median');
if ~isempty(sobelFiltered)
    scatter(sobelFiltered.SSIM, sobelFiltered.PSNR, 'g', 'DisplayName', 'Sobel');
end
xlabel('SSIM');
ylabel('PSNR');
title('Scatter Plot of SSIM vs. PSNR for Each Filter');
legend show;
hold off;
