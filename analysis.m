
% Read the CSV file
data = readtable('C:/emotions.csv');

% Get FFT columns
fft_cols = startsWith(data.Properties.VariableNames, 'fft_');
sample = table2array(data(1, fft_cols))';

% Create a figure directory if it doesn't exist
if ~exist('figures', 'dir')
    mkdir('figures');
end

%% Time-Series Visualization
figure('Position', [100 100 1200 600]);
plot(1:length(sample), sample, 'LineWidth', 1.5);
title('EEG Time-Series Data', 'FontSize', 14);
xlabel('Time Points', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
grid on;
saveas(gcf, 'figures/eeg_time_series.png');

%% Spectral Analysis using Welch's method
sampling_rate = 256;
window_length = 256;
noverlap = window_length/2;
nfft = 512;

[pxx, f] = pwelch(sample, hamming(window_length), noverlap, nfft, sampling_rate);

% Plot Power Spectral Density
figure('Position', [100 100 800 500]);
semilogy(f, pxx, 'LineWidth', 1.5);
grid on;
title('Power Spectral Density', 'FontSize', 14);
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Power/Frequency (dB/Hz)', 'FontSize', 12);
xlim([0 sampling_rate/2]);
saveas(gcf, 'figures/power_spectral_density.png');

%% Frequency Band Analysis
% Define frequency bands
delta = [0.5 4];    % Delta band
theta = [4 8];      % Theta band
alpha = [8 13];     % Alpha band
beta = [13 30];     % Beta band
gamma = [30 100];   % Gamma band

% Function to calculate band power
calcBandPower = @(pxx, f, band) mean(pxx(f >= band(1) & f <= band(2)));

% Calculate power in each band
delta_power = calcBandPower(pxx, f, delta);
theta_power = calcBandPower(pxx, f, theta);
alpha_power = calcBandPower(pxx, f, alpha);
beta_power = calcBandPower(pxx, f, beta);
gamma_power = calcBandPower(pxx, f, gamma);

% Plot frequency bands distribution
figure('Position', [100 100 800 500]);
bands = categorical({'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'});
powers = [delta_power, theta_power, alpha_power, beta_power, gamma_power];
bar(bands, powers);
title('EEG Frequency Band Powers', 'FontSize', 14);
ylabel('Power', 'FontSize', 12);
grid on;
saveas(gcf, 'figures/frequency_bands.png');

%% Time-Frequency Analysis using Spectrogram
window = hamming(256);
noverlap = 128;
nfft = 512;

[s, f, t] = spectrogram(sample, window, noverlap, nfft, sampling_rate);
figure('Position', [100 100 1000 600]);
pcolor(t, f, 10*log10(abs(s)));
shading interp;
title('Time-Frequency Analysis', 'FontSize', 14);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Frequency (Hz)', 'FontSize', 12);
colorbar('Label', 'Power/Frequency (dB/Hz)');
saveas(gcf, 'figures/time_frequency.png');

% Save processed data for Python
processed_data = struct();
processed_data.frequencies = f;
processed_data.psd = pxx;
processed_data.band_powers = powers;
save('processed_eeg_data.mat', 'processed_data');

fprintf('EEG analysis complete. Results saved in figures/ directory\n');