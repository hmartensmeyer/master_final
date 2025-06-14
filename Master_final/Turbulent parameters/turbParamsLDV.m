cwtftinfo2('marr')

%%
% Define grid
x1 = linspace(-5, 5, 200);
x2 = linspace(-5, 5, 200);
[X1, X2] = meshgrid(x1, x2);

% Compute psi
R2 = X1.^2 + X2.^2;
psi = -(1/2*pi)*(R2 - 2) .* exp(-R2 / 2);

% Plot
figure;
surf(X1, X2, psi);
shading interp;
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
zlabel('$\psi(x_1, x_2)$', 'Interpreter', 'latex');
title('Mexican Hat Wavelet', 'Interpreter', 'latex');
colorbar;
view(45, 30);



%% --- SCRIPT TO ESTIMATE TURBULENT PARAMETERS --- %
clear; clc;

%% Step 1: Preprocess the file to replace commas with periods
%origFile   = 'data\SZ_vert_v10p5Hz.txt';   % raw LDV export
origFile = 'AG_SZ_VFDA0_VFDW10.5_H53_US.txt';
tempFile   = 'temp_SZ_vert_v10p5Hz.txt';
textData   = fileread(origFile);
converted  = strrep(textData, ',', '.');
fid        = fopen(tempFile, 'w');
fwrite(fid, converted);
fclose(fid);

%% Step 2: Import the data
opts  = detectImportOptions(tempFile, ...
         'NumHeaderLines',4, 'Delimiter','\t');
T     = readtable(tempFile, opts);
time  = T.AT_ms_   / 1000;    % [s]
u_raw = T.LDA1_m_s_;          % [m/s]
nu = 1.022E-6;

%% Step 2b: Crop the raw velocity
u_trim = u_raw;                      % start with the raw series
for k = 2 : numel(u_trim)
    if u_trim(k) < 0.05 || u_trim(k) > 0.4
        u_trim(k) = u_trim(k-1);
    end
end

%% Step 3: Mean velocity & fluctuations
U_mean    = mean(u_trim);
u_fluct   = u_raw - U_mean;

%% Step 4: Resample onto a uniform grid
dt      = time(end) / length(time);                    % [s]
t_unif  = time(1):dt:time(end);     % uniform time‐base

% Nearest‐neighbor / sample‐and‐hold (ZOH)
u_zoh   = interp1(time, u_trim, t_unif, 'previous');
u_zoh_filt = u_zoh - U_mean;

%% Step 5: Calculate gradient

grad = Jgrad(u_zoh, U_mean*dt, 7);
grad_mean_square = mean(grad.^2);
u_var = mean(u_zoh_filt.^2);
u_rms = sqrt(u_var);

%% PRINT THE DIFFERENT QUANTITIES
lambda = sqrt( u_var / grad_mean_square)
Re_lambda = sqrt(u_var) * lambda / nu
l = lambda*Re_lambda/15;
L_infty = L_int%l/2;
Re_infty = 2*u_rms*L_infty/nu
L_v = (2*L_infty/sqrt(Re_infty))

Fr = u_rms / (sqrt(9.81*L_infty))

%% Weber number calculation
rho = 998.34;
sigma = 72.6E-3; %surface tension

We = rho*u_rms^2*L_infty / (2*sigma)

%% Step 6: Integral length scale from autocorrelation

% 6a) Fluctuations on the uniform ZOH‐resampled grid
u_fluct_zoh = u_zoh - mean(u_zoh);

% 6b) Compute biased autocorrelation
[R, lags]   = xcorr(u_fluct_zoh, 'biased');

% 6c) Convert lags to time (s) and keep only τ ≥ 0
lags_s   = lags * dt;
pos      = lags_s >= 0;
R_pos    = R(pos);
lags_pos = lags_s(pos);

% 6d) Normalize so R(0)=1
R_pos = R_pos / u_var;

% 6e) Find first zero‐crossing (or integrate out to end if none)
zc = find(R_pos < 0, 1);
if isempty(zc)
    zc = numel(R_pos);
end

% 6f) Integral time scale T_int = ∫₀^{τ₀} R(τ) dτ
T_int = trapz(lags_pos(1:zc), R_pos(1:zc));

% 6g) Convert to length scale L = U_mean * T_int
U_mean_zoh = mean(u_zoh);
L_int      = U_mean_zoh * T_int;

% 6h) Display
fprintf('Integral time scale T_int = %.4f s\n', T_int);
fprintf('Integral length scale L_int = %.4f m\n', L_int);

% (Optional) plot the normalized autocorrelation and mark zero crossing
figure;
plot(lags_pos, R_pos, 'LineWidth',1);
hold on
plot(lags_pos(zc), R_pos(zc), 'ro', 'MarkerSize',8, 'LineWidth',1.5);
xlabel('Lag $\tau$ [s]')
ylabel('R($\tau$)')
xlim([0,10])
title('Normalized Autocorrelation of u'' (ZOH series)')
grid on
