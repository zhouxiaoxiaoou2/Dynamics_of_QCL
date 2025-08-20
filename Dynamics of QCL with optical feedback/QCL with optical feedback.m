%% Multi-mode QCL Feedback Simulation - Subplot Version
clear; clc;
close all;

%% Load constants and parameters
global q c hbar eta_3 eta_2 tau_3 tau_2 tau_32 tau_sp beta_sp G0 M alpha_factor;
global N_modes nu_zm omega0 Delta_nu nu_p epsilon_g G_static;
global L_in R2 R epsilon kappa tau_ext alpha_a n_m v_g r1m r2m alpha_m tau_p eta_0 tau_in L_ext n_ext;

%% Physical constants
q = 1.602e-19;          % Elementary charge (C)
c = 3e8;                % Speed of light (m/s)
hbar = 1.055e-34;       % Reduced Planck constant (J.s)

% QCL parameters
eta_3 = 0.54;           % Injection efficiency into ULL
eta_2 = 0.0165;         % Injection efficiency into LLL
tau_3 = 5e-12;          % ULL carrier lifetime (s)
tau_2 = 2.1e-12;        % LLL carrier lifetime (s)
tau_32 = 1.76e-10;      % ULL→LLL relaxation time (s)
tau_sp = 1e-6;          % Spontaneous emission lifetime (s)
beta_sp = 1.627e-4;     % Spontaneous emission factor
G0 = 1.8e4;             % Peak gain factor (s⁻¹)
M = 90;                 % Number of periods in active cavity
alpha_factor = -0.1;     % Linewidth enhancement factor

%% Multi-mode configuration
N_modes = 7;            % Number of modes
nu_zm = [2.712, 2.725, 2.738, 2.752, 2.765, 2.778, 2.791] * 1e12; % Mode frequencies (Hz)
omega0 = 2 * pi * nu_zm; % Angular frequencies (rad/s)
Delta_nu = 5e12;       % 600 GHz. Gain bandwidth (Hz) (varies),assume the interval is the same
nu_p = 2.752e12;        % Peak gain frequency (Hz)
epsilon_g = 1e-7;       % Gain compression coefficient
% Pre-calculate static gain
G_static = G0 ./ (1 + ((nu_p - nu_zm) / (Delta_nu/2)).^2);

%% Cavity parameters
L_in = 3e-3;             % Laser cavity length (m)
R2 = 0.5662;             % Right facet reflectivity
R = 0.7;                 % External target reflectivity
epsilon = 0.5;           % Re-injection coupling factor
kappa = 1.5;             % Feedback coupling coefficient
L_ext = 0.5;               % External cavity length (m)
n_ext = 1.0;             % Refractive index of external cavity 
tau_ext = (2*L_ext*n_ext)/c;  % External cavity delay (s)

% Waveguide and mirror losses
alpha_a = 9e2;          % Waveguide loss (m⁻¹)
n_m = [3.6114, 3.6121, 3.6129, 3.6137, 3.6144, 3.6152, 3.6159]; % Group refractive indices
v_g = c ./ n_m;         % Group velocities (m/s)
r1m = (n_m - 1) ./ (n_m + 1); % Left facet reflectivity
r2m = (n_m - 1) ./ (n_m + 1); % Right facet reflectivity
alpha_m = -log(r1m .* r2m) / (2*L_in); % Mirror losses (m⁻¹) at mode m
tau_p = 1 ./ (v_g .* (alpha_a + alpha_m)); % Photon lifetimes (s)
eta_0 = alpha_m ./ (2 * (alpha_a + alpha_m)); % Output coupling coefficients
tau_in = 2 * L_in * n_m/c;         % Internal round-trip time (s) of each mode

%% Simulation parameters
I0 = 1.8;               % Drive current (A)
tspan = [0, 0.2e-6];      % Simulation time (2 μs for steady-state)
S0 = 1e8 * (1 + 0.5*rand(N_modes, 1));
phi0 = 2*pi*rand(N_modes, 1);  % 加相位扰动
y0 = [1e7; 1e7; reshape([S0 phi0], [], 1)];
history = @(t) y0;

%% Solve DDE
opts = ddeset('RelTol',1e-3,'AbsTol',1e-6);
sol = dde23(@(t,y,Z) lreqns_QCL_OF_multimode(t,y,Z,I0), tau_ext, history, tspan, opts);

% ========= Extract solution vector =========
t = sol.x;
y = sol.y;

S_all = y(3:2:end-1, :);      % Photon number
phi_all = y(4:2:end, :);      % Phase
Tspan = t;                    % Time axis
Ng = length(Tspan);          % FFT points
dt = Tspan(2) - Tspan(1);
Fs = 1 / dt;                  % Sampling frequency (Hz)
f = (-Ng/2:Ng/2-1)*(Fs/Ng);   % Frequency axis (Hz)
delta_N = y(1,:) - y(2,:);   
middle_idx = ceil(N_modes / 2);

scale_factors = (eta_0(:) .* hbar .* omega0(:)) ./ tau_p(:);  % [7×1]
P_out_modes = scale_factors .* S_all;                 % [7×N]

% ========= Construct electric fields for each mode P1 ~ P7 =========
Po1 = sqrt(S_all(1,:));  Pha1 = phi_all(1,:);  wf1 = omega0(1);
Po2 = sqrt(S_all(2,:));  Pha2 = phi_all(2,:);  wf2 = omega0(2);
Po3 = sqrt(S_all(3,:));  Pha3 = phi_all(3,:);  wf3 = omega0(3);
Po4 = sqrt(S_all(4,:));  Pha4 = phi_all(4,:);  wf4 = omega0(4);
Po5 = sqrt(S_all(5,:));  Pha5 = phi_all(5,:);  wf5 = omega0(5);
Po6 = sqrt(S_all(6,:));  Pha6 = phi_all(6,:);  wf6 = omega0(6);
Po7 = sqrt(S_all(7,:));  Pha7 = phi_all(7,:);  wf7 = omega0(7);

P1 = Po1 .* exp(1j * wf1 * Tspan + 1j * Pha1);
P2 = Po2 .* exp(1j * wf2 * Tspan + 1j * Pha2);
P3 = Po3 .* exp(1j * wf3 * Tspan + 1j * Pha3);
P4 = Po4 .* exp(1j * wf4 * Tspan + 1j * Pha4);
P5 = Po5 .* exp(1j * wf5 * Tspan + 1j * Pha5);
P6 = Po6 .* exp(1j * wf6 * Tspan + 1j * Pha6);
P7 = Po7 .* exp(1j * wf7 * Tspan + 1j * Pha7);

% ========= Total field =========
P_sum = P1 + P2 + P3 + P4 + P5 + P6 + P7;
total_power = abs(P_sum).^2;

% ========= Optical spectrum =========
NPfsum = fft(P_sum, Ng) / Ng;
Pof = fftshift(NPfsum);
spectrum_dB = 10 * log10(abs(Pof).^2 / max(abs(Pof).^2));
spectrum_dB(spectrum_dB < -50) = -50;
 

%% Optical Spectrum Calculation (f)
idx = t > max(t)/2;
t_sel = t(idx);
dt = mean(diff(t_sel));
N = length(t_sel);
Fs = 1/dt;

omega_mat = omega0(:) * ones(1, N);     % [7 × N]
t_mat = ones(N_modes, 1) * t_sel;       % [7 × N]
E_field = sum(sqrt(S_all(:,idx)) .* exp(1i * (omega_mat .* t_mat + phi_all(:,idx))), 1);

% Frequency axis
nu_center = nu_p;  % Center frequency Hz
f_axis_Hz = (-N/2:N/2-1)*(Fs/N) + nu_center;
f_axis_offset_GHz = (f_axis_Hz - nu_center) / 1e9;  % GHz offset from center frequency

% FFT processing
spectrum = abs(fftshift(fft(E_field))).^2;
spectrum = spectrum / max(spectrum);  % normalize
spectrum_dB = 10 * log10(spectrum);
spectrum_dB(spectrum_dB < -50) = -50;


%% ---- Power Spectrum from total output power (not E-field) ----
idx = Tspan > max(Tspan) / 2;
P_sel = abs(P_sum(idx)).^2;         % Instantaneous power sequence (Intensity)
% P_sel = P_sel - mean(P_sel);        % Remove DC component to reduce spectral leakage

N_P = length(P_sel);
P_fft = abs(fftshift(fft(P_sel))).^2;
P_fft = P_fft / max(P_fft);         % Normalize
f_axis = (-N_P/2:N_P/2-1) * (Fs / N_P) / 1e9;  % GHz

P_fft_dB = 10 * log10(P_fft);
P_fft_dB(P_fft_dB < -50) = -50;

%% Create master figure with all subplots
figure('Position', [100, 100, 1200, 900], 'Color', 'w');

% Plot 1: Population Levels
subplot(3,3,1);
plot(t*1e9, y(1,:), 'r', t*1e9, y(2,:), 'b', 'LineWidth', 2);
legend('N3', 'N2', 'FontSize', 9); 
xlabel('Time (ns)');
ylabel('Population');
title('(a) Population Levels');
grid on;
set(gca, 'FontSize', 10);

% Plot 2: N3-N2 Difference
subplot(3,3,2);
plot(t*1e9, delta_N/1e7, 'LineWidth', 2); 
title('(b) N3 - N2 Difference');
xlabel('Time (ns)');
ylabel('N3-N2 (×10^7)');
grid on;
set(gca, 'FontSize', 10);

% Plot 3: Output Power vs Time
idx_zoom = t*1e9 > 150 & t*1e9 < 151;
subplot(3,3,3);
plot(t(idx_zoom) * 1e9, total_power(idx_zoom), 'k', 'LineWidth', 2);
xlabel('Time (ns)'); 
ylabel('Output Power ');
title('(c) Output Power vs Time');
grid on;
set(gca, 'FontSize', 10);

% Plot 6: Continuous Absolute Optical Spectrum (no φ)连续
subplot(3,3,4);

% 1. Construct "no phase drift" complex electric field envelope
E_nophase = sum( sqrt(S_all(:,idx)) .* exp(1j*(omega_mat .* t_mat)), 1 );

% 2. FFT and normalize
Nfft = length(E_nophase);
Spec  = fftshift( fft(E_nophase, Nfft) / Nfft );
Pwr   = abs(Spec).^2;
Pwr   = Pwr / max(Pwr);
Pwr_dB= 10 * log10(Pwr);
Pwr_dB(Pwr_dB < -60) = -60;

% 3. Construct true THz frequency axis
f_axis_Hz  = (-Nfft/2 : Nfft/2-1) * (Fs / Nfft) + nu_p;  % Hz
f_axis_THz = f_axis_Hz / 1e12;                         % THz

% 4. Plot
plot(f_axis_THz, Pwr_dB, 'b', 'LineWidth', 1.5);
xlabel('Frequency (THz)',     'FontSize',10);
ylabel('Normalized Optical Power (dB)', 'FontSize',10);
title('(f) Absolute Optical Spectrum',    'FontSize',10);
grid on;
set(gca,'FontSize',10);


% Plot 8: All Mode Photon Numbers
subplot(3,3,5);
plot(t_sel * 1e9, S_all(:,idx));
xlabel('Time (ns)');
ylabel('Photon Number');
title('All Mode Photon Numbers vs Time');
legend(arrayfun(@(m) sprintf('Mode %d', m), 1:N_modes, 'UniformOutput', false));

% plot 9:Power Spectrum (dB)
subplot(3,3,6);
plot(f_axis, P_fft_dB, 'LineWidth', 2);
xlabel('Offset Frequency (GHz)');
ylabel('Power Spectrum (dB)');
title('(i) RF Power Spectrum (from |E|^2)');
ylim([-50 0]);
grid on;
set(gca, 'FontSize', 10);

% Add overall title
sgtitle(sprintf('Multi-mode QCL Feedback Simulation (L_{ext}=%.2fm, \\kappa=%.2f)', L_ext, kappa), 'FontSize', 12, 'FontWeight', 'bold');

% Set save path
folder_path = 'results'; % You can replace with your own path
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
end

% Increase figure size (e.g., 1600x1200 pixels)
set(gcf, 'Position', [100, 100, 1600, 1200]);

% Save as high-resolution PNG
exportgraphics(gcf, 'Fig1_QCL_multimode.png', 'Resolution', 300);

% Save as scalable PDF (for publication)
exportgraphics(gcf, 'Fig1_QCL_multimode.pdf', 'ContentType', 'vector');

mean_photon = mean(S_all(:, idx), 2);  % Average photon number per mode
disp('Average photon number per mode:');
disp(mean_photon);

% Calculate steady-state average power per mode
steady_idx = t > max(t)/2;
P_avg = mean(P_out_modes(:, steady_idx), 2);
disp('Steady-state power per mode (mW):');
disp(P_avg * 1e3); % Convert to mW

% Plot mode power distribution
figure;
bar(1:N_modes, P_avg * 1e3);
xlabel('Mode Index');
ylabel('Power (mW)');
title('Mode Power Distribution');

disp('Static gain per mode:'); disp(G_static);

%% DDE Function (must be at the end or in separate file)
function dy = lreqns_QCL_OF_multimode(t, y, Z, I0)
    global q eta_3 eta_2 tau_3 tau_2 tau_32 tau_sp beta_sp G_static M alpha_factor;
    global omega_m omega0 delta_nu N_modes tau_ext tau_in kappa epsilon_g tau_p;
    
    % Extract variables
    N3 = y(1);
    N2 = y(2);
    S = y(3:2:2*N_modes+1);
    phi = y(4:2:2*N_modes+2);
    
    S_del = Z(3:2:2*N_modes+1);
    phi_del = Z(4:2:2*N_modes+2);
    
    % Gain saturation
    S_total = sum(S);
    G = zeros(N_modes, 1);
    for m = 1:N_modes
        G(m) = G_static(m) / (1 + epsilon_g * S_total);
    end
    
    % Carrier rate equations
    dN3dt = (eta_3 * I0 / q) - sum(G .* (N3 - N2) .* S) - N3 / tau_3;
    dN2dt = (eta_2 * I0 / q) + sum(G .* (N3 - N2) .* S) + N3 / tau_32 + N3 / tau_sp - N2 / tau_2;
    
    % Photon and phase equations for each mode
    dSdt = zeros(N_modes,1);
    dphidt = zeros(N_modes,1);
    
    for m = 1:N_modes
        fb = (2 * kappa / tau_in(m)) * sqrt(S(m) * S_del(m)) * cos(omega0(m) * tau_ext + phi(m) - phi_del(m));
        fb_phi = -(kappa / tau_in(m)) * sqrt(S_del(m) / S(m)) * sin(omega0(m) * tau_ext + phi(m) - phi_del(m));
        dSdt(m) = M * G(m) * (N3 - N2) * S(m) + M * beta_sp * N3 / tau_sp - S(m)/tau_p(m) + fb;
        dphidt(m) = (alpha_factor / 2) * (M * G(m) * (N3 - N2) - 1/tau_p(m)) + fb_phi;
    end
    
    dy = [dN3dt; dN2dt; reshape([dSdt dphidt], [], 1)];
end

