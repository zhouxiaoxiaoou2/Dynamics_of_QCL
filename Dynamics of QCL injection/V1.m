clear; clc; close all;

%% ========== Constants Set  ==========
q = 1.602e-19;
eta = 0.5;
m = 30;
tau_32 = 2.0e-12;
tau_31 = 2.4e-12;
tau_21 = 0.5e-12;
tau_out = 0.5e-12;
tau_p = 3.7e-12;
tau_sp = 7.0e-9;
beta = 1.0e-6;
G0 = 5.3e4;
alphaH = 0.5;
kc = 2.0e10;
Delta_finj = 4.2e9;

% === Define Pump Current ===
I_th = 223e-3; % Threshold current in A
I = 1.5 * I_th; % Injected current

% === Injection ===
S0 = 7.3e6;                       % Steady-state photon number
Rinj_dB = 10;                      % Injection ratio (dB)
Rinj = 10^(Rinj_dB / 10);         % Convert to linear scale
Sinj = Rinj * S0;               % Injected photon number


%% ========== Simulation Settings ==========
dt = 0.1e-12;
T_total = 1e-6; % 1 microsecond
t_uniform = 0:dt:T_total; % For interpolation and FFT


%% ========== Initial Conditions ==========
% Free-running
S0 = 7.3e6; % Steady-state photon number
y0 = [0; 0; 0; S0*1.05; 0.1];


% Define options for ode15 (adjust tolerances if needed)
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
% Solve using ode15
[t_ode, y_ode] = ode15s(@(t,y) QCL_Rate_Eqns(t,y,eta,q,tau_32,tau_31,tau_21,tau_out,...
tau_p,tau_sp,beta,G0,m,alphaH,kc,Sinj,Delta_finj,I), ...
[0 T_total], y0, options);

% Interpolate results to uniform time grid
S = interp1(t_ode, y_ode(:,4), t_uniform);
delta_phi = interp1(t_ode, y_ode(:,5), t_uniform);



%% ========== Select time windows ==========
% For photon number (100-114 ns)
idx_S = (t_uniform >= 100e-9) & (t_uniform <= 114e-9);
t_S = t_uniform(idx_S);
S_sel = S(idx_S)/1e6; % Convert to units of 10^6
% For phase difference (100-114 ns)
idx_phi = (t_uniform >= 100e-9) & (t_uniform <= 114e-9);
t_phi = t_uniform(idx_phi);
delta_phi_sel = delta_phi(idx_phi)/pi; % Convert to units of π


%% ========== Plot:  Photon Number ==========
figure(1);
plot(t_S*1e9, S_sel, 'b', 'LineWidth', 1.5);
xlabel('Time (ns)'); 
ylabel('Photon Number (×10^6)');
title(' Photon Number vs Time ');
xlim([100 114]);
grid on;
set(gca, 'FontSize', 12);


%% ========== Plot:  Phase Difference ==========
figure(2);
plot(t_phi*1e9, delta_phi_sel, 'b', 'LineWidth', 1.5);
xlabel('Time (ns)'); 
ylabel('\Delta\phi (\pi)');
title(' Phase Difference vs Time ');
xlim([100 114]);
grid on;
set(gca, 'FontSize', 12);

%% ========== FFT for Optical Spectrum ==========
% --- Signal windowing ---
window = hamming(length(S_sel));  % Apply Hamming window to suppress spectral leakage
S_win = S_sel .* window;
fs = 1/dt;
N_fft = 2^nextpow2(length(S_sel));
S_fft = abs(fft(S_sel, N_fft)).^2;
S_fft_db = 10*log10(S_fft/max(S_fft)); % Normalized dB
f = (-fs/2:fs/N_fft:fs/2-fs/N_fft)/1e9; % Frequency in GHz

%% ========== Plot: Optical Spectrum ==========
figure(3);
plot(f, fftshift(S_fft_db), 'b', 'LineWidth', 1.5);
xlabel('Offset Frequency (GHz)');
ylabel('Normalized Power (dB)');
title(' Optical Spectrum ');
xlim([-40 40]);
ylim([-120 0]);
grid on;
set(gca, 'FontSize', 12);

%% ========== Beat Note Spectrum ==========
% Calculate envelope via Hilbert transform
envelope = abs(hilbert(S_sel));
dS_dt = gradient(envelope, mean(diff(t_S)));
N_beat = 2^nextpow2(length(dS_dt));
beat_fft = abs(fft(dS_dt, N_beat)).^2;
beat_fft_db = 10*log10(beat_fft/max(beat_fft)); % Normalized dB
f_beat = (0:N_beat/2-1)*fs/N_beat/1e9; % Positive frequencies in GHz
%% ========== Plot: Beat Note Spectrum ==========
figure(4);
plot(f_beat, beat_fft_db(1:N_beat/2), 'b', 'LineWidth', 1.5);
xlabel('Microwave Frequency (GHz)');
ylabel('Normalized Electrical Power (dB)');
title(' Beat Note Spectrum ');
xlim([0 80]);
ylim([-120 0]);
grid on;
set(gca, 'FontSize', 12);



%% ========== Rate Equation Definition ==========
function dydt = QCL_Rate_Eqns(t, y, eta, q, tau_32, tau_31, tau_21, tau_out, ...
tau_p, tau_sp, beta, G0, m, alphaH, kc, Sinj, Delta_finj, I)
% Unpack variables
N3 = y(1); N2 = y(2); N1 = y(3); S = y(4); delta_phi = y(5);
% Carrier equations
dN3 = eta * I/q - N3/tau_32 - N3/tau_31 - G0*(N3 - N2)*S;
dN2 = N3/tau_32 - N2/tau_21 + G0*(N3 - N2)*S;
dN1 = N3/tau_31 + N2/tau_21 - N1/tau_out;
% Photon and phase equations
DeltaN = N3 - N2;
dS = (m*G0*DeltaN - 1/tau_p)*S + m*beta*N3/tau_sp + 2*kc*sqrt(Sinj*S)*cos(delta_phi);
dphi = (alphaH/2)*(m*G0*DeltaN - 1/tau_p) - kc*sqrt(Sinj/S)*sin(delta_phi) - 2*pi*Delta_finj;
dydt = [dN3; dN2; dN1; dS; dphi];
end