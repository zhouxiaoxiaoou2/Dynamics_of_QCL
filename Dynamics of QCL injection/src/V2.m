% scan_dynamics.m
clear; clc; close all;

% === Constants ===
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
I_th = 230e-3;
I = 1.5 * I_th;
S0 = 7.3e6;

% === Time grid ===
dt = 0.1e-12;
T_total = 1e-6;
t_uniform = 0:dt:T_total;

% === Scan parameters ===
Delta_finj_list = [4.2e9];  % Hz
Rinj_dB_list = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 9, 10, 11 12, 14];                     % dB

if ~exist('output', 'dir')
    mkdir('output');
end

% === Loop ===
for DFi = 1:length(Delta_finj_list)
    for Ri = 1:length(Rinj_dB_list)
        Delta_finj = Delta_finj_list(DFi);
        Rinj_dB = Rinj_dB_list(Ri);
        Rinj = 10^(Rinj_dB / 10);
        Sinj = Rinj * S0;

        % Initial condition
        y0 = [0; 0; 0; S0*1.05; 0.1];

        % ODE solve
        options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
        [t_ode, y_ode] = ode15s(@(t,y) QCL_Rate_Eqns(t,y,eta,q,tau_32,tau_31,tau_21,tau_out,...
            tau_p,tau_sp,beta,G0,m,alphaH,kc,Sinj,Delta_finj,I), [0 T_total], y0, options);

        % Interpolation
        S = interp1(t_ode, y_ode(:,4), t_uniform);
        delta_phi = interp1(t_ode, y_ode(:,5), t_uniform);

        % Window
        idx = (t_uniform >= 100e-9) & (t_uniform <= 114e-9);
        t_sel = t_uniform(idx);
        S_sel = S(idx)/1e6;

        % FFT spectrum
        fs = 1/dt;
        N_fft = 2^nextpow2(length(S_sel));
        S_fft = abs(fft(S_sel, N_fft)).^2;
        S_fft_db = 10*log10(S_fft / max(S_fft));
        f_fft = (-fs/2:fs/N_fft:fs/2 - fs/N_fft)/1e9;

        % Beat note
        env = abs(hilbert(S_sel));
        dS = gradient(env, mean(diff(t_sel)));
        N_beat = 2^nextpow2(length(dS));
        beat_fft = abs(fft(dS, N_beat)).^2;
        beat_fft_db = 10*log10(beat_fft / max(beat_fft));
        f_beat = (0:N_beat/2-1)*fs/N_beat/1e9;

        % === Plot & Save ===
prefix = sprintf('Finj%.1fGHz_Rinj%+ddB', Delta_finj/1e9, Rinj_dB);
period_label = classify_periodicity(t_sel, S_sel);
fprintf('[%s] %s\n', prefix, period_label);

% create output directory
fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 400]); % Width 1200, Height 400

% Subplot 1: Time Domain Waveform (1 row 3 columns, 1st)
subplot(1,3,1); % Change to 1 row 3 columns
t_zoom = t_sel(t_sel >= 100e-9 & t_sel <= 104e-9);
S_zoom = S_sel(t_sel >= 100e-9 & t_sel <= 104e-9);
plot(t_zoom*1e9, S_zoom, 'b', 'LineWidth', 1.2); grid on;
ylabel('Photon Number (×10^6)');
title(['Time Domain - ' period_label]);
xlim([100 104]);
h1 = gca;

% Subplot 2: Optical Spectrum (1 row 3 columns, 2nd)
subplot(1,3,2);
plot(f_fft, fftshift(S_fft_db), 'r', 'LineWidth', 1.2); grid on;
ylabel('Norm. Power (dB)');
title('Optical Spectrum');
xlim([-40 40]); ylim([-120 0]);
h2 = gca;

% Subplot 3: Beat Spectrum (1 row 3 columns, 3rd)
subplot(1,3,3);
plot(f_beat, beat_fft_db(1:N_beat/2), 'k', 'LineWidth', 1.2); grid on;
xlabel('Frequency (GHz)'); ylabel('Power (dB)');
title('Beat Spectrum');
xlim([0 80]); ylim([-120 0]);
h3 = gca;

% Adjust subplot positions
set(gcf, 'Position', [100, 100, 1200, 400]); % Ensure sufficient width
set(h1, 'Position', [0.05 0.15 0.25 0.75]);  % [left bottom width height]
set(h2, 'Position', [0.37 0.15 0.25 0.75]);
set(h3, 'Position', [0.69 0.15 0.25 0.75]);

% Save combined figure
saveas(fig, ['output/Horizontal_Combined_' prefix '.png']);
savefig(fig, ['output/Horizontal_Combined_' prefix '.fig']);

% Save individual subplots (optional)
saveas(h1.Parent, ['output/' prefix '_TimeDomain.fig']);
saveas(h2.Parent, ['output/' prefix '_OpticalSpectrum.fig']);
saveas(h3.Parent, ['output/' prefix '_BeatSpectrum.fig']);

close(fig);
        % Save data
        save(['output/Data_' prefix '.mat'], 't_sel', 'S_sel', 'f_fft', 'S_fft_db', 'f_beat', 'beat_fft_db', 'period_label');
    end
end

% ========== Nested Function ==========
function dydt = QCL_Rate_Eqns(t, y, eta, q, tau_32, tau_31, tau_21, tau_out, ...
    tau_p, tau_sp, beta, G0, m, alphaH, kc, Sinj, Delta_finj, I)
    N3 = y(1); N2 = y(2); N1 = y(3); S = y(4); dphi = y(5);
    DeltaN = N3 - N2;
    dN3 = eta*I/q - N3/tau_32 - N3/tau_31 - G0*DeltaN*S;
    dN2 = N3/tau_32 - N2/tau_21 + G0*DeltaN*S;
    dN1 = N3/tau_31 + N2/tau_21 - N1/tau_out;
    dS = (m*G0*DeltaN - 1/tau_p)*S + m*beta*N3/tau_sp + 2*kc*sqrt(Sinj*S)*cos(dphi);
    ddphi = (alphaH/2)*(m*G0*DeltaN - 1/tau_p) - kc*sqrt(Sinj/S)*sin(dphi) - 2*pi*Delta_finj;
    dydt = [dN3; dN2; dN1; dS; ddphi];
end

function period_label = classify_periodicity(t, S)
    S = S - mean(S);
    zero_crossings = find(diff(S > 0));
    time_periods = diff(t(zero_crossings));
    
    if length(time_periods) < 2
        period_label = 'Unclear';
        return;
    end
    
    std_T = std(time_periods);
    mean_T = mean(time_periods);
    rel_std = std_T / mean_T;
    
    if rel_std < 0.05
        period_label = 'P1';
    elseif rel_std < 0.15
        period_label = 'P2';
    elseif rel_std < 0.25
        period_label = 'P4';
    elseif rel_std < 0.35
        period_label = 'P8';
    else
        period_label = 'Chaos';
    end
end