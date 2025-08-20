clc; clear;

% ========= parameter scanning =========
Delta_finj_list = 1e9:0.5e9:10e9;         %  detuning (Hz)
Sinj_factor_list = 0.3:0.3:3.0;            % Intensity
output_dir = 'scan_output_P_class_extended';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% ========= constants =========
q = 1.602e-19; eta = 0.5; m = 30;
tau_32 = 2.0e-12; tau_31 = 2.4e-12; tau_21 = 0.5e-12;
tau_out = 0.5e-12; tau_p = 3.7e-12; tau_sp = 7.0e-9;
beta = 1e-6; G0 = 5.3e4; alphaH = 0.5;
kc0 = 2.0e10; Sinj_base = 7.3e6 * 9.25;
I_th = 223e-3; I = 1.5 * I_th;

% ========= time =========
T_total = 3e-6;             % Total simulation time: 3 microseconds
dt = 0.1e-12;               % Time step
tspan = 0:dt:T_total;       % Time axis
fs = 1/dt;                  % Sampling frequency

% ========== start scanning ==========
for i = 1:length(Delta_finj_list)
    for j = 1:length(Sinj_factor_list)

        Delta_finj = Delta_finj_list(i);
        Sinj = Sinj_base * Sinj_factor_list(j);
        kc = kc0;

        % ========= phase noise =========
S0 = 7.3e6;
N3_ss = (eta*I/q) / (1/tau_32 + 1/tau_31 + G0*S0);
N2_ss = N3_ss * (tau_21/tau_32) / (1 - G0*S0*tau_21);
N1_ss = (N3_ss/tau_31 + N2_ss/tau_21) * tau_out;
phi0 = 0.1 + 0.05*randn;    % 添加扰动
y0 = [N3_ss*1.01; N2_ss*0.99; N1_ss; S0*1.05; phi0];

        opts = odeset('RelTol',1e-6,'AbsTol',1e-9);
        [t_ode, y_ode] = ode15s(@(t,y) QCL_Rate_Eqns_single(t, y, eta, q, ...
            tau_32, tau_31, tau_21, tau_out, tau_p, tau_sp, ...
            beta, G0, m, alphaH, kc, Sinj, Delta_finj, I), ...
            tspan, y0, opts);

        % Interpolation
        t_uniform = tspan;
        S = interp1(t_ode, y_ode(:,4), t_uniform);
        delta_phi = interp1(t_ode, y_ode(:,5), t_uniform);

        % ================= time window =================
        idx_S = (t_uniform >= 100e-9) & (t_uniform <= 114e-9);
        t_S = t_uniform(idx_S);
        S_sel = S(idx_S)/1e6;

        idx_phi = (t_uniform >= 100e-9) & (t_uniform <= 103e-9);
        t_phi = t_uniform(idx_phi);
        phi_sel = delta_phi(idx_phi) / pi;

        % ================= FFT：windowing + dB =================
        window = hamming(length(S_sel));
        S_win = S_sel .* window;
        N_fft = 2^nextpow2(length(S_win));
        f_fft = (-fs/2:fs/N_fft:fs/2-fs/N_fft)/1e9;
        S_fft = abs(fftshift(fft(S_win, N_fft))).^2;
        S_fft_db = 10*log10(S_fft / max(S_fft));

        % ================= Beat Note Extraction =================
        envelope = abs(hilbert(S_sel));
        dS_dt = gradient(envelope, mean(diff(t_S)));
        N_beat = 2^nextpow2(length(dS_dt));
        beat_fft = abs(fft(dS_dt, N_beat)).^2;
        beat_fft_db = 10*log10(beat_fft / max(beat_fft));
        f_beat = (0:N_beat/2-1)*fs/N_beat/1e9;
        [~, idx_peak] = max(beat_fft_db(2:round(end/5)));
        beat_freq_val = f_beat(idx_peak + 1);

        % ================= P classification =================
        label = classify_periodicity(S_sel, t_S);

        % ===== figure plot=====
title_str = sprintf('Injection: \\Deltaf = %.1f GHz, Sinj = %.1f × base, State = %s', ...
                    Delta_finj/1e9, Sinj_factor_list(j), label);
fig = figure('Visible','off', 'Position', [100, 100, 1200, 800]);

subplot(2,2,1);
plot(t_S*1e9, S_sel, 'b');
xlabel('Time (ns)'); ylabel('Photon Number (×10^6)');
title('(a-i) S(t)');

subplot(2,2,2);
plot(t_phi*1e9, phi_sel, 'r');
xlabel('Time (ns)'); ylabel('\phi/\pi');
title('(a-ii) Phase');

subplot(2,2,3);
plot(f_fft, S_fft_db, 'k');
xlabel('Frequency (GHz)'); ylabel('Norm. Power (dB)');
title('(a-iii) FFT');
xlim([-40 40]); ylim([-120 0]);

subplot(2,2,4);
plot(f_beat, beat_fft_db(1:N_beat/2), 'b');
xlabel('GHz'); ylabel('Norm. Elec. Power (dB)');
title(sprintf('(a-iv) Beat Spectrum (%.2f GHz)', beat_freq_val));
xlim([0 50]); ylim([-120 0]);

sgtitle(title_str, 'Interpreter', 'tex');  % More aesthetically pleasing support for Δf

% Clearer file names
basename = sprintf('df%.1fGHz_Sinj%.1f_%s', Delta_finj/1e9, Sinj_factor_list(j), label);
saveas(fig, fullfile(output_dir, [basename '.png']));
savefig(fig, fullfile(output_dir, [basename '.fig']));
close(fig);

    end
end

function label = classify_periodicity(S, t)
    [pks, ~] = findpeaks(S, 'MinPeakProminence', 0.01);
    N = length(pks);
    if N < 8
        label = 'chaos';
        return;
    end

    % Normalize
    pks = pks - mean(pks);
    pks = pks / max(abs(pks));

    % New P1 criterion: all peak values fluctuate very little, and there is no structural difference between groups
    mean_diff = max(pks) - min(pks);
    if std(pks) < 0.1 && mean_diff < 0.05
        label = 'P1';
        return;
    end

    % P2 judgment: odd and even peaks alternate
    odd = pks(1:2:end);
    even = pks(2:2:end);
    if std(odd - mean(odd)) < 0.05 && std(even - mean(even)) < 0.05 && abs(mean(odd) - mean(even)) > 0.05
        label = 'P2';
        return;
    end

    % P4 judgment
    if N >= 12
        groups = {pks(1:4:end), pks(2:4:end), pks(3:4:end), pks(4:4:end)};
        lengths = cellfun(@length, groups);
        min_len = min(lengths);
        groups_trimmed = cellfun(@(v) v(1:min_len), groups, 'UniformOutput', false);
        group_matrix = cell2mat(groups_trimmed);
        if all(std(group_matrix, 0, 2) < 0.05)
            label = 'P4';
            return;
        end
    end

    % P8 判定
    if N >= 20
        groups = arrayfun(@(k) pks(k:8:end), 1:8, 'UniformOutput', false);
        lengths = cellfun(@length, groups);
        min_len = min(lengths);
        groups_trimmed = cellfun(@(v) v(1:min_len), groups, 'UniformOutput', false);
        group_matrix = cell2mat(groups_trimmed);
        if all(std(group_matrix, 0, 2) < 0.05)
            label = 'P8';
            return;
        end
    end

    label = 'chaos';
end

% ========= 方程函数 =========
function dydt = QCL_Rate_Eqns_single(t, y, eta, q, tau_32, tau_31, tau_21, tau_out, ...
    tau_p, tau_sp, beta, G0, m, alphaH, kc, Sinj, Delta_finj, I)

    N3 = y(1); N2 = y(2); N1 = y(3); S = y(4); phi = y(5);

    dN3 = eta * I/q - N3/tau_32 - N3/tau_31 - G0*(N3 - N2)*S;
    dN2 = N3/tau_32 - N2/tau_21 + G0*(N3 - N2)*S;
    dN1 = N3/tau_31 + N2/tau_21 - N1/tau_out;

    DeltaN = N3 - N2;
    dS = (m*G0*DeltaN - 1/tau_p)*S + m*beta*N3/tau_sp + 2*kc*sqrt(Sinj*S)*cos(phi);
    dphi = (alphaH/2)*(m*G0*DeltaN - 1/tau_p) - kc*sqrt(Sinj/S)*sin(phi) - 2*pi*Delta_finj;

    dydt = [dN3; dN2; dN1; dS; dphi];
end
