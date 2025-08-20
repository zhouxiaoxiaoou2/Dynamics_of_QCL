% === Scan Dynamics without classification ===
clear; clc; close all;

%% === Constants ===
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
S_base = 7.3e6 * 9.25;

% === Time grid ===
dt = 0.1e-12;
T_total = 1e-6;
tspan = 0:dt:T_total;       % 时间轴
fs = 1/dt;                  % 采样频率

%% === Scan parameters ===
Delta_finj_list = [7.0e9];
Sinj_list = [0.3, 0.9, 1.5, 2.1, 2.7];

if ~exist('output', 'dir')
    mkdir('output');
end

for DFi = 1:length(Delta_finj_list)
    for Ri = 1:length(Sinj_list)
        Delta_finj = Delta_finj_list(DFi);
        Sinj = Sinj_list(Ri) * S_base;

        S0 = 7.3e6;
        N3_ss = (eta*I/q) / (1/tau_32 + 1/tau_31 + G0*S0);
        N2_ss = N3_ss * (tau_21/tau_32) / (1 - G0*S0*tau_21);
        N1_ss = (N3_ss/tau_31 + N2_ss/tau_21) * tau_out;
        phi0 = 0.1 + 0.05*randn;    % 添加扰动
        y0 = [N3_ss*1.01; N2_ss*0.99; N1_ss; S0*1.05; phi0];

        options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
        [t_ode, y_ode] = ode15s(@(t,y) QCL_Rate_Eqns(t,y,eta,q,tau_32,tau_31,tau_21,tau_out,...
            tau_p,tau_sp,beta,G0,m,alphaH,kc,Sinj,Delta_finj,I), tspan, y0, options);

        % 插值统一时间轴
        t_uniform = tspan;
        S = interp1(t_ode, y_ode(:,4), t_uniform);
        delta_phi = interp1(t_ode, y_ode(:,5), t_uniform);

        idx_S = (t_uniform >= 100e-9) & (t_uniform <= 114e-9);
        t_sel = t_uniform(idx_S);
        S_sel = S(idx_S)/1e6;

        idx_phi = (t_uniform >= 100e-9) & (t_uniform <= 103e-9);
        t_phi = t_uniform(idx_phi);
        phi_sel = delta_phi(idx_phi) / pi;

        window = hamming(length(S_sel))';
        S_win = S_sel .* window;
        N_fft = 2^nextpow2(length(S_win));
        f_fft = (-fs/2:fs/N_fft:fs/2-fs/N_fft)/1e9;
        S_fft = abs(fftshift(fft(S_win, N_fft))).^2;
        S_fft_db = 10*log10(S_fft / max(S_fft));

        env = abs(hilbert(S_sel));
        dS = gradient(env, mean(diff(t_sel)));
        N_beat = 2^nextpow2(length(dS));
        beat_fft = abs(fft(dS, N_beat)).^2;
        beat_fft_db = 10*log10(beat_fft / max(beat_fft));
        f_beat = (0:N_beat/2-1)*fs/N_beat/1e9;

%% === Improved Safe Naming and Folder ===
base_name = sprintf('Finj_%.1fGHz_Sinj_%.2f', Delta_finj/1e9, Sinj/S_base);
safe_name = regexprep(base_name, '[^a-zA-Z0-9=._-]', '_');  % 空格或符号替换成 "_"

% Create output folder for this case
folder_path = fullfile('output', safe_name);
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
end

% Prepare zoomed waveform (FIXED HERE)
t_zoom = t_sel(t_sel >= 100e-9 & t_sel <= 114e-9);
S_zoom = S_sel(t_sel >= 100e-9 & t_sel <= 114e-9);

% Save plots
fig = figure('Position', [100, 100, 1200, 500]);  % 宽度提升到1800
clf;
ax1=subplot(1,3,1);
plot(t_zoom*1e9, S_zoom, 'b', 'LineWidth', 1.2); grid on;
ylabel('Photon Number (×10^6)');
title('Time Domain'); xlim([100 114]);

ax2=subplot(1,3,2);
plot(f_fft, S_fft_db, 'b', 'LineWidth', 1.2); grid on;
ylabel('Norm. Power (dB)');
title('Optical Spectrum'); xlim([-40 40]); ylim([-120 0]);

ax3=subplot(1,3,3);
plot(f_beat, beat_fft_db(1:N_beat/2), 'b', 'LineWidth', 1.2); grid on;
xlabel('Frequency (GHz)'); ylabel('Power (dB)');
title('Beat Spectrum'); xlim([0 80]); ylim([-120 0]);

% 手动调宽每个子图
set(ax1, 'Position', [0.05, 0.15, 0.25, 0.75]);  % 左图
set(ax2, 'Position', [0.37, 0.15, 0.25, 0.75]);  % 中图
set(ax3, 'Position', [0.69, 0.15, 0.25, 0.75]);  % 右图

% Save both PNG and FIG formats
saveas(gcf, fullfile(folder_path, 'Combined_Plots.png'));
saveas(gcf, fullfile(folder_path, 'Combined_Plots.fig'));

% Save data
save(fullfile(folder_path, 'data.mat'), 't_sel', 'S_sel', 'f_fft', 'S_fft_db', 'f_beat', 'beat_fft_db');
    end
end

%% === 3D 光频谱图 ===
figure('Position', [100, 100, 800, 600]);
hold on;
colors = jet(length(Sinj_list));

for Ri = 1:length(Sinj_list)
    % 构造安全文件夹名
    Sinj_ratio = Sinj_list(Ri);                 % 原始注入强度比例
    base_name = sprintf('Finj_%.1fGHz_Sinj_%.2f', Delta_finj/1e9, Sinj_ratio);
    safe_name = regexprep(base_name, '[^a-zA-Z0-9=._-]', '_');
    folder_path = fullfile('output', safe_name);

    % 加载数据文件
    data_file = fullfile(folder_path, 'data.mat');
    if exist(data_file, 'file')
        S = load(data_file, 'f_fft', 'S_fft_db');
        plot3(S.f_fft, Sinj_ratio * ones(size(S.f_fft)), S.S_fft_db, ...
              'Color', colors(Ri,:), 'LineWidth', 1.5);
    else
        warning('File not found: %s', data_file);
    end
end

grid on;
xlabel('Offset Frequency (GHz)');
ylabel('Injection Ratio S_{inj}');
zlabel('Normalized Power (dB)');
title('3D Optical Spectrum vs Injection Ratio');
xlim([-40 40]); zlim([-120 0]);
view(45, 30);
saveas(gcf, 'output/3D_Optical_Spectrum_Curves.png');
savefig(gcf, 'output/3D_Optical_Spectrum_Curves.fig');

%% === Dynamics Function ===
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
