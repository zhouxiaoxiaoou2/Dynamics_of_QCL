% scan_dynamics_4params.m
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
I_th = 230e-3;
I = 1.5 * I_th;
S0 = 7.3e6;

% === Time grid ===
dt = 0.1e-12;
T_total = 1e-6;
t_uniform = 0:dt:T_total;

% === 四参数扫描设置 ===
Delta_finj_list = [1.5e9, 3e9, 6e9];         % 频率失谐 (Hz)
Rinj_dB_list = [-10, 0, 10];                 % 注入功率 (dB)
alphaH_list = [0.5, 2.0, 3.5];               % 线宽增强因子
kc_list = logspace(10, 11, 3);               % 耦合系数 (1e10~1e11)

% === 创建参数组合网格 ===
[DF_grid, R_grid, A_grid, K_grid] = ndgrid(...
    Delta_finj_list, Rinj_dB_list, alphaH_list, kc_list);
param_combinations = [DF_grid(:), R_grid(:), A_grid(:), K_grid(:)];
total_combinations = size(param_combinations, 1);

% === 创建输出目录 ===
output_dir = 'output_4params';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% === 并行计算准备 ===
if isempty(gcp('nocreate'))
    parpool(min([4, feature('numcores')])); % 根据CPU核心数开启并行
end

% === 主扫描循环 (并行) ===
parfor combo_idx = 1:total_combinations
    % 获取当前参数组合
    Delta_finj = param_combinations(combo_idx, 1);
    Rinj_dB = param_combinations(combo_idx, 2);
    alphaH = param_combinations(combo_idx, 3);
    kc = param_combinations(combo_idx, 4);
    
    Rinj = 10^(Rinj_dB / 10);
    Sinj = Rinj * S0;

    % === 模拟求解 ===
    y0 = [0; 0; 0; S0*1.05; 0.1];
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
    [t_ode, y_ode] = ode15s(@(t,y) QCL_Rate_Eqns(t,y,eta,q,tau_32,tau_31,tau_21,tau_out,...
        tau_p,tau_sp,beta,G0,m,alphaH,kc,Sinj,Delta_finj,I), [0 T_total], y0, options);

    % === 数据分析 ===
    S = interp1(t_ode, y_ode(:,4), t_uniform);
    window_idx = (t_uniform >= 100e-9) & (t_uniform <= 114e-9);
    t_sel = t_uniform(window_idx);
    S_sel = S(window_idx)/1e6;

    % FFT计算
    fs = 1/dt;
    N_fft = 2^nextpow2(length(S_sel));
    S_fft = abs(fft(S_sel, N_fft)).^2;
    S_fft_db = 10*log10(S_fft / max(S_fft));
    f_fft = (-fs/2:fs/N_fft:fs/2 - fs/N_fft)/1e9;

    % 拍频计算
    env = abs(hilbert(S_sel));
    dS = gradient(env, mean(diff(t_sel)));
    N_beat = 2^nextpow2(length(dS));
    beat_fft = abs(fft(dS, N_beat)).^2;
    beat_fft_db = 10*log10(beat_fft / max(beat_fft));
    f_beat = (0:N_beat/2-1)*fs/N_beat/1e9;

    % === 动态状态分类 ===
    period_label = classify_periodicity(t_sel, S_sel);
    param_str = sprintf('Δf=%.1fGHz R=%.0fdB α=%.1f kc=%.1e', ...
        Delta_finj/1e9, Rinj_dB, alphaH, kc);
    fprintf('[%d/%d] %s → %s\n', combo_idx, total_combinations, param_str, period_label);

    % === 组合绘图 ===
    fig = figure('Visible', 'off', 'Position', [100, 100, 900, 700]);
    
    % 子图1: 时域波形
    subplot(3,1,1);
    plot(t_sel*1e9, S_sel, 'b', 'LineWidth', 1);
    title(sprintf('Time Domain - %s - %s', param_str, period_label));
    xlabel('Time (ns)'); ylabel('Photon (×10^6)');
    xlim([100 114]); grid on;
    
    % 子图2: 光谱
    subplot(3,1,2);
    plot(f_fft, fftshift(S_fft_db), 'r');
    title('Optical Spectrum');
    xlabel('Freq (GHz)'); ylabel('Power (dB)');
    xlim([-40 40]); ylim([-120 0]); grid on;
    
    % 子图3: 拍频
    subplot(3,1,3);
    plot(f_beat, beat_fft_db(1:N_beat/2), 'k');
    title('Beat Note Spectrum');
    xlabel('Freq (GHz)'); ylabel('Power (dB)');
    xlim([0 80]); ylim([-120 0]); grid on;
    
    % === 保存结果 ===
    fname = sprintf('DF%.1f_R%d_A%.1f_K%.0e', ...
        Delta_finj/1e9, Rinj_dB, alphaH, kc);
    saveas(fig, fullfile(output_dir, [fname '.png']));
    close(fig);
    
    % 保存数据（使用临时变量避免parfor冲突）
    temp_data = struct();
    temp_data.t = t_sel;
    temp_data.S = S_sel;
    temp_data.f_fft = f_fft;
    temp_data.S_fft_db = S_fft_db;
    temp_data.f_beat = f_beat;
    temp_data.beat_fft_db = beat_fft_db;
    temp_data.label = period_label;
    temp_data.parameters = param_combinations(combo_idx, :);
    
    parsave(fullfile(output_dir, [fname '.mat']), temp_data);
end

% === 辅助函数 ===
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
    if length(zero_crossings) < 2
        period_label = 'Unclear';
        return;
    end
    time_periods = diff(t(zero_crossings));
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

function parsave(fname, data)
    save(fname, '-struct', 'data');
end