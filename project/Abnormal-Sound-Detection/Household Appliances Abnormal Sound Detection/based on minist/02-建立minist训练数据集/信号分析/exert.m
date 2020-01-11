% 功能:短时傅里叶变换    
% 日期：2019,6,3
clear;
clc;
close all;
% ======================= input signal ==========================
dir = ['..\洗衣机提取数据\20171030海尔洗衣机实验室数据-故障\no_9关盖脱水电机异常声\关盖脱水电机异常声-',num2str(23),'.mat'];
% dir = ['..\洗衣机提取数据\20171030海尔洗衣机生产线数据-正常\no_3关盖脱水噪声跟着生产线走\关盖脱水噪声跟着生产线走-',num2str(4),'.mat'];
data = load(dir); 
fs = 65536;
left=1;
right=length(data.Channel_1_Data);
% left=fs * 0.0778 * 6;
% right = fs * 0.0778 * 10;
x = data.Channel_1_Data(left : right);%对正常-1限制数据区域
%===========降采样到16000Hz==========
% x = resample(x,16000,fs);
% fs = 16000;

% fid=fopen('D:\abnormal_2.pcm','wb');%存为raw，也就是pcm格式
% fwrite(fid,x,'float');
% ==================== 参数设置 ===================================
% 将帧长，窗长，nfft点数设置成相同。
% nfft: fft的长度，可以和FFT的长度不同，以获得更高的频率分辨率，但必须大>=win的长度
wlen = floor(fs*0.03);             % 帧长，每一帧一般取10~30ms,这里约取30ms,根据采样率算出fs*0.03约等于1966
inc = floor(wlen/3);                  % 帧移,我自己定的，约帧长的三分之一
win = hanning(wlen);        % 窗类型
nfft = wlen;                % nfft的点数
N = length(x);
time = (0: N -1)/fs;
y = enframe(x,win,inc)';   % 帧长*帧数
fn = size(y,2);            % 帧数
frameTime=(((1:fn)-1)*inc+wlen/2)/fs;    % 求出每帧对应的时间，即取这一帧数据中间位置的时间
n2 = 1:wlen/2+1;           % 由于共轭对称，取数据的一半
freq = (n2 - 1)*fs/wlen;   % 计算fft后的频率刻度
Y = fft(y);                % 短时傅里叶变换
clf                        % 初始化图形

% ======================= 时域图  ===============================
% axes('Position',[0.07 0.72 0.9 0.22]);
subplot(2,2,1);
plot(time,x, 'k')
xlim([0 max(time)]);
xlabel('时间/s');
ylabel('幅值');
title('语音信号波形');
% ======================= 语谱图  ===============================
set(gcf, 'Position',[20 100 600 500]);
% axes('Position',[0.1 0.1 0.8 0.5]);
subplot(2,2,4);
imagesc(frameTime, freq(:,3),abs(Y(n2,:)));
axis xy;
ylabel('频率/Hz');
xlabel('时间/s');
title('语谱图');
LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0 ];
black = [0.5 0.7 1];
colors = [LightYellow; MidRed; black];
m = 64;
colormap(SpecColorMap(m, colors));  % m指在着色分配中设置多少等级
% ======================= 傅里叶变换 不分帧，频谱===========================
C=x;
N = length(C);
T=(1/fs)*(N-1);
f=1/T;
vecf=(0:N-1)*f;
C = (C-mean(C)).*hanning(N); % 去直流后加窗
C = 2*abs(fft(C,N)/N);
C = C(1:N/10);
subplot(2,2,3);plot(vecf(1:N/10),C);
axis([50 500 -inf 0.07])
ylabel('声压/Pa');
xlabel('频率/Hz');
title('不分帧频谱');
% ======================= 分帧加窗 频谱 求均值和方差===========================
% 每帧加窗求fft
Y = 2*abs(Y((1:1+nfft/2),:))/wlen; % 每一列是一帧STFT的数组，复数，保留1~nfft/2+1个频率分量
n2 = 1:wlen/2+1;           % 由于共轭对称，取数据的一半
freq = (n2 - 1)*fs/wlen;   % 计算fft后的频率刻度

subplot(2,2,2);
plot(freq, Y);
axis([50 500 -inf inf])
ylabel('声压/Pa');
xlabel('频率/Hz');
title('分帧频谱');


figure(2);
subplot(1,2,1);
plot(1:size(Y,2),mean(Y))   % 均值
axis([-inf inf 0.4e-3 4e-3])
ylabel('均值');
xlabel('帧数');
title('分帧频谱计算各帧的均值');

% 求方差sigmax2,除以n
% n = size(Y,1);
% mu = (1/n).*sum(Y);
% sigmax = (1/n).*sum((Y-mu).^2);
sigmax = var(Y,1);%调用MATLAB工具
subplot(1,2,2);
plot(1:size(Y,2),sigmax)       % 方差
axis([-inf inf 0 1.5e-3])
ylabel('方差');
xlabel('帧数');
title('分帧频谱计算各帧的方差');





