% ����:��ʱ����Ҷ�任    
% ���ڣ�2019,6,3
clear;
clc;
close all;
% ======================= input signal ==========================
dir = ['..\ϴ�»���ȡ����\20171030����ϴ�»�ʵ��������-����\no_9�ظ���ˮ����쳣��\�ظ���ˮ����쳣��-',num2str(23),'.mat'];
% dir = ['..\ϴ�»���ȡ����\20171030����ϴ�»�����������-����\no_3�ظ���ˮ����������������\�ظ���ˮ����������������-',num2str(4),'.mat'];
data = load(dir); 
fs = 65536;
left=1;
right=length(data.Channel_1_Data);
% left=fs * 0.0778 * 6;
% right = fs * 0.0778 * 10;
x = data.Channel_1_Data(left : right);%������-1������������
%===========��������16000Hz==========
% x = resample(x,16000,fs);
% fs = 16000;

% fid=fopen('D:\abnormal_2.pcm','wb');%��Ϊraw��Ҳ����pcm��ʽ
% fwrite(fid,x,'float');
% ==================== �������� ===================================
% ��֡����������nfft�������ó���ͬ��
% nfft: fft�ĳ��ȣ����Ժ�FFT�ĳ��Ȳ�ͬ���Ի�ø��ߵ�Ƶ�ʷֱ��ʣ��������>=win�ĳ���
wlen = floor(fs*0.03);             % ֡����ÿһ֡һ��ȡ10~30ms,����Լȡ30ms,���ݲ��������fs*0.03Լ����1966
inc = floor(wlen/3);                  % ֡��,���Լ����ģ�Լ֡��������֮һ
win = hanning(wlen);        % ������
nfft = wlen;                % nfft�ĵ���
N = length(x);
time = (0: N -1)/fs;
y = enframe(x,win,inc)';   % ֡��*֡��
fn = size(y,2);            % ֡��
frameTime=(((1:fn)-1)*inc+wlen/2)/fs;    % ���ÿ֡��Ӧ��ʱ�䣬��ȡ��һ֡�����м�λ�õ�ʱ��
n2 = 1:wlen/2+1;           % ���ڹ���Գƣ�ȡ���ݵ�һ��
freq = (n2 - 1)*fs/wlen;   % ����fft���Ƶ�ʿ̶�
Y = fft(y);                % ��ʱ����Ҷ�任
clf                        % ��ʼ��ͼ��

% ======================= ʱ��ͼ  ===============================
% axes('Position',[0.07 0.72 0.9 0.22]);
subplot(2,2,1);
plot(time,x, 'k')
xlim([0 max(time)]);
xlabel('ʱ��/s');
ylabel('��ֵ');
title('�����źŲ���');
% ======================= ����ͼ  ===============================
set(gcf, 'Position',[20 100 600 500]);
% axes('Position',[0.1 0.1 0.8 0.5]);
subplot(2,2,4);
imagesc(frameTime, freq(:,3),abs(Y(n2,:)));
axis xy;
ylabel('Ƶ��/Hz');
xlabel('ʱ��/s');
title('����ͼ');
LightYellow = [0.6 0.6 0.6];
MidRed = [0 0 0 ];
black = [0.5 0.7 1];
colors = [LightYellow; MidRed; black];
m = 64;
colormap(SpecColorMap(m, colors));  % mָ����ɫ���������ö��ٵȼ�
% ======================= ����Ҷ�任 ����֡��Ƶ��===========================
C=x;
N = length(C);
T=(1/fs)*(N-1);
f=1/T;
vecf=(0:N-1)*f;
C = (C-mean(C)).*hanning(N); % ȥֱ����Ӵ�
C = 2*abs(fft(C,N)/N);
C = C(1:N/10);
subplot(2,2,3);plot(vecf(1:N/10),C);
axis([50 500 -inf 0.07])
ylabel('��ѹ/Pa');
xlabel('Ƶ��/Hz');
title('����֡Ƶ��');
% ======================= ��֡�Ӵ� Ƶ�� ���ֵ�ͷ���===========================
% ÿ֡�Ӵ���fft
Y = 2*abs(Y((1:1+nfft/2),:))/wlen; % ÿһ����һ֡STFT�����飬����������1~nfft/2+1��Ƶ�ʷ���
n2 = 1:wlen/2+1;           % ���ڹ���Գƣ�ȡ���ݵ�һ��
freq = (n2 - 1)*fs/wlen;   % ����fft���Ƶ�ʿ̶�

subplot(2,2,2);
plot(freq, Y);
axis([50 500 -inf inf])
ylabel('��ѹ/Pa');
xlabel('Ƶ��/Hz');
title('��֡Ƶ��');


figure(2);
subplot(1,2,1);
plot(1:size(Y,2),mean(Y))   % ��ֵ
axis([-inf inf 0.4e-3 4e-3])
ylabel('��ֵ');
xlabel('֡��');
title('��֡Ƶ�׼����֡�ľ�ֵ');

% �󷽲�sigmax2,����n
% n = size(Y,1);
% mu = (1/n).*sum(Y);
% sigmax = (1/n).*sum((Y-mu).^2);
sigmax = var(Y,1);%����MATLAB����
subplot(1,2,2);
plot(1:size(Y,2),sigmax)       % ����
axis([-inf inf 0 1.5e-3])
ylabel('����');
xlabel('֡��');
title('��֡Ƶ�׼����֡�ķ���');





