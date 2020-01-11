% Ƶ��,ȡһ��
% 50��������Ϊһ�����룬2�����ݼ�
% ����ministѵ�����ݼ�

clc;        % ��������д���
close all;
clear all;  % ��չ�����
clear out;

%��������
SRN = 40;  % �����
magic_number = 3331;%��Ϊimages��ͷ�ļ�
data_number = 10000;%ѵ��������������=2000��ѭ����10������
line_number = 100;%50��������Ϊһ��
line_number_fft = 50; 
column_number = 1;%����ͨ����һά��һ��
 
magic_number1 = 2049;%��Ϊlabels��ͷ�ļ�
data_number1 = 10000;%��ǩ������������������ͬ
 
MAG = [magic_number;data_number;line_number_fft;column_number];
MBG = [magic_number1;data_number1];

% fid=fopen���ļ�·�������򿪷�ʽ����
    % w,�򿪺�д�����ݡ����ļ��Ѵ�������£��������򴴽�
    % w����ӡ�b�������Զ����Ƹ�ʽ��
fid_fiture = fopen('..\..\mnist\50\train-images-idx3-ubyte','wb'); %����
fid_label = fopen('..\..\mnist\50\train-labels-idx1-ubyte','wb'); %��ǩ

% COUNT��fwrite��fid��A��precision,skip��machinefmt��
    % fwrite ��������ļ�д������
    % fidΪ�ļ������
    % A�������д���ļ������ݣ�
    % precision�������ݾ��ȣ�
    % machinefmt='b'��ʾBig-endian ordering���������
fwrite(fid_fiture,MAG,'int32','b');
fwrite(fid_label,MBG,'int32','b');
 
% ������ǩ
% Channel_1_Data = zeros(2,1);
% Channel_1_Data(1,1) = 1;
% save('��ǩ.mat','Channel_1_Data');
path_l = '..\..\ϴ�»���ȡ����\��ǩ.mat';
for i = 3:12
    path_p = ['..\..\ϴ�»���ȡ����\20171030����ϴ�»�����������-����\no_3�ظ���ˮ����������������'...
        '\�ظ���ˮ����������������-',num2str(i),'.mat'];
    path_n = ['..\..\ϴ�»���ȡ����\20171030����ϴ�»�ʵ��������-����\no_9�ظ���ˮ����쳣��'...
        '\�ظ���ˮ����쳣��-',num2str(i),'.mat'];
    data_p = load(path_p);  % ������
    data_n = load(path_n);  % ������
    data_l = load(path_l);  % label
    
    for j = 1:1000
 
        A = rand(1,1); %����һ��һ�У�һ��������0~1�������
        if A > 0.5  %������
            begin_ = 1 + line_number*(j-1);
            end_ = line_number*j;
            C = data_p.Channel_1_Data(begin_ : end_);
            
%             CR = C;
%             dFs = 65536;
%             N = length(C);
%             T=(1/dFs)*(N-1);
%             t=0:1/dFs:T; 
%             f=1/T;
%             vecf=(0:N-1)*f;
%             subplot(2,2,1);plot(t,C);xlabel('ʱ�䣨s��');ylabel('��ѹ��pa��');title('ԭ����');
            
            C = awgn(C,SRN);%�������ź���������ΪSRNdB�ĸ�˹������
            
%             subplot(2,2,3);plot(t,C);xlabel('ʱ�䣨s��');ylabel('��ѹ��pa��');title('snr=20');
            C = C-mean(C); % ȥֱ��
%             CR = CR -mean(CR);
            N=length(C);
            C = 2*abs(fft(C,N)/N);
            C = C(1:N/2);
%             CR = 2*abs(fft(CR,N)/N);
%             CR = CR(1:N/2);
%             subplot(2,2,2);plot(vecf(1:N/2),CR);xlabel('Ƶ�ʣ�f��');ylabel('��ѹ��pa��');title('ԭ����');
%             subplot(2,2,4);plot(vecf(1:N/2),C);xlabel('Ƶ�ʣ�f��');ylabel('��ѹ��pa��');title('snr=20');
            
            G = data_l.Channel_1_Data(1);
            fwrite(fid_fiture,C,'float32','b');
            fwrite(fid_label,G,'uint8','b');
        else  %������
            begin_ = 1 + line_number*(j-1);
            end_ = line_number*j;
            E = data_n.Channel_1_Data(begin_:end_);
            E = awgn(E,SRN);%�������ź���������ΪSRNdB�ĸ�˹������
            
            E = E-mean(E); % ȥֱ��
            N=length(E);
            E = 2*abs(fft(E,N)/N);
            E = E(1:N/2);
            
            H = data_l.Channel_1_Data(2);
            fwrite(fid_fiture,E,'float32','b');
            fwrite(fid_label,H,'uint8','b');
        end
    end
end
gzip('..\..\mnist\50\train-images-idx3-ubyte')
gzip('..\..\mnist\50\train-labels-idx1-ubyte')
disp('finished')

