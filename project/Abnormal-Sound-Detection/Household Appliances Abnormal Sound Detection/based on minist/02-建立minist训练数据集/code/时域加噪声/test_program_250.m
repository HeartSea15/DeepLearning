% ʱ�� + ����
% 250��������Ϊһ������
% ����ministѵ�����ݼ�

clc;
close all;
clear out;

%��������
SRN = 60;  % �����
magic_number = 3331;%��Ϊimages��ͷ�ļ�
data_number = 4200;%ѵ��������������21*200
line_number = 250;%250��������Ϊһ��
column_number = 1;%����ͨ����һά��һ��
 
magic_number1 = 2049;%��Ϊlabels��ͷ�ļ�
data_number1 = 4200;%��ǩ������������������ͬ
 
MAG = [magic_number;data_number;line_number;column_number];
MBG = [magic_number1;data_number1];

% fid=fopen���ļ�·�������򿪷�ʽ����
    % w,�򿪺�д�����ݡ����ļ��Ѵ�������£��������򴴽�
    % w����ӡ�b�������Զ����Ƹ�ʽ��
fid_fiture = fopen('..\..\mnist\250\t10k-images-idx3-ubyte','wb'); %����
fid_label = fopen('..\..\mnist\250\t10k-labels-idx1-ubyte','wb'); %��ǩ

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
for i = 1:21
    path_p = ['..\..\ϴ�»���ȡ����\��������\TEST\����-1\',num2str(i), '.mat'];
    path_n = ['..\..\ϴ�»���ȡ����\��������\TEST\�쳣-0\',num2str(i),'.mat'];
    data_p = load(path_p);  % ������
    data_n = load(path_n);  % ������
    data_l = load(path_l);  % label
    
    for j = 1:200
 
        A = rand(1,1); %����һ��һ�У�һ��������0~1�������
        if A > 0.5  %������
            begin_ = 1 + line_number*(j-1);
            end_ = line_number*j;
            C = data_p.Channel_1_Data(begin_ : end_);
            C = C-mean(C);
            C = awgn(C,SRN);%�������ź���������ΪSRNdB�ĸ�˹������
            G = data_l.Channel_1_Data(1);
            fwrite(fid_fiture,C,'float32','b');
            fwrite(fid_label,G,'uint8','b');
        else  %������
            begin_ = 1 + line_number*(j-1);
            end_ = line_number*j;
            E = data_n.Channel_1_Data(begin_:end_);
            E = E-mean(E);
            E = awgn(E,SRN);%�������ź���������ΪSRNdB�ĸ�˹������
            H = data_l.Channel_1_Data(2);
            fwrite(fid_fiture,E,'float32','b');
            fwrite(fid_label,H,'uint8','b');
        end
    end
end
gzip('..\..\mnist\250\t10k-images-idx3-ubyte')
gzip('..\..\mnist\250\t10k-labels-idx1-ubyte')
disp('finished')
