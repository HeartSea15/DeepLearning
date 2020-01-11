% 时域 + 噪声
% 250个数据作为一段输入
% 建立minist训练数据集

clc;
close all;
clear out;

%参数设置
SRN = 60;  % 信噪比
magic_number = 3331;%作为images的头文件
data_number = 4200;%训练特征样本数量21*200
line_number = 250;%250个数据作为一段
column_number = 1;%声音通道，一维，一行
 
magic_number1 = 2049;%作为labels的头文件
data_number1 = 4200;%标签数量，与特征数量相同
 
MAG = [magic_number;data_number;line_number;column_number];
MBG = [magic_number1;data_number1];

% fid=fopen（文件路径，‘打开方式’）
    % w,打开后写入数据。该文件已存在则更新；不存在则创建
    % w后添加“b”，则以二进制格式打开
fid_fiture = fopen('..\..\mnist\250\t10k-images-idx3-ubyte','wb'); %特征
fid_label = fopen('..\..\mnist\250\t10k-labels-idx1-ubyte','wb'); %标签

% COUNT＝fwrite（fid，A，precision,skip，machinefmt）
    % fwrite 向二进制文件写入数据
    % fid为文件句柄，
    % A用来存放写入文件的数据，
    % precision代表数据精度，
    % machinefmt='b'表示Big-endian ordering（大端排序）
fwrite(fid_fiture,MAG,'int32','b');
fwrite(fid_label,MBG,'int32','b');
 
% 制作标签
% Channel_1_Data = zeros(2,1);
% Channel_1_Data(1,1) = 1;
% save('标签.mat','Channel_1_Data');
path_l = '..\..\洗衣机提取数据\标签.mat';
for i = 1:21
    path_p = ['..\..\洗衣机提取数据\整理数据\TEST\正常-1\',num2str(i), '.mat'];
    path_n = ['..\..\洗衣机提取数据\整理数据\TEST\异常-0\',num2str(i),'.mat'];
    data_p = load(path_p);  % 正样本
    data_n = load(path_n);  % 负样本
    data_l = load(path_l);  % label
    
    for j = 1:200
 
        A = rand(1,1); %产生一行一列（一个数）的0~1的随机数
        if A > 0.5  %正样本
            begin_ = 1 + line_number*(j-1);
            end_ = line_number*j;
            C = data_p.Channel_1_Data(begin_ : end_);
            C = C-mean(C);
            C = awgn(C,SRN);%给正常信号添加信噪比为SRNdB的高斯白噪声
            G = data_l.Channel_1_Data(1);
            fwrite(fid_fiture,C,'float32','b');
            fwrite(fid_label,G,'uint8','b');
        else  %负样本
            begin_ = 1 + line_number*(j-1);
            end_ = line_number*j;
            E = data_n.Channel_1_Data(begin_:end_);
            E = E-mean(E);
            E = awgn(E,SRN);%给正常信号添加信噪比为SRNdB的高斯白噪声
            H = data_l.Channel_1_Data(2);
            fwrite(fid_fiture,E,'float32','b');
            fwrite(fid_label,H,'uint8','b');
        end
    end
end
gzip('..\..\mnist\250\t10k-images-idx3-ubyte')
gzip('..\..\mnist\250\t10k-labels-idx1-ubyte')
disp('finished')
