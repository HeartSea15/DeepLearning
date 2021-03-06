% 时域 + 噪声
% 50个数据作为一段输入，2万数据集
% 建立mnist训练数据集

clc;
close all;
clear out;

%参数设置
SRN = 20;  % 信噪比
magic_number = 3331;%作为images的头文件
data_number = 20000;%训练特征样本数量
line_number = 50;%50个数据作为一段
column_number = 1;%声音通道，一维，一行
 
magic_number1 = 2049;%作为labels的头文件
data_number1 = 20000;%标签数量，与特征数量相同
 
MAG = [magic_number;data_number;line_number;column_number];
MBG = [magic_number1;data_number1];

% fid=fopen（文件路径，‘打开方式’）
    % w,打开后写入数据。该文件已存在则更新；不存在则创建
    % w后添加“b”，则以二进制格式打开
fid_fiture = fopen('E:\based on minist\02-建立minist训练数据集\mnist\50\时域加噪声\train-images-idx3-ubyte','wb'); %特征
fid_label = fopen('E:\based on minist\02-建立minist训练数据集\mnist\50\时域加噪声\train-labels-idx1-ubyte','wb'); %标签

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
for i = 3:12
    path_p = ['..\..\洗衣机提取数据\20171030海尔洗衣机生产线数据-正常\no_3关盖脱水噪声跟着生产线走'...
        '\关盖脱水噪声跟着生产线走-',num2str(i),'.mat'];
    path_n = ['..\..\洗衣机提取数据\20171030海尔洗衣机实验室数据-故障\no_9关盖脱水电机异常声'...
        '\关盖脱水电机异常声-',num2str(i),'.mat'];
    data_p = load(path_p);  % 正样本
    data_n = load(path_n);  % 负样本
    data_l = load(path_l);  % label
    
    for j = 1:2000
 
        A = rand(1,1); %产生一行一列（一个数）的0~1的随机数
        if A > 0.5  %正样本
            begin_ = 1 + line_number*(j-1);
            end_ = line_number*j;
            C = data_p.Channel_1_Data(begin_ : end_);
            C = awgn(C,SRN);%给正常信号添加信噪比为SRNdB的高斯白噪声
            G = data_l.Channel_1_Data(1);
            fwrite(fid_fiture,C,'float32','b');
            fwrite(fid_label,G,'uint8','b');
        else  %负样本
            begin_ = 1 + line_number*(j-1);
            end_ = line_number*j;
            E = data_n.Channel_1_Data(begin_:end_);
            E = awgn(E,SRN);%给正常信号添加信噪比为SRNdB的高斯白噪声
            H = data_l.Channel_1_Data(2);
            fwrite(fid_fiture,E,'float32','b');
            fwrite(fid_label,H,'uint8','b');
        end
    end
end
gzip('E:\based on minist\02-建立minist训练数据集\mnist\50\时域加噪声\train-images-idx3-ubyte')
gzip('E:\based on minist\02-建立minist训练数据集\mnist\50\时域加噪声\train-labels-idx1-ubyte')
disp('finished')

