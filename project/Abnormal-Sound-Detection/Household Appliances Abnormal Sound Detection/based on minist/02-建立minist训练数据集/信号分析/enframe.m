function [y_data ] = enframe(x,win,inc)
% in��
    % x�������ź�
    % win������������֡��
    % inc��֡��
% out:
    % ��֡������飨֡����֡����
% ===================================================
 L = length(x(:)); % ���ݵĳ���
 nwin = length(win); % ȡ����, ���ֵĳ�����1
 if (nwin == 1)  % �ж����޴���������Ϊ1����ʾû�д�����
     wlen = win;  % û�У�֡������win
 else
     wlen = nwin; % �д�������֡�����ڴ���
 end
 
 if (nargin <3)  % ���ֻ������������inc = ֡��
    inc = len;
 end
 
 fn = floor((L - wlen)/inc) + 1;  % ֡��
 
 y_data = zeros(fn,wlen); % ��ʼ����fn�У�wlen��
 indf = ((0:(fn - 1))*inc)'; % ÿһ֡������y�п�ʼλ�õ�ָ��
 inds = 1:wlen;              % ÿһ֡������λ��Ϊ1��wlen
 indf_k = indf(:,ones(1,wlen));  % ��indf��չ��fn*wlen�ľ���ÿһ�е���ֵ����ԭindfһ��
 inds_k = inds (ones(fn,1), : ); % ��inds��չ��fn*wlen�ľ���ÿһ�е���ֵ����ԭindsһ��
 y_data(:) = x(indf_k + inds_k);
 
 if (nwin >1)  % ���������д���������ÿ֡���Դ�����
     w = win(:)';
     y_data = y_data.*w(ones(fn,1),:);
end