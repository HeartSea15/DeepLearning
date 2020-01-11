load('a_01.mat');
data=Channel_1_Data;
function state=dataWrite(data,Channel_1_Data)
    fid=fopen(fileName,'wb');
    if(fid>0)
        count=fwrite(fid,data,'float');
        if(count==size(data,1)*size(data,2))
            state=1;
        else
            state=-1;
        end
    end
    fclose(fid);
end