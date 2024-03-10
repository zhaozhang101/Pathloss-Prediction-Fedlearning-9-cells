clear;
MINX=0;MINY=0;MAXX=0;MAXY=0;
min_xlist = [];
max_xlist = [];
min_ylist = [];
max_ylist = [];
list_all = zeros(4,90);
max_power_pos = zeros(2,90);
x_all =[];y_all = [];
for index = 1:10
    if index<10
        data = load(['data_mat\user_0',num2str(index),'.mat']);
    else
        data = load(['data_mat\user_',num2str(index),'.mat']);
    end
    mat = data.data;
    mat_trans = reshape(mat,[6,length(mat)/6]);
    position = mat_trans(1:2,:);
    pathloss = mat_trans(3:6,:);
    pathloss(pathloss==0)=nan;
    [val, idx] = max(pathloss);
    [res, id] = max(val);
    pathloss(idx(id),id);
    max_power_pos(1,index) = position(1,id);
    max_power_pos(2,index) = position(2,id);
    x = position(1,:);
    y = position(2,:);
    xleng(index) = length(x);
    x_all = [x_all,x];
    y_all = [y_all,y];
    min_x=min(x);max_x=max(x);
    min_y=min(y);max_y=max(y);
    list_all(1,index)=min_x;
    list_all(2,index)=min_y;
    list_all(3,index)=max_x;
    list_all(4,index)=max_y;
    if min_x<MINX
        MINX=min_x;
    end
    if min_y<MINY
        MINY=min_y;
    end
    if max_x>MAXX
        MAXX=max_x;
    end
    if max_y>MAXY
        MAXY=max_y;
    end
end
abcx = sort(x_all);
abcy = sort(y_all);