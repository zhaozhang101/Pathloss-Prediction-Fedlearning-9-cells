clear all;close all;
MINX=0;MINY=0;MAXX=0;MAXY=0;
min_xlist = [];
max_xlist = [];
min_ylist = [];
max_ylist = [];
list_all = zeros(4,90);
max_power_pos = zeros(2,90);
for index = 1:90
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
    min_x=min(x);max_x=max(x);
    min_y=min(y);max_y=max(y);
    list_all(1,index)=min_x;
    list_all(2,index)=min_y;
    list_all(3,index)=max_x;
    list_all(4,index)=max_y;
    list_all(5,index)=max(max(pathloss));
    list_all(6,index)=min(min(pathloss));
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
%%
color=zeros(9,3);color(1,:) = [0,0,1];color(2,:) = [1 0 0];color(3,:) = [0 1 0];
color(4,:) = [1 1 0];color(5,:) = [1 0 1];color(6,:) = [0 1 1];color(7,:) = [0.67 0 1];
color(8,:) = [1 0.5 0];color(9,:) = [0.5 0.5 0.5];
figure(1);
for index = 1:10:90
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
    inten = (val-min(val)) / (max(val) - min(val));
    x = position(1,:);
    y = position(2,:);
    scatter(x,y,[],color(idx,:),'.');
    hold on;
end
xlabel('x [m]');
ylabel('y [m]');
x1 = [100,300,300,100]-50;
y1 = [100,100,300,300]-50;
scatter(x1,y1,ones(1,4)*100,'k','h','filled');
title('接入基站');
hold off
%% 所有用户看信号强度
figure(1);
for index = 1:10:90
    
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
    inten = (val-min(val)) / (max(val) - min(val));
    x = position(1,:);
    y = position(2,:);
    s = scatter(x,y,10*ones(length(x),1),color((index+9)/10,:),'filled');
    s.AlphaData = inten*10;
    s.MarkerFaceAlpha = 'flat';
    hold on
    scatter(x1,y1,ones(1,4)*100,'k','h','filled');
    
end
hold off
xlabel('x [m]');
ylabel('y [m]');
title('用户分布');
%% 分基站来看信号强度
figure(3);
for index = 1:10:90
    if index<10
        data = load(['data_mat\user_0',num2str(index),'.mat']);
    else
        data = load(['data_mat\user_',num2str(index),'.mat']);
    end
    mat = data.data;
    mat_trans = reshape(mat,[6,length(mat)/6]);
    position = mat_trans(1:2,:);
    pathloss = mat_trans(3:3,:);
    % 左下；左上；右下；右上；
    pathloss(pathloss==0)=nan;
    [val, idx] = max(pathloss);
    inten = (pathloss-min(pathloss)) / (max(pathloss) - min(pathloss));
    base1 = [65.4,65.1];base2 = [65.4,265];base3 =[265.2,65.1]; base4=[265.2,265];
    x = position(1,:);
    y = position(2,:);
    s = scatter(x,y,10*ones(length(x),1),color((index+9)/10,:),'filled');
    s.AlphaData = inten*10;
    s.MarkerFaceAlpha = 'flat';
    hold on
    scatter(x1,y1,ones(1,4)*100,'k','h','filled');
    
end
hold off
xlabel('x [m]');
ylabel('y [m]');
title('用户分布');
%% 确定基站位置

max_power_pos_x = max_power_pos(1,:);
max_power_pos_y = max_power_pos(2,:);
x_max_t = max_power_pos_x(max_power_pos_x>100);
x_min_t = max_power_pos_x(max_power_pos_x<100);

y_max_t = max_power_pos_y(max_power_pos_y>100);
y_min_t = max_power_pos_y(max_power_pos_y<100);

base_x = [mean(x_max_t),mean(x_min_t)];
base_y = [mean(y_max_t),mean(y_min_t)];

%% 查看路损拟合结果
figure(4);i=1;
for index = 1:10:90
    subplot(3,3,i);i = i + 1;
    if index<10
        data = load(['data_mat\user_0',num2str(index),'.mat']);
    else
        data = load(['data_mat\user_',num2str(index),'.mat']);
    end
    mat = data.data;
    mat_trans = reshape(mat,[6,length(mat)/6]);
    position = mat_trans(1:2,:);
    pathloss = mat_trans(3:3,:);
    % 左下；左上；右下；右上；
    idx = find(pathloss==0);
    pathloss(idx)=[];
    base1 = [65.4,65.1];base2 = [65.4,265];base3 =[265.2,65.1]; base4=[265.2,265];
    x = position(1,:);x(idx)=[];
    y = position(2,:);y(idx)=[];
    trans_pos = [x;y]';
    x = pdist2(trans_pos,base1)/1000;
    syms a
    f = fittype('-20*log10(x)-a');
    PL = pathloss;
    PL(isnan(PL))= -200;
    cfun = fit(x,PL',f);
    
    plot(x,cfun(x),'b');
    hold on
    plot(x,-20*log10(x)-cfun.a+20,'k');
    hold on
    pathloss(isnan(pathloss))=-200;
    scatter(x,pathloss,'r*');
    hold off
end