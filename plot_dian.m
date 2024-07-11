function [] = plot_dian(se)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
    plot(se,'-o');
    % 添加标题
    title('收敛曲线');
    xlabel('迭代次数');
    ylabel('传输速率');
end

