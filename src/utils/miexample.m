clear; close all; clc;

m = 10000; % 10000 data points
n = 3; % into 3 regions
rnd_flag = false; % keep them ordered

% splitting into regions
regions = datasplitind( m, n, false ); 

% plotting arbitrary y data
figure(1); cla(gca);
ph = plotRegions( 1:m, sin((1:m)/m) + rand(1, m), regions );

% The legend
legend(cellstr(num2str((1:n)', 'Region %d')), 'location', 'northwest');
