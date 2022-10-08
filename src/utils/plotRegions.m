function [ ph ] = plotRegions( x, y, regions )
%PLOTREGIONS Plots x/y data but colors according to the array specified
%   in regions, which has the same length as x and y with integer values
%   representing the index of the region. E.g. [1 1 1 2 2 3 3 1 1 ..]
%   can be used in conjunction with datasplitind or crossvalind to
%   visualise sets of data.
%
%   usage: ph = PLOTREGIONS( x, y, regions )
%
%   Vahe Tshitoyan
%   20/08/2017
%
%   Arguments
%   x:          vector of x data
%   y:          vector of y data
%   regions:    vector of the same length as x/y with integer values 
%               representing the index of the region
%
%   Returns
%   ph:         the handle of the plot

% @eq comparison builds a matrix of 0s and 1s. Each column of this matrix
% represents one region, where 1s indicate values that belong to this region
% @rdivide by 0 results in Inf values which are not rendered in the plot
ph = plot(bsxfun(@rdivide, x(:), bsxfun(@eq, regions(:), unique(regions(:))')), y(:));
end

