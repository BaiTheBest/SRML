function [ regions ] = datasplitind( m, n, rnd_flag )
%DATASPLITIND returns a [m x 1] vector of numbers 1 to n of approximately 
% equal amounts. Works similar to crosvalind from bioinformatics toolbox 
% with a k-fold argument
%
%   usage: [ regions ] = datasplitind( m, n, rnd_flag )
%
%   Vahe Tshitoyan
%   04/05/2017
%   
%   Arguments
%   m:          length of the dataset (output vector)
%   n:          number of regions
%   rnd_flag:   if true, will shuffle the output
%   
%   Returns
%   regions:    [m x 1] vector of numbers 1 to n of approximately equal 
%               amounts.

% number of elements for each index 1 to n
subSizes = diff(round(linspace(0, m, n+1)));

% creating the ordered vector with randomized lengths for each region
% this matters only if rnd_flag is false
regions = repelem2(1:n, subSizes(randperm(n)))';

% randomizing the output regions
if rnd_flag; regions = regions(randperm(m));
end

