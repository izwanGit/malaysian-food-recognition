%% KMEANS SEGMENT - K-means Clustering for Ingredient Segmentation
% Segments food into ingredient regions using color-based clustering
%
% Syntax:
%   labeledRegions = kmeansSegment(img)
%   labeledRegions = kmeansSegment(img, mask)
%   labeledRegions = kmeansSegment(img, mask, numClusters)
%
% Inputs:
%   img         - RGB image
%   mask        - Optional binary mask to limit processing area
%   numClusters - Number of clusters/ingredients (default: 5)
%
% Outputs:
%   labeledRegions - Label matrix where each pixel has cluster ID (0 = background)

function labeledRegions = kmeansSegment(img, mask, numClusters)
    %% Default parameters
    if nargin < 2 || isempty(mask)
        mask = true(size(img, 1), size(img, 2));
    end
    if nargin < 3
        numClusters = 5;
    end
    
    %% Ensure proper types
    if isa(img, 'uint8')
        img = im2double(img);
    end
    
    %% Convert to Lab color space for better clustering
    labImg = rgb2lab(img);
    
    %% Extract pixels within mask
    maskIdx = find(mask);
    numPixels = length(maskIdx);
    
    if numPixels < numClusters * 10
        % Too few pixels, return all as one region
        labeledRegions = uint8(mask);
        return;
    end
    
    %% Prepare feature matrix for clustering
    % Use a* and b* channels (color) plus L* (luminance) with reduced weight
    [rows, cols, ~] = size(labImg);
    
    % Get Lab values for masked pixels
    L_vals = labImg(:,:,1);
    a_vals = labImg(:,:,2);
    b_vals = labImg(:,:,3);
    
    % Combine into feature matrix (weight L less for color-based clustering)
    features = zeros(numPixels, 3);
    features(:, 1) = L_vals(maskIdx) * 0.3;  % Reduced weight for luminance
    features(:, 2) = a_vals(maskIdx);
    features(:, 3) = b_vals(maskIdx);
    
    %% Apply k-means clustering
    maxIterations = 100;
    replicates = 3;  % Multiple runs for robustness
    
    try
        [clusterIdx, ~] = kmeans(features, numClusters, ...
                                 'MaxIter', maxIterations, ...
                                 'Replicates', replicates, ...
                                 'Distance', 'sqeuclidean');
    catch
        % Fallback for edge cases
        clusterIdx = ones(numPixels, 1);
    end
    
    %% Create output label matrix
    labeledRegions = zeros(rows, cols, 'uint8');
    labeledRegions(maskIdx) = clusterIdx;
    
    %% Post-processing: Clean up small regions within each cluster
    for k = 1:numClusters
        clusterMask = labeledRegions == k;
        
        % Remove very small components
        clusterMask = bwareaopen(clusterMask, 50);
        
        % Update labels (removed pixels become 0/background)
        labeledRegions(~clusterMask & labeledRegions == k) = 0;
    end
end
