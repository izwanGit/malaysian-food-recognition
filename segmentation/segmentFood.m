%% SEGMENT FOOD - Main Food Segmentation Pipeline
% Segments food regions from images using color and morphological methods
%
% Syntax:
%   mask = segmentFood(img)
%   [mask, labeledRegions] = segmentFood(img)
%   [mask, labeledRegions, segmentedImg] = segmentFood(img)
%
% Inputs:
%   img - RGB image (preprocessed recommended)
%
% Outputs:
%   mask           - Binary mask of food region
%   labeledRegions - Label matrix for different ingredient regions
%   segmentedImg   - RGB image with segmentation overlay

function [mask, labeledRegions, segmentedImg] = segmentFood(img)
    %% Input validation
    if ischar(img) || isstring(img)
        img = imread(img);
    end
    
    if isa(img, 'double')
        img = im2uint8(img);
    end
    
    %% Step 1: HSV-based food region detection
    % Isolate food regions based on color (typically non-white backgrounds)
    hsvMask = hsvThreshold(img);
    
    %% Step 2: Apply morphological operations to clean the mask
    cleanMask = morphologyClean(hsvMask);
    
    %% Step 3: Remove background and keep largest connected region(s)
    % Label connected components
    cc = bwconncomp(cleanMask);
    
    if cc.NumObjects > 0
        % Get properties of each region
        stats = regionprops(cc, 'Area', 'BoundingBox');
        areas = [stats.Area];
        
        % Keep regions that are at least 5% of the largest region
        maxArea = max(areas);
        validIdx = areas >= (maxArea * 0.05);
        
        % Create final mask
        mask = false(size(cleanMask));
        for i = find(validIdx)
            mask(cc.PixelIdxList{i}) = true;
        end
    else
        mask = cleanMask;
    end
    
    %% Step 4: K-means clustering for ingredient segmentation
    if nargout > 1
        % Apply k-means only within the food mask
        numClusters = 5;  % Typical number of ingredient types
        labeledRegions = kmeansSegment(img, mask, numClusters);
    end
    
    %% Step 5: Create visualization if requested
    if nargout > 2
        % Create overlay visualization
        segmentedImg = img;
        
        % Create colored overlay for the mask outline
        maskOutline = bwperim(mask);
        
        % Make outline thicker (dilate)
        se = strel('disk', 2);
        maskOutline = imdilate(maskOutline, se);
        
        % Apply bright green outline
        for c = 1:3
            channel = segmentedImg(:,:,c);
            if c == 2  % Green channel
                channel(maskOutline) = 255;
            else
                channel(maskOutline) = 0;
            end
            segmentedImg(:,:,c) = channel;
        end
        
        % Dim background (but keep it visible - 60% instead of 30%)
        for c = 1:3
            channel = segmentedImg(:,:,c);
            channel(~mask) = uint8(double(channel(~mask)) * 0.6);
            segmentedImg(:,:,c) = channel;
        end
    end
end
