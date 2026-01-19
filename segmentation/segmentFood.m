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
    % HSV (Hue-Saturation-Value) color space is used as it separates chroma 
    % from intensity, making it more robust to lighting variations than RGB.
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
    
    %% Step 4: Refine mask using Active Contours (Snakes)
    % This "shrink-wraps" the mask to the actual food edges
    if sum(mask(:)) > 0
        % ADVANCED OPTIMIZATION:
        % 1. Dilate first: We expand the mask slightly so the "Snake" starts 
        %    outside the food boundary and shrinks tightly onto the edges. 
        %    This prevents the mask from getting stuck inside the food.
        se_expand = strel('disk', 5);
        mask = imdilate(mask, se_expand);
        
        % 2. Run Active Contours (Chan-Vese)
        % Algorithm: Chan-Vese Active Contours (Region-based energy minimization)
        % Reference: T. F. Chan and L. A. Vese, "Active contours without edges," 
        % IEEE Transactions on Image Processing, vol. 10, no. 2, pp. 266-277, 2001.
        % Iterations: 200 (Increased for maximum precision)
        mask = activecontour(img, mask, 200, 'Chan-Vese');
        
        % 3. Final Polish: Fill holes and smooth edges
        mask = imfill(mask, 'holes');              % Ensure solidity
        mask = imclose(mask, strel('disk', 3));    % Smooth out tiny jagged edges
    end

    %% Step 5: K-means clustering for ingredient segmentation
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
        
        % Dim background (Make it darker for "Pop" effect - 0.35 intensity)
        for c = 1:3
            channel = segmentedImg(:,:,c);
            channel(~mask) = uint8(double(channel(~mask)) * 0.35); % Darker cinema-style background
            segmentedImg(:,:,c) = channel;
        end
    end
end
