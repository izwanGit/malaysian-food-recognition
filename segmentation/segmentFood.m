%% SEGMENT FOOD - Main Food Segmentation Pipeline
% Segments food regions from images using multi-stage refinement
%
% Syntax:
%   mask = segmentFood(img)
%   mask = segmentFood(img, foodType)
%   [mask, labeledRegions, segmentedImg] = segmentFood(...)
%
% Inputs:
%   img      - RGB image (preprocessed recommended)
%   foodType - Optional food class (e.g., 'satay', 'nasi_lemak')
%
% Outputs:
%   mask           - Binary mask of food region
%   labeledRegions - Label matrix for different ingredient regions
%   segmentedImg   - RGB image with segmentation overlay

function [mask, labeledRegions, segmentedImg] = segmentFood(img, foodType)
    %% Input validation
    if nargin < 2
        foodType = 'general';
    end
    
    if ischar(img) || isstring(img)
        img = imread(img);
    end
    
    if isa(img, 'double')
        img = im2uint8(img);
    end
    
    %% STEP 0: A++ PRE-PROCESSING - Adaptive Histogram Equalization
    % Enhance contrast while preserving edges
    labImg = rgb2lab(img);
    L = labImg(:,:,1);
    % Mild CLAHE to boost local contrast
    L_eq = adapthisteq(L./100, 'NumTiles', [8 8], 'ClipLimit', 0.01);
    labImg(:,:,1) = L_eq * 100;
    imgEnhanced = lab2rgb(labImg);
    
    % A++ ENHANCEMENT: Bilateral Filtering (REMOVED - Reverting to stable)
    % imgEnhanced = imbilatfilt(imgEnhanced); 
    
    %% STEP 1: Geometry-Aware HSV Thresholding
    hsvMask = hsvThreshold(imgEnhanced, foodType);
    
    %% STEP 2: MULTI-SCALE MORPHOLOGICAL CLEANING
    % Different strategies for different food types
    switch lower(foodType)
        case 'satay'
            % For satay: preserve thin structures
            options.openRadius = 2;
            options.closeRadius = 3;
            options.minArea = 50;  % Small for sticks
        case 'nasi_lemak'
            % For rice: fill holes, smooth boundaries
            options.openRadius = 3;
            options.closeRadius = 8;
            options.minArea = 200;
        otherwise
            options.openRadius = 3;
            options.closeRadius = 6;
            options.minArea = 100;
    end
    
    cleanMask = morphologyClean(hsvMask, options);
    
    %% STEP 2.5: BREAK CONNECTIVITY (REMOVED - Reverting to stable)
    % cleanMask = imerode(cleanMask, seErode);
    
    
    %% STEP 3: A++ CONCAVITY-AWARE SHAPE REFINEMENT
    [rows, cols, ~] = size(img);
    
    % Refine individual regions
    cc = bwconncomp(cleanMask);
    stats = regionprops(cc, 'Solidity', 'ConvexArea', 'Area');
    
    for i = 1:cc.NumObjects
        regionMask = false(rows, cols);
        regionMask(cc.PixelIdxList{i}) = true;
        
        % If region is concave (has indentations), fill them
        if stats(i).Solidity < 0.85
            % Create convex hull
            convexMask = bwconvhull(regionMask);
            
            % Fill only if the convex addition is reasonable (< 30% increase)
            addedArea = sum(convexMask(:)) - stats(i).Area;
            if addedArea < stats(i).Area * 0.3
                regionMask = convexMask;
            end
        end
        
        cleanMask(cc.PixelIdxList{i}) = regionMask(cc.PixelIdxList{i});
    end
    %% STEP 3.5: ADAPTIVE K-MEANS REFINEMENT (Saturation Dominant)
    % Cluster into Food (Colorful) vs Background (Dull)
    
    pixelIdx = find(cleanMask);
    if numel(pixelIdx) > 500
        % Extract features
        hsvMap = rgb2hsv(img);
        S = hsvMap(:,:,2);
        V = hsvMap(:,:,3);
        
        grayImg = rgb2gray(img);
        entropyMap = entropyfilt(grayImg, true(9));
        
        % Features
        featS = S(pixelIdx);
        featE = entropyMap(pixelIdx);
        
        % Normalize features
        featS = (featS - min(featS)) / (max(featS) - min(featS) + eps);
        featE = (featE - min(featE)) / (max(featE) - min(featE) + eps);
        
        % CRITICAL WEIGHTING ADJUSTMENT (Restoring "Best Version"):
        % Saturation is the reliable discriminator for Colorful Food vs Dull Wrapper/Rice.
        % Texture is ignored because wrinkles create fake texture.
        
        featS = featS * 2.5;  % High Saturation Weight
        featE = featE * 1.0;  % Normal Texture Weight
        
        % K-Means Clustering (k=2)
        features = [featS, featE];
        
        try
            [clusterIdx, centers] = kmeans(features, 2, 'Replicates', 3);
            
            % Identify "Food" Cluster
            % Higher Saturation + Texture Score
            % Food = Colorful.
            meanSat1 = centers(1, 1); meanTex1 = centers(1, 2);
            meanSat2 = centers(2, 1); meanTex2 = centers(2, 2);
            
            score1 = meanSat1 + meanTex1;
            score2 = meanSat2 + meanTex2;
            
            foodCluster = 1;
            if score2 > score1
                foodCluster = 2;
            end
            
            % RICE RESCUE MISSION:
            % Rice: High Brightness (V > 0.7), Low Sat (S < 0.3)
            values = V(pixelIdx);
            saturations = S(pixelIdx);
            
            isRice = (values > 0.70) & (saturations < 0.30);
            isFoodCluster = (clusterIdx == foodCluster);
            
            % Keep Food Cluster OR Rice
            keepPixels = pixelIdx(isFoodCluster | isRice);
            
            refinedMask = false(rows, cols);
            refinedMask(keepPixels) = true;
            
            % Cleanup
            refinedMask = imclose(refinedMask, strel('disk', 3));
            refinedMask = imfill(refinedMask, 'holes');
            refinedMask = bwareaopen(refinedMask, 200);
            
            cleanMask = refinedMask;
            
        catch
            % K-means failed
        end
    end
    %% STEP 4: EDGE-GUIDED ACTIVE CONTOURS
    if sum(cleanMask(:)) > 0
        % A++ ENHANCEMENT: Edge-based initialization
        grayImg = rgb2gray(img);
        edges = edge(grayImg, 'canny', [0.03 0.1]);
        
        % Create distance map from edges
        edgeDist = bwdist(edges);
        
        % Refine mask: pixels far from edges but near current boundary
        currentBoundary = bwperim(cleanMask);
        boundaryDist = bwdist(currentBoundary);
        
        % Where we have edges but mask doesn't align, adjust
        misaligned = (edgeDist < 3) & (boundaryDist > 5);
        if any(misaligned(:))
            % Use edges to guide mask expansion/contraction
            se = strel('disk', 2);
            misalignedDilated = imdilate(misaligned, se);
            
            % Expand mask to meet nearby edges
            cleanMask = cleanMask | misalignedDilated;
        end
        
        %% MULTI-PHASE ACTIVE CONTOURS
        % Phase 1: Coarse adjustment
        mask = activecontour(img, cleanMask, 50, 'Chan-Vese');
        
        % Phase 2: Edge-based refinement
        mask = activecontour(grayImg, mask, 100, 'edge');
        
        % Phase 3: Final smoothing
        mask = activecontour(img, mask, 50, 'Chan-Vese');
        
        % [REMOVED] Gradient Barrier - it was removing smooth food regions (rice/tofu)
        % The active contour 'edge' method is sufficient for boundary adhearence.
        
        % CRITICAL SAFETY CHECK:
        % If active contours collapsed the mask to nothing (common with weak edges),
        % revert to the original morphological mask.
        if sum(mask(:)) < 100
             mask = cleanMask;
        end
    else
        mask = cleanMask;
    end
    
    %% STEP 4.5: GRABCUT REFINEMENT (REMOVED - Reverting to stable)
    % Reverting to previous best version without GrabCut.

    
    %% STEP 5: SEMANTIC FILLING (Fix Holes in Solid Objects)
    mask = imfill(mask, 'holes');
    
    %% STEP 6: GLOBAL CONTEXT REFINEMENT
    % Use superpixels helper if available
    try
        if exist('superpixelRefine', 'file')
             mask = superpixelRefine(img, mask);
        else
             mask = bwareaopen(mask, 500);
        end
    catch
        mask = bwareaopen(mask, 500);
    end
    
    %% STEP 7: CONTAINER REMOVAL (Plates/Trays)
    % Detect circular/rectangular containers using Hough Transform
    grayImg = rgb2gray(img);
    edges = edge(grayImg, 'canny');
    
    % Find circles (plates)
    rmin = round(min(rows,cols)/10);
    rmax = round(min(rows,cols)/2);
    [centers, radii, ~] = imfindcircles(edges, [rmin, rmax], ...
        'ObjectPolarity', 'bright', 'Sensitivity', 0.9);
    
    if ~isempty(centers)
        % Create plate mask
        plateMask = false(rows, cols);
        [X, Y] = meshgrid(1:cols, 1:rows);
        
        for i = 1:size(centers, 1)
            distance = sqrt((X - centers(i,1)).^2 + (Y - centers(i,2)).^2);
            plateMask = plateMask | (distance <= radii(i) * 0.95);
        end
        
        % Keep only food that's on the plate (or remove plate itself)
        if strcmpi(foodType, 'general')
            % Keep intersection of food and plate
            mask = mask & plateMask;
        else
            % Remove plate pixels from food mask (assuming plate is white/bright)
            platePixels = plateMask & (grayImg > 200);  % Bright plate areas
            mask = mask & ~platePixels;
        end
    end
    
    %% STEP 8: FINAL POLISH
    % Smooth but preserve details
    mask = imclose(mask, strel('disk', 2));
    mask = imfill(mask, 'holes');
    
    % Remove any remaining tiny noise
    mask = bwareaopen(mask, 100);
    
    %% Output Handling
    if nargout > 1
        numClusters = 5;
        if exist('kmeansSegment', 'file')
            labeledRegions = kmeansSegment(img, mask, numClusters);
        else
            labeledRegions = zeros(size(mask));
        end
    end
    
    if nargout > 2
        segmentedImg = createVisualization(img, mask);
    end
end

%% Helper function for visualization
function visImg = createVisualization(img, mask)
    visImg = img;
    maskOutline = bwperim(mask);
    
    % Thicker, colored outline
    maskOutline = imdilate(maskOutline, strel('disk', 3));
    
    % Green outline
    visImg(:,:,1) = visImg(:,:,1) .* uint8(~maskOutline) + uint8(maskOutline) * 0;
    visImg(:,:,2) = visImg(:,:,2) .* uint8(~maskOutline) + uint8(maskOutline) * 255;
    visImg(:,:,3) = visImg(:,:,3) .* uint8(~maskOutline) + uint8(maskOutline) * 0;
    
    % Dim background
    dimFactor = 0.4;
    for c = 1:3
        channel = visImg(:,:,c);
        channel(~mask) = uint8(double(channel(~mask)) * dimFactor);
        visImg(:,:,c) = channel;
    end
    
    % Highlight food with slight saturation boost
    hsvImg = rgb2hsv(visImg);
    hsvImg(:,:,2) = hsvImg(:,:,2) .* (1 + 0.2 * double(mask));
    hsvImg(:,:,2) = min(hsvImg(:,:,2), 1);
    visImg = hsv2rgb(hsvImg);
end
