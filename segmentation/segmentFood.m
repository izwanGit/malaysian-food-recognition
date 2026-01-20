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
    %% STEP 3.5: ADAPTIVE K-MEANS REFINEMENT (The "Smart 3-Class" Fix)
    % Cluster into 3 classes: [Food, Rice/Ambiguous, Background]
    % Strategy: Identify the "Background" cluster and remove it. Keep the rest.
    
    fprintf('  Debug: Smart 3-Class K-Means...\n');
    pixelIdx = find(cleanMask);
    if numel(pixelIdx) > 1000
        % Extract features
        hsvMap = rgb2hsv(img);
        S = hsvMap(:,:,2);
        V = hsvMap(:,:,3);
        
        grayImg = rgb2gray(img);
        entropyMap = entropyfilt(grayImg, true(9));
        
        featS = S(pixelIdx);
        featE = entropyMap(pixelIdx);
        
        % Normalize
        featS = (featS - min(featS)) / (max(featS) - min(featS) + eps);
        featE = (featE - min(featE)) / (max(featE) - min(featE) + eps);
        
        % Feature 3: Spatial Bias (Centrality)
        % Wrappers/Background are usually at the edges. Food is central.
        [X, Y] = meshgrid(1:cols, 1:rows);
        x_norm = X(pixelIdx) / cols;
        y_norm = Y(pixelIdx) / rows;
        x_center_dist = abs(x_norm - 0.5);
        y_center_dist = abs(y_norm - 0.5);
        spatialFeat = exp(-(x_center_dist.^2 + y_center_dist.^2) * 5); % Sharper Gaussian
        
        % Combined Features: Color (2x), Texture (1x), Spatial (1.5x)
        features = double([featS * 2.0, featE, spatialFeat * 1.5]);
        
        % Ensure no NaNs
        features(isnan(features)) = 0;
        
        try
            % SMART FIX: Use 'EmptyAction','singleton' to prevent crashes if a cluster disappears
            [clusterIdx, centers] = kmeans(features, 3, 'Replicates', 2, ...
                'MaxIter', 100, 'EmptyAction', 'singleton');
            
            % Score Clusters: High Score = Likely Food
            % Score = Sat + Texture + Spatial
            % centers(:,1)=Sat, (:,2)=Tex, (:,3)=Spatial
            scores = centers(:,1) + centers(:,2) + centers(:,3);
            
            [~, sortedIdx] = sort(scores, 'descend');
            
            % sortedIdx(1) = Best Food (Chicken, Sambal)
            % sortedIdx(2) = Ambiguous (Rice, Light Gravy)
            % sortedIdx(3) = Background (Wrapper, Plate, Table)
            
            % Logic: Keep Top 2 clusters. Drop the Worst one.
            keepClusters = sortedIdx(1:2);
            
            % A++ EMERGENCY UPDATE: Broaden to catch "Shadow Rice" AND "Dirty Rice"
            % Sat < 0.45 (allow gravy colors), Val > 0.25 (allow dark shadows)
            isRice = (values > 0.25) & (saturations < 0.45);
            
            % SAUCE RESCUE (Satay/General): Keep Red/Orange Sauce (Sambal/Curry)
            H = hsvMap(:,:,1);
            hueVals = H(pixelIdx);
            % Red is near 0 or 1. Orange is near 0.1.
            isRedSauce = (saturations > 0.5) & (hueVals < 0.12 | hueVals > 0.9);
            
            % Selection Logic
            isInKeepClusters = ismember(clusterIdx, keepClusters);
            
            % Keep pixels that are in Good Clusters OR look like Rice OR look like Sauce
            keepPixels = pixelIdx(isInKeepClusters | isRice | isRedSauce);
            
            refinedMask = false(rows, cols);
            refinedMask(keepPixels) = true;
            
            % Cleanup
            refinedMask = imclose(refinedMask, strel('disk', 3));
            refinedMask = imfill(refinedMask, 'holes');
            refinedMask = bwareaopen(refinedMask, 150); % Lighter cleanup for rice grains
            
            cleanMask = refinedMask;
            fprintf('  Debug: K-Means kept %d pixels (Dropped worst cluster).\n', sum(cleanMask(:)));
        catch
            fprintf('  Debug: K-Means failed, keeping original.\n');
        end
    end
    
    %% STEP 4: EDGE-GUIDED ACTIVE CONTOURS
    currentMaskSize = sum(cleanMask(:));
    if currentMaskSize > 500
        backupMask = cleanMask; 
        
        grayImg = rgb2gray(img);
        % Gentle Active Contour (Gentle on rice)
        mask = activecontour(img, cleanMask, 60, 'Chan-Vese', 'ContractionBias', -0.02);
        mask = activecontour(grayImg, mask, 80, 'edge'); 
        
        % Safety Check
        if sum(mask(:)) < 100
             mask = backupMask;
        elseif sum(mask(:)) < (0.2 * currentMaskSize)
             mask = backupMask;
        end
    else
        mask = cleanMask;
    end
    
    % FINAL EMERGENCY FALLBACK
    if sum(mask(:)) == 0 && currentMaskSize > 0
        mask = cleanMask;
    end
    
    %% STEP 5: SEMANTIC FILLING (Fix Holes in Solid Objects)
    mask = imfill(mask, 'holes');

    %% A++ FIX 5: THE "ONE SHAPE" FINALIZER (Brush Tool Mode)
    % The user wants exactly 1 coherent shape. No islands.
    
    % 1. Create a "Super Glue" mask to bridge meat and rice
    % SATAY FIX: Use MASSIVE glue to connect separate sauce bowls
    if strcmpi(foodType, 'satay')
        glueRadius = 60; % Bridge the gap between sauce and plate
    else
        glueRadius = 50;
    end
    glueSE = strel('disk', glueRadius); 
    gluedMask = imclose(mask, glueSE);
    gluedMask = imfill(gluedMask, 'holes');
    
    % 2. Pick the ABSOLUTE BEST candidate for the food (Central + Large)
    ccGlue = bwconncomp(gluedMask, 4);
    if ccGlue.NumObjects > 0
        statsGlue = regionprops(ccGlue, 'Area', 'Centroid');
        imageCenter = [cols/2, rows/2];
        scores = zeros(1, ccGlue.NumObjects);
        
        for i = 1:ccGlue.NumObjects
            dist = norm(statsGlue(i).Centroid - imageCenter);
            % Heavy Centrality Bias: Area / (dist^2 + 100)
            scores(i) = statsGlue(i).Area / (dist^2 + 100);
        end
        
        [~, bestIdx] = max(scores);
        
        % 3. Enforce 1 Shape: Zero out EVERYTHING else
        oneShapeContainer = false(size(mask));
        oneShapeContainer(ccGlue.PixelIdxList{bestIdx}) = true;
        
        % Safety Margin: Expand container to catch edge rice (Restored & Increased)
        oneShapeContainer = imdilate(oneShapeContainer, strel('disk', 25));
        
        % Filter original result
        mask = mask & oneShapeContainer;
        
        % 4. Solidify: If the user wants a 'brush tool' feel, we should fill internal gaps
        mask = imclose(mask, strel('disk', 5));
        mask = imfill(mask, 'holes');
        
        % 5. Final connected component check (Absolute Safety)
        ccFinal = bwconncomp(mask);
        if ccFinal.NumObjects > 1
             [~, largestFinal] = max(cellfun(@numel, ccFinal.PixelIdxList));
             mask = false(size(mask));
             mask(ccFinal.PixelIdxList{largestFinal}) = true;
        end
    end
    
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
