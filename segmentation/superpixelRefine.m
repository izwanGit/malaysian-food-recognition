%% SUPERPIXEL REFINE - Superpixel-based Semantic Refinement
% Groups pixels into superpixels for better semantic segmentation

function refinedMask = superpixelRefine(img, mask)
    if isa(img, 'uint8')
        img = im2double(img);
    end
    
    %% Create superpixels
    % Use SLIC algorithm to generate approx 300 superpixels
    [L, N] = superpixels(img, 300, 'Compactness', 10);
    
    %% Calculate superpixel features
    idx = label2idx(L);
    grayImg = rgb2gray(img);
    
    refinedMask = false(size(mask));
    
    for labelVal = 1:N
        regionIdx = idx{labelVal};
        
        % Feature 1: Color consistency
        % Check if the standard deviation of color in the region is low
        colorStd = std(double(img(regionIdx)), 0, 1);
        colorUniformity = mean(colorStd) < 0.15;
        
        % Feature 2: Texture
        texture = std(double(grayImg(regionIdx))) < 0.1;
        
        % Feature 3: Mask overlap
        % How much of this superpixel is currently selected?
        maskOverlap = mean(mask(regionIdx));
        
        % Decision: Include superpixel if likely food
        % 1. Strongly selected (> 70%)
        % 2. Moderately selected (> 30%) AND looks consistent (uniform color/texture)
        isFood = (maskOverlap > 0.7) || ... 
                 (maskOverlap > 0.3 && colorUniformity && texture);  
        
        if isFood
            refinedMask(regionIdx) = true;
        end
    end
    
    %% Connect nearby food regions
    % Small morphological close to bridge tiny gaps between superpixels
    refinedMask = imclose(refinedMask, strel('disk', 2));
end
