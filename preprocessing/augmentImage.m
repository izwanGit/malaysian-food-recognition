%% AUGMENT IMAGE - Data Augmentation for Training
% Applies random augmentations to increase training data diversity
%
% Syntax:
%   augmentedImg = augmentImage(img)
%   [augmentedImg, augmentType] = augmentImage(img)
%
% Augmentations applied:
%   - Random rotation (±15 degrees)
%   - Horizontal flip (50% chance)
%   - Brightness adjustment (±20%)
%   - Contrast adjustment (±20%)
%   - Random crop and resize (90-100% scale)

function [augmentedImg, augmentType] = augmentImage(img)
    augmentType = {};
    augmentedImg = img;
    
    % Ensure image is uint8
    if isa(augmentedImg, 'double')
        augmentedImg = im2uint8(augmentedImg);
    end
    
    %% 1. Random Rotation (±15 degrees)
    if rand() > 0.3  % 70% chance
        angle = (rand() - 0.5) * 30;  % ±15 degrees
        augmentedImg = imrotate(augmentedImg, angle, 'bilinear', 'crop');
        augmentType{end+1} = sprintf('rotate(%.1f)', angle);
    end
    
    %% 2. Horizontal Flip (50% chance)
    if rand() > 0.5
        augmentedImg = fliplr(augmentedImg);
        augmentType{end+1} = 'hflip';
    end
    
    %% 3. Brightness Adjustment (±20%)
    if rand() > 0.4  % 60% chance
        brightnessFactor = 0.8 + rand() * 0.4;  % 0.8 to 1.2
        augmentedImg = im2uint8(im2double(augmentedImg) * brightnessFactor);
        augmentType{end+1} = sprintf('brightness(%.2f)', brightnessFactor);
    end
    
    %% 4. Contrast Adjustment (±20%)
    if rand() > 0.5  % 50% chance
        contrastFactor = 0.8 + rand() * 0.4;  % 0.8 to 1.2
        imgDouble = im2double(augmentedImg);
        meanVal = mean(imgDouble(:));
        imgDouble = (imgDouble - meanVal) * contrastFactor + meanVal;
        imgDouble = max(0, min(1, imgDouble));  % Clip to [0, 1]
        augmentedImg = im2uint8(imgDouble);
        augmentType{end+1} = sprintf('contrast(%.2f)', contrastFactor);
    end
    
    %% 5. Random Crop and Resize (90-100% scale)
    if rand() > 0.5  % 50% chance
        [h, w, ~] = size(augmentedImg);
        scale = 0.9 + rand() * 0.1;  % 90% to 100%
        
        newH = round(h * scale);
        newW = round(w * scale);
        
        % Random crop position
        startY = randi([1, h - newH + 1]);
        startX = randi([1, w - newW + 1]);
        
        % Crop
        cropped = augmentedImg(startY:startY+newH-1, startX:startX+newW-1, :);
        
        % Resize back to original size
        augmentedImg = imresize(cropped, [h, w]);
        augmentType{end+1} = sprintf('crop(%.2f)', scale);
    end
    
    %% 6. Color Jitter (slight hue/saturation shift)
    if rand() > 0.6  % 40% chance
        hsvImg = rgb2hsv(augmentedImg);
        
        % Slight hue shift
        hsvImg(:,:,1) = mod(hsvImg(:,:,1) + (rand() - 0.5) * 0.05, 1);
        
        % Slight saturation shift
        satFactor = 0.9 + rand() * 0.2;  % 0.9 to 1.1
        hsvImg(:,:,2) = min(1, hsvImg(:,:,2) * satFactor);
        
        augmentedImg = im2uint8(hsv2rgb(hsvImg));
        augmentType{end+1} = 'colorjitter';
    end
    
    %% 7. Gaussian Blur (simulates different camera focus/quality)
    if rand() > 0.7  % 30% chance
        sigma = 0.2 + rand() * 0.8;  % Blur amount 0.2-1.0 (reduced from 0.5-2.0)
        augmentedImg = imgaussfilt(augmentedImg, sigma);
        augmentType{end+1} = sprintf('blur(%.1f)', sigma);
    end
    
    %% 8. Aspect ratio variation (handles different photo dimensions)
    if rand() > 0.8  % 20% chance
        [h, w, c] = size(augmentedImg);
        stretchFactor = 0.9 + rand() * 0.2;  % 0.9 to 1.1
        if rand() > 0.5
            newW = round(w * stretchFactor);
            augmentedImg = imresize(augmentedImg, [h, newW]);
            augmentedImg = imresize(augmentedImg, [h, w]);  % Back to original
        else
            newH = round(h * stretchFactor);
            augmentedImg = imresize(augmentedImg, [newH, w]);
            augmentedImg = imresize(augmentedImg, [h, w]);
        end
        augmentType{end+1} = sprintf('stretch(%.2f)', stretchFactor);
    end
    
    % Return augmentation info
    if isempty(augmentType)
        augmentType = {'none'};
    end
end
