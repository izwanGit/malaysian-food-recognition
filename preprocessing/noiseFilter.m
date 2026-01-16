%% NOISE FILTER - Noise Reduction
% Applies various noise reduction filters to food images
%
% Syntax:
%   filteredImg = noiseFilter(img)
%   filteredImg = noiseFilter(img, filterType)
%   filteredImg = noiseFilter(img, filterType, kernelSize)
%
% Inputs:
%   img        - RGB image (double or uint8)
%   filterType - 'median' (default), 'average', 'gaussian', or 'bilateral'
%   kernelSize - Filter kernel size (default: 3)
%
% Outputs:
%   filteredImg - Noise-reduced RGB image

function filteredImg = noiseFilter(img, filterType, kernelSize)
    %% Input validation
    if nargin < 2
        filterType = 'median';
    end
    if nargin < 3
        kernelSize = 3;
    end
    
    % Ensure kernel size is odd
    if mod(kernelSize, 2) == 0
        kernelSize = kernelSize + 1;
    end
    
    % Check if input is uint8 and convert if needed
    wasUint8 = isa(img, 'uint8');
    if wasUint8
        img = im2double(img);
    end
    
    %% Apply filter based on type
    switch lower(filterType)
        case 'median'
            % Median filter - good for salt-and-pepper noise
            filteredImg = zeros(size(img), 'like', img);
            for c = 1:size(img, 3)
                filteredImg(:,:,c) = medfilt2(img(:,:,c), [kernelSize, kernelSize]);
            end
            
        case 'average'
            % Average filter - smooths image
            h = fspecial('average', [kernelSize, kernelSize]);
            filteredImg = imfilter(img, h, 'replicate');
            
        case 'gaussian'
            % Gaussian filter - smooth with weighted average
            sigma = (kernelSize - 1) / 6;  % Rule of thumb
            h = fspecial('gaussian', [kernelSize, kernelSize], sigma);
            filteredImg = imfilter(img, h, 'replicate');
            
        case 'bilateral'
            % Bilateral filter - edge-preserving smoothing
            % Use a simplified implementation
            % Degree of smoothing based on kernel size
            degreeOfSmoothing = 0.01 * kernelSize;
            spatialSigma = max(1, kernelSize / 2);
            
            % Convert to uint8 for imbilatfilt
            imgUint8 = im2uint8(img);
            filteredUint8 = imbilatfilt(imgUint8, degreeOfSmoothing, spatialSigma);
            filteredImg = im2double(filteredUint8);
            
        otherwise
            error('noiseFilter:InvalidType', ...
                  'Filter type must be ''median'', ''average'', ''gaussian'', or ''bilateral''');
    end
    
    % Ensure output is within valid range
    filteredImg = max(0, min(1, filteredImg));
    
    % Convert back to uint8 if input was uint8
    if wasUint8
        filteredImg = im2uint8(filteredImg);
    end
end
