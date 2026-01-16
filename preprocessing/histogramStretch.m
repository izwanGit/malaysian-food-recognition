%% HISTOGRAM STRETCH - Contrast Enhancement
% Enhances image contrast using histogram stretching and CLAHE
%
% Syntax:
%   enhancedImg = histogramStretch(img)
%   enhancedImg = histogramStretch(img, method)
%
% Inputs:
%   img    - RGB image (double or uint8)
%   method - 'stretch' (default), 'clahe', or 'both'
%
% Outputs:
%   enhancedImg - Contrast-enhanced RGB image (same type as input)

function enhancedImg = histogramStretch(img, method)
    %% Input validation
    if nargin < 2
        method = 'stretch';
    end
    
    % Check if input is uint8 and convert if needed
    wasUint8 = isa(img, 'uint8');
    if wasUint8
        img = im2double(img);
    end
    
    %% Process each channel
    enhancedImg = zeros(size(img), 'like', img);
    
    switch lower(method)
        case 'stretch'
            % Apply histogram stretching using imadjust
            for c = 1:3
                channel = img(:,:,c);
                % Calculate percentile-based limits for robustness
                lowLimit = prctile(channel(:), 1);
                highLimit = prctile(channel(:), 99);
                % Apply contrast stretching
                enhancedImg(:,:,c) = imadjust(channel, [lowLimit, highLimit], [0, 1]);
            end
            
        case 'clahe'
            % Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            % Convert to LAB color space
            labImg = rgb2lab(img);
            % Apply CLAHE only to L channel
            L = labImg(:,:,1) / 100;  % Normalize L to [0,1]
            L = adapthisteq(L, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
            labImg(:,:,1) = L * 100;  % Scale back
            % Convert back to RGB
            enhancedImg = lab2rgb(labImg);
            
        case 'both'
            % Apply both methods
            % First stretch
            for c = 1:3
                channel = img(:,:,c);
                lowLimit = prctile(channel(:), 1);
                highLimit = prctile(channel(:), 99);
                enhancedImg(:,:,c) = imadjust(channel, [lowLimit, highLimit], [0, 1]);
            end
            % Then CLAHE on luminance
            labImg = rgb2lab(enhancedImg);
            L = labImg(:,:,1) / 100;
            L = adapthisteq(L, 'ClipLimit', 0.01, 'Distribution', 'uniform');
            labImg(:,:,1) = L * 100;
            enhancedImg = lab2rgb(labImg);
            
        otherwise
            error('histogramStretch:InvalidMethod', ...
                  'Method must be ''stretch'', ''clahe'', or ''both''');
    end
    
    % Ensure output is within valid range
    enhancedImg = max(0, min(1, enhancedImg));
    
    % Convert back to uint8 if input was uint8
    if wasUint8
        enhancedImg = im2uint8(enhancedImg);
    end
end
