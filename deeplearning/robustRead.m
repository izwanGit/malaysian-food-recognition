%% Robust Read Function
function img = robustRead(filename)
    try
        img = imread(filename);
        % Handle grayscale
        if size(img,3) == 1
            img = repmat(img, 1, 1, 3);
        end
        % Handle CMYK or Alpha
        if size(img,3) > 3
            img = img(:,:,1:3);
        end
        % Resize immediately to standard size
        img = imresize(img, [224 224]);
    catch
        % Return black image on failure to prevent crash
        img = zeros(224,224,3, 'uint8');
        fprintf('Warning: Failed to read %s\n', filename);
    end
end
