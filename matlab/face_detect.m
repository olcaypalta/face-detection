function bboxes = face_detect(imagePath, outPath)
%FACE_DETECT Face and eye detection with the Viola-Jones detector.
%   bboxes = face_detect('../examples/astronaut.jpg')
%   bboxes = face_detect('group.jpg', 'result.jpg')
%   face_detect('webcam')   % live camera (needs MATLAB Support Package for USB Webcams)
%
%   Requires: Computer Vision Toolbox.

if nargin < 1, imagePath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'examples', 'astronaut.jpg'); end
if nargin < 2, outPath = 'result.jpg'; end

faceDet = vision.CascadeObjectDetector('FrontalFaceCART', 'MergeThreshold', 5, 'MinSize', [40 40]);
eyeDet  = vision.CascadeObjectDetector('EyePairBig');

if strcmpi(imagePath, 'webcam')
    cam = webcam();
    fig = figure('Name', 'Face detection (close window to stop)');
    while ishandle(fig)
        frame = snapshot(cam);
        imshow(annotate(frame, faceDet, eyeDet));
        drawnow;
    end
    clear cam;
    bboxes = [];
    return;
end

img = imread(imagePath);
[out, bboxes] = annotate(img, faceDet, eyeDet);
imwrite(out, outPath);
fprintf('%d face(s) found -> %s\n', size(bboxes, 1), outPath);
end

function [out, bboxes] = annotate(img, faceDet, eyeDet)
bboxes = faceDet(img);
out = img;
if isempty(bboxes), return; end
out = insertObjectAnnotation(out, 'rectangle', bboxes, 'face', 'Color', 'green', 'LineWidth', 2);
for k = 1:size(bboxes, 1)
    roi = imcrop(img, bboxes(k, :));
    eyes = eyeDet(roi);
    if ~isempty(eyes)
        eyes(:, 1:2) = eyes(:, 1:2) + bboxes(k, 1:2) - 1;
        out = insertShape(out, 'rectangle', eyes, 'Color', 'cyan');
    end
end
end
