function face_app()
%FACE_APP Yüz Tanıma Sistemi - enrollment, live multi-face recognition, sighting log.
%   face_app
%
%   Buttons: add person (camera / photo), live recognition, recognize in photo,
%   registered people (photo, ID, rename, remove) and sighting records (delete, export).
%   Detection: Viola-Jones (vision.CascadeObjectDetector). Recognition: HOG + LBP features
%   of the histogram-equalized face, nearest neighbour with cosine similarity.
%   Requires: Computer Vision Toolbox, Image Processing Toolbox and the free
%   "MATLAB Support Package for USB Webcams" (Home > Add-Ons) for camera functions.

here = fileparts(mfilename('fullpath'));
dbDir = fullfile(here, 'database');
if ~isfolder(dbDir), mkdir(dbDir); end
dbFile = fullfile(dbDir, 'people.mat');
logFile = fullfile(dbDir, 'sightings.csv');

THRESHOLD = 0.80;      % cosine similarity on HOG+LBP features
SAMPLES = 12;
COOLDOWN = 60;         % seconds between two log entries of the same person

people = struct('id', {}, 'name', {}, 'created', {}, 'feats', {}, 'photo', {});
if isfile(dbFile), s = load(dbFile); people = s.people; end
logT = table('Size', [0 4], 'VariableTypes', {'string', 'string', 'string', 'double'}, ...
    'VariableNames', {'time', 'id', 'name', 'similarity'});
if isfile(logFile), logT = readtable(logFile, 'TextType', 'string', 'Delimiter', ','); end
lastSeen = containers.Map();

detector = vision.CascadeObjectDetector('FrontalFaceCART', 'MergeThreshold', 6, 'MinSize', [60 60]);

% ---------------------------------------------------------------- main window
fig = uifigure('Name', 'Yüz Tanıma Sistemi', 'Position', [200 120 460 520], 'Color', [0.96 0.97 0.98]);
uilabel(fig, 'Text', 'Yüz Tanıma Sistemi', 'FontSize', 22, 'FontWeight', 'bold', ...
    'Position', [20 455 420 40], 'HorizontalAlignment', 'center');
summary = uilabel(fig, 'Position', [20 425 420 24], 'HorizontalAlignment', 'center', 'FontColor', [0.35 0.42 0.48]);
labels = {'Kişi Ekle (Kamera)', 'Kişi Ekle (Fotoğraftan)', 'Canlı Yüz Tanıma', ...
          'Fotoğrafta Yüz Tanıma', 'Kayıtlı Kişiler', 'Görülme Kayıtları'};
callbacks = {@enrollCamera, @enrollPhoto, @liveRecognition, @recognizePhoto, @peopleWindow, @logWindow};
for ib = 1:numel(labels)
    uibutton(fig, 'Text', labels{ib}, 'FontSize', 15, 'Position', [70 370 - (ib - 1) * 58 320 46], ...
        'ButtonPushedFcn', @(~, ~) callbacks{ib}());
end
refreshSummary();

% ================================================================ helpers
    function refreshSummary()
        summary.Text = sprintf('%d kayıtlı kişi  ·  %d görülme kaydı', numel(people), height(logT));
    end

    function saveDb()
        save(dbFile, 'people');
        writetable(logT, logFile);
    end

    function f = faceFeatures(img, box)
        g = img; if size(g, 3) == 3, g = rgb2gray(g); end
        g = histeq(imresize(imcrop(g, box), [96 96]));
        h = extractHOGFeatures(g, 'CellSize', [12 12]);
        l = extractLBPFeatures(g, 'CellSize', [24 24], 'Upright', false);
        f = [h / (norm(h) + eps), l / (norm(l) + eps)];
        f = f / norm(f);
    end

    function [idx, sim] = identify(f)
        idx = 0; sim = 0;
        for p = 1:numel(people)
            s = max(people(p).feats * f');
            if s > sim, sim = s; idx = p; end
        end
        if sim < THRESHOLD, idx = 0; end
    end

    function id = nextId()
        n = 0;
        for p = 1:numel(people), n = max(n, str2double(people(p).id(2:end))); end
        id = sprintf('P%03d', n + 1);
    end

    function addPerson(name, feats, photo)
        id = nextId();
        people(end + 1) = struct('id', id, 'name', name, 'created', string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
            'feats', feats, 'photo', photo);
        saveDb(); refreshSummary();
        uialert(fig, sprintf('%s %s numarasıyla kaydedildi (%d örnek).', name, id, size(feats, 1)), 'Kişi Ekle', 'Icon', 'success');
    end

    function txt = sentence(r)
        months = {'Ocak','Şubat','Mart','Nisan','Mayıs','Haziran','Temmuz','Ağustos','Eylül','Ekim','Kasım','Aralık'};
        days = {'Pazartesi','Salı','Çarşamba','Perşembe','Cuma','Cumartesi','Pazar'};
        d = datetime(r.time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
        dayIdx = mod(weekday(d) - 2, 7) + 1;                     % weekday(): 1 = Sunday
        txt = sprintf('%s (%s), %d %s %d %s, saat %s''de kamerada görüldü.', r.name, r.id, day(d), ...
            months{month(d)}, year(d), days{dayIdx}, char(datetime(d, 'Format', 'HH:mm:ss')));
    end

    function out = annotate(img, boxes, names, known)
        out = img;
        if isempty(boxes), return; end
        colors = repmat([230 70 60], size(boxes, 1), 1);
        colors(known, :) = repmat([40 200 90], nnz(known), 1);
        out = insertObjectAnnotation(out, 'rectangle', boxes, names, 'Color', colors, ...
            'LineWidth', 3, 'FontSize', 16, 'TextBoxOpacity', 0.9);
    end

    function name = askName()
        name = '';
        answer = inputdlg('Kişinin adı soyadı:', 'Kişi Ekle', [1 45]);
        if ~isempty(answer) && strlength(strtrim(answer{1})) > 0, name = strtrim(answer{1}); end
    end

    function cam = openCamera()
        cam = [];
        try
            cam = webcam();
        catch err
            uialert(fig, ['Kamera açılamadı: ' err.message newline ...
                'Home > Add-Ons bölümünden "MATLAB Support Package for USB Webcams" kurulmalıdır.'], 'Kamera');
        end
    end

% ================================================================ actions
    function enrollCamera()
        name = askName(); if isempty(name), return; end
        cam = openCamera(); if isempty(cam), return; end
        w = uifigure('Name', ['Kişi Ekle - ' name], 'Position', [260 120 680 560]);
        ax = uiaxes(w, 'Position', [10 50 660 500]); axis(ax, 'off');
        st = uilabel(w, 'Position', [10 10 660 30], 'FontSize', 14);
        feats = zeros(0, 0); photo = []; t0 = tic;
        while isvalid(w) && size(feats, 1) < SAMPLES
            frame = fliplr(snapshot(cam));
            boxes = detector(frame);
            if size(boxes, 1) == 1 && toc(t0) > 0.3
                f = faceFeatures(frame, boxes);
                if isempty(feats), feats = f; photo = imresize(imcrop(frame, boxes), [160 160]);
                else, feats(end + 1, :) = f; end %#ok<AGROW>
                t0 = tic;
            end
            if isempty(boxes), st.Text = 'Yüz görünmüyor - kameraya bakın';
            elseif size(boxes, 1) > 1, st.Text = 'Kayıt sırasında kadrajda yalnızca bir kişi olmalı';
            else, st.Text = sprintf('Kaydediliyor: %d/%d (başınızı hafifçe sağa-sola çevirin)', size(feats, 1), SAMPLES);
            end
            imshow(annotate(frame, boxes, repmat({name}, size(boxes, 1), 1), true(size(boxes, 1), 1)), 'Parent', ax);
            drawnow;
        end
        clear cam;
        if isvalid(w), delete(w); end
        if size(feats, 1) >= 5, addPerson(name, feats, photo);
        else, uialert(fig, 'Yeterli yüz örneği alınamadı, kayıt yapılmadı.', 'Kişi Ekle'); end
    end

    function enrollPhoto()
        [files, folder] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Resimler'}, 'Kişinin fotoğrafları', 'MultiSelect', 'on');
        if isequal(files, 0), return; end
        files = cellstr(files);
        name = askName(); if isempty(name), return; end
        feats = []; photo = [];
        for k = 1:numel(files)
            img = imread(fullfile(folder, files{k}));
            boxes = detector(img);
            if isempty(boxes), continue; end
            [~, b] = max(boxes(:, 3) .* boxes(:, 4));
            feats = [feats; faceFeatures(img, boxes(b, :))]; %#ok<AGROW>
            if isempty(photo), photo = imresize(imcrop(img, boxes(b, :)), [160 160]); end
        end
        if isempty(feats), uialert(fig, 'Fotoğraflarda yüz bulunamadı.', 'Kişi Ekle'); return; end
        addPerson(name, feats, photo);
    end

    function [boxes, names, known] = labelFaces(img, doLog)
        boxes = detector(img);
        n = size(boxes, 1); names = cell(n, 1); known = false(n, 1);
        for k = 1:n
            [idx, sim] = identify(faceFeatures(img, boxes(k, :)));
            if idx > 0
                p = people(idx); known(k) = true;
                names{k} = sprintf('%s (%s) %.2f', p.name, p.id, sim);
                if doLog && (~isKey(lastSeen, p.id) || toc(lastSeen(p.id)) > COOLDOWN)
                    lastSeen(p.id) = tic;
                    logT(end + 1, :) = {string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), string(p.id), string(p.name), sim};
                    saveDb(); refreshSummary();
                end
            else
                names{k} = 'Bilinmeyen';
            end
        end
    end

    function liveRecognition()
        if isempty(people), uialert(fig, 'Önce en az bir kişi ekleyin.', 'Canlı Yüz Tanıma'); return; end
        cam = openCamera(); if isempty(cam), return; end
        w = uifigure('Name', 'Canlı Yüz Tanıma (pencereyi kapatınca durur)', 'Position', [260 120 680 560]);
        ax = uiaxes(w, 'Position', [10 50 660 500]); axis(ax, 'off');
        st = uilabel(w, 'Position', [10 10 660 30], 'FontSize', 13);
        while isvalid(w)
            frame = fliplr(snapshot(cam));
            [boxes, names, known] = labelFaces(frame, true);
            imshow(annotate(frame, boxes, names, known), 'Parent', ax);
            last = 'Henüz kayıt yok';
            if height(logT) > 0, last = sentence(logT(end, :)); end
            if isvalid(st), st.Text = sprintf('%d yüz, %d tanındı  ·  Son kayıt: %s', size(boxes, 1), nnz(known), last); end
            drawnow;
        end
        clear cam;
    end

    function recognizePhoto()
        [file, folder] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Resimler'});
        if isequal(file, 0), return; end
        img = imread(fullfile(folder, file));
        [boxes, names, known] = labelFaces(img, false);
        w = uifigure('Name', file); ax = uiaxes(w, 'Position', [10 10 540 400]); axis(ax, 'off');
        imshow(annotate(img, boxes, names, known), 'Parent', ax);
    end

    function peopleWindow()
        w = uifigure('Name', 'Kayıtlı Kişiler', 'Position', [240 160 720 400]);
        tbl = uitable(w, 'Position', [10 50 480 340], 'ColumnName', {'Numara', 'Ad Soyad', 'Örnek', 'Kayıt tarihi'}, ...
            'ColumnWidth', {70, 180, 60, 150}, 'SelectionType', 'row', 'CellSelectionCallback', @onSelect);
        img = uiimage(w, 'Position', [510 190 200 200]);
        uibutton(w, 'Text', 'Yeniden Adlandır', 'Position', [10 10 150 30], 'ButtonPushedFcn', @(~, ~) renameSel());
        uibutton(w, 'Text', 'Seçili Kişiyi Sil', 'Position', [170 10 150 30], 'ButtonPushedFcn', @(~, ~) removeSel());
        sel = 0; fill();
        function fill()
            data = cell(numel(people), 4);
            for p = 1:numel(people)
                data(p, :) = {people(p).id, people(p).name, size(people(p).feats, 1), char(people(p).created)};
            end
            tbl.Data = data; refreshSummary();
        end
        function onSelect(~, ev)
            if isempty(ev.Indices), return; end
            sel = ev.Indices(1, 1);
            img.ImageSource = people(sel).photo;
        end
        function renameSel()
            if sel == 0, return; end
            a = inputdlg('Yeni ad:', 'Yeniden Adlandır', [1 45], {people(sel).name});
            if ~isempty(a) && strlength(strtrim(a{1})) > 0
                people(sel).name = strtrim(a{1}); saveDb(); fill();
            end
        end
        function removeSel()
            if sel == 0, return; end
            q = uiconfirm(w, sprintf('%s (%s) veritabanından çıkarılsın mı?', people(sel).name, people(sel).id), ...
                'Kişiyi Sil', 'Options', {'Evet', 'Hayır'}, 'DefaultOption', 2);
            if strcmp(q, 'Evet')
                people(sel) = []; sel = 0; saveDb(); fill();
            end
        end
    end

    function logWindow()
        w = uifigure('Name', 'Görülme Kayıtları', 'Position', [220 160 820 420]);
        tbl = uitable(w, 'Position', [10 50 800 360], 'ColumnName', {'Kayıt', 'Benzerlik'}, ...
            'ColumnWidth', {680, 90}, 'SelectionType', 'row', 'Multiselect', 'on');
        uibutton(w, 'Text', 'Seçili Kayıtları Sil', 'Position', [10 10 160 30], 'ButtonPushedFcn', @(~, ~) delSel());
        uibutton(w, 'Text', 'Tümünü Sil', 'Position', [180 10 120 30], 'ButtonPushedFcn', @(~, ~) delAll());
        uibutton(w, 'Text', 'Excel''e Aktar', 'Position', [310 10 120 30], 'ButtonPushedFcn', @(~, ~) export());
        order = []; fill();
        function fill()
            order = height(logT):-1:1;           % newest first
            data = cell(numel(order), 2);
            for k = 1:numel(order)
                data(k, :) = {sentence(logT(order(k), :)), round(logT.similarity(order(k)), 3)};
            end
            tbl.Data = data; refreshSummary();
        end
        function delSel()
            if isempty(tbl.Selection), return; end
            rows = unique(tbl.Selection(:, 1));
            q = uiconfirm(w, sprintf('%d kayıt silinsin mi?', numel(rows)), 'Kayıt Sil', 'Options', {'Evet', 'Hayır'}, 'DefaultOption', 2);
            if strcmp(q, 'Evet'), logT(order(rows), :) = []; saveDb(); fill(); end
        end
        function delAll()
            q = uiconfirm(w, 'Tüm görülme kayıtları silinsin mi?', 'Tümünü Sil', 'Options', {'Evet', 'Hayır'}, 'DefaultOption', 2);
            if strcmp(q, 'Evet'), logT(:, :) = []; saveDb(); fill(); end
        end
        function export()
            [f, p] = uiputfile('*.xlsx', 'Kaydet', sprintf('gorulme_kayitlari_%s.xlsx', char(datetime('now', 'Format', 'yyyyMMdd'))));
            if isequal(f, 0), return; end
            out = logT; out.aciklama = strings(height(out), 1);
            for k = 1:height(out), out.aciklama(k) = sentence(out(k, :)); end
            writetable(out, fullfile(p, f));
        end
    end
end
