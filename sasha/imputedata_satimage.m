clear
load satimage_data.mat

[m,d]=size(Xtr);

med = median(Xtr,'omitnan');
Xtr = fillmissing(Xtr,'constant',med);
med = median(Xte,'omitnan');
Xte = fillmissing(Xte,'constant',med);