clear
clc

copyfile satimage_data.mat newData.mat
load newData.mat

XtestTemp = Xte;
Ytest = yte;
XtrTemp = Xtr;
Ytr = ytr;

[mTr,dTr]=size(XtrTemp);

[mTe,dTe]=size(XtestTemp);

medTr = median(XtrTemp,'omitnan');
XtrTemp = fillmissing(XtrTemp,'constant',medTr);

medTe = median(XtestTemp,'omitnan');
XtestTemp = fillmissing(XtestTemp,'constant',medTe);

XtestTemp = XtestTemp * XtrTemp';
XtrTemp = XtrTemp * XtrTemp';

fprintf('Done kernelizing\n');

tempm1 = size(XtrTemp,1);
tempm2 = size(XtestTemp,1);
tempd = size(XtrTemp,2);

Xtr = zeros(tempm1, tempd);
Xtest = zeros(tempm2, tempd);

for firstIter = 1:tempm1
    Xtr(firstIter,:) = XtrTemp(firstIter,:);
end

for secondIter = 1:tempm2
    Xtest(secondIter,:) = XtestTemp(secondIter,:);
end

fprintf('Done filling\n');

d = size(Xtr,2);

% random train/val split
idx=randperm(mTr);
Xtr=Xtr(idx,:);
ytr=double(ytr(idx));
m_train=round(0.7*mTr);
Xtrain = Xtr(1:m_train,:);
Xval = Xtr(m_train+1:end,:);
Ytrain = ytr(1:m_train,:);
Yval = ytr(m_train+1:end,:);

% extract individual classes
[mVal,dVal]=size(Xval);
[mTrain,dTrain]=size(Xtrain);
[mTest,dTest]=size(Xtest);
classCountVal = zeros(1,6);
classCountTrain = zeros(1,6);
classCountTest = zeros(1,6);

for i = 1:mTrain
    switch Ytrain(i)
        case 1
            classCountTrain(1) = classCountTrain(1) + 1;
        case 2
            classCountTrain(2) = classCountTrain(2) + 1;
        case 3
            classCountTrain(3) = classCountTrain(3) + 1;
        case 4
            classCountTrain(4) = classCountTrain(4) + 1;
        case 5
            classCountTrain(5) = classCountTrain(5) + 1;
        case 6
            classCountTrain(6) = classCountTrain(6) + 1;
    end
end

x1Train = zeros(classCountTrain(1),d);
x2Train = zeros(classCountTrain(2),d);
x3Train = zeros(classCountTrain(3),d);
x4Train = zeros(classCountTrain(4),d);
x5Train = zeros(classCountTrain(5),d);
x6Train = zeros(classCountTrain(6),d);

y1Train = zeros(classCountTrain(1),1);
y2Train = zeros(classCountTrain(2),1);
y3Train = zeros(classCountTrain(3),1);
y4Train = zeros(classCountTrain(4),1);
y5Train = zeros(classCountTrain(5),1);
y6Train = zeros(classCountTrain(6),1);

indx1Train = 1;
indx2Train = 1;
indx3Train = 1;
indx4Train = 1;
indx5Train = 1;
indx6Train = 1;

for i = 1:mTrain
    switch Ytrain(i)
        case 1
            x1Train(indx1Train,:) = Xtrain(i,:);
            y1Train(indx1Train) = Ytrain(i);
            indx1Train = indx1Train + 1;
        case 2
            x2Train(indx2Train,:) = Xtrain(i,:);
            y2Train(indx2Train) = Ytrain(i);
            indx2Train = indx2Train + 1;
        case 3
            x3Train(indx3Train,:) = Xtrain(i,:);
            y3Train(indx3Train) = Ytrain(i);
            indx3Train = indx3Train + 1;
        case 4
            x4Train(indx4Train,:) = Xtrain(i,:);
            y4Train(indx4Train) = Ytrain(i);
            indx4Train = indx4Train + 1;
        case 5
            x5Train(indx5Train,:) = Xtrain(i,:);
            y5Train(indx5Train) = Ytrain(i);
            indx5Train = indx5Train + 1;
        case 6
            x6Train(indx6Train,:) = Xtrain(i,:);
            y6Train(indx6Train) = Ytrain(i);
            indx6Train = indx6Train + 1;
    end
end

for i = 1:mTest
    switch Ytest(i)
        case 1
            classCountTest(1) = classCountTest(1) + 1;
        case 2
            classCountTest(2) = classCountTest(2) + 1;
        case 3
            classCountTest(3) = classCountTest(3) + 1;
        case 4
            classCountTest(4) = classCountTest(4) + 1;
        case 5
            classCountTest(5) = classCountTest(5) + 1;
        case 6
            classCountTest(6) = classCountTest(6) + 1;
    end
end

x1Test = zeros(classCountTest(1),d);
x2Test = zeros(classCountTest(2),d);
x3Test = zeros(classCountTest(3),d);
x4Test = zeros(classCountTest(4),d);
x5Test = zeros(classCountTest(5),d);
x6Test = zeros(classCountTest(6),d);

y1Test = zeros(classCountTest(1),1);
y2Test = zeros(classCountTest(2),1);
y3Test = zeros(classCountTest(3),1);
y4Test = zeros(classCountTest(4),1);
y5Test = zeros(classCountTest(5),1);
y6Test = zeros(classCountTest(6),1);

indx1Test = 1;
indx2Test = 1;
indx3Test = 1;
indx4Test = 1;
indx5Test = 1;
indx6Test = 1;

for i = 1:mTest
    switch Ytest(i,1)
        case 1
            x1Test(indx1Test,:) = Xtest(i,:);
            y1Test(indx1Test) = Ytest(i);
            indx1Test = indx1Test + 1;
        case 2
            x2Test(indx2Test,:) = Xtest(i,:);
            y2Test(indx2Test) = Ytest(i);
            indx2Test = indx2Test + 1;
        case 3
            x3Test(indx3Test,:) = Xtest(i,:);
            y3Test(indx3Test) = Ytest(i);
            indx3Test = indx3Test + 1;
        case 4
            x4Test(indx4Test,:) = Xtest(i,:);
            y4Test(indx4Test) = Ytest(i);
            indx4Test = indx4Test + 1;
        case 5
            x5Test(indx5Test,:) = Xtest(i,:);
            y5Test(indx5Test) = Ytest(i);
            indx5Test = indx5Test + 1;
        case 6
            x6Test(indx6Test,:) = Xtest(i,:);
            y6Test(indx6Test) = Ytest(i);
            indx6Test = indx6Test + 1;
    end
end

for i = 1:mVal
    switch Yval(i)
        case 1
            classCountVal(1) = classCountVal(1) + 1;
        case 2
            classCountVal(2) = classCountVal(2) + 1;
        case 3
            classCountVal(3) = classCountVal(3) + 1;
        case 4
            classCountVal(4) = classCountVal(4) + 1;
        case 5
            classCountVal(5) = classCountVal(5) + 1;
        case 6
            classCountVal(6) = classCountVal(6) + 1;
    end
end

x1Val = zeros(classCountVal(1),d);
x2Val = zeros(classCountVal(2),d);
x3Val = zeros(classCountVal(3),d);
x4Val = zeros(classCountVal(4),d);
x5Val = zeros(classCountVal(5),d);
x6Val = zeros(classCountVal(6),d);

y1Val = zeros(classCountVal(1),1);
y2Val = zeros(classCountVal(2),1);
y3Val = zeros(classCountVal(3),1);
y4Val = zeros(classCountVal(4),1);
y5Val = zeros(classCountVal(5),1);
y6Val = zeros(classCountVal(6),1);

indx1Val = 1;
indx2Val = 1;
indx3Val = 1;
indx4Val = 1;
indx5Val = 1;
indx6Val = 1;

for i = 1:mVal
    switch Yval(i,1)
        case 1
            x1Val(indx1Val,:) = Xval(i,:);
            y1Val(indx1Val) = Yval(i);
            indx1Val = indx1Val + 1;
        case 2
            x2Val(indx2Val,:) = Xval(i,:);
            y2Val(indx2Val) = Yval(i);
            indx2Val = indx2Val + 1;
        case 3
            x3Val(indx3Val,:) = Xval(i,:);
            y3Val(indx3Val) = Yval(i);
            indx3Val = indx3Val + 1;
        case 4
            x4Val(indx4Val,:) = Xval(i,:);
            y4Val(indx4Val) = Yval(i);
            indx4Val = indx4Val + 1;
        case 5
            x5Val(indx5Val,:) = Xval(i,:);
            y5Val(indx5Val) = Yval(i);
            indx5Val = indx5Val + 1;
        case 6
            x6Val(indx6Val,:) = Xval(i,:);
            y6Val(indx6Val) = Yval(i);
            indx6Val = indx6Val + 1;
    end
end

xTrMain = cell(1,6);
xValMain = cell(1,6);
xTeMain = cell(1,6);
yTrMain = cell(1,6);
yValMain = cell(1,6);
yTeMain = cell(1,6);

xTrMain{1} = x1Train;
xTrMain{2} = x2Train;
xTrMain{3} = x3Train;
xTrMain{4} = x4Train;
xTrMain{5} = x5Train;
xTrMain{6} = x6Train;

xValMain{1} = x1Val;
xValMain{2} = x2Val;
xValMain{3} = x3Val;
xValMain{4} = x4Val;
xValMain{5} = x5Val;
xValMain{6} = x6Val;

xTeMain{1} = x1Test;
xTeMain{2} = x2Test;
xTeMain{3} = x3Test;
xTeMain{4} = x4Test;
xTeMain{5} = x5Test;
xTeMain{6} = x6Test;

xMAIN = {xTrMain, xValMain, xTeMain};

yTrMain{1} = y1Train;
yTrMain{2} = y2Train;
yTrMain{3} = y3Train;
yTrMain{4} = y4Train;
yTrMain{5} = y5Train;
yTrMain{6} = y6Train;

yValMain{1} = y1Val;
yValMain{2} = y2Val;
yValMain{3} = y3Val;
yValMain{4} = y4Val;
yValMain{5} = y5Val;
yValMain{6} = y6Val;

yTeMain{1} = y1Test;
yTeMain{2} = y2Test;
yTeMain{3} = y3Test;
yTeMain{4} = y4Test;
yTeMain{5} = y5Test;
yTeMain{6} = y6Test;

yMAIN = {yTrMain, yValMain, yTeMain};

save newData.mat

fprintf('Done cleaning\n');