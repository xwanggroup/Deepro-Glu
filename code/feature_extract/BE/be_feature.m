clear all
clc
input=importdata('train_data.xlsx');
data1=input.textdata(:,1);
label1=input.data(:,1);
data=data1(:,1);
label=label1(:,1);
num=numel(data);
sign0=find(label==0);
sign1=find(label==1);
sequence=[data(sign1,:);data(sign0,:)];
% [a1,a2]=size(sign0);
% sequence_label=[ones(a1,1);zeros(a1,1)];
output1=[];
input=sequence;
for i=1:num
    protein=input{i};
    output =exchange_matrix(protein);
    output1=[output1;output];
end
matrix=output1;
[num1,column]=size(matrix);
feature=[];
vector=[];
for i=1:num1
    A=matrix(i,:);
    for j=1:column
        feature=zeros(1,21);
        feature(A(j))=1;
        vector=[vector,feature];
        feature=[];
    end
    vec{i,1}=vector;
    vector=[];
    A=[];
end
matrix_train=cell2mat(vec);
save  be_train.mat matrix_train

